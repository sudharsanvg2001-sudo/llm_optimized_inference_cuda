#include "tensor_parallel.cuh"
#include <stdlib.h>
#include <stdio.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

/* ===================================================================
 * 2D Tiled Matrix Multiplication (GEMM) Kernel for Projections
 * Computes: C [M x N] = A [M x K] * B [K x N]
 * =================================================================== */
#define TILE_DIM 16

__global__ void gemm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K
) {
    __shared__ float s_a[TILE_DIM][TILE_DIM];
    __shared__ float s_b[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = 0.0f;

    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_DIM + threadIdx.x;
        int b_row = t * TILE_DIM + threadIdx.y;

        s_a[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        s_b[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            acc += s_a[threadIdx.y][k] * s_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

static void launch_gemm(const float *d_a, const float *d_b, float *d_c,
                        int M, int N, int K, cudaStream_t stream) {
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    dim3 block(TILE_DIM, TILE_DIM);
    gemm_tiled_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, M, N, K);
}

/* Helper to initialize pseudo-random weights on host and copy to device */
static float* alloc_and_fill_device_weights(size_t elements, unsigned int seed) {
    float *h_w = (float *)malloc(elements * sizeof(float));
    for (size_t i = 0; i < elements; i++) {
        seed = seed * 1103515245u + 12345u;
        h_w[i] = ((float)(seed & 0x7fff) / 32767.0f - 0.5f) * 0.05f;
    }
    float *d_w;
    CUDA_CHECK(cudaMalloc((void **)&d_w, elements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, elements * sizeof(float), cudaMemcpyHostToDevice));
    free(h_w);
    return d_w;
}

/* ===================================================================
 * Tensor Parallel Worker Lifecycle
 * =================================================================== */

TensorParallelWorker tp_worker_init(int rank, int world_size, int total_blocks) {
    TensorParallelWorker w;
    w.rank = rank;
    w.world_size = world_size;
    w.num_local_heads = N_HEADS / world_size;
    w.head_start = rank * w.num_local_heads;

    CUDA_CHECK(cudaStreamCreate(&w.stream));

    /* Initialize local sharded block pool */
    w.local_pool = gpu_block_pool_init(total_blocks, BLOCK_SIZE_TOKENS, w.num_local_heads, HEAD_DIM);

    int local_dim = w.num_local_heads * HEAD_DIM;

    /* Column-parallel weight shards: [HIDDEN_SIZE x local_dim] */
    w.d_w_q = alloc_and_fill_device_weights((size_t)HIDDEN_SIZE * local_dim, 100 + rank * 10 + 1);
    w.d_w_k = alloc_and_fill_device_weights((size_t)HIDDEN_SIZE * local_dim, 100 + rank * 10 + 2);
    w.d_w_v = alloc_and_fill_device_weights((size_t)HIDDEN_SIZE * local_dim, 100 + rank * 10 + 3);

    /* Row-parallel weight shard: [local_dim x HIDDEN_SIZE] */
    w.d_w_o = alloc_and_fill_device_weights((size_t)local_dim * HIDDEN_SIZE, 100 + rank * 10 + 4);

    /* Allocate activation buffers */
    size_t local_act_bytes = (size_t)MAX_BATCH_SIZE * local_dim * sizeof(float);
    CUDA_CHECK(cudaMalloc((void **)&w.d_q_local, local_act_bytes));
    CUDA_CHECK(cudaMalloc((void **)&w.d_k_local, local_act_bytes));
    CUDA_CHECK(cudaMalloc((void **)&w.d_v_local, local_act_bytes));
    CUDA_CHECK(cudaMalloc((void **)&w.d_attn_local, local_act_bytes));

    size_t partial_out_bytes = (size_t)MAX_BATCH_SIZE * HIDDEN_SIZE * sizeof(float);
    CUDA_CHECK(cudaMalloc((void **)&w.d_partial_out, partial_out_bytes));

    return w;
}

void tp_worker_free(TensorParallelWorker *w) {
    gpu_block_pool_free(&w->local_pool);

    if (w->d_w_q) cudaFree(w->d_w_q);
    if (w->d_w_k) cudaFree(w->d_w_k);
    if (w->d_w_v) cudaFree(w->d_w_v);
    if (w->d_w_o) cudaFree(w->d_w_o);

    if (w->d_q_local) cudaFree(w->d_q_local);
    if (w->d_k_local) cudaFree(w->d_k_local);
    if (w->d_v_local) cudaFree(w->d_v_local);
    if (w->d_attn_local) cudaFree(w->d_attn_local);
    if (w->d_partial_out) cudaFree(w->d_partial_out);

    cudaStreamDestroy(w->stream);
}

/* ===================================================================
 * Forward Pass with Tensor Parallelism
 * =================================================================== */

void tp_worker_forward_decode(
    TensorParallelWorker *worker,
    const float *d_x,            /* [batch_size x HIDDEN_SIZE] */
    const int *d_block_tables,   /* Shared block tables */
    const int *d_seq_lens,
    int batch_size,
    float *d_out_accum           /* In/out buffer for All-Reduce sum */
) {
    int local_dim = worker->num_local_heads * HEAD_DIM;

    /* 1. Column Parallel Projections for Local Heads: Q, K, V */
    launch_gemm(d_x, worker->d_w_q, worker->d_q_local, batch_size, local_dim, HIDDEN_SIZE, worker->stream);
    launch_gemm(d_x, worker->d_w_k, worker->d_k_local, batch_size, local_dim, HIDDEN_SIZE, worker->stream);
    launch_gemm(d_x, worker->d_w_v, worker->d_v_local, batch_size, local_dim, HIDDEN_SIZE, worker->stream);

    /* 2. Write new K, V tokens into the sharded local block pool */
    dim3 grid_append(batch_size, worker->num_local_heads);
    dim3 block_append(WARP_SIZE);
    paged_kv_append_kernel<<<grid_append, block_append, 0, worker->stream>>>(
        worker->d_k_local,
        worker->d_v_local,
        worker->local_pool.d_k_pool,
        worker->local_pool.d_v_pool,
        d_block_tables,
        d_seq_lens,
        worker->num_local_heads,
        HEAD_DIM,
        BLOCK_SIZE_TOKENS,
        MAX_BLOCKS_PER_SEQ
    );

    /* 3. Execute Local PagedAttention Decode Kernel */
    dim3 grid_attn(batch_size, worker->num_local_heads);
    dim3 block_attn(WARP_SIZE);
    paged_attention_decode_kernel<<<grid_attn, block_attn, 0, worker->stream>>>(
        worker->d_q_local,
        worker->local_pool.d_k_pool,
        worker->local_pool.d_v_pool,
        d_block_tables,
        d_seq_lens,
        worker->d_attn_local,
        worker->num_local_heads,
        HEAD_DIM,
        BLOCK_SIZE_TOKENS,
        MAX_BLOCKS_PER_SEQ
    );

    /* 4. Row Parallel Output Projection: local_dim -> HIDDEN_SIZE */
    launch_gemm(worker->d_attn_local, worker->d_w_o, worker->d_partial_out,
                batch_size, HIDDEN_SIZE, local_dim, worker->stream);

    CUDA_CHECK(cudaStreamSynchronize(worker->stream));

    /* 5. Accumulate into d_out_accum for All-Reduce */
    #pragma omp critical(tp_accum_lock)
    {
        /* Simple elementwise addition of partial outputs */
        size_t total_elements = (size_t)batch_size * HIDDEN_SIZE;
        float *h_part = (float *)malloc(total_elements * sizeof(float));
        float *h_accum = (float *)malloc(total_elements * sizeof(float));

        CUDA_CHECK(cudaMemcpy(h_part, worker->d_partial_out, total_elements * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_accum, d_out_accum, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < total_elements; i++) {
            h_accum[i] += h_part[i];
        }

        CUDA_CHECK(cudaMemcpy(d_out_accum, h_accum, total_elements * sizeof(float), cudaMemcpyHostToDevice));
        free(h_part);
        free(h_accum);
    }
}

/* ===================================================================
 * All-Reduce Collective Communication
 * =================================================================== */
void tp_allreduce_sum(float *d_data, int count, int rank, int world_size) {
    (void)rank;
    (void)world_size;
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, d_data, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
    /* When running standalone in-process, accumulation is performed directly */
    (void)d_data;
    (void)count;
#endif
}
