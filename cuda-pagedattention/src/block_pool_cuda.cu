#include "paged_attention_cuda.cuh"
#include <stdlib.h>
#include <string.h>

/* ===================================================================
 * Host Centralized Scheduler (Control Plane)
 * =================================================================== */

CentralScheduler scheduler_init(int total_blocks) {
    CentralScheduler sched;
    sched.total_blocks = total_blocks;
    sched.n_free = total_blocks;

    sched.free_stack = (int *)malloc((size_t)total_blocks * sizeof(int));
    sched.ref_counts = (int *)calloc((size_t)total_blocks, sizeof(int));

    /* Initialize free stack: 0, 1, ..., total_blocks - 1 */
    for (int i = 0; i < total_blocks; i++) {
        sched.free_stack[i] = i;
    }

    size_t table_bytes = (size_t)MAX_BATCH_SIZE * MAX_BLOCKS_PER_SEQ * sizeof(int);
    sched.h_block_tables = (int *)malloc(table_bytes);
    memset(sched.h_block_tables, -1, table_bytes);

    CUDA_CHECK(cudaMalloc((void **)&sched.d_block_tables, table_bytes));

    size_t lens_bytes = (size_t)MAX_BATCH_SIZE * sizeof(int);
    sched.h_seq_lens = (int *)calloc(MAX_BATCH_SIZE, sizeof(int));
    CUDA_CHECK(cudaMalloc((void **)&sched.d_seq_lens, lens_bytes));

    return sched;
}

void scheduler_free(CentralScheduler *sched) {
    if (sched->free_stack) free(sched->free_stack);
    if (sched->ref_counts) free(sched->ref_counts);
    if (sched->h_block_tables) free(sched->h_block_tables);
    if (sched->h_seq_lens) free(sched->h_seq_lens);

    if (sched->d_block_tables) cudaFree(sched->d_block_tables);
    if (sched->d_seq_lens) cudaFree(sched->d_seq_lens);

    sched->free_stack = NULL;
    sched->n_free = 0;
}

int scheduler_alloc_block(CentralScheduler *sched) {
    if (sched->n_free <= 0) return -1;
    sched->n_free--;
    int block_id = sched->free_stack[sched->n_free];
    sched->ref_counts[block_id] = 1;
    return block_id;
}

void scheduler_release_block(CentralScheduler *sched, int block_id) {
    if (block_id < 0 || block_id >= sched->total_blocks) return;
    sched->ref_counts[block_id]--;
    if (sched->ref_counts[block_id] <= 0) {
        sched->ref_counts[block_id] = 0;
        sched->free_stack[sched->n_free] = block_id;
        sched->n_free++;
    }
}

int scheduler_fork_sequence(CentralScheduler *sched, int src_seq, int dst_seq) {
    if (src_seq >= MAX_BATCH_SIZE || dst_seq >= MAX_BATCH_SIZE) return -1;

    int cur_len = sched->h_seq_lens[src_seq];
    sched->h_seq_lens[dst_seq] = cur_len;

    int n_blocks = (cur_len + BLOCK_SIZE_TOKENS - 1) / BLOCK_SIZE_TOKENS;
    for (int b = 0; b < n_blocks; b++) {
        int phys_id = sched->h_block_tables[src_seq * MAX_BLOCKS_PER_SEQ + b];
        sched->h_block_tables[dst_seq * MAX_BLOCKS_PER_SEQ + b] = phys_id;
        if (phys_id >= 0) {
            sched->ref_counts[phys_id]++; /* Shared block reference */
        }
    }
    return 0;
}

void scheduler_sync_metadata_to_device(CentralScheduler *sched, int active_seqs) {
    if (active_seqs > MAX_BATCH_SIZE) active_seqs = MAX_BATCH_SIZE;

    size_t table_bytes = (size_t)active_seqs * MAX_BLOCKS_PER_SEQ * sizeof(int);
    CUDA_CHECK(cudaMemcpy(sched->d_block_tables, sched->h_block_tables, table_bytes, cudaMemcpyHostToDevice));

    size_t lens_bytes = (size_t)active_seqs * sizeof(int);
    CUDA_CHECK(cudaMemcpy(sched->d_seq_lens, sched->h_seq_lens, lens_bytes, cudaMemcpyHostToDevice));
}

/* ===================================================================
 * GPU DRAM Block Pool (Data Plane)
 * =================================================================== */

GpuBlockPool gpu_block_pool_init(int total_blocks, int block_size, int num_local_heads, int head_dim) {
    GpuBlockPool pool;
    pool.total_blocks = total_blocks;
    pool.block_size_tokens = block_size;
    pool.num_local_heads = num_local_heads;
    pool.head_dim = head_dim;
    pool.block_stride_elements = (size_t)block_size * num_local_heads * head_dim;

    size_t total_elements = (size_t)total_blocks * pool.block_stride_elements;
    size_t total_bytes = total_elements * sizeof(float);

    CUDA_CHECK(cudaMalloc((void **)&pool.d_k_pool, total_bytes));
    CUDA_CHECK(cudaMalloc((void **)&pool.d_v_pool, total_bytes));

    CUDA_CHECK(cudaMemset(pool.d_k_pool, 0, total_bytes));
    CUDA_CHECK(cudaMemset(pool.d_v_pool, 0, total_bytes));

    return pool;
}

void gpu_block_pool_free(GpuBlockPool *pool) {
    if (pool->d_k_pool) cudaFree(pool->d_k_pool);
    if (pool->d_v_pool) cudaFree(pool->d_v_pool);
    pool->d_k_pool = NULL;
    pool->d_v_pool = NULL;
}
