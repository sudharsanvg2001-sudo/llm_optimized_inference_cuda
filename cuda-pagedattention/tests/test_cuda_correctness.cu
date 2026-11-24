#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#include "config.h"
#include "paged_attention_cuda.cuh"
#include "tensor_parallel.cuh"

static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  [PASS] %s\n", msg); } \
    else      { printf("  [FAIL] %s\n", msg); g_failures++; } \
} while (0)

/* ===================================================================
 * Reference CPU Contiguous Attention for Correctness Comparison
 * =================================================================== */
static void cpu_contiguous_attention(
    const float *q, const float *k, const float *v, float *out,
    int cur_len, int num_heads, int head_dim
) {
    float scale = 1.0f / sqrtf((float)head_dim);
    float *scores = (float *)malloc(cur_len * sizeof(float));

    for (int h = 0; h < num_heads; h++) {
        const float *qh = &q[h * head_dim];

        float max_s = -1e30f;
        for (int t = 0; t < cur_len; t++) {
            const float *kt = &k[t * num_heads * head_dim + h * head_dim];
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += qh[d] * kt[d];
            dot *= scale;
            scores[t] = dot;
            if (dot > max_s) max_s = dot;
        }

        float sum_exp = 0.0f;
        for (int t = 0; t < cur_len; t++) {
            scores[t] = expf(scores[t] - max_s);
            sum_exp += scores[t];
        }

        float *oh = &out[h * head_dim];
        for (int d = 0; d < head_dim; d++) oh[d] = 0.0f;

        for (int t = 0; t < cur_len; t++) {
            const float *vt = &v[t * num_heads * head_dim + h * head_dim];
            float w = scores[t] / sum_exp;
            for (int d = 0; d < head_dim; d++) oh[d] += w * vt[d];
        }
    }
    free(scores);
}

/* ===================================================================
 * Test 1: PagedAttention Kernel vs Contiguous Reference
 * =================================================================== */
static void test_paged_vs_contiguous(void) {
    printf("\n=== Test 1: CUDA PagedAttention vs. Contiguous Reference ===\n");
    const int CUR_LEN = 37; /* Non-multiple of block size (16) to test edge handling */
    const int BATCH = 1;

    /* Allocate and fill synthetic Q, K, V */
    size_t q_elements = (size_t)BATCH * N_HEADS * HEAD_DIM;
    size_t kv_elements = (size_t)CUR_LEN * N_HEADS * HEAD_DIM;

    float *h_q = (float *)malloc(q_elements * sizeof(float));
    float *h_k = (float *)malloc(kv_elements * sizeof(float));
    float *h_v = (float *)malloc(kv_elements * sizeof(float));
    float *h_out_ref = (float *)malloc(q_elements * sizeof(float));
    float *h_out_paged = (float *)malloc(q_elements * sizeof(float));

    for (size_t i = 0; i < q_elements; i++) h_q[i] = ((float)(rand() % 1000) / 1000.0f - 0.5f);
    for (size_t i = 0; i < kv_elements; i++) {
        h_k[i] = ((float)(rand() % 1000) / 1000.0f - 0.5f);
        h_v[i] = ((float)(rand() % 1000) / 1000.0f - 0.5f);
    }

    /* Compute CPU reference */
    cpu_contiguous_attention(h_q, h_k, h_v, h_out_ref, CUR_LEN, N_HEADS, HEAD_DIM);

    /* Setup CUDA PagedAttention */
    CentralScheduler sched = scheduler_init(100);
    GpuBlockPool pool = gpu_block_pool_init(100, BLOCK_SIZE_TOKENS, N_HEADS, HEAD_DIM);

    /* Allocate blocks for sequence 0 */
    int num_blocks = (CUR_LEN + BLOCK_SIZE_TOKENS - 1) / BLOCK_SIZE_TOKENS;
    for (int b = 0; b < num_blocks; b++) {
        int phys_id = scheduler_alloc_block(&sched);
        sched.h_block_tables[0 * MAX_BLOCKS_PER_SEQ + b] = phys_id;
    }
    sched.h_seq_lens[0] = CUR_LEN;
    scheduler_sync_metadata_to_device(&sched, 1);

    /* Copy K, V into paged layout in host buffer, then to GPU pool */
    size_t block_stride = (size_t)BLOCK_SIZE_TOKENS * N_HEADS * HEAD_DIM;
    size_t pool_bytes = (size_t)100 * block_stride * sizeof(float);
    float *h_k_pool = (float *)calloc(100 * block_stride, sizeof(float));
    float *h_v_pool = (float *)calloc(100 * block_stride, sizeof(float));

    for (int t = 0; t < CUR_LEN; t++) {
        int b = t / BLOCK_SIZE_TOKENS;
        int off = t % BLOCK_SIZE_TOKENS;
        int phys_id = sched.h_block_tables[b];
        size_t dst_off = (size_t)phys_id * block_stride + (size_t)off * N_HEADS * HEAD_DIM;
        size_t src_off = (size_t)t * N_HEADS * HEAD_DIM;
        memcpy(&h_k_pool[dst_off], &h_k[src_off], N_HEADS * HEAD_DIM * sizeof(float));
        memcpy(&h_v_pool[dst_off], &h_v[src_off], N_HEADS * HEAD_DIM * sizeof(float));
    }
    CUDA_CHECK(cudaMemcpy(pool.d_k_pool, h_k_pool, pool_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pool.d_v_pool, h_v_pool, pool_bytes, cudaMemcpyHostToDevice));

    float *d_q, *d_out;
    CUDA_CHECK(cudaMalloc((void **)&d_q, q_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_out, q_elements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_q, h_q, q_elements * sizeof(float), cudaMemcpyHostToDevice));

    /* Launch CUDA PagedAttention Kernel */
    dim3 grid(BATCH, N_HEADS);
    dim3 block(WARP_SIZE);
    paged_attention_decode_kernel<<<grid, block>>>(
        d_q, pool.d_k_pool, pool.d_v_pool,
        sched.d_block_tables, sched.d_seq_lens,
        d_out, N_HEADS, HEAD_DIM, BLOCK_SIZE_TOKENS, MAX_BLOCKS_PER_SEQ
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out_paged, d_out, q_elements * sizeof(float), cudaMemcpyDeviceToHost));

    /* Check difference */
    float max_diff = 0.0f;
    for (size_t i = 0; i < q_elements; i++) {
        float diff = fabsf(h_out_ref[i] - h_out_paged[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  max |contiguous_ref - paged_cuda_out| = %.8f\n", max_diff);
    CHECK(max_diff < 1e-4f, "CUDA PagedAttention matches contiguous reference exact to numerical tolerance");

    free(h_q); free(h_k); free(h_v); free(h_out_ref); free(h_out_paged);
    free(h_k_pool); free(h_v_pool);
    cudaFree(d_q); cudaFree(d_out);
    scheduler_free(&sched);
    gpu_block_pool_free(&pool);
}

/* ===================================================================
 * Test 2: Copy-on-Write (CoW) Prompt Sharing
 * =================================================================== */
static void test_cow_sharing(void) {
    printf("\n=== Test 2: Copy-on-Write (CoW) Memory Sharing ===\n");
    CentralScheduler sched = scheduler_init(100);

    /* Sequence 0 allocates 3 blocks for prompt */
    int b0 = scheduler_alloc_block(&sched);
    int b1 = scheduler_alloc_block(&sched);
    int b2 = scheduler_alloc_block(&sched);
    sched.h_block_tables[0 * MAX_BLOCKS_PER_SEQ + 0] = b0;
    sched.h_block_tables[0 * MAX_BLOCKS_PER_SEQ + 1] = b1;
    sched.h_block_tables[0 * MAX_BLOCKS_PER_SEQ + 2] = b2;
    sched.h_seq_lens[0] = 40; /* 40 tokens */

    CHECK(sched.ref_counts[b0] == 1 && sched.ref_counts[b1] == 1 && sched.ref_counts[b2] == 1,
          "Initial prompt blocks have ref_count = 1");

    /* Fork Sequence 1 from Sequence 0 (Parallel Sampling) */
    scheduler_fork_sequence(&sched, 0, 1);

    CHECK(sched.ref_counts[b0] == 2 && sched.ref_counts[b1] == 2 && sched.ref_counts[b2] == 2,
          "After forking for parallel sampling, prompt blocks have ref_count = 2 (Zero Memory Duplication)");

    /* Release Sequence 0 */
    for (int b = 0; b < 3; b++) {
        scheduler_release_block(&sched, sched.h_block_tables[0 * MAX_BLOCKS_PER_SEQ + b]);
    }
    CHECK(sched.ref_counts[b0] == 1 && sched.n_free == 97,
          "Releasing Sequence 0 decrements ref_counts to 1; blocks remain valid for Sequence 1");

    /* Release Sequence 1 */
    for (int b = 0; b < 3; b++) {
        scheduler_release_block(&sched, sched.h_block_tables[1 * MAX_BLOCKS_PER_SEQ + b]);
    }
    CHECK(sched.ref_counts[b0] == 0 && sched.n_free == 100,
          "Releasing Sequence 1 returns all blocks to the free stack (n_free = 100)");

    scheduler_free(&sched);
}

/* ===================================================================
 * Main Test Harness
 * =================================================================== */
int main(void) {
    printf("=====================================================\n");
    printf("   CUDA PagedAttention & Distributed Correctness Tests\n");
    printf("=====================================================\n");

    test_paged_vs_contiguous();
    test_cow_sharing();

    printf("\n=====================================================\n");
    if (g_failures == 0) {
        printf(">>> ALL TESTS PASSED SUCCESSFULLY! <<<\n");
    } else {
        printf(">>> %d TEST(S) FAILED <<<\n", g_failures);
    }
    return g_failures;
}
