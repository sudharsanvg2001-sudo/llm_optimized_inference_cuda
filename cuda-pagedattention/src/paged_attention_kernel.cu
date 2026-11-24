#include "paged_attention_cuda.cuh"
#include <math.h>

/* ===================================================================
 * 1. Decode PagedAttention Kernel with Online Softmax (Warp-Level)
 *
 * Each Threadblock is 1 Warp (32 threads) dedicated to (seq_id, head_id).
 * Thread d (0 <= d < 32) handles dimension d of HEAD_DIM = 32.
 * Uses FlashAttention-style Online Softmax to process variable context
 * lengths without temporary DRAM or shared-memory score buffers.
 * =================================================================== */
__global__ void paged_attention_decode_kernel(
    const float* __restrict__ d_q,             /* [batch_size x local_heads x head_dim] */
    const float* __restrict__ d_k_pool,        /* [total_blocks x block_size x local_heads x head_dim] */
    const float* __restrict__ d_v_pool,        /* [total_blocks x block_size x local_heads x head_dim] */
    const int*   __restrict__ d_block_tables,  /* [max_batch x max_blocks_per_seq] */
    const int*   __restrict__ d_seq_lens,      /* [batch_size] */
    float*       __restrict__ d_out,           /* [batch_size x local_heads x head_dim] */
    int num_local_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq
) {
    int seq_id  = blockIdx.x;
    int head_id = blockIdx.y;
    int tid     = threadIdx.x; /* 0 <= tid < 32 (WARP_SIZE) */

    int cur_len = d_seq_lens[seq_id];
    if (cur_len <= 0) return;

    /* Stride multipliers for flat block pool indexing */
    size_t head_stride  = head_dim;
    size_t token_stride = (size_t)num_local_heads * head_stride;
    size_t block_stride = (size_t)block_size * token_stride;

    /* 1. Load Query Vector Q for this sequence and head into registers */
    size_t q_offset = ((size_t)seq_id * num_local_heads + head_id) * head_dim + tid;
    float q_reg = d_q[q_offset];

    float scale = rsqrtf((float)head_dim);

    /* 2. Online Softmax State */
    float m_prev = -1e30f; /* running maximum */
    float l_prev = 0.0f;   /* running denominator sum */
    float acc_v  = 0.0f;   /* running weighted value accumulator for dimension tid */

    int total_blocks = (cur_len + block_size - 1) / block_size;

    /* 3. Loop over all historical physical blocks */
    for (int b = 0; b < total_blocks; b++) {
        int physical_block_id = d_block_tables[seq_id * max_blocks_per_seq + b];
        size_t block_base = (size_t)physical_block_id * block_stride + (size_t)head_id * head_stride + tid;

        int tokens_in_this_block = block_size;
        if (b == total_blocks - 1) {
            int rem = cur_len % block_size;
            if (rem != 0) tokens_in_this_block = rem;
        }

        /* Iterate through tokens in the current block */
        for (int t = 0; t < tokens_in_this_block; t++) {
            size_t token_mem_offset = block_base + (size_t)t * token_stride;

            /* Coalesced 128-byte read of key vector K */
            float k_reg = d_k_pool[token_mem_offset];

            /* Compute partial dot product Q . K */
            float dot = q_reg * k_reg;

            /* Warp-level parallel reduction across all 32 threads */
            float score = warp_reduce_sum(dot) * scale;

            /* Broadcast reduced scalar score to all 32 threads */
            score = __shfl_sync(0xffffffff, score, 0);

            /* Coalesced 128-byte read of value vector V */
            float v_reg = d_v_pool[token_mem_offset];

            /* Online Softmax Update */
            float m_curr = fmaxf(m_prev, score);
            float alpha  = expf(m_prev - m_curr);
            float beta   = expf(score - m_curr);

            l_prev = l_prev * alpha + beta;
            acc_v  = acc_v  * alpha + beta * v_reg;
            m_prev = m_curr;
        }
    }

    /* 4. Final normalization and write to global output DRAM */
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    float final_out = acc_v * inv_l;

    size_t out_offset = ((size_t)seq_id * num_local_heads + head_id) * head_dim + tid;
    d_out[out_offset] = final_out;
}

/* ===================================================================
 * 2. Fused KV Cache Append Kernel
 * Writes newly generated (K, V) projections into the assigned
 * physical block in GPU DRAM at position cur_pos.
 * =================================================================== */
__global__ void paged_kv_append_kernel(
    const float* __restrict__ d_k_new,         /* [batch_size x local_heads x head_dim] */
    const float* __restrict__ d_v_new,         /* [batch_size x local_heads x head_dim] */
    float*       __restrict__ d_k_pool,
    float*       __restrict__ d_v_pool,
    const int*   __restrict__ d_block_tables,
    const int*   __restrict__ d_seq_lens,
    int num_local_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq
) {
    int seq_id  = blockIdx.x;
    int head_id = blockIdx.y;
    int tid     = threadIdx.x; /* 0 <= tid < 32 */

    int cur_pos = d_seq_lens[seq_id];
    int logical_block = cur_pos / block_size;
    int token_offset  = cur_pos % block_size;

    int physical_block_id = d_block_tables[seq_id * max_blocks_per_seq + logical_block];

    size_t head_stride  = head_dim;
    size_t token_stride = (size_t)num_local_heads * head_stride;
    size_t block_stride = (size_t)block_size * token_stride;

    size_t pool_offset = (size_t)physical_block_id * block_stride +
                         (size_t)token_offset * token_stride +
                         (size_t)head_id * head_stride + tid;

    size_t new_offset = ((size_t)seq_id * num_local_heads + head_id) * head_dim + tid;

    d_k_pool[pool_offset] = d_k_new[new_offset];
    d_v_pool[pool_offset] = d_v_new[new_offset];
}

/* ===================================================================
 * 3. Batched Copy-on-Write Block Cloning Kernel
 * Clones physical blocks for parallel sampling / beam search branching.
 * =================================================================== */
__global__ void batched_block_copy_kernel(
    const float* __restrict__ d_src_pool,
    float*       __restrict__ d_dst_pool,
    const int*   __restrict__ d_src_block_ids,
    const int*   __restrict__ d_dst_block_ids,
    int num_blocks_to_copy,
    size_t block_elements
) {
    int block_idx = blockIdx.y;
    if (block_idx >= num_blocks_to_copy) return;

    int src_id = d_src_block_ids[block_idx];
    int dst_id = d_dst_block_ids[block_idx];

    size_t base_src = (size_t)src_id * block_elements;
    size_t base_dst = (size_t)dst_id * block_elements;

    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < block_elements;
         i += (size_t)gridDim.x * blockDim.x) {
        d_dst_pool[base_dst + i] = d_src_pool[base_src + i];
    }
}
