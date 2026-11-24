#ifndef PAGED_ATTENTION_CUDA_CUH
#define PAGED_ATTENTION_CUDA_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include "config.h"

/*Physical Block Pool Management (GPU DRAM Data Plane)
*/

typedef struct {
    float *d_k_pool;          /* [TOTAL_BLOCKS x BLOCK_SIZE_TOKENS x local_heads x HEAD_DIM] */
    float *d_v_pool;          /* [TOTAL_BLOCKS x BLOCK_SIZE_TOKENS x local_heads x HEAD_DIM] */
    int total_blocks;
    int block_size_tokens;
    int num_local_heads;
    int head_dim;
    size_t block_stride_elements; /* BLOCK_SIZE_TOKENS * num_local_heads * HEAD_DIM */
} GpuBlockPool;

/*
  Centralized Scheduler Memory Manager (Host Control Plane)
 */

typedef struct {
    int *free_stack;          /* Stack of free physical block IDs */
    int n_free;               /* Number of currently available blocks */
    int *ref_counts;          /* Reference count per block for Copy-on-Write [TOTAL_BLOCKS] */
    int total_blocks;

    /* Global page table on host, mirrored to GPU DRAM */
    int *h_block_tables;      /* [MAX_BATCH_SIZE x MAX_BLOCKS_PER_SEQ] */
    int *d_block_tables;
    int *h_seq_lens;          /* [MAX_BATCH_SIZE] */
    int *d_seq_lens;
} CentralScheduler;

/* Host Manager API */
CentralScheduler scheduler_init(int total_blocks);
void             scheduler_free(CentralScheduler *sched);
int              scheduler_alloc_block(CentralScheduler *sched);
void             scheduler_release_block(CentralScheduler *sched, int block_id);
int              scheduler_fork_sequence(CentralScheduler *sched, int src_seq, int dst_seq);
void             scheduler_sync_metadata_to_device(CentralScheduler *sched, int active_seqs);

/* GPU Pool API */
GpuBlockPool gpu_block_pool_init(int total_blocks, int block_size, int num_local_heads, int head_dim);
void         gpu_block_pool_free(GpuBlockPool *pool);

/*
  CUDA Kernel Declarations
 */

/* Warp-level PagedAttention for decode phase (1 query token per sequence) */
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
);

/* Fused KV Cache Append Kernel */
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
);

/* Batched Copy-on-Write Block Cloning Kernel */
__global__ void batched_block_copy_kernel(
    const float* __restrict__ d_src_pool,
    float*       __restrict__ d_dst_pool,
    const int*   __restrict__ d_src_block_ids,
    const int*   __restrict__ d_dst_block_ids,
    int num_blocks_to_copy,
    size_t block_elements
);

/* Device Warp Reduction Utility */
__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = (other > val) ? other : val;
    }
    return val;
}

#endif
