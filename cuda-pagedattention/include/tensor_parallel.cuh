#ifndef TENSOR_PARALLEL_CUH
#define TENSOR_PARALLEL_CUH

#include <cuda_runtime.h>
#include "config.h"
#include "paged_attention_cuda.cuh"

/* 
 Megatron-LM Style Tensor Parallelism Worker Shard Aligned with Section 4.6 of the vLLM Paper.
 */

typedef struct {
    int rank;
    int world_size;
    int num_local_heads;
    int head_start;
    
    /* Device Memory for Local Shards */
    GpuBlockPool local_pool;
    
    /* Local Model Weights */
    float *d_w_q;       /* [HIDDEN_SIZE x (num_local_heads * HEAD_DIM)] */
    float *d_w_k;       /* [HIDDEN_SIZE x (num_local_heads * HEAD_DIM)] */
    float *d_w_v;       /* [HIDDEN_SIZE x (num_local_heads * HEAD_DIM)] */
    float *d_w_o;       /* [(num_local_heads * HEAD_DIM) x HIDDEN_SIZE] */

    /* Activation Buffers */
    float *d_q_local;   /* [batch_size x num_local_heads x HEAD_DIM] */
    float *d_k_local;
    float *d_v_local;
    float *d_attn_local;/* [batch_size x num_local_heads x HEAD_DIM] */
    float *d_partial_out;/* [batch_size x HIDDEN_SIZE] */

    cudaStream_t stream;
} TensorParallelWorker;

/* Worker Initialization and Cleanup */
TensorParallelWorker tp_worker_init(int rank, int world_size, int total_blocks);
void                 tp_worker_free(TensorParallelWorker *worker);

/* Forward pass for a batch of decode tokens on one worker */
void tp_worker_forward_decode(
    TensorParallelWorker *worker,
    const float *d_x,            /* [batch_size x HIDDEN_SIZE] input token embeddings */
    const int *d_block_tables,   /* Page tables broadcast by scheduler */
    const int *d_seq_lens,
    int batch_size,
    float *d_out_accum           /* In/out buffer where All-Reduce sums results */
);

/* All-Reduce Collective across workers */
void tp_allreduce_sum(float *d_data, int count, int rank, int world_size);

#endif
