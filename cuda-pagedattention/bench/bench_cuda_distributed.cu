#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "config.h"
#include "paged_attention_cuda.cuh"
#include "tensor_parallel.cuh"


static void benchmark_distributed_tp(int world_size, int batch_size, int decode_steps) {
    printf("\n--- Running Configuration: World Size W = %d (Local Heads = %d) ---\n",
           world_size, N_HEADS / world_size);

    int local_heads = N_HEADS / world_size;
    size_t sharded_vram_bytes = (size_t)TOTAL_BLOCKS * BLOCK_SIZE_TOKENS * local_heads * HEAD_DIM * sizeof(float) * 2;
    printf("  KV Cache VRAM per Worker : %.2f MB (%.2f KB per block)\n",
           (double)sharded_vram_bytes / (1024.0 * 1024.0),
           (double)sharded_vram_bytes / TOTAL_BLOCKS / 1024.0);

    /* Initialize Distributed Workers */
    TensorParallelWorker *workers = (TensorParallelWorker *)malloc(world_size * sizeof(TensorParallelWorker));
    for (int r = 0; r < world_size; r++) {
        workers[r] = tp_worker_init(r, world_size, TOTAL_BLOCKS);
    }

    /* Centralized Scheduler */
    CentralScheduler sched = scheduler_init(TOTAL_BLOCKS);
    int blocks_per_seq = 4;
    for (int s = 0; s < batch_size; s++) {
        for (int b = 0; b < blocks_per_seq; b++) {
            int phys_id = scheduler_alloc_block(&sched);
            sched.h_block_tables[s * MAX_BLOCKS_PER_SEQ + b] = phys_id;
        }
        sched.h_seq_lens[s] = 48; /* 48 tokens context */
    }
    scheduler_sync_metadata_to_device(&sched, batch_size);

    /* Input tokens & Output Accumulator */
    size_t input_bytes = (size_t)batch_size * HIDDEN_SIZE * sizeof(float);
    float *d_x, *d_out_accum;
    CUDA_CHECK(cudaMalloc((void **)&d_x, input_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out_accum, input_bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Warmup */
    for (int r = 0; r < world_size; r++) {
        tp_worker_forward_decode(&workers[r], d_x, sched.d_block_tables, sched.d_seq_lens, batch_size, d_out_accum);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed Execution */
    CUDA_CHECK(cudaEventRecord(start));
    for (int step = 0; step < decode_steps; step++) {
        CUDA_CHECK(cudaMemset(d_out_accum, 0, input_bytes));
        for (int r = 0; r < world_size; r++) {
            tp_worker_forward_decode(&workers[r], d_x, sched.d_block_tables, sched.d_seq_lens, batch_size, d_out_accum);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    long long total_tokens = (long long)batch_size * decode_steps;
    double throughput = (double)total_tokens / ((double)total_ms / 1000.0);

    printf("  Total Elapsed Time       : %.2f ms\n", total_ms);
    printf("  Throughput Achieved      : %.2f tokens/sec\n", throughput);
    printf("  Latency per Decode Step  : %.4f ms\n", total_ms / decode_steps);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_out_accum));

    for (int r = 0; r < world_size; r++) {
        tp_worker_free(&workers[r]);
    }
    free(workers);
    scheduler_free(&sched);
}

int main(void) {
    printf("====================================================================\n");
    printf("   Megatron-LM Tensor Parallelism Scaling Sweep (vLLM Section 4.6)\n");
    printf("====================================================================\n");

    const int BATCH_SIZE = 128;
    const int DECODE_STEPS = 20;

    /* Sweep world sizes: 1, 2, 4 */
    benchmark_distributed_tp(1, BATCH_SIZE, DECODE_STEPS);
    benchmark_distributed_tp(2, BATCH_SIZE, DECODE_STEPS);
    benchmark_distributed_tp(4, BATCH_SIZE, DECODE_STEPS);

    printf("\n====================================================================\n");
    printf("Distributed Scaling Verification Completed.\n");
    return 0;
}
