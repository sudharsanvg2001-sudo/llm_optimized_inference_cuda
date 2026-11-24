#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "config.h"
#include "paged_attention_cuda.cuh"


static void run_scaling_benchmark(void) {
    const int DECODE_STEPS = 30;
    const int INITIAL_PROMPT_LEN = 64;

    int batch_sizes[] = {1, 4, 16, 64, 128, 256, 512, 1024, 2048};
    int n_configs = (int)(sizeof(batch_sizes) / sizeof(batch_sizes[0]));

    printf("=========================================================================================\n");
    printf("   CUDA PagedAttention Concurrency Scaling Sweep (RTX 5000 Ada - 16 GB Pool)\n");
    printf("   Hidden: %d | Heads: %d | HeadDim: %d | Pool: %d Blocks (~16 GB)\n",
           HIDDEN_SIZE, N_HEADS, HEAD_DIM, TOTAL_BLOCKS);
    printf("=========================================================================================\n");
    printf("%-10s | %-12s | %-16s | %-16s | %-18s\n",
           "Batch Size", "Total Tokens", "Time (ms)", "Throughput (tok/s)", "Latency (ms/tok)");
    printf("-----------------------------------------------------------------------------------------\n");

    FILE *csv = fopen("results/cuda_scaling.csv", "w");
    if (csv) {
        fprintf(csv, "batch_size,decode_steps,total_tokens,elapsed_ms,throughput_tok_s,latency_ms\n");
    }

    /* Initialize GPU Resources */
    CentralScheduler sched = scheduler_init(TOTAL_BLOCKS);
    GpuBlockPool pool = gpu_block_pool_init(TOTAL_BLOCKS, BLOCK_SIZE_TOKENS, N_HEADS, HEAD_DIM);

    /* Allocate device input/output buffers for maximum batch size */
    size_t q_bytes = (size_t)MAX_BATCH_SIZE * N_HEADS * HEAD_DIM * sizeof(float);
    float *d_q, *d_out;
    CUDA_CHECK(cudaMalloc((void **)&d_q, q_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, q_bytes));

    /* CUDA Timers */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int c = 0; c < n_configs; c++) {
        int batch = batch_sizes[c];
        if (batch > MAX_BATCH_SIZE) continue;

        /* Reset and allocate physical blocks for this batch */
        sched.n_free = TOTAL_BLOCKS;
        for (int i = 0; i < TOTAL_BLOCKS; i++) {
            sched.free_stack[i] = i;
            sched.ref_counts[i] = 0;
        }

        int blocks_per_seq = (INITIAL_PROMPT_LEN + DECODE_STEPS + BLOCK_SIZE_TOKENS - 1) / BLOCK_SIZE_TOKENS;
        for (int s = 0; s < batch; s++) {
            for (int b = 0; b < blocks_per_seq; b++) {
                int phys_id = scheduler_alloc_block(&sched);
                sched.h_block_tables[s * MAX_BLOCKS_PER_SEQ + b] = phys_id;
            }
            sched.h_seq_lens[s] = INITIAL_PROMPT_LEN;
        }
        scheduler_sync_metadata_to_device(&sched, batch);

        /* Warmup 2 iterations */
        dim3 grid(batch, N_HEADS);
        dim3 block(WARP_SIZE);
        for (int w = 0; w < 2; w++) {
            paged_attention_decode_kernel<<<grid, block>>>(
                d_q, pool.d_k_pool, pool.d_v_pool,
                sched.d_block_tables, sched.d_seq_lens,
                d_out, N_HEADS, HEAD_DIM, BLOCK_SIZE_TOKENS, MAX_BLOCKS_PER_SEQ
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Benchmark Decode Loop */
        CUDA_CHECK(cudaEventRecord(start));
        for (int step = 0; step < DECODE_STEPS; step++) {
            paged_attention_decode_kernel<<<grid, block>>>(
                d_q, pool.d_k_pool, pool.d_v_pool,
                sched.d_block_tables, sched.d_seq_lens,
                d_out, N_HEADS, HEAD_DIM, BLOCK_SIZE_TOKENS, MAX_BLOCKS_PER_SEQ
            );
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        long long total_tokens = (long long)batch * DECODE_STEPS;
        double throughput = (double)total_tokens / ((double)elapsed_ms / 1000.0);
        double latency_per_step_ms = (double)elapsed_ms / DECODE_STEPS;

        printf("%-10d | %-12lld | %-16.2f | %-16.2f | %-18.4f\n",
               batch, total_tokens, elapsed_ms, throughput, latency_per_step_ms);

        if (csv) {
            fprintf(csv, "%d,%d,%lld,%.2f,%.2f,%.4f\n",
                    batch, DECODE_STEPS, total_tokens, elapsed_ms, throughput, latency_per_step_ms);
        }
    }

    if (csv) {
        fclose(csv);
        printf("\nDetailed sweep results saved to: results/cuda_scaling.csv\n");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_out));

    scheduler_free(&sched);
    gpu_block_pool_free(&pool);
}

int main(void) {
    run_scaling_benchmark();
    return 0;
}
