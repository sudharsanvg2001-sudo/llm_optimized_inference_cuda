#ifndef CONFIG_H
#define CONFIG_H

/* ===================================================================
 * Model & Serving Configuration for PagedAttention (CUDA Implementation)
 * Scaled 1.1B - 1.3B Parameter Configuration (20 GB VRAM Budget)
 * Aligned with LLaMA-1B / OPT-1.3B Architecture for Published Baselines
 * =================================================================== */

#define HIDDEN_SIZE         2048     /* d_model: Scaled from 256 -> 2048 */
#define N_HEADS             64       /* Total attention heads: Scaled from 8 -> 64 */
#define HEAD_DIM            (HIDDEN_SIZE / N_HEADS) /* 32: Exactly 1 GPU Warp (32 threads) */
#define N_LAYERS            16       /* 16 layers (1.1B LLaMA) or 24 layers (1.3B OPT) */
#define MLP_HIDDEN          8192     /* 4x d_model intermediate dimension */
#define VOCAB_SIZE          32000    /* Standard vocabulary dimension for logits */

#define BLOCK_SIZE_TOKENS   16       /* Tokens per physical block ("page") */
#define TOTAL_BLOCKS        32768    /* ~8.4 GB KV pool in GPU DRAM (32,768 blocks x 256 KB) */
#define VRAM_BUDGET_MB      16384    /* 16 GB VRAM budget (16 x 1024 MB) */
#define MAX_BATCH_SIZE      2048     /* Concurrency support up to 2048 sequences */
#define MAX_SEQ_LEN         2048     /* Standard 2K context length (tokens) */
#define MAX_BLOCKS_PER_SEQ  (MAX_SEQ_LEN / BLOCK_SIZE_TOKENS) /* 128 blocks */

#define WARP_SIZE           32       /* NVIDIA GPU hardware warp dimension */

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#endif 
