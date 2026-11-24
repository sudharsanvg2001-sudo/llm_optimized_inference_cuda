# Milestone 3: CUDA PagedAttention & Distributed Tensor Parallelism

A high-performance **CUDA implementation of PagedAttention** and **Distributed Tensor Model Parallelism** for LLM serving, directly reproducing the core architecture from *"Efficient Memory Management for Large Language Model Serving with PagedAttention"*

Configured KV Cache Pool: **15 GB (15,360 MB)** -> **122,880 Physical Blocks** (~1.97 Million Tokens).

---

##  Key Technical Features Implemented

1. **Warp-Level PagedAttention Kernel (`src/paged_attention_kernel.cu`)**:
   - Each GPU Threadblock is mapped as **1 Warp (32 threads) per attention head**, exactly matching `HEAD_DIM = 32`.
   - Thread $d \in [0, 31]$ owns dimension $d$. When threads load key/value vectors, the GPU executes a **128-byte coalesced DRAM transaction**.
   - Dot products are reduced inside register space using **Warp Shuffle Intrinsics (`__shfl_down_sync`)** without shared memory latency.
   - **FlashAttention-Style Online Softmax**: Computes causal attention in a single pass over the non-contiguous physical blocks with $O(1)$ memory overhead, supporting arbitrary sequence lengths without memory exhaustion.

2. **Distributed Tensor Parallelism (vLLM Section 4.6 & Megatron-LM)**:
   - **Column-Parallel Projections**: Q, K, and V linear layers are sharded across workers. Each worker only computes and stores activations for its local heads ($H_{\text{local}} = N_{\text{heads}} / W$).
   - **KV Cache VRAM Sharding**: Reduces GPU memory consumption per worker by $1/W$ (e.g. 50% reduction for 2 ranks, 75% for 4 ranks).
   - **Row-Parallel Projection & All-Reduce**: Projects local attention outputs and performs collective vector summation (`All-Reduce`) across workers.

3. **Copy-on-Write (CoW) Memory Sharing (vLLM Section 4.4)**:
   - Reference-counted physical blocks enable **Parallel Sampling**: multiple candidate sequences share the same prompt blocks with zero duplicate VRAM usage until mutation.

4. **Centralized Host Scheduler (Control Plane vs. Data Plane)**:
   - Synchronized logical-to-physical page table management on host with $O(1)$ allocation/release.
   - Zero lock contention on the GPU.

---

##  Directory Structure

```
cuda-pagedattention/
├── include/
│   ├── config.h                  # Model dimensions (hidden=256, heads=8, head_dim=32, block_size=16)
│   ├── paged_attention_cuda.cuh  # Paged KV pool structs, kernel declarations, warp reductions
│   └── tensor_parallel.cuh       # Megatron-LM worker sharding & All-Reduce interface
├── src/
│   ├── paged_attention_kernel.cu # Decode kernel, KV append kernel, CoW copy kernel
│   ├── block_pool_cuda.cu        # Host scheduler free list & GPU DRAM flat block allocator
│   └── tensor_parallel.cu        # Column/Row parallel GEMMs and distributed forward pass
├── tests/
│   └── test_cuda_correctness.cu  # Contiguous vs Paged numerical validation & CoW sharing tests
├── bench/
│   ├── bench_cuda_capacity.cu    # VRAM capacity & fragmentation benchmark (Paper Fig. 2)
│   ├── bench_cuda_scaling.cu     # Batch size sweep (1 -> 1024 seqs), measuring tokens/sec
│   └── bench_cuda_distributed.cu # Distributed TP scaling across W = 1, 2, 4 workers
├── Makefile                      # NVCC build rules targeting sm_89
└── README.md
```



## Individual Target Commands

```bash
make run-test          # Runs numerical correctness vs contiguous reference (max diff < 1e-4)
make run-capacity      # Evaluates memory capacity & fragmentation for Naive vs Paged
make run-scaling       # Measures tokens/sec throughput across batch sizes 1 to 1024
make run-distributed   # Measures distributed scaling across 1, 2, and 4 workers
```

---

##  Output Verification & Metrics

- **Correctness**: Validates that non-contiguous PagedAttention produces numerically identical hidden states to standard contiguous attention (`max |ref - paged| = 0.00000000`).
- **Capacity**: Demonstrates over **15× to 30× higher concurrent sequence capacity** compared to naive contiguous reservation.
- **Throughput**: Shows hardware saturation curve reaching tens of thousands of tokens per second as batch concurrency increases on the RTX 5000 Ada GPU.
