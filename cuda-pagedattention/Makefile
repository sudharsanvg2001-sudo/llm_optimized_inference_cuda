
NVCC        ?= nvcc
ARCH        ?= sm_89

NVCCFLAGS   = -O3 -arch=$(ARCH) -std=c++17 -Iinclude -Xcompiler -Wall,-Wextra -Xcompiler -fopenmp
LDFLAGS     = -lcudart -lm -Xcompiler -fopenmp

SRC_CORE    = src/paged_attention_kernel.cu src/block_pool_cuda.cu src/tensor_parallel.cu
OBJ_CORE    = $(SRC_CORE:.cu=.o)

BIN_DIR     = bin
RESULTS_DIR = results

TARGETS     = $(BIN_DIR)/test_cuda_correctness \
              $(BIN_DIR)/bench_cuda_capacity \
              $(BIN_DIR)/bench_cuda_scaling \
              $(BIN_DIR)/bench_cuda_distributed

all: $(TARGETS)

# Object file pattern rule
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Executables
$(BIN_DIR)/test_cuda_correctness: $(OBJ_CORE) tests/test_cuda_correctness.o
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(OBJ_CORE) tests/test_cuda_correctness.o -o $@ $(LDFLAGS)

$(BIN_DIR)/bench_cuda_capacity: bench/bench_cuda_capacity.o
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) bench/bench_cuda_capacity.o -o $@ $(LDFLAGS)

$(BIN_DIR)/bench_cuda_scaling: $(OBJ_CORE) bench/bench_cuda_scaling.o
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(OBJ_CORE) bench/bench_cuda_scaling.o -o $@ $(LDFLAGS)

$(BIN_DIR)/bench_cuda_distributed: $(OBJ_CORE) bench/bench_cuda_distributed.o
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(OBJ_CORE) bench/bench_cuda_distributed.o -o $@ $(LDFLAGS)

# Convenient Run Targets
run-test: $(BIN_DIR)/test_cuda_correctness
	./$(BIN_DIR)/test_cuda_correctness

run-capacity: $(BIN_DIR)/bench_cuda_capacity
	./$(BIN_DIR)/bench_cuda_capacity 16384

run-scaling: $(BIN_DIR)/bench_cuda_scaling
	@mkdir -p $(RESULTS_DIR)
	./$(BIN_DIR)/bench_cuda_scaling

run-distributed: $(BIN_DIR)/bench_cuda_distributed
	./$(BIN_DIR)/bench_cuda_distributed

# Master Milestone 3 Target
milestone3: all
	@echo "\n======================================================="
	@echo "   1. RUNNING NUMERICAL & SHARING CORRECTNESS TESTS"
	@echo "======================================================="
	./$(BIN_DIR)/test_cuda_correctness
	@echo "\n======================================================="
	@echo "   2. RUNNING MEMORY CAPACITY & FRAGMENTATION BENCHMARK (16 GB)"
	@echo "======================================================="
	./$(BIN_DIR)/bench_cuda_capacity 16384
	@echo "\n======================================================="
	@echo "   3. RUNNING DISTRIBUTED TENSOR PARALLELISM BENCHMARK"
	@echo "======================================================="
	./$(BIN_DIR)/bench_cuda_distributed
	@echo "\n======================================================="
	@echo "   4. RUNNING BATCH CONCURRENCY SCALING SWEEP"
	@echo "======================================================="
	@mkdir -p $(RESULTS_DIR)
	./$(BIN_DIR)/bench_cuda_scaling
	@echo "\n>>> Milestone 3 Execution Complete! Results in $(RESULTS_DIR)/ <<<"

clean:
	rm -rf $(OBJ_CORE) tests/*.o bench/*.o $(BIN_DIR) $(RESULTS_DIR)

.PHONY: all run-test run-capacity run-scaling run-distributed milestone3 clean
