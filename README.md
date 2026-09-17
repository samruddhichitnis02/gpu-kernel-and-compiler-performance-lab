# GPU Kernel and Compiler Performance Lab

A GPU systems project focused on understanding how machine learning workloads are transformed into efficient GPU execution.

The project studies numerical precision, matrix multiplication, memory bandwidth, attention mechanisms, operator fusion, custom kernels, and compiler assisted execution using PyTorch and CUDA enabled NVIDIA hardware.

## Project Overview

Modern AI workloads are constrained by two major factors:

1. How quickly the GPU performs mathematical operations.
2. How quickly the GPU moves data through memory.

This project evaluates both limitations through reproducible experiments on an NVIDIA GeForce RTX 3060 Laptop GPU. It connects high level machine learning operations with the lower level execution behavior that determines GPU performance.

The work is motivated by AI compiler and machine learning systems engineering, where compiler transformations, kernel implementations, memory reuse, numerical precision, and hardware utilization directly affect the speed of training and inference workloads.

## Objectives

- Measure GPU throughput across FP32, TF32, FP16, and BF16 precision.
- Compare achieved throughput with theoretical hardware limits.
- Analyze how matrix size affects GPU utilization.
- Identify compute bound and memory bound workloads.
- Measure the quadratic memory cost of naive attention.
- Compare naive attention with fused or memory efficient attention.
- Analyze operator fusion and compiler assisted execution.
- Measure GPU behavior during sustained workloads.
- Preserve hardware provenance for every measurement.
- Produce reproducible benchmark results, logs, and figures.

## Hardware and Software

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- Architecture: NVIDIA Ampere
- VRAM: 6 GB
- GPU UUID: `GPU-20606811-3bb2-e6b5-ec74-b477d9c1d1d8`
- Driver Version: 610.88
- PyTorch Version: 2.6.0
- PyTorch CUDA Runtime: 12.4
- BF16 Support: Available
- Default GPU Power Limit: 80 W

All measurements in this repository were collected on the GPU identified by the UUID above.

## Experiments

### 1. Precision Aware Matrix Multiplication

Dense square matrix multiplication is benchmarked using:

- FP32
- TF32
- FP16
- BF16

The following matrix sizes are evaluated:

```text
1024 × 1024
4096 × 4096
8192 × 8192
16384 × 16384
```

Each configuration includes GPU warm up iterations followed by repeated timed executions.

The benchmark records:

- Average execution latency
- Achieved TFLOPS
- Peak GPU memory allocation
- Numerical precision
- Matrix size
- Repetition count
- GPU UUID
- Execution status

The number of floating point operations for a square matrix multiplication is approximated as:

```text
FLOPs = 2 × N³
```

Achieved throughput is calculated as:

```text
TFLOPS = (2 × N³) / execution_time / 10¹²
```

The precision comparison shows how reduced precision and Tensor Core execution affect throughput, memory consumption, and workload scaling.

### 2. Workload Scaling and GPU Utilization

The benchmark evaluates how matrix size affects GPU utilization.

Small matrices may not provide enough parallel work to fully occupy the GPU. As the matrix size increases, more GPU execution resources become active and throughput approaches a performance plateau.

The performance curve is visualized in:

```text
figures/achieved_tflops.png
```

The plot compares achieved TFLOPS across matrix sizes for all tested precisions.

### 3. Theoretical Peak Comparison

Measured throughput is compared against the theoretical peak throughput reported for the GPU and precision being evaluated.

The comparison includes:

- Achieved TFLOPS
- Theoretical peak TFLOPS
- Percentage of theoretical peak
- Matrix size
- Numerical precision
- GPU UUID

The percentage of theoretical peak is calculated as:

```text
Percentage of theoretical peak =
achieved TFLOPS / theoretical peak TFLOPS × 100
```

The difference between theoretical and achieved performance is analyzed using factors such as:

- Kernel launch overhead
- Matrix dimensions
- GPU utilization
- Memory movement
- Software overhead
- Power limits
- Thermal behavior
- Numerical precision

### 4. Compute Bound and Memory Bound Workloads

Two workload types are evaluated:

- A large matrix multiplication representing a compute bound operation.
- A large elementwise operation representing a memory bound operation.

The memory bound benchmark measures effective memory bandwidth:

```text
Effective bandwidth =
bytes transferred / execution time
```

The results are compared with the GPU specified memory bandwidth.

Arithmetic intensity is calculated as:

```text
Arithmetic intensity = FLOPs / bytes transferred
```

The arithmetic intensity analysis determines whether each operation is limited primarily by available computation or available memory bandwidth.

This provides an experimental connection to the GPU roofline model.

### 5. Naive Scaled Dot Product Attention

Scaled dot product attention is implemented using query, key, and value tensors:

```text
Attention(Q, K, V) =
softmax(QKᵀ / √d) V
```

The naive implementation materializes the complete sequence by sequence attention matrix.

The experiment measures:

- Forward latency
- Peak GPU memory
- Sequence length
- Batch size
- Head dimension
- Out of memory boundary
- Memory growth as sequence length increases

The attention matrix has dimensions:

```text
sequence length × sequence length
```

Therefore, its memory requirement grows quadratically with sequence length.

The measured memory curve is fitted to confirm the quadratic component using observed data rather than assuming quadratic behavior.

### 6. Fused and Memory Efficient Attention

The naive attention implementation is compared with a fused or memory efficient attention implementation available in the software stack.

The comparison evaluates:

- Forward latency
- Peak GPU memory
- Sequence length limits
- Out of memory boundary
- Speedup relative to naive attention
- Memory savings

Fused attention reduces unnecessary intermediate tensor materialization and can combine multiple operations into fewer GPU kernels.

This reduces:

- Global memory traffic
- Kernel launch overhead
- Peak memory requirements
- Unnecessary movement of intermediate tensors

### 7. Compilation and Operator Fusion

The project compares eager execution with compiler assisted execution for a representative machine learning workload.

The workload includes multiple operations that can potentially be fused:

```text
matrix multiplication → bias addition → activation
```

The comparison measures:

- Eager execution latency
- Compilation time
- Compiled execution latency
- Repeated execution latency after compilation
- Peak memory usage
- Numerical equivalence
- Execution speedup

Operator fusion can reduce:

- Kernel launch overhead
- Intermediate tensor allocation
- Global memory reads
- Global memory writes
- Movement of data between separate operations

This experiment demonstrates how compiler transformations can improve AI workload performance without changing the mathematical result.

### 8. GPU Kernel Performance

The project analyzes how GPU execution changes with:

- Numerical precision
- Matrix dimensions
- Memory access patterns
- Kernel launch configuration
- Intermediate tensor materialization
- Operation fusion
- Compilation overhead
- GPU utilization

Where supported by the environment, framework generated or custom kernels are compared with eager framework implementations.

Potential kernel workloads include:

- Elementwise addition
- Fused bias and activation
- Tiled matrix multiplication
- Softmax
- Attention suboperations

The analysis focuses on the relationship between algorithm structure, memory access, tiling, kernel launch behavior, and observed performance.

### 9. Sustained GPU Load

A sustained compute workload is executed for 20 minutes while sampling:

- GPU clock
- Memory clock
- Temperature
- Power draw
- GPU utilization
- Elapsed time

The thermal experiment evaluates:

- Peak throughput during the first 30 seconds
- Steady state throughput during the final five minutes
- Clock changes over time
- Temperature behavior
- Power behavior
- Throttling
- Throttle onset time

This demonstrates why short benchmarks may overestimate the performance available during long training and inference workloads.

## Repository Structure

```text
.
├── nvidia-smi-q.txt
├── scripts/
│   ├── matmul_benchmark.py
│   ├── plot_tflops.py
│   ├── bandwidth_benchmark.py
│   ├── attention_benchmark.py
│   ├── fused_attention_benchmark.py
│   ├── compile_comparison.py
│   └── thermal_load.py
├── results/
│   ├── matmul_results.csv
│   ├── bandwidth_results.csv
│   ├── attention_results.csv
│   ├── fused_attention_results.csv
│   ├── compile_results.csv
│   └── thermal_results.csv
├── figures/
│   ├── achieved_tflops.png
│   ├── bandwidth_comparison.png
│   ├── attention_memory.png
│   ├── attention_latency.png
│   ├── compilation_speedup.png
│   └── thermal_behavior.png
├── logs/
│   └── thermal_load.csv
├── METRICS.md
├── RUN_LOG.txt
├── AI_USE.md
└── README.md
```

## Reproducibility

Run all commands from the repository root:

```powershell
python scripts\matmul_benchmark.py
python scripts\plot_tflops.py
python scripts\bandwidth_benchmark.py
python scripts\attention_benchmark.py
python scripts\fused_attention_benchmark.py
python scripts\compile_comparison.py
python scripts\thermal_load.py
```

The raw measurements are stored as CSV files.

The generated figures are stored in the `figures` directory.

The GPU hardware report is stored in:

```text
nvidia-smi-q.txt
```

All experiments include the GPU UUID:

```text
GPU-20606811-3bb2-e6b5-ec74-b477d9c1d1d8
```

The experiment methodology, metrics, observations, and conclusions are documented in:

```text
METRICS.md
```

## Engineering Principles

This project follows these engineering principles:

- Measure real execution instead of relying only on theoretical specifications.
- Record hardware provenance for every experiment.
- Separate warm up time from steady state execution time.
- Use repeated measurements to reduce timing variance.
- Report successful runs and out of memory results.
- Compare latency, memory usage, bandwidth, and throughput.
- Preserve raw measurements in machine readable formats.
- Analyze both performance and failure boundaries.
- Treat unsupported software features as valid tooling findings.
- Keep benchmark scripts reproducible and easy to rerun.
- Avoid making compiler claims that are not supported by implemented experiments.

## Engineering Findings

The experiments demonstrate several important GPU systems principles:

- Reduced precision can increase matrix multiplication throughput.
- Matrix size affects how efficiently the GPU is utilized.
- Theoretical peak throughput is higher than application level achieved throughput.
- Memory bound operations are limited primarily by data movement.
- Naive attention becomes expensive because the attention matrix grows quadratically.
- Fused attention reduces intermediate memory traffic and improves sequence length scalability.
- Operator fusion can reduce kernel launches and global memory transfers.
- Compilation introduces a one time cost that can be evaluated against repeated execution savings.
- Sustained workloads may run below short benchmark performance because of power and thermal limits.
- GPU performance depends on the interaction between hardware, software, workload shape, and numerical format.

## Systems and Performance Focus

This project develops practical experience with the performance layers involved in AI compiler and machine learning systems engineering:

- Mapping high level tensor operations to GPU execution
- Selecting numerical formats for performance
- Measuring Tensor Core utilization
- Understanding arithmetic intensity
- Identifying compute and bandwidth bottlenecks
- Reducing intermediate tensor materialization
- Studying memory reuse
- Analyzing kernel launch overhead
- Applying operator fusion
- Evaluating compiler assisted execution
- Measuring compilation overhead
- Understanding workload scaling
- Comparing eager and optimized execution
- Investigating attention performance for long sequence workloads
- Reproducing performance results across hardware configurations

The project connects mathematical operations in machine learning models with the execution behavior of GPU kernels and compiled workloads.

## Limitations

This repository is a GPU performance and compiler systems study.

It does not claim to implement a complete production compiler, MLIR backend, Apache TVM compiler, FlashInfer runtime, FlashAttention implementation, or autonomous kernel generation system.

The measurements are specific to:

- The NVIDIA GeForce RTX 3060 Laptop GPU
- Its 6 GB VRAM capacity
- Its 80 W default power limit
- The installed NVIDIA driver
- The installed PyTorch and CUDA software stack
- The laptop cooling system
- The surrounding room temperature
- The workload and benchmark configuration

Results may differ on other GPUs, driver versions, CUDA versions, PyTorch versions, or system configurations.

## Author
Samruddhi Chitnis
