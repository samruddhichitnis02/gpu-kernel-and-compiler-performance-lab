# HW2.5 Measurement Summary

All measurements were collected on:

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- GPU UUID: `GPU-20606811-3bb2-e6b5-ec74-b477d9c1d1d8`
- VRAM: 6 GB
- Driver: 610.88
- PyTorch: 2.6.0+cu124
- CUDA runtime: 12.4
- Default power limit: 80 W

## Table HW2.5.1 — Summary Table

| Measurement | Your GPU | Notes |
|---|---:|---|
| Peak achieved TFLOPS, BF16 | 24.25 TFLOPS | N = 4096; UUID-labelled in `results/matmul_results_with_percentages.csv` |
| Percentage of theoretical peak, BF16 | 46.35% | Theoretical peak: 52.32 TFLOPS |
| Effective bandwidth | 315.77 GB/s | FP32 vector addition; approximately 93.98% of the 336 GB/s reference |
| Naive attention OOM length | 43,008 tokens failed | Largest tested success: 40,960 tokens; FP32, batch size 1, head dimension 64 |
| Fused attention OOM length | Not observed through 163,840 tokens | 163,840 was the largest tested successful length |
| Steady-state / peak throughput | 94.33% | 21.47 TFLOPS final five-minute average / 22.76 TFLOPS initial peak |
| Throttle onset | Power ceiling by approximately 10 seconds | Approximately 80 W power limit; no clear temperature-based throttling observed |

## Part B — Precision Results

| Precision | N = 1024 | N = 4096 | N = 8192 | N = 16384 |
|---|---:|---:|---:|---:|
| FP32 | 6.05 TFLOPS | 5.21 TFLOPS | 6.88 TFLOPS | 6.92 TFLOPS |
| TF32 | 7.25 TFLOPS | 11.75 TFLOPS | 11.70 TFLOPS | 11.19 TFLOPS |
| FP16 | 14.47 TFLOPS | 23.98 TFLOPS | 23.34 TFLOPS | 21.34 TFLOPS |
| BF16 | 13.96 TFLOPS | 24.25 TFLOPS | 23.86 TFLOPS | 22.85 TFLOPS |

The highest measured BF16 throughput was 24.25 TFLOPS at N = 4096.

## Part C — Roofline Measurements

| Operation | Average time | Throughput or bandwidth | Arithmetic intensity |
|---|---:|---:|---:|
| FP32 vector addition | 0.002550 seconds | 315.77 GB/s | 0.0833 FLOPs/byte |
| FP32 matrix multiplication, N = 4096 | 0.026217 seconds | 5.24 TFLOPS | 682.67 FLOPs/byte |

The vector addition is memory-bound because it performs very little computation per byte moved.

The matrix multiplication is compute-bound because it performs many floating-point operations per byte transferred.

## Part D — Attention Results

### Naive Attention

| Sequence length | Average latency | Peak allocated memory |
|---:|---:|---:|
| 512 | 0.333 ms | 0.010 GB |
| 1024 | 0.232 ms | 0.017 GB |
| 2048 | 0.545 ms | 0.041 GB |
| 4096 | 1.795 ms | 0.137 GB |
| 8192 | 6.698 ms | 0.516 GB |
| 16384 | 28.990 ms | 2.024 GB |

Configuration:

- Batch size: 1
- Head dimension: 64
- Data type: FP32

Quadratic fit:

- Quadratic coefficient: `7.4665 × 10⁻⁹ GB/token²`
- Equivalent coefficient: `8.0171 bytes/token²`
- Intercept: `0.021607 GB`
- R-squared: `1.000000`

### Fused Attention

The fused implementation used FP16 with batch size 1, one attention head, and head dimension 64.

At sequence length 16,384:

- Naive latency: 13.230 ms
- Fused latency: 4.806 ms
- Speedup: 2.75×
- Naive memory: 1.016 GB
- Fused workspace memory: 0.016 GB
- Memory reduction: approximately 98.45%

The fused implementation successfully ran through 163,840 tokens without an out-of-memory failure.

## Part E — Sustained Load

The sustained workload repeatedly executed 4096 × 4096 FP16 matrix multiplication for approximately 20 minutes.

| Measurement | Result |
|---|---:|
| Duration | 1,200.58 seconds |
| Samples | 237 |
| Initial peak throughput | 22.76 TFLOPS |
| Final five-minute average | 21.47 TFLOPS |
| Steady-state percentage | 94.33% |
| Initial temperature | 49°C |
| Maximum temperature | 77°C |
| Average utilization | 96.26% |
| Maximum power draw | Approximately 80 W |
| Memory clock | 7001 MHz |

The GPU reached its power ceiling early in the run. The measurements do not show clear temperature-based throttling.

## Related Files

- `results/matmul_results_with_percentages.csv`
- `results/part_c_results.csv`
- `results/part_c_bandwidth_precisions.csv`
- `results/part_d_naive_attention.csv`
- `results/part_d_naive_boundary.csv`
- `results/part_d_fused_forced_comparison.csv`
- `results/part_d_fused_boundary.csv`
- `logs/part_e_thermal_log.csv`
- `nvidia-smi-q.txt`

Every measurement is traceable to GPU UUID `GPU-20606811-3bb2-e6b5-ec74-b477d9c1d1d8`.