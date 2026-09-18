# scripts/part_c_benchmark.py

import csv
import torch

DEVICE = "cuda"
DTYPE = torch.float32

def time_cuda_operation(operation, warmup = 5, repetitions = 20):
    for _ in range(warmup):
        operation()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repetitions):
        operation()
    end.record()

    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return (total_ms / repetitions) / 1000.0


def main():
    print("GPU:", torch.cuda.get_device_name(0))
    print("UUID: GPU-20606811-3bb2-e6b5-ec74-b477d9c1d1d8")

    results = []

    # --------------------------------------------------
    # Memory-bound operation: C = A + B
    # --------------------------------------------------
    num_elements = 64 * 1024 * 1024

    A = torch.ones(num_elements, device=DEVICE, dtype=DTYPE)
    B = torch.ones(num_elements, device=DEVICE, dtype=DTYPE)

    C = torch.empty_like(A)

    def vector_add():
        torch.add(A, B, out=C)

    time_seconds = time_cuda_operation(
        vector_add,
        warmup=5,
        repetitions=20
    )

    bytes_moved = 3 * num_elements * 4
    bandwidth_gbs = bytes_moved / time_seconds / 1e9

    vector_add_flops = num_elements
    vector_add_arithmetic_intensity = vector_add_flops / bytes_moved

    print("\nMEMORY-BOUND: VECTOR ADD")
    print("Elements:", num_elements)
    print("Average time:", time_seconds, "seconds")
    print("Effective bandwidth:", bandwidth_gbs, "GB/s")
    print("Arithmetic intensity:", vector_add_arithmetic_intensity, "FLOPs/byte")

    results.append({
        "operation": "vector_add",
        "time_seconds": time_seconds,
        "effective_bandwidth_gbs": bandwidth_gbs,
        "arithmetic_intensity_flops_per_byte": vector_add_arithmetic_intensity,
        "repetitions": 20
    })

    del A, B, C
    torch.cuda.empty_cache()

    # --------------------------------------------------
    # Compute-bound operation: C = A @ B
    # --------------------------------------------------
    N = 4096

    A = torch.randn((N, N), device=DEVICE, dtype=DTYPE)
    B = torch.randn((N, N), device=DEVICE, dtype=DTYPE)

    def matrix_multiply():
        torch.matmul(A, B)

    time_seconds = time_cuda_operation(
        matrix_multiply,
        warmup=3,
        repetitions=10
    )

    matmul_flops = 2 * (N ** 3)
    matmul_bytes = 3 * (N ** 2) * 4
    matmul_arithmetic_intensity = matmul_flops / matmul_bytes
    achieved_tflops = matmul_flops / time_seconds / 1e12

    print("\nCOMPUTE-BOUND: MATRIX MULTIPLICATION")
    print("Matrix size:", N, "x", N)
    print("Average time:", time_seconds, "seconds")
    print("Achieved TFLOPS:", achieved_tflops)
    print("Arithmetic intensity:", matmul_arithmetic_intensity, "FLOPs/byte")

    results.append({
        "operation": "matmul",
        "time_seconds": time_seconds,
        "effective_bandwidth_gbs": "",
        "arithmetic_intensity_flops_per_byte": matmul_arithmetic_intensity,
        "repetitions": 10
    })

    with open("results/part_c_results.csv", "w", newline="") as file:
        fieldnames = [
            "operation",
            "time_seconds",
            "effective_bandwidth_gbs",
            "arithmetic_intensity_flops_per_byte",
            "repetitions"
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to results/part_c_results.csv")


if __name__ == "__main__":
    main()