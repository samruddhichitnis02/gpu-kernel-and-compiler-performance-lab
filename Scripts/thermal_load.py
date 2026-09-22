import csv
import os
import subprocess
import time

import torch


DURATION_SECONDS = 20 * 60
SAMPLE_INTERVAL_SECONDS = 5

MATRIX_SIZE = 4096
DTYPE = torch.float16

OUTPUT_FILE = "logs/part_e_thermal_log.csv"


def read_gpu_metrics():
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,clocks.gr,clocks.mem,"
        "temperature.gpu,power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    values = [value.strip() for value in result.stdout.strip().split(",")]

    return {
        "nvidia_timestamp": values[0],
        "gpu_clock_mhz": values[1],
        "memory_clock_mhz": values[2],
        "temperature_c": values[3],
        "power_w": values[4],
        "gpu_utilization_percent": values[5],
        "memory_used_mb": values[6],
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    os.makedirs("logs", exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_uuid = torch.cuda.get_device_properties(0).uuid

    print("GPU:", gpu_name)
    print("GPU UUID:", gpu_uuid)
    print("Matrix size:", MATRIX_SIZE)
    print("Data type:", DTYPE)
    print("Duration:", DURATION_SECONDS, "seconds")
    print("Sampling interval:", SAMPLE_INTERVAL_SECONDS, "seconds")
    print()

    a = torch.randn(
        MATRIX_SIZE,
        MATRIX_SIZE,
        device="cuda",
        dtype=DTYPE,
    )

    b = torch.randn(
        MATRIX_SIZE,
        MATRIX_SIZE,
        device="cuda",
        dtype=DTYPE,
    )

    print("Performing warm-up...")
    for _ in range(10):
        c = torch.matmul(a, b)

    torch.cuda.synchronize()
    del c

    fieldnames = [
        "gpu_name",
        "gpu_uuid",
        "elapsed_seconds",
        "nvidia_timestamp",
        "gpu_clock_mhz",
        "memory_clock_mhz",
        "temperature_c",
        "power_w",
        "gpu_utilization_percent",
        "memory_used_mb",
        "throughput_tflops",
    ]

    start_time = time.monotonic()
    next_sample_time = start_time
    total_operations = 0

    print("Starting 20-minute sustained load...")
    print("Press Ctrl+C to stop early.")
    print()

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        try:
            while True:
                current_time = time.monotonic()

                if current_time - start_time >= DURATION_SECONDS:
                    break

                # Run repeated matrix multiplications.
                interval_operations = 0
                interval_start = time.monotonic()

                while (
                    time.monotonic() - interval_start
                    < SAMPLE_INTERVAL_SECONDS
                ):
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()

                    interval_operations += 1
                    total_operations += 1

                    del c

                interval_end = time.monotonic()
                interval_duration = interval_end - interval_start

                # A matrix multiplication performs approximately 2*N^3 FLOPs.
                flops_per_matmul = 2 * (MATRIX_SIZE ** 3)

                throughput_tflops = (
                    interval_operations
                    * flops_per_matmul
                    / interval_duration
                    / 1e12
                )

                metrics = read_gpu_metrics()
                elapsed_seconds = interval_end - start_time

                row = {
                    "gpu_name": gpu_name,
                    "gpu_uuid": gpu_uuid,
                    "elapsed_seconds": round(elapsed_seconds, 2),
                    "nvidia_timestamp": metrics["nvidia_timestamp"],
                    "gpu_clock_mhz": metrics["gpu_clock_mhz"],
                    "memory_clock_mhz": metrics["memory_clock_mhz"],
                    "temperature_c": metrics["temperature_c"],
                    "power_w": metrics["power_w"],
                    "gpu_utilization_percent": (
                        metrics["gpu_utilization_percent"]
                    ),
                    "memory_used_mb": metrics["memory_used_mb"],
                    "throughput_tflops": round(
                        throughput_tflops,
                        4,
                    ),
                }

                writer.writerow(row)
                file.flush()

                print(
                    f"Elapsed: {elapsed_seconds:7.1f}s | "
                    f"Temperature: {metrics['temperature_c']:>5} C | "
                    f"Power: {metrics['power_w']:>6} W | "
                    f"GPU utilization: "
                    f"{metrics['gpu_utilization_percent']:>3}% | "
                    f"Throughput: "
                    f"{throughput_tflops:>7.2f} TFLOPS"
                )

        except KeyboardInterrupt:
            print("\nStopped early by user.")

    del a, b
    torch.cuda.empty_cache()

    print()
    print("Total matrix multiplications:", total_operations)
    print("Log saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()