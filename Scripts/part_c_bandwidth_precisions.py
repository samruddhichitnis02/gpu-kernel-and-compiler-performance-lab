# scripts/part_c_bandwidth_precisions.py

import csv
import torch

DEVICE = "cuda"
NUM_ELEMENTS = 64 * 1024 * 1024
WARMUP_REPETITIONS = 5
TIMED_REPETITIONS = 20

PRECISIONS = {
    "FP32": torch.float32,
    "FP16": torch.float16,
    "BF16": torch.bfloat16,
}


def measure_bandwidth(dtype_name, dtype):
    print(f"\nTesting {dtype_name}")

    A = torch.ones(NUM_ELEMENTS, device=DEVICE, dtype=dtype)
    B = torch.ones(NUM_ELEMENTS, device=DEVICE, dtype=dtype)
    C = torch.empty_like(A)

    def vector_add():
        torch.add(A, B, out=C)

    # Warm-up
    for _ in range(WARMUP_REPETITIONS):
        vector_add()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(TIMED_REPETITIONS):
        vector_add()

    end.record()
    torch.cuda.synchronize()

    total_time_ms = start.elapsed_time(end)
    average_time_seconds = (
        total_time_ms / TIMED_REPETITIONS / 1000
    )

    bytes_per_value = torch.tensor([], dtype=dtype).element_size()

    # Read A + read B + write C
    total_bytes_moved = (
        3 * NUM_ELEMENTS * bytes_per_value
    )

    effective_bandwidth_gbs = (
        total_bytes_moved
        / average_time_seconds
        / 1e9
    )

    arithmetic_intensity = (
        NUM_ELEMENTS / total_bytes_moved
    )

    print("Data type:", dtype)
    print("Bytes per value:", bytes_per_value)
    print("Average time:", average_time_seconds, "seconds")
    print("Effective bandwidth:", effective_bandwidth_gbs, "GB/s")
    print("Arithmetic intensity:", arithmetic_intensity, "FLOPs/byte")

    result = {
        "precision": dtype_name,
        "elements": NUM_ELEMENTS,
        "bytes_per_value": bytes_per_value,
        "warmup_repetitions": WARMUP_REPETITIONS,
        "timed_repetitions": TIMED_REPETITIONS,
        "average_time_seconds": average_time_seconds,
        "effective_bandwidth_gbs": effective_bandwidth_gbs,
        "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
    }

    del A, B, C
    torch.cuda.empty_cache()

    return result


def main():
    print("GPU:", torch.cuda.get_device_name(0))
    print("UUID: GPU-20606811-3bb2-e6b5-ec74-b477d9c1d1d8")

    results = []

    for precision_name, dtype in PRECISIONS.items():
        try:
            result = measure_bandwidth(precision_name, dtype)
            results.append(result)

        except Exception as error:
            print(f"{precision_name} failed:")
            print(type(error).__name__, error)

    with open(
        "results/part_c_bandwidth_precisions.csv",
        "w",
        newline=""
    ) as file:
        fieldnames = [
            "precision",
            "elements",
            "bytes_per_value",
            "warmup_repetitions",
            "timed_repetitions",
            "average_time_seconds",
            "effective_bandwidth_gbs",
            "arithmetic_intensity_flops_per_byte",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(
        "\nResults saved to "
        "results/part_c_bandwidth_precisions.csv"
    )


if __name__ == "__main__":
    main()