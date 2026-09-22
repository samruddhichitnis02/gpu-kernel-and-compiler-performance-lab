import csv
import math
import os
import time

import torch


# Part D configuration
BATCH_SIZE = 1
HEAD_DIM = 64
DTYPE = torch.float32

SEQUENCE_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]

# More repetitions for smaller workloads, fewer for larger workloads
TIMED_REPETITIONS = {
    512: 20,
    1024: 10,
    2048: 5,
    4096: 3,
    8192: 1,
    16384: 1,
}

WARMUP_REPETITIONS = 2

OUTPUT_FILE = "results/part_d_naive_attention.csv"


def naive_attention(q, k, v):
    """
    Naive scaled dot-product attention.

    The important line is:
        scores = q @ k.transpose(-2, -1)

    This creates the complete S x S attention matrix.
    """
    scale = 1.0 / math.sqrt(HEAD_DIM)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, v)

    return output


def benchmark_sequence_length(sequence_length):
    print(f"\nTesting sequence length: {sequence_length}")

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Q, K, V shapes:
        # [batch size, sequence length, head dimension]
        q = torch.randn(
            BATCH_SIZE,
            sequence_length,
            HEAD_DIM,
            device="cuda",
            dtype=DTYPE,
        )

        k = torch.randn(
            BATCH_SIZE,
            sequence_length,
            HEAD_DIM,
            device="cuda",
            dtype=DTYPE,
        )

        v = torch.randn(
            BATCH_SIZE,
            sequence_length,
            HEAD_DIM,
            device="cuda",
            dtype=DTYPE,
        )

        # Warm-up runs
        with torch.inference_mode():
            for _ in range(WARMUP_REPETITIONS):
                output = naive_attention(q, k, v)
                del output

        torch.cuda.synchronize()

        # Reset memory statistics after warm-up
        torch.cuda.reset_peak_memory_stats()

        repetitions = TIMED_REPETITIONS[sequence_length]

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.inference_mode():
            start_event.record()

            for _ in range(repetitions):
                output = naive_attention(q, k, v)
                del output

            end_event.record()

        torch.cuda.synchronize()

        total_time_seconds = start_event.elapsed_time(end_event) / 1000.0
        average_time_seconds = total_time_seconds / repetitions

        peak_allocated_gb = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
        )

        peak_reserved_gb = (
            torch.cuda.max_memory_reserved() / (1024 ** 3)
        )

        print(f"Status: SUCCESS")
        print(f"Average forward latency: {average_time_seconds:.6f} seconds")
        print(f"Timed repetitions: {repetitions}")
        print(f"Peak allocated memory: {peak_allocated_gb:.3f} GB")
        print(f"Peak reserved memory: {peak_reserved_gb:.3f} GB")

        result = {
            "sequence_length": sequence_length,
            "batch_size": BATCH_SIZE,
            "head_dim": HEAD_DIM,
            "dtype": str(DTYPE),
            "warmup_repetitions": WARMUP_REPETITIONS,
            "timed_repetitions": repetitions,
            "status": "success",
            "average_latency_seconds": average_time_seconds,
            "peak_allocated_gb": peak_allocated_gb,
            "peak_reserved_gb": peak_reserved_gb,
            "error": "",
        }

        del q, k, v
        torch.cuda.empty_cache()

        return result

    except torch.cuda.OutOfMemoryError as error:
        print("Status: OUT_OF_MEMORY")
        print("This sequence length does not fit in GPU memory.")

        torch.cuda.empty_cache()

        return {
            "sequence_length": sequence_length,
            "batch_size": BATCH_SIZE,
            "head_dim": HEAD_DIM,
            "dtype": str(DTYPE),
            "warmup_repetitions": WARMUP_REPETITIONS,
            "timed_repetitions": TIMED_REPETITIONS[sequence_length],
            "status": "out_of_memory",
            "average_latency_seconds": "",
            "peak_allocated_gb": "",
            "peak_reserved_gb": "",
            "error": str(error).split("\n")[0],
        }

    except RuntimeError as error:
        print("Status: RUNTIME_ERROR")
        print(error)

        torch.cuda.empty_cache()

        return {
            "sequence_length": sequence_length,
            "batch_size": BATCH_SIZE,
            "head_dim": HEAD_DIM,
            "dtype": str(DTYPE),
            "warmup_repetitions": WARMUP_REPETITIONS,
            "timed_repetitions": TIMED_REPETITIONS[sequence_length],
            "status": "runtime_error",
            "average_latency_seconds": "",
            "peak_allocated_gb": "",
            "peak_reserved_gb": "",
            "error": str(error).split("\n")[0],
        }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    os.makedirs("results", exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_uuid = torch.cuda.get_device_properties(0).uuid

    print("GPU:", gpu_name)
    print("GPU UUID:", gpu_uuid)
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("Batch size:", BATCH_SIZE)
    print("Head dimension:", HEAD_DIM)
    print("Data type:", DTYPE)
    print("Attention implementation: naive")
    print("")

    results = []

    for sequence_length in SEQUENCE_LENGTHS:
        result = benchmark_sequence_length(sequence_length)
        result["gpu_name"] = gpu_name
        result["gpu_uuid"] = gpu_uuid
        results.append(result)

    fieldnames = [
        "gpu_name",
        "gpu_uuid",
        "sequence_length",
        "batch_size",
        "head_dim",
        "dtype",
        "warmup_repetitions",
        "timed_repetitions",
        "status",
        "average_latency_seconds",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "error",
    ]

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()