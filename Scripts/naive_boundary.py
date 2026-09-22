import csv
import math
import os

import torch


BATCH_SIZE = 1
HEAD_DIM = 64
DTYPE = torch.float32

# Boundary-search lengths
SEQUENCE_LENGTHS = [
    18432,
    20480,
    22528,
    24576,
    26624,
    28672,
    30720,
    32768,
    34816,
    36864,
    38_912,
    40_960,
    43_008,
    45_056,
]

OUTPUT_FILE = "results/part_d_naive_boundary.csv"


def naive_attention(q, k, v):
    scale = 1.0 / math.sqrt(HEAD_DIM)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)

    return output


def test_sequence_length(sequence_length):
    print(f"\nTesting sequence length: {sequence_length}")

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

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

        with torch.inference_mode():
            output = naive_attention(q, k, v)

        torch.cuda.synchronize()

        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        print("Status: SUCCESS")
        print(f"Peak memory: {peak_memory_gb:.3f} GB")

        del q, k, v, output
        torch.cuda.empty_cache()

        return {
            "sequence_length": sequence_length,
            "status": "success",
            "peak_memory_gb": peak_memory_gb,
            "error": "",
        }

    except torch.cuda.OutOfMemoryError as error:
        print("Status: OUT_OF_MEMORY")

        torch.cuda.empty_cache()

        return {
            "sequence_length": sequence_length,
            "status": "out_of_memory",
            "peak_memory_gb": "",
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
    print("Batch size:", BATCH_SIZE)
    print("Head dimension:", HEAD_DIM)
    print("Data type:", DTYPE)

    results = []

    for sequence_length in SEQUENCE_LENGTHS:
        result = test_sequence_length(sequence_length)
        result["gpu_name"] = gpu_name
        result["gpu_uuid"] = gpu_uuid
        result["batch_size"] = BATCH_SIZE
        result["head_dim"] = HEAD_DIM
        result["dtype"] = str(DTYPE)
        results.append(result)

    fieldnames = [
        "gpu_name",
        "gpu_uuid",
        "sequence_length",
        "batch_size",
        "head_dim",
        "dtype",
        "status",
        "peak_memory_gb",
        "error",
    ]

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()


