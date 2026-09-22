import csv
import math
import os

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


BATCH_SIZE = 1
NUMBER_OF_HEADS = 1
HEAD_DIM = 64
DTYPE = torch.float16

SEQUENCE_LENGTHS = [
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    49152,
    65536,
    73728,
    81920,
    90112,
    98304,
    114688,
    131072,
    147456,
    163840,
]


TIMED_REPETITIONS = {
    512: 20,
    1024: 10,
    2048: 5,
    4096: 3,
    8192: 1,
    16384: 1,
    32768: 1,
    49152 : 1,
    65536 : 1,
    73728 : 1,
    81920 : 1,
    90112 : 1,
    98304 : 1,
    114688 : 1,
    131072 : 1,
    147456 : 1,
    163840 : 1,
    
}

WARMUP_REPETITIONS = 2

OUTPUT_FILE = "results/part_d_fused_boundary.csv"


def naive_attention(q, k, v):
    """
    Naive attention.

    This explicitly creates:
    1. The S x S score matrix
    2. The S x S softmax matrix
    """
    scale = 1.0 / math.sqrt(HEAD_DIM)

    scores = torch.matmul(
        q,
        k.transpose(-2, -1),
    ) * scale

    weights = torch.softmax(scores, dim=-1)

    output = torch.matmul(weights, v)

    return output


def fused_attention(q, k, v):
    """
    Force PyTorch to use Flash Attention or
    memory-efficient attention.

    The ordinary math fallback is disabled.
    """
    with sdpa_kernel([
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
    ]):
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )

    return output


def create_inputs(sequence_length):
    """
    Tensor shape:

    [batch size, number of heads, sequence length, head dimension]

    For this experiment:

    [1, 1, S, 64]
    """
    q = torch.randn(
        BATCH_SIZE,
        NUMBER_OF_HEADS,
        sequence_length,
        HEAD_DIM,
        device="cuda",
        dtype=DTYPE,
    )

    k = torch.randn(
        BATCH_SIZE,
        NUMBER_OF_HEADS,
        sequence_length,
        HEAD_DIM,
        device="cuda",
        dtype=DTYPE,
    )

    v = torch.randn(
        BATCH_SIZE,
        NUMBER_OF_HEADS,
        sequence_length,
        HEAD_DIM,
        device="cuda",
        dtype=DTYPE,
    )

    return q, k, v


def benchmark_method(method_name, method, sequence_length):
    print(
        f"\nTesting {method_name} attention "
        f"at sequence length {sequence_length}"
    )

    q = None
    k = None
    v = None
    output = None

    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        q, k, v = create_inputs(sequence_length)

        # Warm-up runs
        with torch.inference_mode():
            for _ in range(WARMUP_REPETITIONS):
                output = method(q, k, v)
                del output
                output = None

        torch.cuda.synchronize()

        # Reset statistics after warm-up
        torch.cuda.reset_peak_memory_stats()

        repetitions = TIMED_REPETITIONS[sequence_length]

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.inference_mode():
            start_event.record()

            for _ in range(repetitions):
                output = method(q, k, v)
                del output
                output = None

            end_event.record()

        torch.cuda.synchronize()

        total_time_seconds = (
            start_event.elapsed_time(end_event) / 1000.0
        )

        average_latency_seconds = (
            total_time_seconds / repetitions
        )

        peak_memory_gb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        print("Status: SUCCESS")
        print(
            f"Average latency: "
            f"{average_latency_seconds:.6f} seconds"
        )
        print(
            f"Peak allocated memory: "
            f"{peak_memory_gb:.3f} GB"
        )

        result = {
            "method": method_name,
            "sequence_length": sequence_length,
            "status": "success",
            "average_latency_seconds": (
                average_latency_seconds
            ),
            "peak_memory_gb": peak_memory_gb,
            "timed_repetitions": repetitions,
            "error": "",
        }

        del q, k, v
        torch.cuda.empty_cache()

        return result

    except torch.cuda.OutOfMemoryError as error:
        print("Status: OUT_OF_MEMORY")
        print("This configuration exceeded GPU memory.")

        torch.cuda.empty_cache()

        return {
            "method": method_name,
            "sequence_length": sequence_length,
            "status": "out_of_memory",
            "average_latency_seconds": "",
            "peak_memory_gb": "",
            "timed_repetitions": (
                TIMED_REPETITIONS[sequence_length]
            ),
            "error": str(error).split("\n")[0],
        }

    except RuntimeError as error:
        print("Status: RUNTIME_ERROR")
        print(error)

        torch.cuda.empty_cache()

        return {
            "method": method_name,
            "sequence_length": sequence_length,
            "status": "runtime_error",
            "average_latency_seconds": "",
            "peak_memory_gb": "",
            "timed_repetitions": (
                TIMED_REPETITIONS[sequence_length]
            ),
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
    print("Number of heads:", NUMBER_OF_HEADS)
    print("Head dimension:", HEAD_DIM)
    print("Data type:", DTYPE)
    print("Tensor shape: [batch, heads, sequence, head_dimension]")
    print("Ordinary math fallback: DISABLED")
    print("")

    results = []

    methods = [
        # ("naive_fp16", naive_attention),
        ("forced_fused_fp16", fused_attention),
    ]

    for sequence_length in SEQUENCE_LENGTHS:
        for method_name, method in methods:
            result = benchmark_method(
                method_name,
                method,
                sequence_length,
            )

            result["gpu_name"] = gpu_name
            result["gpu_uuid"] = gpu_uuid
            result["batch_size"] = BATCH_SIZE
            result["number_of_heads"] = NUMBER_OF_HEADS
            result["head_dim"] = HEAD_DIM
            result["dtype"] = str(DTYPE)
            result["warmup_repetitions"] = (
                WARMUP_REPETITIONS
            )

            results.append(result)

    fieldnames = [
        "gpu_name",
        "gpu_uuid",
        "method",
        "sequence_length",
        "batch_size",
        "number_of_heads",
        "head_dim",
        "dtype",
        "warmup_repetitions",
        "timed_repetitions",
        "status",
        "average_latency_seconds",
        "peak_memory_gb",
        "error",
    ]

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print("")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()