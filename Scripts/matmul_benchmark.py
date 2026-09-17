import csv
import time
import torch

GPU_UUID = "GPU-20606811-3bb2-e6b5-ec74-b477d9c1d1d8"
OUTPUT_FILE = r"C:\Users\samru\Study\Study_Abroad\Universities\SJSU\Semester3\Data_266_Generative_AI\Assignments\GPU_Assignment1\results\matmul_results.csv"

SIZES = [1024, 4096, 8192, 16384]
PRECISIONS = ["FP32", "TF32", "FP16", "BF16"]

REPS = {
    1024: 20,
    4096: 10,
    8192: 3,
    16384: 1,
}

WARMUP = 3


def run_matmul(n, precision):
    torch.cuda.empty_cache()

    if precision == "FP32":
        dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = False

    elif precision == "TF32":
        dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True

    elif precision == "FP16":
        dtype = torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True

    elif precision == "BF16":
        dtype = torch.bfloat16
        torch.backends.cuda.matmul.allow_tf32 = True

    try:
        a = torch.randn((n, n), device = "cuda", dtype = dtype)
        b = torch.randn((n, n), device = "cuda", dtype = dtype)

        for _ in range(WARMUP):
            _ = torch.matmul(a, b)

        torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(REPS[n]):
            _ = torch.matmul(a, b)

        torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) / REPS[n]
        tflops = (2 * n**3) / elapsed / 1e12

        allocated_gb = torch.cuda.max_memory_allocated() / 1024**3

        print(
            f"{precision:5s} | N={n:5d} | "
            f"time={elapsed:.6f} s | "
            f"TFLOPS={tflops:.3f} | "
            f"memory={allocated_gb:.2f} GB"
        )

        del a, b
        torch.cuda.empty_cache()

        return {
            "gpu_uuid": GPU_UUID,
            "gpu_name": torch.cuda.get_device_name(0),
            "precision": precision,
            "matrix_size": n,
            "repetitions": REPS[n],
            "average_time_seconds": elapsed,
            "achieved_tflops": tflops,
            "peak_memory_gb": allocated_gb,
            "status": "success",
        }

    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            print(f"{precision:5s} | N={n:5d} | OUT OF MEMORY")
            torch.cuda.empty_cache()

            return {
                "gpu_uuid": GPU_UUID,
                "gpu_name": torch.cuda.get_device_name(0),
                "precision": precision,
                "matrix_size": n,
                "repetitions": REPS[n],
                "average_time_seconds": "",
                "achieved_tflops": "",
                "peak_memory_gb": "",
                "status": "OOM",
            }

        raise


def main():
    torch.cuda.reset_peak_memory_stats()

    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU UUID:", GPU_UUID)
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print()

    rows = []

    for precision in PRECISIONS:
        for n in SIZES:
            torch.cuda.reset_peak_memory_stats()
            result = run_matmul(n, precision)
            rows.append(result)

    fieldnames = [
        "gpu_uuid",
        "gpu_name",
        "precision",
        "matrix_size",
        "repetitions",
        "average_time_seconds",
        "achieved_tflops",
        "peak_memory_gb",
        "status",
    ]

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()