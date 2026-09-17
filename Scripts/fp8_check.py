# scripts/fp8_check.py

import torch

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)

try:
    A = torch.randn((1024, 1024), device="cuda", dtype=torch.float8_e4m3fn)
    B = torch.randn((1024, 1024), device="cuda", dtype=torch.float8_e4m3fn)

    torch.cuda.synchronize()
    C = A @ B
    torch.cuda.synchronize()

    print("FP8 matrix multiplication: SUCCESS")
    print("Output dtype:", C.dtype)

except Exception as error:
    print("FP8 matrix multiplication: NOT SUPPORTED")
    print("Error type:", type(error).__name__)
    print("Error message:", error)