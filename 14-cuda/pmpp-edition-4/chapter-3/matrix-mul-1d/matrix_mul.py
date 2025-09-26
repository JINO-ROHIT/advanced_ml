from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

import os
def set_cuda_arch():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}.{minor}"
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        print(f"[INFO] Using CUDA arch {arch} for compilation")
    else:
        print("[WARN] CUDA not available, compiling for CPU only")


def compile_extension():
    cuda_source = (
        Path(__file__).parent / "matrix_mul.cu"
    ).read_text()

    cpp_sources = [
        "torch::Tensor matrixRowMul(torch::Tensor M, torch::Tensor N);",
        "torch::Tensor matrixColMul(torch::Tensor M, torch::Tensor N);",
    ]

    return load_inline(
        name="matrixMul_extension",
        cpp_sources=cpp_sources,
        cuda_sources=cuda_source,
        functions=["matrixRowMul", "matrixColMul"],
        with_cuda=True,
    )


def main():
    set_cuda_arch()
    ext = compile_extension()

    DEVICE, DTYPE = "cuda", torch.float32

    M = torch.randn(4, 4).to(DEVICE, DTYPE)
    N = torch.randn(4, 4).to(DEVICE, DTYPE)

    P_row = ext.matrixRowMul(M, N)
    P_col = ext.matrixColMul(M, N)

    torch_P = torch.matmul(M, N)

    all_close = torch.allclose(P_row, torch_P, rtol=1e-3, atol=1e-3) & torch.allclose(
        P_col, torch_P, rtol=1e-3, atol=1e-3
    )

    print(f"All close: {all_close}")
    print()
    print(P_row[:4, :4])
    print()
    print(P_col[:4, :4])
    print()
    print(torch_P[:4, :4])


if __name__ == "__main__":
    main()