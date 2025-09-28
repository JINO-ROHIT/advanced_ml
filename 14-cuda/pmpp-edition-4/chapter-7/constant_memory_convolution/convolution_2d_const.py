import os
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from triton.testing import do_bench

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
os.environ["MAX_JOBS"] = "4"


def load_conv2d_extension():
    return load(
        name="conv2d_torch",
        sources=[
            str(Path(__file__).parent / "convolution_2d_const.cu"),
        ],
        verbose=True,
        with_cuda=True,
    )

def get_filter_radius_from_header():
    header_path = Path(__file__).parent / "conv2d.cuh"

    with open(header_path, "r") as f:
        for line in f:
            if line.strip().startswith("#define FILTER_RADIUS"):
                # Extract the value after #define FILTER_RADIUS
                return int(line.split()[2])

    raise ValueError("FILTER_RADIUS not found in header file")

@torch.inference_mode
def main():
    conv2d_extension = load_conv2d_extension()

    device = "cuda"
    height = 4096
    width = 4096
    r = (
        get_filter_radius_from_header()
    )  ##this is a hack we do so we don't have the same filter value in both files
    print(f"R value: {r}")
    kernel_size = 2 * r + 1

    input_tensor = torch.randn(height, width, device=device, dtype=torch.float32)
    kernel_tensor = torch.randn(
        kernel_size, kernel_size, device=device, dtype=torch.float32
    )

    input_torch = input_tensor.unsqueeze(0).unsqueeze(0)
    kernel_torch = kernel_tensor.unsqueeze(0).unsqueeze(0)

    torch_output = F.conv2d(input_torch, kernel_torch, padding=r).squeeze()
    custom_output = (
        conv2d_extension.conv2d_torch(
            input_tensor, kernel_tensor, r
        )
    )

    print(custom_output[:10, :10])
    print()
    print(torch_output[:10, :10])

    # assert torch.allclose(
    #     custom_output, torch_output, rtol=1e-3, atol=1e-2
    # ), "Your function output differs from torch."


    custom_conv2d = partial(
        conv2d_extension.conv2d_torch,
        input_tensor,
        kernel_tensor,
        r,
    )

    torch_conv2d = partial(
        F.conv2d,
        input=input_torch,
        weight=kernel_torch,
        padding=r,
    )

    custom_conv2d_time = do_bench(custom_conv2d, warmup=1, rep=1)
    torch_conv2d_time = do_bench(torch_conv2d, warmup=1, rep=1)

    print(f"Custom Conv2d kernel time: {custom_conv2d_time:.4f} ms")
    print(f"Torch Conv2d kernel time: {torch_conv2d_time:.4f} ms")


if __name__ == "__main__":
    main()