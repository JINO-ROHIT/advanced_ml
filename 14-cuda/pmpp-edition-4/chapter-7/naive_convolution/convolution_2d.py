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
            str(Path(__file__).parent / "convolution_2d.cu"),
        ],
        verbose=True,
        with_cuda=True,
    )

@torch.inference_mode
def main():
    conv2d_extension = load_conv2d_extension()

    device = "cuda"
    height = 4096
    width = 4096
    r = 7
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

    # print(custom_output[:10, :10])
    # print()
    # print(torch_output)

    assert torch.allclose(
        custom_output, torch_output, rtol=1e-5, atol=1e-5
    ), "Your function output differs from torch."

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

    custom_conv2d_time = do_bench(custom_conv2d, warmup=2, rep=10)
    torch_conv2d_time = do_bench(torch_conv2d, warmup=2, rep=10)

    print(f"Custom Conv2d kernel time: {custom_conv2d_time:.4f} ms")
    print(f"Torch Conv2d kernel time: {torch_conv2d_time:.4f} ms")


if __name__ == "__main__":
    main()