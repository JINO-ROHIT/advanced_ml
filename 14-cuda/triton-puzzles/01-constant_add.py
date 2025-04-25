import triton
import triton.language as tl
import torch


@triton.jit
def add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * B0
    range = offsets + tl.arange(0, B0)
    x = tl.load(x_ptr + range, mask = range < N0)
    z = x + 10
    tl.store(z_ptr + range, z, mask = range < N0)