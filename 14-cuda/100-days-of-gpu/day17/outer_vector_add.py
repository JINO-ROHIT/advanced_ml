import triton
import triton.language as tl
import torch

def add_vec_spec(x, y):
    return x[None, :] + y[:, None]

@triton.jit
def add_vec_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    pid = tl.program_id(0)
    x_start = pid * B0 
    y_start = pid * B1
    
    x_offsets = x_start + tl.arange(0, B0)
    y_offsets = y_start + tl.arange(0, B1)
    
    maskx = x_offsets < N0
    masky = y_offsets < N1

    row1 = tl.load(x_ptr + x_offsets, mask = maskx)
    row2 = tl.load(y_ptr + y_offsets, mask = masky)
    out = add_vec_spec(row1, row2)

    # print(out.shape)
    # print(x_offsets.shape)

    z_offsets = x_offsets[None, :]  + y_offsets[:, None] * N0
    print(z_offsets)
    # print(z_offsets.shape)

    maskz = maskx[None, :] & masky[:, None]
    #print(maskz)
    tl.store(z_ptr + z_offsets, out, mask = maskz)