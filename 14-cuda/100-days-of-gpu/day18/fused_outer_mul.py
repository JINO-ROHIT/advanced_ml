import triton
import triton.language as tl
import torch


def mul_relu_block(x, y):
    z = x[None, :] * y[:, None]
    op = tl.where(z > 0, z, tl.zeros((x.shape[0], y.shape[0]), dtype=tl.float32))
    return op
    
@triton.jit
def mul_relu_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    
    x_start = pid_0 * B0 
    y_start = pid_1 * B1
    
    x_offsets = x_start + tl.arange(0, B0)
    y_offsets = y_start + tl.arange(0, B1)
    
    maskx = x_offsets < N0
    masky = y_offsets < N1

    row1 = tl.load(x_ptr + x_offsets, mask = maskx)
    row2 = tl.load(y_ptr + y_offsets, mask = masky)
    out = mul_relu_block(row1, row2)

    # print(out.shape)
    # print(x_offsets.shape)

    z_offsets = x_offsets[None, :]  + y_offsets[:, None] * N0
    #print(z_offsets)
    # print(z_offsets.shape)

    maskz = maskx[None, :] & masky[:, None]
    #print(maskz)
    tl.store(z_ptr + z_offsets, out, mask = maskz)