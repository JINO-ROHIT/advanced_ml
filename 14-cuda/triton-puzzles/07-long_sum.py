import triton
import triton.language as tl
import torch

'''
N0 - 4
N1 - 200

so (4, 200) or 4 batches of 200 elements

B0 - 1 --> blocks across rows
B1 - 32 ---> blocks across columns

so the idea is to assign one program per row , so 4 programs processing 32 elements each time and saving into z_ptr.
'''
@triton.jit
def sum_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)

    rows_offset = pid_0 * B0 + tl.arange(0, B0)
    #print(range_0)
    mask_0 = rows_offset < N0
    _sum = tl.arange(0, B0) * 0.0
    #print(_sum)

    for elements in tl.static_range(0, T, B1):
        cols_offset = elements + tl.arange(0, B1)
        #print(range_1)
        range_block = rows_offset[:, None] * T + cols_offset[None, :]  # (B0, B1)
        mask_block = mask_0[:, None] & (cols_offset[None, :] < T)

        x_block = tl.load(x_ptr + range_block, mask = mask_block, other = 0.0)
        _sum += tl.sum(x_block, axis = 1)

    tl.store(z_ptr + rows_offset, _sum, mask = mask_0)

    return