import triton
import triton.language as tl
import torch

"""long softmax - numerically stable softmax using blocks
TO-DO verify the logic
"""

@triton.jit
def softmax_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)
    log2_e = 1.44269504 # exp2 is faster than exp --> log₂(e) so we can write exp(x) as exp2(x * log2_e)
    
    row_offsets = pid_0 * B0 + tl.arange(0, B0)
    mask_row = row_offsets < N0
    _max = (tl.arange(0, B0) + 1) * (-1e10) # init a very small max like -inf
    _sum = tl.arange(0, B0) * 0 # init 0

    for _col in tl.static_range(0, T, B1):
        col_offsets = _col + tl.arange(0, B1)
        col_block = row_offsets[:, None] * T + col_offsets[None, :]  # (B0, B1)
        mask_col = mask_row[:, None] & (col_offsets[None, :] < T)

        x_block = tl.load(x_ptr + col_block, mask = mask_col, other=0.0)
        block_max = tl.max(x_block, axis = 1)
        new_max = tl.maximum(_max, block_max)

        _sum *= tl.exp2(_max - new_max)  # correcting the sum in terms of new max since we calculate block wise
        _max = new_max

        x_block -= _max[:, None]
        x_block = tl.exp2(x_block * log2_e)

        _sum += tl.sum(x_block, axis=1)

    for _col in tl.static_range(0, T, B1):
        col_offsets = _col + tl.arange(0, B1)
        col_block = row_offsets[:, None] * T + col_offsets[None, :]  # (B0, B1)
        mask_col = mask_row[:, None] & (col_offsets[None, :] < T)

        x_block = tl.load(x_ptr + col_block, mask = mask_col, other=0.0)
        x_block -= _max[:, None]
        x_block = tl.exp2(x_block * log2_e)
        x_block /= _sum[:, None]

        tl.store(z_ptr + col_block, x_block, mask = mask_col)