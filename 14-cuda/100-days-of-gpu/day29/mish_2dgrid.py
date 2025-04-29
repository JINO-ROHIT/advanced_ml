## Mish formula

# ```
# Mish(x) = x ∗ Tanh(Softplus(x))
# ```

import torch
import triton
import triton.language as tl
import time
import os

DEVICE = torch.device("cuda:0")

@triton.jit
def tanh_kernel(x):
    return (tl.exp(x) - tl.exp(-x)) / (tl.exp(x) + tl.exp(-x))

@triton.jit
def softplus_kernel(x):
    return tl.log(1 + tl.exp(x))

@triton.jit
def mish_kernel(input, output, rows, cols, stride_rows, stride_cols, ROWS_BLOCK: tl.constexpr, COLS_BLOCK: tl.constexpr):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    
    offset_rows = pid_row * ROWS_BLOCK + tl.arange(0, ROWS_BLOCK)
    offset_cols = pid_col * COLS_BLOCK + tl.arange(0, COLS_BLOCK)
    
    mask = (offset_rows[:, None] < rows) & (offset_cols[None, :] < cols)
    input_ptrs = input + offset_rows[:,None] * stride_rows + offset_cols[None,:] * stride_cols
    row = tl.load(input_ptrs, mask = mask)

    out = row * tanh_kernel(softplus_kernel(row))

    output_ptrs = output + offset_rows[:,None] * stride_rows + offset_cols[None,:] * stride_cols
    tl.store(output_ptrs, out, mask = mask)

def mish_forward(input, output):
    rows, cols = input.shape
    ROWS_BLOCK = 32
    COLS_BLOCK = 64
    
    grid = (triton.cdiv(rows, ROWS_BLOCK),triton.cdiv(cols, COLS_BLOCK))

    mish_kernel[grid](
        input,
        output,
        rows,
        cols,
        input.stride(0),
        input.stride(1),
        ROWS_BLOCK,
        COLS_BLOCK
    )
    return output

@triton.jit
def mish_kernel_backward(input, output, rows, cols, stride_rows, stride_cols, ROWS_BLOCK: tl.constexpr, COLS_BLOCK: tl.constexpr):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    
    offset_rows = pid_row * ROWS_BLOCK + tl.arange(0, ROWS_BLOCK)
    offset_cols = pid_col * COLS_BLOCK + tl.arange(0, COLS_BLOCK)
    
    mask = (offset_rows[:, None] < rows) & (offset_cols[None, :] < cols)
    input_ptrs = input + offset_rows[:,None] * stride_rows + offset_cols[None,:] * stride_cols
    row = tl.load(input_ptrs, mask = mask)

    ### d/dx[Mish(x)] = tanh(softplus(x)) + x * (1 - tanh²(softplus(x))) * sigmoid(x)
    tanh_sp = row * tanh_kernel(softplus_kernel(row))
    out = tanh_sp + row * (1 - tanh_sp * tanh_sp) * tl.sigmoid(row)
    
    output_ptrs = output + offset_rows[:,None] * stride_rows + offset_cols[None,:] * stride_cols
    tl.store(output_ptrs, out, mask = mask)

def mish_backward(input, output):
    rows, cols = input.shape
    ROWS_BLOCK = 32
    COLS_BLOCK = 64
    
    grid = (triton.cdiv(rows, ROWS_BLOCK),triton.cdiv(cols, COLS_BLOCK))

    mish_kernel_backward[grid](
        input,
        output,
        rows,
        cols,
        input.stride(0),
        input.stride(1),
        ROWS_BLOCK,
        COLS_BLOCK
    )
    return output

class Mish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return mish_forward(input, torch.zeros_like(input))
        
    @staticmethod
    def backward(ctx, out):
        input = ctx.saved_tensors[0]
        return mish_backward(input, torch.zeros_like(input))
    
def test():
    x = torch.randn((4, 4), device = 'cuda', requires_grad = True)

    out = Mish.apply(x)

    loss = out.mean()
    loss.backward()

    triton_grad = x.grad
    x.grad.zero_() # remember to zero out

    out_torch = torch.nn.functional.mish(x)

    loss1 = out_torch.mean()
    loss1.backward()

    torch_grad = x.grad

    torch.allclose(triton_grad, torch_grad)