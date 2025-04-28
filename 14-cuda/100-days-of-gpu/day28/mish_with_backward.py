## Mish formula

# ```
# Mish(x) = x ∗ Tanh(Softplus(x))
# ```

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")

@triton.jit
def tanh_kernel(x):
    return (tl.exp(x) - tl.exp(-x)) / (tl.exp(x) + tl.exp(-x))

@triton.jit
def softplus_kernel(x):
    return tl.log(1 + tl.exp(x))

@triton.jit
def mish_kernel(input, output, n, BLOCK_SIZE: tl.constexpr, num_warps = 32):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    row = tl.load(input + offsets, mask = mask)
    out = row * tanh_kernel(softplus_kernel(row))
    tl.store(output + offsets, out, mask = mask)

def mish_forward(input, output, BLOCK_SIZE):
    num_elements = input.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)

    mish_kernel[grid](
        input,
        output,
        num_elements,
        BLOCK_SIZE = 512
    )
    return output

@triton.jit
def mish_kernel_backward(input, output, n, BLOCK_SIZE: tl.constexpr, num_warps = 32):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    row = tl.load(input + offsets, mask = mask)

    ### d/dx[Mish(x)] = tanh(softplus(x)) + x * (1 - tanh²(softplus(x))) * sigmoid(x)
    tanh_sp = row * tanh_kernel(softplus_kernel(row))
    out = tanh_sp + row * (1 - tanh_sp * tanh_sp) * tl.sigmoid(row)
    
    tl.store(output + offsets, out, mask = mask)

def mish_backward(input, output):
    num_elements = input.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)

    mish_kernel_backward[grid](
        input,
        output,
        num_elements,
        BLOCK_SIZE = 512
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