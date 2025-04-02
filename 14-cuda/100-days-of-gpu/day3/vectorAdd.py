import torch

import triton
import triton.language as tl

#DEVICE = triton.runtime.driver.active.get_active_torch_device()
DEVICE = torch.device("cuda:0") # to run on colab/kaggle


@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements, 
               BLOCK_SIZE: tl.constexpr,
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    #print(x.device, y.device, output.device)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


size = 1000
x = torch.randn(size, device = DEVICE)
y = torch.randn(size, device = DEVICE)
output = add(x, y)
print(output)