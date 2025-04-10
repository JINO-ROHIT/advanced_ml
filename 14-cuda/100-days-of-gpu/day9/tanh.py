# import triton
# import triton.language as tl
# import torch

# @triton.jit 
# def _tanh_kernel(
#     input_ptr, output_ptr,
#     input_row_stride, output_row_stride,    
#     n_rows, n_cols,                         
#     BLOCK_SIZE: tl.constexpr,               
# ): 
#     row_start = tl.program_id(0) 
#     row_step = tl.num_programs(0) 

#     for row_idx in tl.range(row_start, n_rows, row_step):
#         row_start_ptr = input_ptr + row_idx * input_row_stride
#         out_row_start_ptr = output_ptr + row_idx * output_row_stride
#         col_offsets = tl.arange(0, BLOCK_SIZE)
#         input_ptrs = row_start_ptr + col_offsets
#         mask = col_offsets < n_cols
#         row = tl.load(input_ptrs, mask=mask, other=None) 
#         exp_pos = tl.exp(row)
#         exp_neg = tl.exp(-row)
#         result = (exp_pos - exp_neg) / (exp_pos + exp_neg)
#         tl.store(out_row_start_ptr + col_offsets, result, mask = mask)

# def solution(input, output, n: int, m: int):
#     BLOCK_SIZE = triton.next_power_of_2(n)

#     grid = (m,)
#     _tanh_kernel[grid](
#         input, output,
#         input.stride(0), output.stride(0),
#         m, n, BLOCK_SIZE
#     )
#     return output

# def test_tanh():
#     n, m = 16, 8
#     input_tensor = torch.randn((m, n), dtype=torch.float32, device='cuda')
#     output_tensor = torch.empty_like(input_tensor, device='cuda')

#     solution(input_tensor, output_tensor, n, m)
#     expected_output = torch.tanh(input_tensor)

#     assert torch.allclose(output_tensor, expected_output, atol=1e-2), "Triton and PyTorch results do not match!"

# if __name__ == "__main__":
#     test_tanh()
#     print("All tests passed!")

## the above implementation has some bug on large elements, need to debug, meanwhile here is a working version.

import triton
import triton.language as tl

@triton.jit
def tanh_kernel(
    input, 
    output,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis = 0)
    block_id = pid * BLOCK_SIZE
    offsets = block_id + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    curr_input = tl.load(input + offsets, mask = mask)
    out = (tl.exp(curr_input) - tl.exp(-curr_input) ) / (tl.exp(curr_input) + tl.exp(-curr_input))
    tl.store(output + offsets, out, mask = mask)

# Note: input, output are all float32 device tensors
def solution(input, output, n: int, m: int):
    num_elements = input.numel()
    grid = lambda meta : (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    tanh_kernel[grid](input, output, num_elements, BLOCK_SIZE = 1024)
    
    return output