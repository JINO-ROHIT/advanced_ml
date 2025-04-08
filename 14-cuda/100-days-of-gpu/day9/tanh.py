import triton
import triton.language as tl

@triton.jit 
def _tanh_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,    
    n_rows, n_cols,                         
    BLOCK_SIZE: tl.constexpr,               
): 
    row_start = tl.program_id(0) 
    row_step = tl.num_programs(0) 

    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        out_row_start_ptr = output_ptr + row_idx * output_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=None) 
        exp_pos = tl.exp(row)
        exp_neg = tl.exp(-row)
        result = (exp_pos - exp_neg) / (exp_pos + exp_neg)
        tl.store(out_row_start_ptr + col_offsets, result, mask = mask)

def solution(input, output, n: int, m: int):
    BLOCK_SIZE = triton.next_power_of_2(n)

    grid = (m,)
    _tanh_kernel[grid](
        input, output,
        input.stride(0), output.stride(0),
        m, n, BLOCK_SIZE
    )
    return output