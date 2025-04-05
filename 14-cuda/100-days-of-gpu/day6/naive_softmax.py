import triton
import triton.language as tl
import torch

@triton.jit
def naive_softmax_kernel(input_ptr, input_row_stride, output_ptr, output_row_stride, n_rows, n_cols:tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    
    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        offsets = tl.arange(0, n_cols)
        mask = offsets < n_cols
        row = tl.load(row_start_ptr + offsets, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + offsets, softmax_output, mask=mask)

def softmax(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    grid = (n_rows,)  # One program per row
    naive_softmax_kernel[grid](
        x, x.stride(0), y, y.stride(0), n_rows, n_cols
    )
    return y

if __name__ == "__main__":
    x = torch.randn(4, 4, device="cuda")
    y = softmax(x)
    print("Input:")
    print(x)
    print("Softmax Output:")
    print(y)
    y_triton = softmax(x)
    y_torch = torch.softmax(x, dim=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)