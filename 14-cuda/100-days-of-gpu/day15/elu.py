import triton
import triton.language as tl

@triton.jit
def elu_kernel(input, output, n, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    row = tl.load(input + offsets, mask = mask)
    out = tl.where(row > 0, row, alpha * ( tl.exp(row) - 1) )
    tl.store(output + offsets, out, mask = mask)

def solution(input, output, n: int, m: int, alpha: int):
    num_elements = input.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)

    elu_kernel[grid](
        input,
        output,
        num_elements,
        alpha,
        BLOCK_SIZE = 1024
    )
    return output