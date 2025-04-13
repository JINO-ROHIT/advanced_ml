import triton
import triton.language as tl

@triton.jit
def softplus_kernel(input, output, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    row = tl.load(input + offsets, mask = mask)
    # ln(1 + e(x))
    out = tl.log(1 + tl.exp(row))
    tl.store(output + offsets, out, mask = mask)

def solution(input, output, n: int, m: int):
    num_elements = input.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)

    softplus_kernel[grid](
        input,
        output,
        num_elements,
        BLOCK_SIZE = 1024
    )
    return output