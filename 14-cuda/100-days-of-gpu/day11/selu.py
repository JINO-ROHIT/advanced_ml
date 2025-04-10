import triton
import triton.language as tl

@triton.jit
def selu_kernel(input, output, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    row = tl.load(input + offsets, mask = mask)
    # 1.0507 ∗ (max(0, x) + min(0, 1.67326 ∗ (exp(x) − 1)))
    out = 1.0507 * (tl.maximum(0, row) + 1.67326 * tl.minimum(0, tl.exp(row) - 1))
    tl.store(output + offsets, out, mask = mask)

def solution(input, output, n: int, m: int):
    num_elements = input.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)

    selu_kernel[grid](
        input,
        output,
        num_elements,
        BLOCK_SIZE = 1024
    )
    return output