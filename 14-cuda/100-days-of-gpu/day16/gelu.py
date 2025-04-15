import triton
import triton.language as tl

@triton.jit
def gelu_kernel(input, output, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    row = tl.load(input + offsets, mask = mask)

    # 0.5x * (1 + tanh(2 / 3.14 * (x + 0.044715x^3)))
    
    inter = tl.sqrt( (2 / 3.14) * (row + 0.044715 * row * row * row) )
    tanh = (tl.exp(inter) - tl.exp(-inter) ) / (tl.exp(inter) + tl.exp(-inter))
    out = (0.5 * row) * (1 + tanh)
    tl.store(output + offsets, out, mask = mask)

def solution(input, output, n: int, m: int):
    num_elements = input.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)

    gelu_kernel[grid](
        input,
        output,
        num_elements,
        BLOCK_SIZE = 1024,
    )
    return output