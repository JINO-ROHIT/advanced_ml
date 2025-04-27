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


############## BENCHMARKING STUFF ##############


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names = ['size'],
        x_vals = [2**i for i in range(12, 22, 2)], 
        line_arg = 'provider',  
        line_vals = ['triton', 'torch'], 
        line_names = ['Triton', 'PyTorch'], 
        styles = [('blue', '-'), ('green', '-')],
        ylabel = 'GB/s', 
        plot_name = 'mish-performance',  
        args={},
    )
)

def benchmark_mish(size, provider):
    input = torch.randn(size, device = 'cuda', dtype = torch.float32)
    output = torch.empty_like(input)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        mish_torch = torch.nn.Mish().cuda()
        mish_torch(input)
        torch.cuda.synchronize()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: mish_torch(input),
            quantiles=quantiles
        )
    else:
        mish_forward(input, output)
        torch.cuda.synchronize()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: mish_forward(input, output),
            quantiles=quantiles
        )
        
    # Bytes processed: read input + write output (float32 = 4 bytes)
    bytes_processed = size * 4 * 2
    gbps = lambda ms: (bytes_processed) * 1e-9 / (ms * 1e-3)
    
    # Verify correctness
    if provider == 'triton':
        torch_mish = torch.nn.Mish().cuda()(input)
        triton_mish = mish_forward(input, output)
        assert torch.allclose(torch_mish, triton_mish, atol = 1e-4), "Output mismatch"
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    benchmark_mish.run(save_path = '.', print_data = True)