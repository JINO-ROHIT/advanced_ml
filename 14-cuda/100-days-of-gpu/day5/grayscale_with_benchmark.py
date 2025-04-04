import numpy as np
import torch
import triton
import triton.language as tl
import time
from PIL import Image
import requests
import matplotlib.pyplot as plt

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
print(f"Image size: {image.size}")  # (640, 480)

@triton.jit
def convert_to_grayscale(x_ptr, out_ptr, height, width, BLOCK_SIZE: tl.constexpr):
    pid_h = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)
    
    block_start_h = pid_h * BLOCK_SIZE
    block_start_w = pid_w * BLOCK_SIZE
    
    offset_h = block_start_h + tl.arange(0, BLOCK_SIZE)
    offset_w = block_start_w + tl.arange(0, BLOCK_SIZE)
    
    mask_h = offset_h < height
    mask_w = offset_w < width
    
    h_indices = offset_h[:, None]
    w_indices = offset_w[None, :]
    
    mask = mask_h[:,None] & mask_w[None,:]
    indices = h_indices * width + w_indices
    
    r = tl.load(x_ptr + indices, mask=mask)
    g = tl.load(x_ptr + indices + height * width, mask=mask)
    b = tl.load(x_ptr + indices + 2 * height * width, mask=mask)
    
    # Standard grayscale conversion weights
    grayscale = 0.2126 * r + 0.7152 * g + 0.0722 * b
    tl.store(out_ptr + indices, grayscale, mask=mask)

# Triton 
def greyscale_conversion(image_tensor: torch.Tensor, block_size=16):
    if image_tensor.dim() == 3 and image_tensor.shape[2] == 3:
        image_tensor = image_tensor.permute(2, 0, 1)
    
    if image_tensor.device.type == 'cpu':
        image_tensor = image_tensor.cuda()
    
    image_tensor = image_tensor.contiguous()
    
    c, h, w = image_tensor.shape
    assert c == 3, "Input must be a 3-channel image"
    
    output_img = torch.empty((h, w), dtype=torch.float32, device='cuda')
    
    grid = lambda meta: (
        triton.cdiv(h, meta['BLOCK_SIZE']), 
        triton.cdiv(w, meta['BLOCK_SIZE']),
    )
    
    convert_to_grayscale[grid](
        image_tensor, 
        output_img,
        h, w, 
        BLOCK_SIZE=block_size
    )
    
    return output_img

# PyTorch grayscale conversion
def torch_grayscale(image_tensor: torch.Tensor):
    if image_tensor.dim() == 3 and image_tensor.shape[2] == 3:
        image_tensor = image_tensor.permute(2, 0, 1)
    
    if image_tensor.device.type == 'cpu':
        image_tensor = image_tensor.cuda()
    
    r, g, b = image_tensor[0], image_tensor[1], image_tensor[2]
    grayscale = 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    return grayscale

# Benchmark with different block sizes
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['block_size'],                             
        x_vals=[8, 16, 32, 64],
        x_log=True,
        line_arg='provider',                              
        line_vals=['triton', 'torch'],                    
        line_names=['Triton', 'Torch'],                   
        styles=[('blue', '-'), ('green', '-')],           
        ylabel='GB/s',                                    
        plot_name='grayscale-performance-vs-block-size(P100)',
        args={},                                          
    )
)
def benchmark_grayscale(block_size, provider):
    img_tensor = torch.from_numpy(np.array(image)).float().cuda()
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_grayscale(img_tensor),
            quantiles=quantiles
        )
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: greyscale_conversion(img_tensor, block_size),
            quantiles=quantiles
        )
    
    gbps = lambda ms: (img_tensor.numel() * 4) * 1e-9 / (ms * 1e-3)
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark_grayscale.run(show_plots=True, print_data=True)