import numpy as np 
import torch 
import triton 
import triton.language as tl  
import requests  
from PIL import Image  

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw) 


print(image.size) #(640, 480)

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
    
    grayscale = 0.299 * r + 0.587 * g + 0.114 * b
    tl.store(out_ptr + indices, grayscale, mask=mask)

def greyscale_conversion(image_tensor: torch.Tensor): 
    if image_tensor.device.type == 'cpu':
        image_tensor = image_tensor.cuda()
    
    image_tensor = image_tensor.contiguous()
    
    # Assuming input is CHW format (3, H, W)
    c, h, w = image_tensor.shape
    assert c == 3, "Input must be a 3-channel image in CHW format"
    
    # Create output tensor for grayscale image
    output_img = torch.empty((h, w), dtype=torch.float32, device='cuda')
    
    grid = lambda meta: (
        triton.cdiv(h, meta['BLOCK_SIZE']), 
        triton.cdiv(w, meta['BLOCK_SIZE']),
    )
    
    convert_to_grayscale[grid](
        image_tensor,  
        output_img,
        h, w, 
        BLOCK_SIZE=16
    )
    
    return output_img

# (assuming 'image' is a numpy array in HWC format)
img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).cuda().contiguous()  # Convert HWC to CHW
greyscale_image = greyscale_conversion(img_tensor)


import matplotlib.pyplot as plt
plt.imshow(greyscale_image.cpu().numpy(),cmap='gray')  
plt.show()  