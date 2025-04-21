import triton
import triton.language as tl
import torch


@triton.jit
def conv2d_kernel(x_ptr, k_ptr, z_ptr, N0, H, W, KH: tl.constexpr, KW: tl.constexpr, B0: tl.constexpr):
    pid_0 = tl.program_id(0)

    batch_offset = pid_0 * H * W
    
    h_idx = tl.arange(0, KH)  # [0, 1, ..., KH-1]
    w_idx = tl.arange(0, KW)  # [0, 1, ..., KW-1]
    kernel = tl.load(k_ptr + h_idx[:, None] * KW + w_idx[None, :])  # shape: [KH, KW]
    
    # Loop over every pixel (i,j) in the output image (same size as input here)
    for i in tl.static_range(0, H):
        for j in tl.static_range(0, W):
            # For each (i,j), we extract a window from the input x_ptr
            h_pos = i + h_idx  
            w_pos = j + w_idx 
    
            h_mask = h_pos < H
            w_mask = w_pos < W
            mask = h_mask[:, None] & w_mask[None, :]
    
            # Compute the flattened indices for input window
            flat_indices = h_pos[:, None] * W + w_pos[None, :]  # shape: [KH, KW]
    
            x_window = tl.load(x_ptr + batch_offset + flat_indices, mask = mask, other = 0.0)
    
            # Element-wise multiply with kernel and sum for convolution
            conv_result = tl.sum(x_window * kernel)
    
            out_index = batch_offset + i * W + j
            tl.store(z_ptr + out_index, conv_result)