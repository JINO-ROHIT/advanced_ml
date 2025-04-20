import triton
import triton.language as tl
import torch

"""
TO-DO verify the logic
"""
@triton.jit
def flash_attention_kernel(q_ptr, k_ptr, v_ptr, z_ptr, N, BLOCK_SIZE: tl.constexpr):
    # constant used to compute e^x as exp2(x * log2(e))
    LOG2_E = 1.44269504

    # Outer loop: loop over query blocks
    for q_start in tl.static_range(0, N, BLOCK_SIZE):
        q_idx = q_start + tl.arange(0, BLOCK_SIZE)
        q_mask = q_idx < N
        q_block = tl.load(q_ptr + q_idx, mask=q_mask)

        row_max = (-1e10) + tl.zeros([BLOCK_SIZE], dtype=tl.float32) 
        row_sum_exp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)       
        output_block = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  

        # Inner loop: loop over key/value blocks
        for k_start in tl.static_range(0, N, BLOCK_SIZE):
            k_idx = k_start + tl.arange(0, BLOCK_SIZE)
            k_mask = k_idx < N
            k_block = tl.load(k_ptr + k_idx, mask=k_mask)

            # Compute attention scores (dot product): shape (BLOCK_SIZE, BLOCK_SIZE)
            attn_scores = q_block[:, None] * k_block[None, :]
            attn_scores = tl.where(k_mask[None, :], attn_scores, -1e10)


            new_row_max = tl.maximum(row_max, tl.max(attn_scores, axis=1))
            exp_scale = tl.exp2((row_max - new_row_max) * LOG2_E)
            exp_scores = tl.exp2((attn_scores - new_row_max[:, None]) * LOG2_E)
            new_row_sum_exp = row_sum_exp * exp_scale + tl.sum(exp_scores, axis=1)
            attn_probs = exp_scores / new_row_sum_exp[:, None]

            v_block = tl.load(v_ptr + k_idx, mask=k_mask)

            output_block = output_block * (exp_scale * row_sum_exp / new_row_sum_exp)
            output_block += tl.sum(attn_probs * v_block[None, :], axis=1)
            row_max = new_row_max
            row_sum_exp = new_row_sum_exp

        tl.store(z_ptr + q_idx, output_block, mask=q_mask)