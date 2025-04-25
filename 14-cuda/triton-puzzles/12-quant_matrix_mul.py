import triton
import triton.language as tl
import torch

@triton.jit
def quant_dot_kernel(scale_ptr, offset_ptr, weight_ptr, activation_ptr,
                     z_ptr, N0, N1, MID, B0: tl.constexpr, B1: tl.constexpr):
    
    pid_0 = tl.program_id(0) 
    pid_1 = tl.program_id(1)


    row_ids = tl.arange(0, B0) + pid_0 * B0
    col_ids = tl.arange(0, B1) + pid_1 * B1


    row_mask = row_ids < N0
    col_mask = col_ids < N1

    acc = tl.zeros((B0, B1), dtype=tl.float32)

    BLOCK_SIZE = 64  # Must be divisible by FPINT * GROUP

    for offset_mid in tl.static_range(0, MID, BLOCK_SIZE):
        FPINT = 8   
        GROUP = 8  

        packed_mid_ids = offset_mid // FPINT + tl.arange(0, BLOCK_SIZE // FPINT)
        packed_weight_offsets = row_ids[:, None] * (MID // FPINT) + packed_mid_ids[None, :]
        packed_weight_mask = row_mask[:, None] & (packed_mid_ids < MID // FPINT)[None, :]

        packed_weights = tl.load(weight_ptr + packed_weight_offsets, mask=packed_weight_mask, other=0)

        # Unpack 4-bit ints using bit shifting
        bit_offsets = tl.arange(0, 8) * 4
        bit_mask = 0xF  # 2^4 - 1
        unpacked = (packed_weights[:, :, None] >> bit_offsets) & bit_mask
        weights = tl.reshape(unpacked, (B0, BLOCK_SIZE))  # now shape: (B0, BLOCK_SIZE)

        packed_offset_ids = offset_mid // (FPINT * GROUP) + tl.arange(0, BLOCK_SIZE // (FPINT * GROUP))
        packed_offset_offsets = row_ids[:, None] * (MID // FPINT // GROUP) + packed_offset_ids[None, :]
        packed_offset_mask = row_mask[:, None] & (packed_offset_ids < MID // FPINT // GROUP)[None, :]

        packed_offsets = tl.load(offset_ptr + packed_offset_offsets, mask=packed_offset_mask, other=0)

        unpacked_offsets = (packed_offsets[:, :, None] >> bit_offsets) & bit_mask
        offsets = tl.reshape(unpacked_offsets, (B0, BLOCK_SIZE // GROUP))


        scale_ids = offset_mid // GROUP + tl.arange(0, BLOCK_SIZE // GROUP)
        scale_offsets = row_ids[:, None] * (MID // GROUP) + scale_ids[None, :]
        scale_mask = row_mask[:, None] & (scale_ids < MID // GROUP)[None, :]

        scales = tl.load(scale_ptr + scale_offsets, mask=scale_mask, other=0.0)

        weights = tl.reshape(weights, (B0, BLOCK_SIZE // GROUP, GROUP))
        offsets = offsets[:, :, None]
        scales = scales[:, :, None]

        dequantized_weights = (weights - offsets) * scales
        x = tl.reshape(dequantized_weights, (B0, BLOCK_SIZE))  # shape (B0, BLOCK_SIZE)

        mid_ids = offset_mid + tl.arange(0, BLOCK_SIZE)
        mid_mask = mid_ids < MID

        activation_offsets = mid_ids[:, None] * N1 + col_ids[None, :]
        activation_mask = mid_mask[:, None] & col_mask[None, :]

        y = tl.load(activation_ptr + activation_offsets, mask=activation_mask, other=0.0)  # shape (BLOCK_SIZE, B1)

        acc += tl.dot(x, y)

    output_offsets = row_ids[:, None] * N1 + col_ids[None, :]
    output_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(z_ptr + output_offsets, acc, mask=output_mask)