import torch
import triton
import triton.language as tl

@triton.jit
def dot_kernel(x_ptr, y_ptr, z_ptr, N0, N1, N2, MID, B0: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr):
    # Get program IDs for each dimension (parallel over B2 x B0 x B1)
    pid_0 = tl.program_id(0)  # over rows (N0)
    pid_1 = tl.program_id(1)  # over cols (N1)
    pid_2 = tl.program_id(2)  # over batch (N2)

    row_ids = pid_0 * B0 + tl.arange(0, B0)  
    col_ids = pid_1 * B1 + tl.arange(0, B1)  
    batch_id = pid_2 * B2 + tl.arange(0, B2)  

    mask_row = row_ids < N0
    mask_col = col_ids < N1
    mask_batch = batch_id < N2

    acc = tl.zeros((B2, B0, B1), dtype=tl.float32)

    BLOCK_SIZE = 16

    for offset in tl.static_range(0, MID, BLOCK_SIZE):
        mid_ids = offset + tl.arange(0, BLOCK_SIZE)
        mask_mid = mid_ids < MID

        x_offsets = (
            batch_id[:, None, None] * (N0 * MID)
            + row_ids[None, :, None] * MID
            + mid_ids[None, None, :]
        )
        x_mask = (
            mask_batch[:, None, None]
            & mask_row[None, :, None]
            & mask_mid[None, None, :]
        )
        x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        y_offsets = (
            batch_id[:, None, None] * (MID * N1)
            + mid_ids[None, :, None] * N1
            + col_ids[None, None, :]
        )
        y_mask = (
            mask_batch[:, None, None]
            & mask_mid[None, :, None]
            & mask_col[None, None, :]
        )
        y = tl.load(y_ptr + y_offsets, mask=y_mask, other=0.0)

        acc += tl.dot(x, y)

    z_offsets = (
        batch_id[:, None, None] * (N0 * N1)
        + row_ids[None, :, None] * N1
        + col_ids[None, None, :]
    )
    z_mask = (
        mask_batch[:, None, None]
        & mask_row[None, :, None]
        & mask_col[None, None, :]
    )

    tl.store(z_ptr + z_offsets, acc, mask=z_mask)