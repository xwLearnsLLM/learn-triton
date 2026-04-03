import torch

import triton
import triton.language as tl
from triton.runtime import driver


def naive_softmax (x: torch.Tensor) :
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator[:, None]
    return ret


@triton.jit
def softmax_kernel(
    x_ptr       , y_ptr, 
    x_row_stride, y_row_stride, 
    n_rows      , n_cols, 
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr
) :
    row_start = tl.program_id(0)
    row_step  = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages) :
        col_offsets = tl.arange(0, BLOCK_SIZE)
        x_ptrs = (x_ptr + row_idx * x_row_stride) + col_offsets
        y_ptrs = (y_ptr + row_idx * y_row_stride) + col_offsets
        mask = col_offsets < n_cols
        data_row = tl.load(x_ptrs, mask=mask, other=-float('inf'))
        data_row -= tl.max(data_row, axis=0)
        data_row = tl.exp(data_row)
        data_row = data_row / tl.sum(data_row, axis=0)
        tl.store(y_ptrs, data_row, mask=mask)


def triton_softmax (x, y) :
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    SIZE_SMEM = driver.active.utils.get_device_properties(0)["max_shared_mem"]
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    kernel = softmax_kernel.warmup(
        x, y, 
        x.stride(0), y.stride(0),
        n_rows, n_cols, 
        BLOCK_SIZE = BLOCK_SIZE, 
        num_stages = num_stages, 
        num_warps  = 8, 
        grid = (1, )
    )
    kernel._init_handles()
    num_programs = 2
    kernel[(num_programs, 1, 1)](x, y, x.stride(0), y.stride(0), n_rows, n_cols)
    return y


if __name__ == '__main__' :
    x = torch.randn(3, 9, device='cuda')
    y = torch.empty_like(x)
    print(f'triton_softmax(x) = \n{triton_softmax(x, y)}')
    print(f'torch.softmax(x)  = \n{torch.softmax(x, dim=-1)}')
