import torch
import triton
import triton.language as tl

@triton.jit
def gather_kernel(
    src_ptr, index_ptr, out_ptr,
    n_indices, src_stride,
    BLOCK_SIZE: tl.constexpr
):
    # 每个 block 处理一个连续的 chunk
    pid = tl.program_id(0)
    
    # 当前 block 处理的索引范围
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    
    # 创建 mask 防止越界
    mask = offsets < n_indices
    
    # 加载索引值
    idx = tl.load(index_ptr + offsets, mask=mask, other=0)
    
    # 根据索引从 src 中 gather 数据
    # 假设 src 是 1D tensor，index 也是 1D
    gathered = tl.load(src_ptr + idx * src_stride, mask=mask, other=0)
    
    # 存储结果
    tl.store(out_ptr + offsets, gathered, mask=mask)


def triton_gather(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    实现类似 src[index] 的 gather 操作
    
    Args:
        src: 源张量，形状为 (N,) 或更复杂形状
        index: 索引张量，形状为 (M,)
    
    Returns:
        结果张量，形状与 index 相同
    """
    assert src.is_cuda and index.is_cuda, "Tensors must be on CUDA"
    assert index.dtype in [torch.int32, torch.int64], "Index must be integer type"
    
    # 确保 index 是连续的 1D 张量
    index = index.contiguous().view(-1)
    n_indices = index.numel()
    
    # 处理输出
    output = torch.empty_like(index, dtype=src.dtype)
    
    # 获取 stride，处理多维 src 的第一维
    src_stride = src.stride(0) if src.dim() > 0 else 1
    
    # 转换为 int32 以兼容 Triton (如果必要)
    if index.dtype == torch.int64:
        index = index.to(torch.int32)
    
    # 计算 grid
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_indices, BLOCK_SIZE),)
    
    # 启动 kernel
    gather_kernel[grid](
        src, index, output,
        n_indices, src_stride,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


# ==================== 测试 ====================

if __name__ == "__main__":
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available, falling back to CPU (Triton won't work)")
        exit()
    
    print(f"Using device: {device}")
    
    # 测试数据（与问题中的例子一致）
    src = torch.tensor([0.8980, -0.6503, 0.7372, -2.9284, -0.3910, 
                        -0.3241, 0.5410, -0.3818, 2.0485, -0.9609], 
                       device=device, dtype=torch.float32)
    index = torch.tensor([4, 7, 2], device=device, dtype=torch.int64)
    
    print(f"src: {src}")
    print(f"index: {index}")
    
    # PyTorch 参考结果
    expected = src[index]
    print(f"PyTorch src[index]: {expected}")
    
    # Triton 实现
    result = triton_gather(src, index)
    print(f"Triton gather: {result}")
    
    # 验证正确性
    assert torch.allclose(result, expected), "Results don't match!"
    print("✓ Test passed!")
    
    # 性能测试
    print("\n--- Performance Test ---")
    large_src = torch.randn(1000000, device=device)
    large_index = torch.randint(0, 1000000, (500000,), device=device)
    
    # Warm up
    for _ in range(10):
        _ = triton_gather(large_src, large_index)
        _ = large_src[large_index]
    
    torch.cuda.synchronize()
    
    import time
    
    # Triton timing
    n_iters = 100
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = triton_gather(large_src, large_index)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_iters * 1000
    print(f"Triton gather: {triton_time:.3f} ms")
    
    # PyTorch timing
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = large_src[large_index]
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / n_iters * 1000
    print(f"PyTorch gather: {torch_time:.3f} ms")
