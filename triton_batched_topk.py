import torch
import triton
import triton.language as tl



@triton.jit
def kernel_BatchedTopK (
    iptr_score,                 # shape = (B, N), bfloat16
    optr_topk_idxs,             # shape = (B, K), int32
    B: tl.constexpr,            # batch-size
    N: tl.constexpr,            # will get top-K on this dim
    K: tl.constexpr,            # top-K
    BLK_SIZE: tl.constexpr,
) :
    batch_id = tl.program_id(0)                                             # 每个 block 处理一个 batch
    idxs = tl.arange(0, BLK_SIZE)                                           # 范围 (block大小)
    score_bf16 = tl.load(                                                   # 加载 bf16 分数
        iptr_score +                                                        # batched_score 基地址
        (batch_id * N) +                                                    # 当前 batch 在 iptr_score 的偏移
        idxs,                                                               # 范围
        mask  = (idxs < N),                                                 # 掩码
        other = float('-inf')                                               # 被掩码的无效值，取负无穷
    )
    score_f32 = ((score_bf16.to(tl.float32).to(tl.int32, bitcast=True) & 0xFFFF0000) | idxs.to(tl.int32)).to(tl.float32, bitcast=True) # bf16转fp32，并在低16位编码索引信息
    sorted_score_f32 = tl.sort(score_f32, descending=True, dim=0)           # 使用 triton sort 进行降序排序，高16位是分数，低16位是索引
    sorted_idxs_i32  = sorted_score_f32.to(tl.int32, bitcast=True) & 0xFFFF # 转 int32, 提取前 K 个的索引 (从低16位)
    tl.store(                                                               # 写入topk结果
        optr_topk_idxs +
        (batch_id * K) + 
        idxs,
        sorted_idxs_i32,
        mask = (idxs < K)
    )



# triton kernel: ragged_batched_topk_kernel
# 对 batched_scores 做变长批量 topk，每个 batch (i, j, k) 的有效长度和 topk 数量由 batched_M[j] 和 batched_K[j] 决定
# 技巧：由于 triton 只有 sort 并没有 argsort，所以这里用 sort 实现 argsort 的效果。具体来讲：将 bf16 转为 fp32，然后把索引编码到低16位，排序后提取低16位得到 topk 索引
@triton.jit
def kernel_BatchedRaggedTopK (
    batched_scores_ptr,
    batched_M_ptr,
    batched_K_ptr,
    batched_topk_ptr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B3: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    total_blocks = B1 * B2 * B3                                            # 计算总 block 数
    block_id = tl.program_id(0)                                            # 获取当前 block id
    if block_id >= total_blocks:                                           # 边界检查
        return
    k_idx = block_id % B3                                                  # 解码 k 维度索引
    tmp = block_id // B3                                                   # 临时变量用于继续解码
    j_idx = tmp % B2                                                       # 解码 j 维度索引（batch_2）
    i_idx = tmp // B2                                                      # 解码 i 维度索引
    M = tl.load(batched_M_ptr + j_idx)                                     # 加载当前 batch_2 的有效长度 M
    K = tl.load(batched_K_ptr + j_idx)                                     # 加载当前 batch_2 的 topk 数量 K
    batch_offset = ((i_idx * B2 + j_idx) * B3 + k_idx) * N                 # 计算当前 batch 在输入中的偏移量
    offs_n = tl.arange(0, BLOCK_N)                                         # 创建索引范围 [0, BLOCK_N)
    mask_valid = offs_n < M                                                # 有效数据掩码（前 M 个有效）
    mask_n = offs_n < N                                                    # 总范围掩码（防止越界）
    score_bf16 = tl.load(batched_scores_ptr + batch_offset + offs_n, mask=mask_n, other=float('-inf'))  # 加载 bf16 分数，越界位置设为 -inf
    score_bf16 = tl.where(mask_valid, score_bf16, float('-inf'))           # 将无效数据（M 之后）设为 -inf
    score_fp32 = score_bf16.to(tl.float32)                                 # 将 bf16 转为 fp32
    indices = tl.arange(0, BLOCK_N)                                        # 使用 tl.arange 创建索引
    score_as_int = score_fp32.to(tl.int32, bitcast=True)                   # 将 fp32 转为 int32，准备修改低16位
    score_high = score_as_int & 0xFFFF0000                                 # 清除低16位，保留高16位（分数部分）
    encoded = score_high | (indices.to(tl.int32) & 0xFFFF)                 # 加上索引（放到低16位），注意索引需要是 int32
    encoded_fp32 = encoded.to(tl.float32, bitcast=True)                    # 转回 fp32
    sorted_encoded = tl.sort(encoded_fp32, descending=True, dim=0)         # 使用 triton sort 进行降序排序，高16位是分数，低16位是索引
    sorted_as_int = sorted_encoded.to(tl.int32, bitcast=True)              # 将排序结果转回 int32
    topk_indices = (sorted_as_int & 0xFFFF).to(tl.int32)                   # 提取前 K 个的索引（从低16位）
    out_offset = ((i_idx * B2 + j_idx) * B3 + k_idx) * N                   # 计算输出偏移量
    offs_k = tl.arange(0, BLOCK_N)                                         # 创建输出索引范围
    mask_k = offs_k < K                                                    # 只存储前 K 个结果
    tl.store(batched_topk_ptr + out_offset + offs_k, topk_indices, mask=mask_k)  # 存储前 K 个索引



# 函数: triton_batched_topk
# 对 batched_score 的第二个维度求 topk，返回 topk 的索引，dtype=torch.int32
def triton_batched_topk (batched_score: torch.Tensor, K: int) -> torch.Tensor :
    B, N = batched_score.shape
    assert batched_score.dtype == torch.bfloat16, f"Expected bfloat16, got {batched_score.dtype}"
    assert N <= 65536, f"N must be <= 65536, got {N}"
    assert K <= 65536, f"K must be <= 65536, got {K}"
    assert K <= N    , f"K must be <= N, got K={K}, N={N}"
    topk_idxs = torch.empty((B, K), dtype=torch.int32, device=batched_score.device)                       # 分配输出内存
    kernel_BatchedTopK[(B,)](batched_score, topk_idxs, B=B, N=N, K=K, BLK_SIZE=triton.next_power_of_2(N)) # 每个 batch 分配一个 block
    return topk_idxs



# 函数: triton_ragged_batched_topk
# 对 batched_scores 做变长批量 topk，每个 batch_2 的有效长度和 topk 数量可以不同，返回 topk 的索引，dtype=torch.int32
# 具体来说：对于每个 batch (i, j, k)，在 batched_scores[i, j, k, :batched_M[j]] 中找出 topk=batched_K[j]，结果放在 batched_topk[i, j, k, :batched_K[j]]
def triton_ragged_batched_topk (batched_scores: torch.Tensor, batched_M: torch.Tensor, batched_K: torch.Tensor) -> torch.Tensor :
    """
    在最后一个维度上找出 topk 索引，但是对于每个 batch_size_2，候选数量不一样，topk也不一样。
    具体来说是对于每个 batch (i, j, k)，在 batched_scores[i, j, k, :batched_M[j]] 中找出 topk=batched_K[j]
    并把 topk 索引放在 return batched_topk[i, j, k, :batched_K[j]]
    实现方式：将 batched_scores 每个维度中后面的无效数据 mask 成极小值，然后使用 tl.sort 降序排序取前K个

    Args:
        batched_scores: 需要被选择的分数, shape=(B1, B2, B3, N), dtype=torch.bfloat16
        batched_M: 每个batch_2的候选总数, shape=(B2,), dtype=torch.int32, batched_M[j] <= N
        batched_K: 每个batch_2需要选出的数量, shape=(B2,), dtype=torch.int32, batched_K[j] <= batched_M[j]
    Returns:
        batched_topk: TopK结果索引, shape=(B1, B2, B3, N), dtype=torch.int32, 只有前batched_K[j]个有效
    """
    B1, B2, B3, N = batched_scores.shape                                   # 解包输入 shape
    assert batched_scores.dtype == torch.bfloat16, f"Expected bfloat16, got {batched_scores.dtype}"  # 验证数据类型为 bfloat16
    assert batched_M.dtype == torch.int32, f"batched_M must be int32, got {batched_M.dtype}"         # 验证 batched_M 类型
    assert batched_K.dtype == torch.int32, f"batched_K must be int32, got {batched_K.dtype}"         # 验证 batched_K 类型
    assert batched_M.shape == (B2,), f"batched_M shape must be ({B2},), got {batched_M.shape}"       # 验证 batched_M 形状
    assert batched_K.shape == (B2,), f"batched_K shape must be ({B2},), got {batched_K.shape}"       # 验证 batched_K 形状
    assert N < 32768, f"N must be < 32768, got {N}"                        # 确保 N 在有效范围内（索引编码到低16位）
    device = batched_scores.device                                         # 获取设备信息
    batched_M = batched_M.to(device)                                       # 确保 batched_M 在同设备
    batched_K = batched_K.to(device)                                       # 确保 batched_K 在同设备
    topk_idxs = torch.empty((B1, B2, B3, N), dtype=torch.int32, device=device)  # 分配输出内存，shape=(B1, B2, B3, N)
    BLOCK_N = triton.next_power_of_2(N)                                    # 确定 block 大小（需要是2的幂次，且 >= N）
    total_blocks = B1 * B2 * B3                                            # 计算总 block 数
    grid = (total_blocks,)                                                 # 每个 (i, j, k) 组合一个 block
    kernel_BatchedRaggedTopK[grid](batched_scores, batched_M, batched_K, topk_idxs, B1=B1, B2=B2, B3=B3, N=N, BLOCK_N=BLOCK_N)  # 启动 kernel
    return topk_idxs    



if __name__ == "__main__" :    # 简单测试并和 torch 标准实现对比

    # 测试 triton_batched_topk -------------------------------------------------------------
    B, N, K = 3, 1024, 9
    batched_score = torch.randn(B, N, dtype=torch.bfloat16, device='cuda')
    triton_idxs = triton_batched_topk(batched_score, K)         # triton 实现
    _, torch_idxs = torch.topk(batched_score, K, dim=-1)        # torch 实现
    print(triton_idxs)
    print(torch_idxs)
    print('\n')

    # 测试 triton_ragged_batched_topk -------------------------------------------------------------
    B1, B2, B3, N = 1, 4, 2, 16
    batched_scores = torch.randn(B1, B2, B3, N, dtype=torch.bfloat16, device='cuda')   # 生成随机分数
    batched_M = torch.tensor([14,  9, 16, 1], dtype=torch.int32, device='cuda')
    batched_K = torch.tensor([ 9,  6,  4, 0], dtype=torch.int32, device='cuda')
    triton_topk = triton_ragged_batched_topk(batched_scores, batched_M, batched_K)     # Triton 实现
    _, torch_topk = torch.topk(batched_scores, 16, dim=-1)                             # torch 实现
    print(triton_topk)
    print(torch_topk)
    print('\n')



