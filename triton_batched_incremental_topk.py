import torch
import triton
import triton.language as tl



@triton.jit
def kernel_BatchedRaggedIncrementalTopK (
    iop_score,             # (B1, B2, SIZE_N), bf16  - 候选分数
    iop_selected,          # (B1, B2, SIZE_M), int32 - 已选候选的全局索引 , 取值范围 (0~N-1)
    op_inc_selected,       # (B1, B2, SIZE_K), int32 - 输出: 新增候选的全局索引
    # op_discard_ids,        # (B1, B2, SIZE_K), int32 - 输出: 淘汰的局部索引 (0~M-1)
    ip_n,                  # (B1,), int32
    ip_m,                  # (B1,), int32
    ip_k,                  # (B1,), int32
    ip_dst_ids,            # (B1, SIZE_D), int32
    ip_src_ids,            # (B1, SIZE_S), int32
    op_dst_src_ids,        # (2, B2, SIZE_X), int32
    B2      : tl.constexpr,
    B1_2    : tl.constexpr,
    SIZE_N  : tl.constexpr,
    SIZE_M  : tl.constexpr,
    SIZE_K  : tl.constexpr,
    SIZE_N_2: tl.constexpr,
    SIZE_M_2: tl.constexpr,
    SIZE_D  : tl.constexpr,
    SIZE_S  : tl.constexpr,
    SIZE_X  : tl.constexpr
) :
    i_batch = tl.program_id(0)          # 每个block处理一个batch 
    i_b1    = i_batch // B2
    i_b2    = i_batch %  B2
    
    N = tl.load(ip_n + i_b1)            # 该batch的 N
    M = tl.load(ip_m + i_b1)            # 该batch的 M
    K = tl.load(ip_k + i_b1)            # 该batch的 K
    
    valid = (N>0) & (M>0) & (K>0) & (N<=SIZE_N) & (M<=SIZE_M) & (N<=SIZE_N_2) & (M<=SIZE_M_2) & (N>=(M+K)) & (M>=K)
    N = tl.where(valid, N, 0)
    M = tl.where(valid, M, 0)
    K = tl.where(valid, K, 0)

    tl.debug_barrier()
    
    iop_score       += i_batch * SIZE_N    # 当前batch的scores基址
    iop_selected    += i_batch * SIZE_M    # 当前batch的old_selected基址
    op_inc_selected += i_batch * SIZE_K    # 输出inc_selected基址
    # op_discard_ids  += i_batch * SIZE_K    # 输出discard_ids基址
    ip_dst_ids      += i_b1 * SIZE_D 
    ip_src_ids      += i_b1 * SIZE_S 
    k_start = tl.sum(tl.where(tl.arange(0,B1_2)<i_b1, tl.load(ip_k+tl.arange(0,B1_2)), 0))   # 计算前缀和 k_start = sum(ip_k[:i_b1])
    op_dst_ids = op_dst_src_ids + (i_b2 * SIZE_X) + k_start
    op_src_ids = op_dst_ids + (B2 * SIZE_X)

    tl.debug_barrier()
    
    f32_smaller = float('-2e5')
    f32_small   = float('-1e5')
    f32_large   = float('1e5')
    f32_larger  = float('2e5')

    tl.debug_barrier()
    
    n_range = tl.arange(0, SIZE_N_2)       # N个位置
    m_range = tl.arange(0, SIZE_M_2)       # M个位置
    m_bf16_smaller = tl.full((SIZE_M_2,), f32_smaller, dtype=tl.bfloat16)

    tl.debug_barrier()

    # 把 score 钳制到 -1e8 ~ 1e8 范围内，之后就可以放心的使用 -1e9 作为降序排序的无效填充值， 使用 1e9 作为升序排序的无效填充值
    n_bf16_score = tl.load((iop_score+n_range), mask=(n_range<N))
    n_bf16_score = tl.maximum(n_bf16_score, f32_small)
    n_bf16_score = tl.minimum(n_bf16_score, f32_large)
    tl.store((iop_score+n_range), n_bf16_score, mask=(n_range<N))

    tl.debug_barrier()
    
    # old_sel 按照分数从低到高排序，得到排序后的索引 m_i32_old_id_asc (取值范围 0~M-1)
    m_i32_old_sel = tl.load((iop_selected+m_range), mask=(m_range<M), other=0)                                                     # 加载M个全局索引
    m_f32_old_score = tl.load((iop_score+m_i32_old_sel), mask=(m_range<M), other=f32_larger).to(tl.float32)                        #
    m_f32_old_score_id = ((m_f32_old_score.to(tl.uint32, bitcast=True)&0xFFFF0000) | (m_range&0xFFFF)).to(tl.float32, bitcast=True) # 高16bit分数,低16bit局部索引, 转回float32
    m_f32_old_score_id_asc = tl.sort(m_f32_old_score_id, descending=False)                                                         # 升序排序 (分数低的在前)
    m_i32_old_id_asc = m_f32_old_score_id_asc.to(tl.int32, bitcast=True) & 0xFFFF

    # m_i32_old_id_asc[0:K] 写入 op_discard_ids
    #tl.store((op_discard_ids+m_range), m_i32_old_id_asc, mask=(m_range<K))

    tl.debug_barrier()

    # 把 score 中的 old_sel mask 掉。将 f32_smaller 写入 iop_score[old_sel] (scatter store) , 再读上来，得到的就是 mask 掉 old_sel 的 scores
    tl.store((iop_score+m_i32_old_sel), m_bf16_smaller, mask=(m_range<M))
    n_f32_score = tl.load((iop_score+n_range), mask=(n_range<N), other=f32_smaller).to(tl.float32)

    tl.debug_barrier()

    # 
    n_f32_score_id = ((n_f32_score.to(tl.uint32, bitcast=True)&0xFFFF0000) | (n_range&0xFFFF)).to(tl.float32, bitcast=True)
    n_f32_score_id_dsc = tl.sort(n_f32_score_id, descending=True)
    n_i32_id_dsc = n_f32_score_id_dsc.to(tl.int32, bitcast=True) & 0xFFFF 

    tl.debug_barrier()

    #
    tl.store((op_inc_selected+n_range), n_i32_id_dsc, mask=(n_range<K))

    tl.debug_barrier()

    #
    m_i32_inc_sel = tl.load((op_inc_selected+m_range), mask=(m_range<K), other=0)
    tl.store((iop_selected+m_i32_old_id_asc), m_i32_inc_sel, mask=(m_range<K))

    tl.debug_barrier()

    # gather src block table 并写入 op_src_ids
    m_i32_inc_sel = tl.maximum(m_i32_inc_sel, 0)
    m_i32_inc_sel = tl.minimum(m_i32_inc_sel, N-1)
    src_ids = tl.load((ip_src_ids+m_i32_inc_sel), mask=(m_range<K), other=0)
    tl.store((op_src_ids+m_range), src_ids, mask=(m_range<K))

    tl.debug_barrier()

    # gather dst block table 并写入 op_dst_ids
    m_i32_old_id_asc = tl.maximum(m_i32_old_id_asc, 0)
    m_i32_old_id_asc = tl.minimum(m_i32_old_id_asc, M-1)
    dst_ids = tl.load((ip_dst_ids+m_i32_old_id_asc), mask=(m_range<K), other=0)
    tl.store((op_dst_ids+m_range), dst_ids, mask=(m_range<K))



# 函数: batched_ragged_incremental_topk_with_block_scatter
def batched_ragged_incremental_topk_with_block_scatter (
    scores  : torch.Tensor,        # (B1, B2, SIZE_N), bf16
    selected: torch.Tensor,        # (B1, B2, SIZE_M), int32
    tensor_n: torch.Tensor,        # (B1,), int32
    tensor_m: torch.Tensor,        # (B1,), int32
    tensor_k: torch.Tensor,        # (B1,), int32
    dst_ids: torch.Tensor,         # (B1, SIZE_D), int32
    src_ids: torch.Tensor,         # (B1, SIZE_S), int32
    dst_src_ids: torch.Tensor      # (2, B2, SIZE_X), int32
) :
    assert scores.dtype == torch.bfloat16 and scores.is_contiguous()
    assert selected.dtype == torch.int32  and selected.is_contiguous()
    assert tensor_n.dtype == torch.int32  and tensor_n.is_contiguous()
    assert tensor_m.dtype == torch.int32  and tensor_m.is_contiguous()
    assert tensor_k.dtype == torch.int32  and tensor_k.is_contiguous()
    assert     dst_ids.dtype == torch.int32  and     dst_ids.is_contiguous()
    assert     src_ids.dtype == torch.int32  and     src_ids.is_contiguous()
    assert dst_src_ids.dtype == torch.int32  and dst_src_ids.is_contiguous()

    B1, B2, SIZE_N = scores.shape
    _ , _ , SIZE_M = selected.shape
    _ ,     SIZE_D = dst_ids.shape
    _ ,     SIZE_S = src_ids.shape
    _ , _ , SIZE_X = dst_src_ids.shape
    SIZE_K = tensor_k.max().item()

    assert (B1, B2, SIZE_M) == selected.shape
    assert B1 == tensor_n.shape[0]
    assert B1 == tensor_m.shape[0]
    assert B1 == tensor_k.shape[0]
    assert B1 == dst_ids.shape[0]
    assert B1 == src_ids.shape[0]
    assert (2, B2, SIZE_X) == dst_src_ids.shape
    
    inc_selected = torch.empty((B1, B2, SIZE_K), dtype=torch.int32, device=scores.device)
    
    kernel_BatchedRaggedIncrementalTopK[(B1*B2,)](       # 启动 kernel , 每个 batch 一个 block
        scores, selected, inc_selected,
        tensor_n, tensor_m, tensor_k,
        dst_ids, src_ids, dst_src_ids,
        B2,
        triton.next_power_of_2(B1),
        SIZE_N, SIZE_M, SIZE_K,
        triton.next_power_of_2(tensor_n.max().item()),
        triton.next_power_of_2(tensor_m.max().item()),
        SIZE_D, SIZE_S, SIZE_X
    )

    return inc_selected



# 简单测试 
if True :
    if   torch.cuda.is_available() :
        device = 'cuda'
    elif torch.npu.is_available() :
        device = 'npu'
    else :
        assert False, 'cuda or npu not available'
    

    tensor_n = torch.tensor([13, 17, 16, 4, 0, 10, 10], device=device, dtype=torch.int32)
    tensor_m = torch.tensor([ 7, 10,  8, 3, 0,  5,  7], device=device, dtype=torch.int32)
    tensor_k = torch.tensor([ 3,  4,  2, 0, 0,  5,  2], device=device, dtype=torch.int32)
    B1 = tensor_n.shape[0]
    B2 = 10
    SIZE_D = tensor_m.max().item()
    SIZE_S = tensor_n.max().item()
    SIZE_X = tensor_k.sum().item()
    
    scores   = torch.randn((B1, B2, tensor_n.max().item()), device=device, dtype=torch.bfloat16)  # 生成随机分数
    selected = torch.empty((B1, B2, tensor_m.max().item()), device=device, dtype=torch.int32)
    for (i_b1, M) in enumerate(tensor_m) :                                                        # 生成随机排列的 selected
        for i_b2  in range(B2) :
            selected[i_b1, i_b2, :M] = torch.randperm(tensor_n[i_b1].item())[:M]
    
    dst_ids = torch.arange(B1*SIZE_D, dtype=torch.int32, device=device).view(B1, SIZE_D)
    src_ids = torch.arange(B1*SIZE_S, dtype=torch.int32, device=device).view(B1, SIZE_S)
    dst_src_ids = torch.zeros((2, B2, SIZE_X), dtype=torch.int32, device=device)

    scores_old   = scores.clone()
    selected_old = selected.clone()
    
    inc_selected = batched_ragged_incremental_topk_with_block_scatter(scores, selected, tensor_n, tensor_m, tensor_k, dst_ids, src_ids, dst_src_ids) # 调用算子
    
    
if __name__ == '__main__' :
    def is_no_duplicates_on_last_dim (src_tensor):
        sorted_tensor, _ = torch.sort(src_tensor, dim=-1)
        duplicates = (sorted_tensor[..., :-1] == sorted_tensor[..., 1:])
        has_duplicate = duplicates.any(dim=-1)
        return not has_duplicate.any()
    
    all_K = 0
    for (i_b1, (N, M, K)) in enumerate(zip(tensor_n, tensor_m, tensor_k)) :
        for i_b2  in range(B2) :
            print(f'\nbatch({i_b1},{i_b2}):  N={N}, M={M}, K={K}')
            print(f'    scores        = {scores_old  [i_b1, i_b2, :N]}')
            print(f'    selected(old) = {selected_old[i_b1, i_b2, :M].tolist()}')
            print(f'    selected(new) = {selected    [i_b1, i_b2, :M].tolist()}')
            print(f'    selected(inc) = {inc_selected[i_b1, i_b2, :K].tolist()}')
            print(f'    dst_ids       = {dst_ids     [i_b1      , : ].tolist()}')
            print(f'    src_ids       = {src_ids     [i_b1      , : ].tolist()}')
            print(f'    dst_src_ids   = {dst_src_ids [   :, i_b2, all_K:all_K+K].tolist()}')
            assert is_no_duplicates_on_last_dim(selected_old[i_b1, i_b2, :M]), 'duplicates_on_last_dim(old_sel)'
            assert is_no_duplicates_on_last_dim(selected    [i_b1, i_b2, :M]), 'duplicates_on_last_dim(new_sel)'
            assert is_no_duplicates_on_last_dim(inc_selected[i_b1, i_b2, :K]), 'duplicates_on_last_dim(inc_sel)'
        all_K += K

    