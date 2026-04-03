import torch
import triton
import triton.language as tl

from flash_attn import flash_attn_with_kvcache # 用来计算参考结果



@triton.jit
def kernel_page_gqa_decode (
    o_ptr,                               # (BSZ, N_KV_HEAD, N_Q_PER_KV_HEAD, HEAD_DIM), bf16
    q_ptr,                               # (BSZ, N_KV_HEAD, N_Q_PER_KV_HEAD, HEAD_DIM), bf16
    k_ptr,                               # (N_BLOCK, BLOCK_SIZE,  N_KV_HEAD, HEAD_DIM), bf16
    v_ptr,                               # (N_BLOCK, BLOCK_SIZE,  N_KV_HEAD, HEAD_DIM), bf16
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    seq_lens_ptr,                        # (BSZ, )                  , int32
    block_tables_ptr,                    # (BSZ, MAX_N_BLOCK_OF_SEQ), int32
    stride_block_tables_0,
    N_Q_PER_KV_HEAD   : tl.constexpr,
    N_Q_PER_KV_HEAD_2 : tl.constexpr,
    HEAD_DIM          : tl.constexpr,
    HEAD_DIM_2        : tl.constexpr,
    MAX_SEQ_LEN       : tl.constexpr,
    SCALE             : tl.constexpr,
    BLOCK_SIZE        : tl.constexpr,
    TILE_SIZE         : tl.constexpr
):
    i_batch  = tl.program_id(0)
    i_kvhead = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + i_batch)               # 获取当前 batch 的序列长度
    block_tables_ptr += i_batch * stride_block_tables_0     # 定位到当前 batch 的 block table 的起始
    
    o_ptr += i_batch * stride_q0 + i_kvhead * stride_q1     # 指针定位到 o_ptr[i_batch, i_kvhead, :, :]
    q_ptr += i_batch * stride_q0 + i_kvhead * stride_q1     # 指针定位到 q_ptr[i_batch, i_kvhead, :, :]
    k_ptr +=                       i_kvhead * stride_k2     # 指针定位到 k_ptr[:, :   , i_kvhead, :]
    v_ptr +=                       i_kvhead * stride_k2     # 指针定位到 v_ptr[:, :   , i_kvhead, :]

    offs_d    = tl.arange(0, HEAD_DIM_2)                                           # shape: (HEAD_DIM,)
    offs_qh   = tl.arange(0, N_Q_PER_KV_HEAD_2)                                    # shape: (N_Q_PER_KV_HEAD,)
    offs_qh_d =  offs_qh[:, None] * stride_q2      +  offs_d[None, :]              # shape: (N_Q_PER_KV_HEAD, HEAD_DIM)
    mask_qh_d = (offs_qh[:, None]<N_Q_PER_KV_HEAD) & (offs_d[None, :]<HEAD_DIM)    # shape: (N_Q_PER_KV_HEAD, HEAD_DIM)
    
    q = tl.load(q_ptr+offs_qh_d, mask=mask_qh_d, other=0.0)                        # shape: (N_Q_PER_KV_HEAD, HEAD_DIM)
    q = q.to(tl.float32) * SCALE                                                   # shape: (N_Q_PER_KV_HEAD, HEAD_DIM)

    m_i   = tl.full ([N_Q_PER_KV_HEAD_2,], value=float("-inf"), dtype=tl.float32)  # shape: (N_Q_PER_KV_HEAD,)
    l_i   = tl.zeros([N_Q_PER_KV_HEAD_2,]                     , dtype=tl.float32)  # shape: (N_Q_PER_KV_HEAD,)
    acc_o = tl.zeros([N_Q_PER_KV_HEAD_2, HEAD_DIM_2]            , dtype=tl.float32)  # shape: (N_Q_PER_KV_HEAD, HEAD_DIM)

    offs_n = tl.arange(0, TILE_SIZE)                                               # shape: (TILE_SIZE)

    for blk_idx in range(tl.cdiv(MAX_SEQ_LEN, BLOCK_SIZE)) :

        # 计算 KV 块的偏移
        phy_off = tl.load(block_tables_ptr + blk_idx) * stride_k0
        k_blk_ptr = k_ptr + phy_off
        v_blk_ptr = v_ptr + phy_off

        for micro_block_id in range(tl.cdiv(BLOCK_SIZE, TILE_SIZE)) :

            # 加载 KV 块
            k_offs = (offs_n[None, :] * stride_k1 + offs_d[:, None])                    # shape: (HEAD_DIM, TILE_SIZE)
            k_mask = (offs_n[None, :] < seq_len) & (offs_d[:, None] < HEAD_DIM)         # shape: (HEAD_DIM, TILE_SIZE)
            v_offs = (offs_n[:, None] * stride_k1 + offs_d[None, :])                    # shape: (TILE_SIZE, HEAD_DIM)
            v_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM)         # shape: (TILE_SIZE, HEAD_DIM)
            k_block = tl.load(k_blk_ptr+k_offs, mask=k_mask, other=0.0).to(tl.float32)  # shape: (HEAD_DIM, TILE_SIZE)
            v_block = tl.load(v_blk_ptr+v_offs, mask=v_mask, other=0.0).to(tl.float32)  # shape: (TILE_SIZE, HEAD_DIM)
            
            k_blk_ptr += stride_k1 * TILE_SIZE
            v_blk_ptr += stride_k1 * TILE_SIZE

            # 注意力计算
            score = tl.dot(q, k_block)                                                   # shape: (N_Q_PER_KV_HEAD, TILE_SIZE)
            score = tl.where((offs_n[None, :]<seq_len), score, float("-inf"))            # shape: (N_Q_PER_KV_HEAD, TILE_SIZE)
            m_ij  = tl.max(score, axis=-1)                                               # shape: (N_Q_PER_KV_HEAD,)
            m_new = tl.maximum(m_i, m_ij)                                                # shape: (N_Q_PER_KV_HEAD,)
            alpha = tl.exp(m_i - m_new)                                                  # shape: (N_Q_PER_KV_HEAD,)
            weight= tl.exp(score - m_new[:, None])                                       # shape: (N_Q_PER_KV_HEAD, TILE_SIZE)
            weight= tl.where((offs_n[None, :]<seq_len), weight, 0.0)                     # shape: (N_Q_PER_KV_HEAD, TILE_SIZE)
            l_ij  = tl.sum(weight, axis=-1)                                              # shape: (N_Q_PER_KV_HEAD,)
            l_new = alpha * l_i + l_ij                                                   # shape: (N_Q_PER_KV_HEAD,)
            acc_o = acc_o * alpha[:, None]                                               # shape: (N_Q_PER_KV_HEAD, HEAD_DIM)
            w_x_v = tl.dot(weight, v_block)                                              # shape: (N_Q_PER_KV_HEAD, HEAD_DIM)
            acc_o = acc_o + w_x_v                                                        # shape: (N_Q_PER_KV_HEAD, HEAD_DIM)
            m_i = m_new
            l_i = l_new

            seq_len = tl.maximum(seq_len-TILE_SIZE, 0)

    acc_o = acc_o / l_i[:, None]
    acc_o = acc_o.to(o_ptr.dtype.element_ty)
    
    tl.store(o_ptr+offs_qh_d, acc_o, mask=mask_qh_d)




def triton_page_gqa_decode (
    q,                       # (bsz, num_kv_heads, num_q_per_kv_head, head_dim), bf16
    k_physical_blocks,       # (num_blocks, block_size, num_kv_heads, head_dim), bf16
    v_physical_blocks,       # (num_blocks, block_size, num_kv_heads, head_dim), bf16
    seq_lens,                # (bsz, )                    , int32
    block_tables,            # (bsz, max_n_block_of_a_seq), int32
    softmax_scale = None,
) :
    assert q.is_contiguous() and k_physical_blocks.is_contiguous() and v_physical_blocks.is_contiguous() and seq_lens.is_contiguous() and block_tables.is_contiguous()

    (bsz, num_kv_heads, num_q_per_kv_head, head_dim) = q.shape
    (num_blocks, block_size, _, _) = k_physical_blocks.shape 

    assert block_size in [16, 32, 64, 128, 256, 512, 1024]
    assert (num_blocks, block_size, num_kv_heads, head_dim) == k_physical_blocks.shape 
    assert (num_blocks, block_size, num_kv_heads, head_dim) == v_physical_blocks.shape 
    assert bsz == seq_lens.shape[0]
    assert bsz == block_tables.shape[0]

    softmax_scale = (head_dim ** -0.5) if (softmax_scale is None) else softmax_scale

    attn_o = torch.empty_like(q)  # 输出形状与 Q 相同

    N_Q_PER_KV_HEAD_2 = max(triton.next_power_of_2(num_q_per_kv_head), 16)
    HEAD_DIM_2        = max(triton.next_power_of_2(head_dim         ), 16)
    TILE_SIZE         = min(max(2048//HEAD_DIM_2, 16), block_size)

    kernel_page_gqa_decode [(bsz, num_kv_heads)] (
        attn_o, q, k_physical_blocks, v_physical_blocks, 
        q.stride(0), q.stride(1), q.stride(2),
        k_physical_blocks.stride(0), k_physical_blocks.stride(1), k_physical_blocks.stride(2),
        seq_lens,
        block_tables,
        block_tables.stride(0),
        N_Q_PER_KV_HEAD   = num_q_per_kv_head,
        N_Q_PER_KV_HEAD_2 = N_Q_PER_KV_HEAD_2,
        HEAD_DIM          = head_dim,
        HEAD_DIM_2        = HEAD_DIM_2,
        MAX_SEQ_LEN       = seq_lens.max().item(),
        SCALE             = softmax_scale,
        BLOCK_SIZE        = block_size,
        TILE_SIZE         = TILE_SIZE
    )

    return attn_o




if __name__ == "__main__" :
    device = 'cuda'

    bsz, num_kv_heads, num_q_per_kv_head, block_size, head_dim = 32, 8, 5, 256, 64
    num_blocks   = 64

    seq_lens     = torch.tensor([ num_blocks*block_size  for _ in range(bsz)], dtype=torch.int32, device=device)
    block_tables = torch.tensor([list(range(num_blocks)) for _ in range(bsz)], dtype=torch.int32, device=device)

    num_blocks   = int(block_tables.max().item()) + 1

    q                 = torch.randn(bsz, num_kv_heads, num_q_per_kv_head, head_dim, dtype=torch.bfloat16, device=device)
    k_physical_blocks = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v_physical_blocks = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    


    # 精度 =====================================================================================================
    attn_o_fa = flash_attn_with_kvcache(q.view(bsz, 1, num_kv_heads*num_q_per_kv_head, -1), k_physical_blocks, v_physical_blocks, cache_seqlens=seq_lens, block_table=block_tables, softmax_scale=(head_dim**-0.5), causal=True).view_as(q)
    attn_o_triton = triton_page_gqa_decode(q, k_physical_blocks, v_physical_blocks, seq_lens, block_tables)
    
    print(
        f'attn_o.shape = {tuple(attn_o_fa.shape)} \n'
        f'MAX - MIN    = { (attn_o_fa.max() - attn_o_fa.min()).item()         :.8f} \n'
        f'MAE_TRITON   = { (attn_o_fa - attn_o_triton).abs().mean().item()    :.8f} \n'
    )



    # 测量性能 =====================================================================================================
    N_TIME = 10
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(N_TIME) : triton_page_gqa_decode(q, k_physical_blocks, v_physical_blocks, seq_lens, block_tables)
    end_event.record()
    torch.cuda.synchronize()
    latency_triton = start_event.elapsed_time(end_event) / N_TIME

    print(
        f"triton_page_gqa_decode mean latency: {latency_triton:.3f} ms \n"
    )

