import torch
import triton
import triton.language as tl



@triton.jit
def triton_kernel_flash_gqa_decode (
    o_ptr,                               # (BSZ, N_KV_HEAD, N_Q_PER_KV_HEAD, HEAD_DIM), bf16
    q_ptr,                               # (BSZ, N_KV_HEAD, N_Q_PER_KV_HEAD, HEAD_DIM), bf16
    k_ptr,                               # (BSZ, N_KV_HEAD, SEQ_LEN        , HEAD_DIM), bf16
    v_ptr,                               # (BSZ, N_KV_HEAD, SEQ_LEN        , HEAD_DIM), bf16
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    N_Q_PER_KV_HEAD   : tl.constexpr,
    N_Q_PER_KV_HEAD_2 : tl.constexpr,
    HEAD_DIM          : tl.constexpr,
    HEAD_DIM_2        : tl.constexpr,
    SEQ_LEN           : tl.constexpr,
    SCALE             : tl.constexpr,
    BLOCK_N           : tl.constexpr,
):
    i_batch  = tl.program_id(0)
    i_kvhead = tl.program_id(1)
    
    o_ptr += i_batch * stride_q0 + i_kvhead * stride_q1     # 指针定位到 o_ptr[i_batch, i_kvhead, :, :]
    q_ptr += i_batch * stride_q0 + i_kvhead * stride_q1     # 指针定位到 q_ptr[i_batch, i_kvhead, :, :]
    k_ptr += i_batch * stride_k0 + i_kvhead * stride_k1     # 指针定位到 k_ptr[i_batch, i_kvhead, :, :]
    v_ptr += i_batch * stride_k0 + i_kvhead * stride_k1     # 指针定位到 v_ptr[i_batch, i_kvhead, :, :]

    offs_d    = tl.arange(0, HEAD_DIM_2)                                           # shape: (HEAD_DIM,)
    offs_qh   = tl.arange(0, N_Q_PER_KV_HEAD_2)                                    # shape: (N_Q_PER_KV_HEAD,)
    offs_qh_d =  offs_qh[:, None] * stride_q2      +  offs_d[None, :]              # shape: (N_Q_PER_KV_HEAD, HEAD_DIM)
    mask_qh_d = (offs_qh[:, None]<N_Q_PER_KV_HEAD) & (offs_d[None, :]<HEAD_DIM)    # shape: (N_Q_PER_KV_HEAD, HEAD_DIM)
    
    q = tl.load(q_ptr+offs_qh_d, mask=mask_qh_d, other=0.0)                        # shape: (N_Q_PER_KV_HEAD, HEAD_DIM)
    q = q.to(tl.float32) * SCALE                                                   # shape: (N_Q_PER_KV_HEAD, HEAD_DIM)
    q = q[:, None, :]                                                              # shape: (N_Q_PER_KV_HEAD, 1, HEAD_DIM)

    m_i   = tl.full ([N_Q_PER_KV_HEAD_2,], value=float("-inf"), dtype=tl.float32)  # shape: (N_Q_PER_KV_HEAD,)
    l_i   = tl.zeros([N_Q_PER_KV_HEAD_2,]                     , dtype=tl.float32)  # shape: (N_Q_PER_KV_HEAD,)
    acc_o = tl.zeros([N_Q_PER_KV_HEAD_2, HEAD_DIM]            , dtype=tl.float32)  # shape: (N_Q_PER_KV_HEAD, HEAD_DIM)

    for start_n in range(tl.cdiv(SEQ_LEN, BLOCK_N)):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)                         # shape: (BLOCK_N)

        # 加载 KV 块
        kv_offs = (offs_n[:, None] * stride_k2 + offs_d[None, :])                  # shape: (BLOCK_N, HEAD_DIM)
        kv_mask = (offs_n[:, None] < SEQ_LEN) & (offs_d[None, :] < HEAD_DIM)       # shape: (BLOCK_N, HEAD_DIM)
        k_block = tl.load(k_ptr+kv_offs, mask=kv_mask, other=0.0).to(tl.float32)   # shape: (BLOCK_N, HEAD_DIM)
        v_block = tl.load(v_ptr+kv_offs, mask=kv_mask, other=0.0).to(tl.float32)   # shape: (BLOCK_N, HEAD_DIM)

        # 注意力计算
        q_x_k = q * k_block[None, :, :]                                            # shape: (N_Q_PER_KV_HEAD, BLOCK_N, HEAD_DIM)
        score = tl.sum(q_x_k, axis=-1)                                             # shape: (N_Q_PER_KV_HEAD, BLOCK_N)
        m_ij  = tl.max(score, axis=-1)                                             # shape: (N_Q_PER_KV_HEAD,)
        m_new = tl.maximum(m_i, m_ij)                                              # shape: (N_Q_PER_KV_HEAD,)
        alpha = tl.exp(m_i - m_new)                                                # shape: (N_Q_PER_KV_HEAD,)
        weight= tl.exp(score - m_new[:, None])                                     # shape: (N_Q_PER_KV_HEAD, BLOCK_N)
        l_ij  = tl.sum(weight, axis=-1)                                            # shape: (N_Q_PER_KV_HEAD,)
        l_new = alpha * l_i + l_ij                                                 # shape: (N_Q_PER_KV_HEAD,)
        acc_o = acc_o * alpha[:, None]                                             # shape: (N_Q_PER_KV_HEAD, HEAD_DIM)
        w_x_v = weight[:, :, None] * v_block[None, :, :]                           # shape: (N_Q_PER_KV_HEAD, BLOCK_N, HEAD_DIM)
        acc_o = acc_o + tl.sum(w_x_v, axis=1)                                      # shape: (N_Q_PER_KV_HEAD, HEAD_DIM)
        m_i = m_new
        l_i = l_new

    acc_o = acc_o / l_i[:, None]
    acc_o = acc_o.to(o_ptr.dtype.element_ty)
    
    tl.store(o_ptr+offs_qh_d, acc_o, mask=mask_qh_d)




def triton_flash_gqa_decode (
    q,                       # (bsz, num_kv_heads, num_q_per_kv_head, head_dim)
    k,                       # (bsz, num_kv_heads, seq_len          , head_dim)
    v,                       # (bsz, num_kv_heads, seq_len          , head_dim)
    softmax_scale = None
) :
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    (bsz, num_kv_heads, num_q_per_kv_head, head_dim) = q.shape
    (_  , _           , seq_len          , _       ) = k.shape

    assert (bsz, num_kv_heads, seq_len, head_dim) == k.shape 
    assert (bsz, num_kv_heads, seq_len, head_dim) == v.shape 

    softmax_scale = (head_dim ** -0.5) if (softmax_scale is None) else softmax_scale

    attn_o = torch.empty_like(q)  # 输出形状与 Q 相同

    triton_kernel_flash_gqa_decode[(bsz, num_kv_heads)](
        attn_o, q, k, v, 
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        N_Q_PER_KV_HEAD   = num_q_per_kv_head,
        N_Q_PER_KV_HEAD_2 = triton.next_power_of_2(num_q_per_kv_head),
        HEAD_DIM          = head_dim,
        HEAD_DIM_2        = triton.next_power_of_2(head_dim),
        SEQ_LEN           = seq_len,
        SCALE             = softmax_scale,
        BLOCK_N           = 32,
    )

    return attn_o




def sdpa_decode (
    q,                       # (bsz, num_kv_heads, num_q_per_kv_head, head_dim)
    k,                       # (bsz, num_kv_heads, seq_len          , head_dim)
    v,                       # (bsz, num_kv_heads, seq_len          , head_dim)
) :
    (bsz, num_kv_heads, num_q_per_kv_head, head_dim) = q.shape
    (_  , _           , seq_len          , _       ) = k.shape

    assert (bsz, num_kv_heads, seq_len, head_dim) == k.shape 
    assert (bsz, num_kv_heads, seq_len, head_dim) == v.shape 

    qx = q.view(bsz, num_kv_heads*num_q_per_kv_head, 1, head_dim) # (bsz, num_q_heads         , head_dim)
    kx = k.repeat_interleave(num_q_per_kv_head, dim=1)            # (bsz, num_q_heads, seq_len, head_dim)
    vx = v.repeat_interleave(num_q_per_kv_head, dim=1)            # (bsz, num_q_heads, seq_len, head_dim)

    return torch.nn.functional.scaled_dot_product_attention(qx, kx, vx, attn_mask=None, dropout_p=0.0, is_causal=False).view_as(q)




if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("*** need CUDA")
        exit()

    batch_size, num_kv_heads, num_q_per_kv_head, seq_len, head_dim = 7, 8, 5, 5000, 128

    q = torch.randn(batch_size, num_kv_heads, num_q_per_kv_head, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(batch_size, num_kv_heads, seq_len          , head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(batch_size, num_kv_heads, seq_len          , head_dim, device='cuda', dtype=torch.bfloat16)

    # 预热
    for _ in range(10) :
        out_gqa  = triton_flash_gqa_decode(q, k, v)
        out_sdpa = sdpa_decode(q, k, v)

    # 测量 triton_flash_gqa_decode 时延
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(100):
        out_gqa = triton_flash_gqa_decode(q, k, v)
    end_event.record()
    torch.cuda.synchronize()
    gqa_time = start_event.elapsed_time(end_event) / 100

    # 测量 SDPA 时延
    start_event.record()
    for _ in range(100):
        out_sdpa = sdpa_decode(q, k, v)
    end_event.record()
    torch.cuda.synchronize()
    sdpa_time = start_event.elapsed_time(end_event) / 100

    # 计算输出差异
    diff = (out_gqa - out_sdpa).abs().max().item()
    print(f"GQA 输出形状 : {out_gqa.shape}")
    print(f"SDPA 输出形状: {out_sdpa.shape}")
    print(f"最大绝对误差 : {diff:.6f}")
    print(f"Flash Decode 平均时延: {gqa_time:.3f} ms")
    print(f"PyTorch SDPA 平均时延: {sdpa_time:.3f} ms")
    print(f"加速比: {sdpa_time / gqa_time:.2f}x")
