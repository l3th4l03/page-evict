import triton
import triton.language as tl
import torch
import math

BLOCK_SIZE = 64  # tokens per block

@triton.jit
def _gather_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Table_ptr, Output_ptr,
    #strides for K/V buffer:
    stride_k_slot, stride_k_head, stride_k_dim,
    # strides for Q/Output:
    stride_q_head, stride_o_head,
    # sizes
    num_active,
    num_kv_groups,  # num_q_heads // num_kv_heads (4 for Llama)
    scale,          # 1 / sqrt(head_dim), pre-computed on CPU
    # compile-time constants
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # ── Step 1: Identify which head this program handles ──
    q_head = tl.program_id(0)            # which query head (0..31)
    kv_head = q_head // num_kv_groups    # GQA: map to KV head (0..7)

    # ── Step 2: Load the query vector for this head ──
    # Q layout: [1, num_q_heads, 1, head_dim] → we want Q[0, q_head, 0, :]
    # Pointer arithmetic: Q_ptr + q_head * stride_q_head + dim_offset
    dim_range = tl.arange(0, HEAD_DIM)   # [0, 1, 2, ..., 127]
    q = tl.load(Q_ptr + q_head * stride_q_head + dim_range).to(tl.float32)

    # ── Step 3: Initialize online softmax accumulators ──
    # These track the running softmax state across ALL key blocks
    m_i = -float('inf')                              # running max score
    l_i = 0.0                                        # running sum of exp(scores - max)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)     # running weighted sum of V

    # ── Step 4: Loop over all key blocks ──
    # Each iteration processes BLOCK_SIZE (64) keys from the mapping table
    for block_start in range(0, num_active, BLOCK_SIZE):
        # Offsets for this block's entries in the mapping table
        offs = block_start + tl.arange(0, BLOCK_SIZE)  # [block_start, ..., block_start+63]
        mask = offs < num_active  # mask out-of-bounds for the last partial block

        # ── Step 4a: Load physical slot indices from mapping table ──
        # table[offs] → which physical slots in K/V buffer to read
        slots = tl.load(Table_ptr + offs, mask=mask, other=0).to(tl.int64)

        # ── Step 4b: Gather K vectors from scattered physical slots ──
        # For each slot s, load K_buffer[s, kv_head, :] using pointer math
        # slots[:, None] broadcasts: [BLOCK_SIZE, 1] * stride → [BLOCK_SIZE, HEAD_DIM]
        k_ptrs = (K_ptr
                  + slots[:, None] * stride_k_slot    # jump to the right slot
                  + kv_head * stride_k_head            # jump to the right KV head
                  + dim_range[None, :] * stride_k_dim) # offset within the head_dim
        k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
        # k shape: [BLOCK_SIZE, HEAD_DIM]

        # ── Step 4c: Compute attention scores: Q · K^T / sqrt(d) ──
        # Element-wise multiply then sum across HEAD_DIM → one score per key
        scores = tl.sum(q[None, :] * k, axis=1) * scale  # [BLOCK_SIZE]
        scores = tl.where(mask, scores, -float('inf'))    # mask invalid positions

        # ── Step 4d: Online softmax update ──
        # This is the FlashAttention trick: merge this block's scores with
        # the running softmax from previous blocks without materializing
        # the full attention matrix
        block_max = tl.max(scores, axis=0)             # max score in this block
        m_new = tl.maximum(m_i, block_max)             # new global max
        alpha = tl.exp(m_i - m_new)                    # rescale factor for old accumulators
        p = tl.exp(scores - m_new)                     # new block's exp(scores)
        l_i = alpha * l_i + tl.sum(p, axis=0)          # update running sum

        # ── Step 4e: Gather V vectors and accumulate weighted sum ──
        v_ptrs = (V_ptr
                  + slots[:, None] * stride_k_slot
                  + kv_head * stride_k_head
                  + dim_range[None, :] * stride_k_dim)
        v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
        # v shape: [BLOCK_SIZE, HEAD_DIM]

        # Weighted sum: rescale old accumulator + add new block's contribution
        # p[:, None] broadcasts [BLOCK_SIZE, 1] × [BLOCK_SIZE, HEAD_DIM] → [BLOCK_SIZE, HEAD_DIM]
        # then sum across BLOCK_SIZE → [HEAD_DIM]
        acc = alpha * acc + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new  # update running max

    # ── Step 5: Normalize and store output ──
    output = acc / l_i  # divide by softmax denominator

    # Store: Output[0, q_head, 0, :] = output
    o_off = q_head * stride_o_head + dim_range
    tl.store(Output_ptr + o_off, output)


def gather_attention(
    Q: torch.Tensor,              # [1, num_q_heads, 1, head_dim]
    K_buffer: torch.Tensor,       # [max_slots, num_kv_heads, head_dim]
    V_buffer: torch.Tensor,       # [max_slots, num_kv_heads, head_dim]
    mapping_table: torch.Tensor,  # [num_active] int64 tensor
    num_active: int,
):
    num_q_heads = Q.shape[1]
    head_dim = Q.shape[3]
    num_kv_heads = K_buffer.shape[1]
    num_kv_groups = num_q_heads // num_kv_heads  # GQA: 32 // 8 = 4
    scale = 1.0 / math.sqrt(head_dim)            # pre-compute on CPU

    # output same shape/device/dtype as Q
    output = torch.empty_like(Q)

    # launch grid: one program per query head
    # each program loops over all key blocks internally (FlashAttention style)
    grid = (num_q_heads,)

    _gather_attention_kernel[grid](
        Q.data_ptr(), K_buffer.data_ptr(), V_buffer.data_ptr(),
        mapping_table.data_ptr(), output.data_ptr(),
        # K/V strides
        K_buffer.stride(0), K_buffer.stride(1), K_buffer.stride(2),
        # Q/Output strides
        Q.stride(1), output.stride(1),
        # sizes
        num_active,
        num_kv_groups,
        scale,
        # compile-time constants
        HEAD_DIM=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
