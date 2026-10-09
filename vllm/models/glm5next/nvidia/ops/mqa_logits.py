# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fp8 MQA-logits kernels for the kpool indexer's 16-head shape.

DeepGEMM's ``fp8_fp4_(paged_)mqa_logits`` only accept num_heads in {32, 64};
GLM-5.3-Flash ships ``index_n_heads=16``, so without these kernels the
attention layer zero-pads Q and the per-head weights to 32 heads and spends
half the FLOPs of the dominant indexer GEMM on zero head slots. Both kernels
compute the same logits as their DeepGEMM counterparts, for any
``num_heads >= 16``:

    logits[m, n] = k_scale[n] * sum_h weights[m, h] * (q[m, h, :] . k[n, :])

One program per (query row, KV tile): Q and weights are loaded once per
program, the per-head dots run on fp8 tensor cores (``tl.dot`` with fp32
accumulation), the head-weighted sum reduces the per-head dots, and the
per-K dequant scale is applied once after that sum. Only each row's valid
KV range is written — ``[cu_seqlen_ks, cu_seqlen_ke)`` prefill,
``[0, context_len)`` decode — matching the DeepGEMM calls'
``clean_logits=False``; the downstream topk kernels bound their scans by
the same ranges.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.worker.workspace import current_workspace_manager

# KV positions per prefill program.
_PREFILL_BLOCK_V = 128


@triton.jit
def _fp8_mqa_logits_prefill_kernel(
    q_ptr,  # fp8 [M, H, D]
    k_ptr,  # fp8 [N, D]
    k_scale_ptr,  # fp32 [N]
    w_ptr,  # fp32 [M, H]
    row_ks_ptr,  # int32 [M]
    row_ke_ptr,  # int32 [M]
    logits_ptr,  # fp32 [M, N]
    stride_qm,
    stride_qh,
    stride_kn,
    stride_wm,
    stride_lm,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    tl.static_assert(HEAD_SIZE % 4 == 0)
    row = tl.program_id(0)
    kv_start = tl.program_id(1) * BLOCK_V
    ks = tl.load(row_ks_ptr + row)
    ke = tl.load(row_ke_ptr + row)
    if (kv_start >= ke) | (kv_start + BLOCK_V <= ks):
        return

    h = tl.arange(0, NUM_HEADS)
    d = tl.arange(0, HEAD_SIZE)
    q = tl.load(q_ptr + row * stride_qm + h[:, None] * stride_qh + d[None, :])
    w = tl.load(w_ptr + row * stride_wm + h).to(tl.float32)  # [H]

    cols = kv_start + tl.arange(0, BLOCK_V)
    mask = (cols >= ks) & (cols < ke)
    k = tl.load(
        k_ptr + cols.to(tl.int64)[:, None] * stride_kn + d[None, :],
        mask=mask[:, None],
        other=0.0,
    )  # [BLOCK_V, D] fp8
    s = tl.dot(k, tl.trans(q))  # [BLOCK_V, H] fp32
    s = tl.sum(s * w[None, :], axis=1)
    scale = tl.load(k_scale_ptr + cols, mask=mask, other=0.0)
    s = s * scale
    tl.store(logits_ptr + row.to(tl.int64) * stride_lm + cols, s, mask=mask)


def fp8_mqa_logits_prefill(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute MQA logits for a single sequence without KV paging.

    Same contract as DeepGEMM's non-paged fp8 MQA logits with
    ``clean_logits=False``.

    Args:
        q: Query tensor of shape [M, H, D] float8_e4m3fn; the per-token
            scale is already folded into ``weights``.
        kv: Tuple ``(k, k_scale)`` — k is [N, D] float8_e4m3fn and
            k_scale is [N] (or [N, 1]) float32, one dequant scale per K.
        weights: Weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query
            position, shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query
            position, shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    k, k_scale = kv
    m, num_heads, head_size = q.shape
    assert q.dtype == torch.float8_e4m3fn and q.stride(-1) == 1
    assert k.dtype == torch.float8_e4m3fn and k.stride(-1) == 1
    assert k.stride(0) == head_size, k.stride()
    assert weights.dtype == torch.float32 and weights.shape == (m, num_heads)
    assert cu_seqlen_ks.dtype == torch.int32 and cu_seqlen_ke.dtype == torch.int32
    k_scale = k_scale.reshape(-1)
    assert k_scale.dtype == torch.float32 and k_scale.shape[0] == k.shape[0]

    logits = torch.empty((m, k.shape[0]), dtype=torch.float32, device=q.device)
    if m == 0 or k.shape[0] == 0:
        return logits
    _fp8_mqa_logits_prefill_kernel[(m, triton.cdiv(k.shape[0], _PREFILL_BLOCK_V))](
        q,
        k,
        k_scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        weights.stride(0),
        logits.stride(0),
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        BLOCK_V=_PREFILL_BLOCK_V,
        num_warps=4,
    )
    return logits


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    q_ptr,  # fp8 [B, NEXT_N, H, D]
    kv_val_ptr,  # fp8 block-flat [num_blocks, page * (D + 4)]
    kv_scale_ptr,  # fp32 block-flat [num_blocks, page * (D + 4) // 4]
    w_ptr,  # fp32 [B * NEXT_N, H]
    ctx_lens_ptr,  # int32 [B * NEXT_N]
    block_tables_ptr,  # int32 [B, max_blocks]
    logits_ptr,  # fp32 [B * NEXT_N, max_model_len]
    stride_q_b,
    stride_q_n,
    stride_q_h,
    stride_w_row,
    stride_kvblk_fp8,
    stride_kvblk_f32,
    scale_region_off,  # page * D // 4 (fp32 offset of the scale region)
    stride_bt_b,
    stride_logits_row,
    max_blocks,
    max_model_len,
    NEXT_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,  # pools per cache page; one KV tile per page
    N_SPLITS: tl.constexpr,  # KV-tile parallelism factor (grid dim 1)
):
    tl.static_assert(HEAD_SIZE % 4 == 0)
    row = tl.program_id(0)
    split = tl.program_id(1)
    b = row // NEXT_N
    seq_len = tl.load(ctx_lens_ptr + row)
    seq_len = tl.minimum(tl.maximum(seq_len, 0), max_model_len)

    h = tl.arange(0, NUM_HEADS)
    d = tl.arange(0, HEAD_SIZE)
    # Keep q/kv fp8 -> fp8 MMA (f32 accumulation), not the slow f32 path.
    q = tl.load(
        q_ptr
        + b * stride_q_b
        + (row % NEXT_N) * stride_q_n
        + h[:, None] * stride_q_h
        + d[None, :]
    )
    w = tl.load(w_ptr + row * stride_w_row + h).to(tl.float32)  # [H]
    logits_row_ptr = logits_ptr + row.to(tl.int64) * stride_logits_row

    cols = tl.arange(0, PAGE_SIZE)
    for kv_start in tl.range(split * PAGE_SIZE, seq_len, N_SPLITS * PAGE_SIZE):
        logical_blk = kv_start // PAGE_SIZE
        # Guard against a wild page -> fault; seq_len keeps valid tiles
        # within the block-table width.
        blk_ok = logical_blk < max_blocks
        mask = ((kv_start + cols) < seq_len) & blk_ok
        page = tl.load(
            block_tables_ptr + b * stride_bt_b + logical_blk,
            mask=blk_ok,
            other=0,
        ).to(tl.int64)

        # Block-flat layout: values region, then scales region.
        kv = tl.load(
            kv_val_ptr + page * stride_kvblk_fp8 + cols[:, None] * HEAD_SIZE + d[None, :],
            mask=mask[:, None],
            other=0.0,
        )  # [PAGE_SIZE, D] fp8
        sc = tl.load(
            kv_scale_ptr + page * stride_kvblk_f32 + scale_region_off + cols,
            mask=mask,
            other=0.0,
        )  # [PAGE_SIZE]
        s = tl.dot(kv, tl.trans(q))  # [PAGE_SIZE, H] fp32
        s = tl.sum(s * w[None, :], axis=1)
        s = s * sc
        tl.store(logits_row_ptr + kv_start + cols, s, mask=mask)


def fp8_mqa_logits_paged(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute MQA logits using the paged fp8 indexer K cache.

    Same contract as DeepGEMM's paged fp8 MQA logits with
    ``clean_logits=False``; no schedule metadata is needed, so the launch
    has no host sync.

    Args:
        q: Query tensor of shape [B, next_n, H, D] float8_e4m3fn; the
            per-token scale is already folded into ``weights``.
        kv_cache: Paged KV-cache of shape [num_blocks, page, 1, D+4],
            dtype `torch.uint8`, with the last 4 bytes per (block, pool)
            storing the float dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B, next_n] (or flat [B * next_n]),
            dtype int32; effective context length per query row, in cache
            (pool) units.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; one
            cache page per entry.
        max_model_len: Logits width, in cache (pool) units.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    batch_size, next_n, num_heads, head_size = q.shape
    rows = batch_size * next_n
    assert q.dtype == torch.float8_e4m3fn and q.stride(-1) == 1
    assert kv_cache.dtype == torch.uint8 and kv_cache.is_contiguous()
    assert weights.dtype == torch.float32 and weights.shape[0] == rows
    assert block_tables.dtype == torch.int32 and block_tables.shape[0] == batch_size
    cl = context_lens.reshape(-1)
    assert cl.dtype == torch.int32 and cl.numel() == rows

    num_blocks = kv_cache.shape[0]
    page = kv_cache.shape[1]
    kv_flat = kv_cache.reshape(num_blocks, -1)
    kv_val = kv_flat.view(torch.float8_e4m3fn)
    kv_scale = kv_flat.view(torch.float32)

    (logits,) = current_workspace_manager().get_simultaneous(
        ((rows, max_model_len), torch.float32)
    )
    if rows == 0:
        return logits
    # Memory-bound over the KV range: split each row's keys across programs
    # so few-row / long-context launches still fill the GPU. All terms are
    # static at launch, so the grid stays CUDA-graph-safe.
    tiles = triton.cdiv(max_model_len, page)
    n_splits = max(
        1, min(num_compute_units(q.device.index), tiles, max(1, 1024 // rows))
    )
    _fp8_paged_mqa_logits_kernel[(rows, n_splits)](
        q,
        kv_val,
        kv_scale,
        weights,
        cl,
        block_tables,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        weights.stride(0),
        kv_val.stride(0),
        kv_scale.stride(0),
        (page * head_size) // 4,
        block_tables.stride(0),
        logits.stride(0),
        block_tables.shape[1],
        max_model_len,
        NEXT_N=next_n,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        PAGE_SIZE=page,
        N_SPLITS=n_splits,
        num_warps=4,
        num_stages=2,
    )
    return logits
