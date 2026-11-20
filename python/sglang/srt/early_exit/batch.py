# SPDX-License-Identifier: Apache-2.0
"""One-request views of batched decode metadata for conditional middle layers.

KV indptr values retain offsets into the shared flat index array. GDN query
offsets instead address the row's compact one-token tensor and must be rebased.
Views still reference live graph input buffers; scratch is private to a stream.
RoPE positions are [tokens], MRoPE positions are [3, tokens]. Both slice on
the final axis; MRoPE preserves its original axis stride for the Triton kernel.
"""

import copy
from dataclasses import replace

import torch


def prepare_execution_actions(action, output):
    """Choose exclusive batched/row IF bodies without reading a GPU decision.

    The final output slot controls the batched no-skip body. It is enabled only
    for a full eight-request capture with every source action equal to zero.
    A padded -1 row disables this path and retains its skipped row bodies.
    Source actions remain intact for policy statistics.
    """
    size = action.shape[0]
    if size == output.numel() - 1:
        all_zero = (action == 0).all()
        output[:size].copy_(torch.where(all_zero, -1, action))
        output[-1].copy_(torch.where(all_zero, 0, -1))
    else:
        output[:size].copy_(action)
        output[-1].fill_(-1)


def validate_positions(positions, num_tokens):
    if (
        positions.ndim not in (1, 2)
        or (positions.ndim == 2 and positions.shape[0] != 3)
        or positions.shape[-1] != num_tokens
        or positions.stride(-1) != 1
    ):
        raise ValueError(
            "early-exit positions require [tokens] or [3, tokens] with unit token stride"
        )


def row_positions(positions, index):
    # Do not materialize a contiguous copy: graph replay must read live input
    # slots. MRoPE's kernel accepts the preserved stride between its three axes.
    return positions[..., index : index + 1]


def row_batch(batch, index):
    if not batch.forward_mode.is_decode() or batch.spec_info is not None:
        raise ValueError(
            "Per-request early exit supports ordinary decode without speculative tokens"
        )
    result = copy.copy(batch)
    result.batch_size = 1
    for name in (
        "input_ids",
        "req_pool_indices",
        "seq_lens",
        "orig_seq_lens",
        "out_cache_loc",
        "seq_lens_cpu",
        "mamba_track_indices",
        "mamba_track_mask",
    ):
        value = getattr(batch, name, None)
        if value is not None:
            setattr(result, name, value[index : index + 1])
    for name in ("positions", "mrope_positions"):
        value = getattr(batch, name, None)
        if value is not None:
            setattr(result, name, row_positions(value, index))
    result.seq_lens_sum = None  # Decode backends consume the GPU length view.
    result.num_token_non_padded = None  # This single-row view has no DP padding.
    return result


def row_backend(backend, index, scratch):
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    full, linear = backend.full_attn_backend, backend.linear_attn_backend
    if not isinstance(full, TritonAttnBackend) or not isinstance(
        linear, GDNAttnBackend
    ):
        raise ValueError(
            "Per-request early exit requires Triton full attention and GDN"
        )
    if (
        full.use_mla
        or full.use_sliding_window_kv_pool
        or (full.sliding_window_size is not None and full.sliding_window_size > 0)
    ):
        raise ValueError("Per-request early exit supports ordinary full KV attention")
    metadata = full.forward_metadata
    key = (metadata.attn_logits.shape, metadata.attn_logits.dtype)
    if key not in scratch:
        scratch[key] = (
            torch.empty_like(metadata.attn_logits[:1]),
            torch.empty_like(metadata.attn_lse[:1]),
        )
    logits, lse = scratch[key]
    full_row = copy.copy(full)
    full_row.forward_metadata = replace(
        metadata,
        attn_logits=logits,
        attn_lse=lse,
        kv_indptr=metadata.kv_indptr[index : index + 2],
        num_kv_splits=metadata.num_kv_splits[index : index + 1],
    )
    gdn = linear.forward_metadata
    if gdn.is_target_verify or gdn.has_mamba_track_mask:
        raise ValueError("Per-request early exit expects ordinary GDN decode metadata")
    linear_row = copy.copy(linear)
    linear_row.forward_metadata = replace(
        gdn,
        mamba_cache_indices=gdn.mamba_cache_indices[index : index + 1],
        query_start_loc=gdn.query_start_loc[index : index + 2]
        - gdn.query_start_loc[index : index + 1],
    )
    result = copy.copy(backend)
    result.full_attn_backend, result.linear_attn_backend = full_row, linear_row
    result.attn_backend_list = [full_row, linear_row]
    return result
