# SPDX-License-Identifier: Apache-2.0
"""Position validation and private full-attention repair scratch."""

import copy
from dataclasses import replace

import torch


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


def repair_backend(backend, scratch):
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    full, linear = backend.full_attn_backend, backend.linear_attn_backend
    if not isinstance(full, TritonAttnBackend) or not isinstance(
        linear, GDNAttnBackend
    ):
        raise ValueError("Batched early exit requires Triton full attention and GDN")
    if (
        full.use_mla
        or full.use_sliding_window_kv_pool
        or (full.sliding_window_size is not None and full.sliding_window_size > 0)
    ):
        raise ValueError("Batched early exit supports ordinary full KV attention")
    metadata = full.forward_metadata
    key = (metadata.attn_logits.shape, metadata.attn_logits.dtype)
    if key not in scratch:
        scratch[key] = (
            torch.empty_like(metadata.attn_logits),
            torch.empty_like(metadata.attn_lse),
        )
    logits, lse = scratch[key]
    repair_full = copy.copy(full)
    repair_full.forward_metadata = replace(
        metadata,
        attn_logits=logits,
        attn_lse=lse,
    )
    gdn = linear.forward_metadata
    if gdn.is_target_verify or gdn.has_mamba_track_mask:
        raise ValueError("Batched early exit expects ordinary GDN decode metadata")
    repair_linear = copy.copy(linear)
    repair_linear.forward_metadata = replace(gdn)
    result = copy.copy(backend)
    result.full_attn_backend = repair_full
    result.linear_attn_backend = repair_linear
    result.attn_backend_list = [repair_full, repair_linear]
    return result
