# SPDX-License-Identifier: Apache-2.0
"""Device-only partition into no-skip and common-width skip groups.

All tensors have a capture-time shape. GPU counts select exactly one capacity
per nonempty group; unused capacity bodies are CUDA IF nodes, not masked GEMMs.
"""

import copy
from dataclasses import replace

import torch


def capacities(size):
    return tuple(sorted({value for value in (1, 2, 4, 8, size) if value <= size}))


def partition(proposed, actions, group_actions, counts, indices, buckets):
    """Keep source order within each group and exclude original padding."""
    size = proposed.numel()
    rows = torch.arange(size, device=proposed.device)
    members = torch.stack((proposed == 0, proposed > 0))
    counts.copy_(members.sum(-1))
    order = torch.argsort(torch.where(members, rows, rows + size), dim=-1)
    indices.copy_(torch.where(rows < counts[:, None], order, -1))
    minimum = torch.where(proposed > 0, proposed, 12).amin()
    group_actions[0].copy_(torch.where(counts[0] > 0, 0, -1))
    group_actions[1].copy_(torch.where(counts[1] > 0, minimum, -1))
    actions.copy_(torch.where(proposed > 0, minimum, proposed))
    # The largest capacity may be a non-power-of-two explicitly captured size.
    buckets.copy_(counts)
    for capacity in reversed(capacities(size)):
        buckets.copy_(torch.where(counts <= capacity, capacity, buckets))
    buckets.copy_(torch.where(counts > 0, buckets, 0))


def branch_actions(group_actions, buckets, output, widths, size):
    """Map (pass, capacity, segment) to a scalar IF enable value, 0 or -1."""
    # Stack scalar comparisons once, rather than dispatching tensor kernels
    # separately for every candidate IF body during graph replay.
    capacity_matches = torch.stack(
        [buckets == capacity for capacity in capacities(size)], dim=-1
    )
    main = torch.stack(
        [(group_actions >= 0) & (group_actions < width) for width in widths], dim=-1
    )
    repair = torch.stack([group_actions[1] >= width for width in widths])
    segments = torch.cat((main, repair[None]), dim=0)
    selected_capacities = torch.cat((capacity_matches, capacity_matches[1:2]), dim=0)
    enabled = selected_capacities[:, :, None] & segments[:, None, :]
    count = enabled.numel()
    output[:count].copy_(torch.where(enabled.flatten(), 0, -1))
    return count


def gather_rows(value, indices, *, fill=0, positions=False):
    axis = -1 if positions else 0
    selected = value.index_select(axis, indices.clamp_min(0))
    valid = indices >= 0
    shape = [1] * value.ndim
    shape[axis] = indices.numel()
    return torch.where(valid.reshape(shape), selected, fill)


def merge_rows(packed, original, members):
    ranks = (members.to(torch.int64).cumsum(0) - 1).clamp_min(0)
    return torch.where(members[:, None], packed[ranks], original)


def packed_batch(batch, indices):
    result = copy.copy(batch)
    for name in (
        "input_ids",
        "req_pool_indices",
        "seq_lens",
        "orig_seq_lens",
        "out_cache_loc",
        "mamba_track_indices",
        "mamba_track_mask",
    ):
        value = getattr(batch, name, None)
        if value is not None:
            setattr(
                result,
                name,
                gather_rows(value, indices, fill=1 if "seq_lens" in name else 0),
            )
    for name in ("positions", "mrope_positions"):
        value = getattr(batch, name, None)
        if value is not None:
            setattr(result, name, gather_rows(value, indices, positions=True))
    result.seq_lens_cpu = None
    result.seq_lens_sum = None
    result.num_token_non_padded = None
    return result


def bucket_batch(batch, capacity):
    result = copy.copy(batch)
    result.batch_size = capacity
    for name in (
        "input_ids",
        "req_pool_indices",
        "seq_lens",
        "orig_seq_lens",
        "out_cache_loc",
        "mamba_track_indices",
        "mamba_track_mask",
    ):
        value = getattr(batch, name, None)
        if value is not None:
            setattr(result, name, value[:capacity])
    for name in ("positions", "mrope_positions"):
        value = getattr(batch, name, None)
        if value is not None:
            setattr(result, name, value[..., :capacity])
    return result


def packed_backend(backend, batch, indices, scratch):
    """Rebuild ragged KV indices and GDN row indices on the capture stream."""
    from sglang.srt.early_exit.masked_kv import MaskedKVPool

    full, linear = backend.full_attn_backend, backend.linear_attn_backend
    metadata, gdn = full.forward_metadata, linear.forward_metadata
    size = batch.batch_size
    if "kv_indptr" not in scratch:
        scratch["kv_indptr"] = torch.zeros_like(metadata.kv_indptr)
        scratch["kv_indices"] = torch.empty_like(metadata.kv_indices)
        scratch["logits"] = torch.empty_like(metadata.attn_logits)
        scratch["lse"] = torch.empty_like(metadata.attn_lse)
    packed_full = copy.copy(full)
    packed_full.kv_indptr = scratch["kv_indptr"]
    packed_full._fill_kv_indptr_and_indices(
        size, batch.seq_lens, batch.req_pool_indices, scratch["kv_indices"]
    )
    packed_full.token_to_kv_pool = MaskedKVPool(full.token_to_kv_pool)
    packed_full.forward_metadata = replace(
        metadata,
        kv_indptr=scratch["kv_indptr"],
        kv_indices=scratch["kv_indices"],
        attn_logits=scratch["logits"],
        attn_lse=scratch["lse"],
    )
    packed_linear = copy.copy(linear)
    packed_linear.forward_metadata = replace(
        gdn,
        mamba_cache_indices=gather_rows(
            gdn.mamba_cache_indices[:size], indices, fill=-1
        ),
        query_start_loc=torch.arange(
            size + 1, dtype=torch.int32, device=indices.device
        ),
    )
    result = copy.copy(backend)
    result.full_attn_backend, result.linear_attn_backend = packed_full, packed_linear
    result.attn_backend_list = [packed_full, packed_linear]
    return result


def bucket_backend(backend, batch, scratch):
    full, linear = backend.full_attn_backend, backend.linear_attn_backend
    size = batch.batch_size
    if size not in scratch:
        scratch[size] = torch.empty(
            size, dtype=torch.int32, device=batch.seq_lens.device
        )
    splits = scratch[size]
    full.get_num_kv_splits(splits, batch.seq_lens)
    bucket_full = copy.copy(full)
    bucket_full.forward_metadata = replace(
        full.forward_metadata,
        kv_indptr=full.forward_metadata.kv_indptr[: size + 1],
        num_kv_splits=splits,
    )
    bucket_linear = copy.copy(linear)
    bucket_linear.forward_metadata = replace(
        linear.forward_metadata,
        mamba_cache_indices=linear.forward_metadata.mamba_cache_indices[:size],
        query_start_loc=linear.forward_metadata.query_start_loc[: size + 1],
    )
    result = copy.copy(backend)
    result.full_attn_backend, result.linear_attn_backend = bucket_full, bucket_linear
    result.attn_backend_list = [bucket_full, bucket_linear]
    return result
