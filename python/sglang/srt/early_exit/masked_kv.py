# SPDX-License-Identifier: Apache-2.0
"""KV stores for packed groups: padding never writes the shared dummy slot.

Two streams may run the same layer for disjoint real requests. Suppressing the
padding stores avoids a race between their native reserved-slot-zero writes.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _store(
    K,
    V,
    KC,
    VC,
    LOC,
    KS: tl.constexpr,
    VS: tl.constexpr,
    KCS: tl.constexpr,
    VCS: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(LOC + row)
    key = tl.load(K + row * KS + offset, (slot > 0) & (offset < KD), 0)
    value = tl.load(V + row * VS + offset, (slot > 0) & (offset < VD), 0)
    tl.store(KC + slot * KCS + offset, key, (slot > 0) & (offset < KD))
    tl.store(VC + slot * VCS + offset, value, (slot > 0) & (offset < VD))


class MaskedKVPool:
    def __init__(self, pool):
        full_pool = getattr(pool, "full_kv_pool", pool)
        if (
            full_pool.dtype != torch.bfloat16
            or full_pool.store_dtype != full_pool.dtype
        ):
            raise ValueError("Split early exit requires BF16 KV storage")
        if full_pool.kv_cache_layout != "nhd":
            raise ValueError("Split early exit requires standard NHD KV storage")
        self.pool = pool

    def __getattr__(self, name):
        return getattr(self.pool, name)

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None):
        if cache_k.dtype != torch.bfloat16 or cache_v.dtype != torch.bfloat16:
            raise ValueError("Split early exit expects BF16 keys and values")
        key = cache_k.view(cache_k.shape[0], -1)
        value = cache_v.view(cache_v.shape[0], -1)
        kc, vc = self.pool.get_kv_buffer(layer.layer_id)
        if kc.ndim != 3 or vc.ndim != 3 or kc.stride(-1) != 1 or vc.stride(-1) != 1:
            raise ValueError("Split early exit expects contiguous NHD KV rows")
        _store[(loc.numel(), triton.cdiv(max(key.shape[1], value.shape[1]), 256))](
            key,
            value,
            kc,
            vc,
            loc,
            key.stride(0),
            value.stride(0),
            kc.stride(0),
            vc.stride(0),
            key.shape[1],
            value.shape[1],
            256,
        )
