# SPDX-License-Identifier: Apache-2.0
"""Deferred prefix caches, repaired by causal chunk prefill on GPU re-entry.

Experimental static-request batches only: reset() must accompany runner.clear().
Each request's missing prefixes are contiguous suffixes. A decreasing applied
skip width flushes them, in layer order. No routing value is read by Python.
"""

import copy
from dataclasses import replace

import torch
import triton
import triton.language as tl

from sglang.srt.early_exit.masked_kv import MaskedKVPool
from sglang.srt.early_exit.split import capacities
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context


@triton.jit
def plan_reentry(
    Actions,
    Indices,
    Seq,
    Starts,
    Cursors,
    Stops,
    Predicates,
    Stats,
    Widths,
    Offset: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(Indices + row)
    valid = slot >= 0
    slot = tl.maximum(slot, 0)
    action = tl.load(Actions + row)
    position = tl.load(Seq + row) - 1
    segments = tl.arange(0, 4)
    widths = tl.load(Widths + segments)
    old = tl.load(Starts + slot * 4 + segments)
    reentry = valid & (tl.sum(((old >= 0) & (action < widths)).to(tl.int32), 0) > 0)
    cursor = tl.where(reentry & (old >= 0), old, position)
    tl.store(Cursors + row * 4 + segments, cursor)
    tl.store(Stops + row * 4 + segments, position)
    tl.store(
        Predicates + Offset + row * 4 + segments,
        tl.where(reentry & (cursor < position), 0, -1),
    )
    start = tl.where(reentry, -1, old)
    start = tl.where(valid & (action >= widths) & (start < 0), position, start)
    tl.store(Starts + slot * 4 + segments, start, mask=valid)
    tl.store(Stats + row * 8, tl.load(Stats + row * 8) + reentry.to(tl.int64))
    tl.store(Stats + row * 8 + 1, tl.load(Stats + row * 8 + 1) + tl.maximum(action, 0))
    tl.store(
        Stats + row * 8 + 5, tl.load(Stats + row * 8 + 5) + (action > 0).to(tl.int64)
    )


@triton.jit
def save_boundary(
    H,
    R,
    Positions,
    Locations,
    Actions,
    HistoryH,
    HistoryR,
    HistoryP,
    HIDDEN: tl.constexpr,
    TOKENS: tl.constexpr,
    NPOS: tl.constexpr,
    B: tl.constexpr,
):
    row, block = tl.program_id(0), tl.program_id(1)
    offsets = block * B + tl.arange(0, B)
    slot = tl.load(Locations + row)
    keep = (tl.load(Actions + row) > 0) & (slot > 0)
    tl.store(
        HistoryH + slot * HIDDEN + offsets,
        tl.load(H + row * HIDDEN + offsets, offsets < HIDDEN, other=0),
        keep & (offsets < HIDDEN),
    )
    tl.store(
        HistoryR + slot * HIDDEN + offsets,
        tl.load(R + row * HIDDEN + offsets, offsets < HIDDEN, other=0),
        keep & (offsets < HIDDEN),
    )
    if block == 0:
        for dimension in range(NPOS):
            tl.store(
                HistoryP + dimension * TOKENS + slot,
                tl.load(Positions + dimension * tl.num_programs(0) + row),
                keep,
            )


class GraphPrefillKernel:
    """Single-request, <=64-token FLA chunks with device-only ragged length."""

    def __init__(self, chunk_indices, chunk_offsets):
        self.chunk_indices = chunk_indices
        self.chunk_offsets = chunk_offsets

    def extend(
        self, q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc, **kwargs
    ):
        from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule_fwd
        from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd

        # The normal chunk wrapper applies input_guard before these low-level
        # kernels. Replay bypasses it to keep ragged lengths on the GPU, so
        # restore its packed layout contract for projection slices and gates.
        q, k, v, g, beta = (value.contiguous() for value in (q, k, v, g, beta))
        _, output, _, _, states, _ = chunk_gated_delta_rule_fwd(
            q=l2norm_fwd(q),
            k=l2norm_fwd(k),
            v=v,
            g=g,
            beta=beta,
            scale=k.shape[-1] ** -0.5,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            chunk_indices=self.chunk_indices,
            chunk_offsets=self.chunk_offsets,
        )
        return output.to(q.dtype), None, states


class DeferredCache:
    PREDICATE_OFFSET = 48  # Existing main/repair branch reservation for batch8.

    def __init__(self, runtime, capacity=16):
        if capacity not in (1, 16, 32, 64):
            raise ValueError("lazy prefill capacity must be 1,16,32,64")
        self.runtime, self.capacity = runtime, capacity
        self.storage = None
        self.starts = None
        self.replays = {}
        self.widths = torch.tensor(runtime.widths, dtype=torch.int32, device="cuda")
        self.cursors = torch.zeros((8, 4), dtype=torch.int32, device="cuda")
        self.stops = torch.zeros_like(self.cursors)
        # reentries, skipped layers, replayed layers, chunks, capacity slots,
        # enqueued positions, replayed positions (summed across segments), unused.
        self.stats = torch.zeros((8, 8), dtype=torch.int64, device="cuda")

    def reset(self):
        if self.starts is not None:
            self.starts.fill_(-1)
        self.stats.zero_()

    def allocate(self, backend, hidden):
        if self.storage is not None:
            return
        pool = backend.full_attn_backend.token_to_kv_pool
        slots = backend.linear_attn_backend.req_to_token_pool.mamba_pool.size + 1
        self.starts = torch.full((slots, 4), -1, dtype=torch.int32, device="cuda")
        tokens = pool.size + pool.page_size
        self.storage = (
            hidden.new_zeros((tokens, hidden.shape[-1])),
            hidden.new_zeros((tokens, hidden.shape[-1])),
            torch.zeros((3, tokens), dtype=torch.int64, device="cuda"),
        )

    def make_replay(self, backend, batch, row, positions):
        key = (batch.batch_size, row, positions.ndim)
        if key in self.replays:
            return self.replays[key]
        cap = self.capacity
        full, linear = backend.full_attn_backend, backend.linear_attn_backend
        replay_batch = copy.copy(batch)
        replay_batch.batch_size = 1
        replay_batch.forward_mode = ForwardMode.EXTEND
        replay_batch.extend_num_tokens = cap
        replay_batch.num_token_non_padded = None
        replay_batch.num_token_non_padded_cpu = cap
        replay_batch.req_pool_indices = torch.zeros_like(
            batch.req_pool_indices[row : row + 1]
        )
        replay_batch.extend_prefix_lens = torch.zeros(
            1, dtype=torch.int32, device="cuda"
        )
        replay_batch.extend_seq_lens = torch.ones_like(replay_batch.extend_prefix_lens)
        replay_batch.extend_seq_lens_cpu = [cap]
        replay_batch.seq_lens_cpu = None
        replay_batch.seq_lens_sum = None
        replay_batch.out_cache_loc = torch.zeros(
            cap, dtype=batch.out_cache_loc.dtype, device="cuda"
        )
        replay_batch.seq_lens = torch.ones_like(replay_batch.extend_prefix_lens)
        replay_batch.positions = torch.zeros(cap, dtype=positions.dtype, device="cuda")
        replay_batch.mamba_track_mask = None
        replay_batch.mamba_track_indices = None
        replay_full = copy.copy(full)
        replay_full.kv_indptr = torch.zeros(2, dtype=torch.int32, device="cuda")
        kv_indices = torch.zeros_like(full.forward_metadata.kv_indices)
        qo = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        replay_full.token_to_kv_pool = MaskedKVPool(full.token_to_kv_pool)
        replay_full.forward_metadata = replace(
            full.forward_metadata,
            attn_logits=None,
            attn_lse=None,
            kv_indptr=replay_full.kv_indptr,
            kv_indices=kv_indices,
            qo_indptr=qo,
            max_extend_len=cap,
            num_kv_splits=None,
        )
        replay_linear = copy.copy(linear)
        replay_linear.forward_metadata = replace(
            linear.forward_metadata,
            query_start_loc=qo,
            mamba_cache_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
        )
        replay_linear.kernel_dispatcher = copy.copy(linear.kernel_dispatcher)
        replay_linear.kernel_dispatcher.extend_kernel = GraphPrefillKernel(
            torch.zeros((1, 2), dtype=torch.int64, device="cuda"),
            # Exactly one sequence and one <=64-token chunk. Build outside
            # capture; generic offset preparation creates host-side tensors.
            torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
        )
        replay_backend = copy.copy(backend)
        replay_backend.full_attn_backend = replay_full
        replay_backend.linear_attn_backend = replay_linear
        replay_backend.attn_backend_list = [replay_full, replay_linear]
        self.replays[key] = (replay_batch, replay_backend)
        return replay_batch, replay_backend

    def prepare(self, model, hidden, residual, positions, batch, backend):
        runtime = self.runtime
        size = batch.batch_size
        self.allocate(backend, hidden)
        indices = backend.linear_attn_backend.forward_metadata.mamba_cache_indices[
            :size
        ]
        plan_reentry[(size,)](
            runtime.action,
            indices,
            batch.seq_lens,
            self.starts,
            self.cursors,
            self.stops,
            runtime.branch_actions,
            self.stats,
            self.widths,
            Offset=self.PREDICATE_OFFSET,
        )
        history_h, history_r, history_p = self.storage
        save_boundary[(size, triton.cdiv(hidden.shape[-1], 256))](
            hidden,
            residual,
            positions,
            batch.out_cache_loc,
            runtime.action,
            history_h,
            history_r,
            history_p,
            HIDDEN=hidden.shape[-1],
            TOKENS=history_h.shape[0],
            NPOS=1 if positions.ndim == 1 else 3,
            B=256,
        )
        main_regions = 8 * len(capacities(size))
        for row in range(size):
            rb, replay_backend = self.make_replay(backend, batch, row, positions)
            begin = runtime.source
            for segment, width in enumerate(runtime.widths):
                marker = main_regions + row * 4 + segment
                if runtime.oracle and not runtime.building_graph:
                    while (
                        int(
                            runtime.branch_actions[
                                self.PREDICATE_OFFSET + row * 4 + segment
                            ]
                        )
                        == 0
                    ):
                        self.replay_body(
                            model,
                            hidden,
                            positions,
                            batch,
                            backend,
                            indices,
                            row,
                            segment,
                            width,
                            begin,
                            rb,
                            replay_backend,
                        )
                else:
                    runtime.regions.marker(marker * 2)
                    self.replay_body(
                        model,
                        hidden,
                        positions,
                        batch,
                        backend,
                        indices,
                        row,
                        segment,
                        width,
                        begin,
                        rb,
                        replay_backend,
                    )
                    runtime.regions.marker(marker * 2 + 1)
                begin = runtime.source + width

    def replay_body(
        self,
        model,
        hidden,
        positions,
        batch,
        backend,
        indices,
        row,
        segment,
        width,
        begin,
        rb,
        replay_backend,
    ):
        runtime, cap = self.runtime, self.capacity
        history_h, history_r, history_p = self.storage
        cursor = self.cursors[row, segment : segment + 1]
        stop = self.stops[row, segment : segment + 1]
        length = (stop - cursor).clamp(1, cap)
        steps = torch.arange(cap, device=hidden.device)
        valid = steps < length
        original_positions = (cursor.long() + steps).clamp_max(
            backend.full_attn_backend.req_to_token.shape[1] - 1
        )
        req = batch.req_pool_indices[row : row + 1].long()
        slots = backend.full_attn_backend.req_to_token[req, original_positions]
        slots = torch.where(valid, slots, 0).long()
        rb.req_pool_indices.copy_(batch.req_pool_indices[row : row + 1])
        rb.out_cache_loc.copy_(slots)
        rb.extend_prefix_lens.copy_(cursor)
        rb.extend_seq_lens.copy_(length)
        rb.seq_lens.copy_(cursor + length)
        rb.positions.copy_(original_positions)
        replay_full = replay_backend.full_attn_backend
        replay_full.forward_metadata.qo_indptr[1:].copy_(length)
        replay_full._fill_kv_indptr_and_indices(
            1, cursor, rb.req_pool_indices, replay_full.forward_metadata.kv_indices
        )
        replay_backend.linear_attn_backend.forward_metadata.mamba_cache_indices.copy_(
            indices[row : row + 1].clamp_min(0)
        )
        replay_positions = (
            history_p[0, slots]
            if positions.ndim == 1
            else history_p[:, slots].contiguous()
        )
        rb.mrope_positions = replay_positions if positions.ndim == 2 else None
        h, r = history_h[slots], history_r[slots]
        with forward_context(ForwardContext(attn_backend=replay_backend)):
            for layer in range(begin, runtime.source + width):
                h, r = model.layers[layer](
                    positions=replay_positions,
                    hidden_states=h,
                    residual=r,
                    forward_batch=rb,
                )
        # Padding is written only to the private history sink, never KV0.
        # Duplicate slot0 stores have identical zeros.
        history_h.index_copy_(0, slots, torch.where(valid[:, None], h, 0))
        history_r.index_copy_(0, slots, torch.where(valid[:, None], r, 0))
        self.stats[row, 2].add_(length[0] * (runtime.source + width - begin))
        self.stats[row, 3].add_(1)
        self.stats[row, 4].add_(cap)
        self.stats[row, 6].add_(length[0])
        cursor.add_(length)
        runtime.branch_actions[self.PREDICATE_OFFSET + row * 4 + segment].copy_(
            torch.where(cursor[0] < stop[0], 0, -1)
        )

    def metrics(self, batch):
        """Post-timing accounting; never consumed by a routing decision."""
        size = len(batch.reqs)
        slots = [request.mamba_pool_idx for request in batch.reqs]
        starts = self.starts[slots].cpu().tolist()
        ends = self.stops[:size, 0].cpu().tolist()
        stats = self.stats[:size].cpu().tolist()
        deltas = (5, 1, 1, 4)
        debt = sum(
            (end + 1 - start) * delta
            for row, end in zip(starts, ends)
            for start, delta in zip(row, deltas)
            if start >= 0
        )
        skipped, replayed = sum(row[1] for row in stats), sum(row[2] for row in stats)
        if skipped != replayed + debt:
            raise ValueError(f"lazy layer accounting: {skipped} != {replayed} + {debt}")
        return dict(
            row_stats=stats,
            pending_layer_tokens=debt,
            skipped_layer_tokens=skipped,
            replayed_layer_tokens=replayed,
            reentry_events=sum(row[0] for row in stats),
            prefill_chunks=sum(row[3] for row in stats),
            chunk_capacity=self.capacity,
            prefill_capacity_slots=sum(row[4] for row in stats),
            replayed_segment_tokens=sum(row[6] for row in stats),
            pending_starts=starts,
        )
