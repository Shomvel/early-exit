# SPDX-License-Identifier: Apache-2.0
"""Batch-wide Qwen3.5 GPU routing with concurrent cache backfill.

The main stream bypasses a prefix of the marked middle layers. The repair
stream evaluates that prefix from the saved boundary activation, writing its
KV and GDN state. CUDA IF nodes select complementary regions. A device event
joins repair before the decode graph completes, so the next token sees every
cache update. Python runs this topology only during warmup and graph capture.
"""

import hashlib
import logging
import os
from contextlib import contextmanager
from pathlib import Path

import torch

from sglang.srt.early_exit.batch import repair_backend, validate_positions
from sglang.srt.early_exit.conditional import ConditionalRegions
from sglang.srt.early_exit.policy import (
    FIVE_CLASS,
    FOUR_WIDTH_BCE,
    load_classifier,
    select_action,
)
from sglang.srt.early_exit.split import (
    branch_actions,
    bucket_backend,
    bucket_batch,
    capacities,
    gather_rows,
    merge_rows,
    packed_backend,
    packed_batch,
    partition,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_attn_backend,
)

logger = logging.getLogger(__name__)


class EarlyExitRuntime:
    def __init__(self, checkpoint, config):
        path = Path(checkpoint)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        self.checkpoint_format = saved.get("format")
        if self.checkpoint_format not in (FIVE_CLASS, FOUR_WIDTH_BCE):
            raise ValueError(
                "early exit requires a five_class_shared_mlp_v1 or four_width_bce_mlp_v1 checkpoint"
            )
        self.source = saved["source"]  # Count of executed layers, one-based.
        self.widths = tuple(saved["skips"])
        if self.source != 32 or self.widths != (5, 6, 7, 11):
            raise ValueError("this runtime supports source32 and widths5/6/7/11")
        if tuple(saved["actions"]) != (0, *self.widths):
            raise ValueError("early exit expects actions0/5/6/7/11 in width order")
        if config.num_hidden_layers != 64 or config.hidden_size != 5120:
            raise ValueError("early exit expects the calibrated 64-layer 27B model")
        self.net = load_classifier(saved, config.hidden_size).cuda()
        assert all(
            parameter.dtype == torch.float32 for parameter in self.net.parameters()
        )
        self.threshold = next(
            p["threshold"] for p in saved["policies"] if p["goal"] == 0.99
        )
        self.mode = os.environ.get("SGLANG_EARLY_EXIT_CACHE", "async")
        if self.mode not in ("async", "serial", "lazy"):
            raise ValueError("SGLANG_EARLY_EXIT_CACHE must be async, serial or lazy")
        self.regions = ConditionalRegions()
        self.routing = "split_skip"
        self.max_batch_size = 8
        self.action = torch.zeros(self.max_batch_size, dtype=torch.int32, device="cuda")
        self.proposed_action = torch.full_like(self.action, -1)
        self.group_actions = torch.full((2,), -1, dtype=torch.int32, device="cuda")
        self.group_counts = torch.zeros(2, dtype=torch.int64, device="cuda")
        self.group_capacities = torch.zeros_like(self.group_counts)
        self.group_indices = torch.full(
            (2, self.max_batch_size), -1, dtype=torch.int64, device="cuda"
        )
        self.branch_actions = torch.full(
            (12 * len(capacities(self.max_batch_size)) + 4 * self.max_batch_size,),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        self.actions = torch.tensor(saved["actions"], dtype=torch.int32, device="cuda")
        forced = int(os.environ.get("SGLANG_EARLY_EXIT_FORCE", "-1"))
        if forced not in (-1, 0, *self.widths):
            raise ValueError("early-exit force must be -1, 0, 5, 6, 7, or 11")
        self.force = torch.full(
            (self.max_batch_size,), forced, dtype=torch.int32, device="cuda"
        )
        self.histogram = torch.zeros(5, dtype=torch.int64, device="cuda")
        self.executed = torch.zeros(8, dtype=torch.int64, device="cuda")
        self.batch_executed = torch.zeros(8, dtype=torch.int64, device="cuda")
        self.group_executed = torch.zeros((2, 8), dtype=torch.int64, device="cuda")
        self.group_action_histogram = torch.zeros(
            (2, 5), dtype=torch.int64, device="cuda"
        )
        self.group_size_histogram = torch.zeros(
            (2, 9), dtype=torch.int64, device="cuda"
        )
        self.bucket_histogram = torch.zeros_like(self.group_size_histogram)
        self.row_histogram = torch.zeros(
            (self.max_batch_size, 5), dtype=torch.int64, device="cuda"
        )
        self.row_executed = torch.zeros(
            (self.max_batch_size, 8), dtype=torch.int64, device="cuda"
        )
        self.batched_main_executed = torch.zeros((), dtype=torch.int64, device="cuda")
        self.repair_stream = torch.cuda.Stream()
        self.states = {}
        self.scratch = {}
        self.capture_batch_size = None
        self.converted = []
        self.building_graph = False
        self.oracle = False  # Test-only eager scheduling; never used for timing.
        self.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.lazy = None
        if self.mode == "lazy":
            from sglang.srt.early_exit.lazy import DeferredCache

            self.lazy = DeferredCache(
                self, int(os.environ.get("SGLANG_EARLY_EXIT_CHUNK", "16"))
            )

    @classmethod
    def from_environment(cls, config, pp_group):
        path = os.environ.get("SGLANG_EARLY_EXIT_CHECKPOINT")
        fixed = os.environ.get("SGLANG_EARLY_EXIT_FIXED_SKIP")
        if not path and fixed is None:
            return None
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        if pp_group.world_size != 1 or get_attention_tp_size() != 1:
            raise ValueError("GPU early exit currently requires TP1/PP1")
        if fixed is not None:
            if path:
                raise ValueError("Fixed skipping and classifier routing are separate modes")
            from sglang.srt.early_exit.fixed import FixedSkipRuntime

            return FixedSkipRuntime(int(fixed), config)
        routing = os.environ.get("SGLANG_EARLY_EXIT_ROUTING", "batch_min")
        if routing == "batch_min":
            from sglang.srt.early_exit.common import BatchWideRuntime

            return BatchWideRuntime(path, config)
        if routing != "split_skip":
            raise ValueError("early-exit routing must be batch_min or split_skip")
        return cls(path, config)

    def select(self, hidden, residual, valid):
        # Native training observes the complete BF16 residual sum at source32.
        value = (hidden + residual).float()
        value = value * torch.rsqrt(
            value.square().mean(-1, keepdim=True).clamp_min(1e-12)
        )
        selected = select_action(
            self.net(value), self.actions, self.threshold, self.checkpoint_format
        )
        size = hidden.shape[0]
        chosen = torch.where(self.force[:size] >= 0, self.force[:size], selected)
        self.proposed_action[:size].copy_(torch.where(valid, chosen, -1))
        partition(
            self.proposed_action[:size],
            self.action[:size],
            self.group_actions,
            self.group_counts,
            self.group_indices[:, :size],
            self.group_capacities,
        )
        branch_actions(
            self.group_actions,
            self.group_capacities,
            self.branch_actions,
            self.widths,
            size,
        )
        matches = (self.action[:size, None] == self.actions[None, :]).to(torch.int64)
        self.histogram.add_(matches.sum(0))
        self.row_histogram[:size].add_(matches)
        self.group_action_histogram.add_(
            (self.group_actions[:, None] == self.actions).to(torch.int64)
        )
        bins = torch.arange(9, device=hidden.device)
        self.group_size_histogram.add_(
            (self.group_counts[:, None] == bins).to(torch.int64)
        )
        self.bucket_histogram.add_(
            (self.group_capacities[:, None] == bins).to(torch.int64)
        )

    def validate_decode(self, batch, num_tokens, positions):
        if not self.building_graph and not self.oracle:
            raise RuntimeError(
                "early exit decode requires a captured CUDA graph; eager fallback is unsupported"
            )
        if (
            not 1 <= batch.batch_size <= self.max_batch_size
            or num_tokens != batch.batch_size
        ):
            raise ValueError(
                "GPU split early exit supports decode batch sizes 1 through 8"
            )
        if batch.spec_info is not None:
            raise ValueError(
                "GPU split early exit does not support speculative decoding"
            )
        validate_positions(positions, num_tokens)
        validate_positions(batch.positions, num_tokens)
        if getattr(batch, "mrope_positions", None) is not None:
            validate_positions(batch.mrope_positions, num_tokens)

    def forward_middle(self, model, hidden, residual, positions, batch):
        self.validate_decode(batch, hidden.shape[0], positions)
        size = batch.batch_size
        backend = get_attn_backend()
        valid = (
            backend.linear_attn_backend.forward_metadata.mamba_cache_indices[:size] >= 0
        )
        self.select(hidden, residual, valid)
        self.capture_batch_size = size
        if self.lazy is not None:
            self.lazy.prepare(model, hidden, residual, positions, batch, backend)
        if size not in self.states:
            self.states[size] = [
                (torch.empty_like(hidden), torch.empty_like(residual)) for _ in range(3)
            ]
        no_skip_state, skip_state, repair_state = self.states[size]
        batches, backends = [], []
        for group in range(2):
            indices = self.group_indices[group, :size]
            packed = packed_batch(batch, indices)
            packed_positions = gather_rows(positions, indices, positions=True)
            packed_scratch = self.scratch.setdefault((size, group, "packed"), {})
            grouped_backend = packed_backend(backend, packed, indices, packed_scratch)
            group_batches, group_backends = [], []
            for capacity in capacities(size):
                subbatch = bucket_batch(packed, capacity)
                subscratch = self.scratch.setdefault((size, group, "splits"), {})
                subbackend = bucket_backend(grouped_backend, subbatch, subscratch)
                if group == 1 and self.lazy is None:
                    repair_scratch = self.scratch.setdefault(
                        (size, capacity, "repair"), {}
                    )
                    repair = repair_backend(subbackend, repair_scratch)
                else:
                    repair = None
                group_batches.append((subbatch, packed_positions[..., :capacity]))
                group_backends.append((subbackend, repair))
            batches.append(group_batches)
            backends.append(group_backends)
            group_states = (no_skip_state,) if group == 0 else (skip_state,)
            if group == 1 and self.lazy is None:
                group_states += (repair_state,)
            for state in group_states:
                state[0].copy_(gather_rows(hidden, indices))
                state[1].copy_(gather_rows(residual, indices))
        current = torch.cuda.current_stream()
        if self.lazy is None:
            self.repair_stream.wait_stream(current)
        asynchronous = self.mode == "async" and torch.cuda.is_current_stream_capturing()
        if self.lazy is None:
            with torch.cuda.stream(self.repair_stream):
                for bucket_index, (subbatch, subpositions) in enumerate(batches[1]):
                    with forward_context(
                        ForwardContext(attn_backend=backends[1][bucket_index][1])
                    ):
                        self._segments(
                            model,
                            repair_state,
                            subpositions,
                            subbatch,
                            group=1,
                            bucket_index=bucket_index,
                            repair=True,
                        )
            if not asynchronous:
                current.wait_stream(self.repair_stream)
        for group, state in enumerate((no_skip_state, skip_state)):
            for bucket_index, (subbatch, subpositions) in enumerate(batches[group]):
                with forward_context(
                    ForwardContext(attn_backend=backends[group][bucket_index][0])
                ):
                    self._segments(
                        model,
                        state,
                        subpositions,
                        subbatch,
                        group=group,
                        bucket_index=bucket_index,
                        repair=False,
                    )
        for group, state in enumerate((no_skip_state, skip_state)):
            members = (
                (self.proposed_action[:size] == 0)
                if group == 0
                else (self.proposed_action[:size] > 0)
            )
            hidden = merge_rows(state[0], hidden, members)
            residual = merge_rows(state[1], residual, members)
        for layer_idx in range(self.source + self.widths[-1], model.end_layer):
            hidden, residual = model.layers[layer_idx](
                positions=positions,
                hidden_states=hidden,
                residual=residual,
                forward_batch=batch,
            )
        if self.lazy is None:
            current.wait_stream(self.repair_stream)
        return hidden, residual

    def _segments(self, model, state, positions, batch, *, group, bucket_index, repair):
        size = self.capture_batch_size
        bucket_count = len(capacities(size))
        pass_index = 2 if repair else group
        members = (
            (self.proposed_action[:size] == 0)
            if group == 0
            else (self.proposed_action[:size] > 0)
        )
        begin = self.source
        for segment, width in enumerate(self.widths):
            region = segment + (4 if repair else 0)
            marker = (pass_index * bucket_count + bucket_index) * 4 + segment
            if (
                self.oracle
                and not self.building_graph
                and int(self.branch_actions[marker]) != 0
            ):
                begin = self.source + width
                continue
            self.regions.marker(2 * marker)
            hidden, residual = (value[: batch.batch_size] for value in state)
            for layer_idx in range(begin, self.source + width):
                hidden, residual = model.layers[layer_idx](
                    positions=positions,
                    hidden_states=hidden,
                    residual=residual,
                    forward_batch=batch,
                )
            # Every consumer outside the conditional body reads stable state.
            # A bypass therefore retains the preceding boundary activation.
            state[0][: batch.batch_size].copy_(hidden)
            state[1][: batch.batch_size].copy_(residual)
            selected_group = 1 if repair else pass_index
            self.executed[region].add_(members.sum())
            self.row_executed[:size, region].add_(members)
            self.batch_executed[region].add_(1)
            self.group_executed[selected_group, region].add_(1)
            if pass_index == 0 and segment == len(self.widths) - 1:
                self.batched_main_executed.add_(1)
            self.regions.marker(2 * marker + 1)
            begin = self.source + width

    def convert(self, graph):
        count = (8 if self.lazy is not None else 12) * len(
            capacities(self.capture_batch_size)
        )
        indices = list(range(count))
        kinds = [0] * count
        if self.lazy is not None:
            indices += list(
                range(
                    self.lazy.PREDICATE_OFFSET,
                    self.lazy.PREDICATE_OFFSET + 4 * self.capture_batch_size,
                )
            )
            kinds += [1] * (4 * self.capture_batch_size)
        moved = self.regions.convert(
            graph, self.branch_actions, [1] * len(indices), indices, kinds
        )
        self.converted.append(moved)
        logger.info(
            "Early-exit CUDA IF regions: %s; cache=%s; threshold=%s; batch=%s; routing=%s",
            moved,
            self.mode,
            self.threshold,
            self.capture_batch_size,
            self.routing,
        )

    @contextmanager
    def graph_build(self):
        self.building_graph = True
        try:
            yield
        finally:
            self.building_graph = False


def get_runtime(model):
    for module in model.modules():
        runtime = getattr(module, "early_exit", None)
        if runtime is not None:
            return runtime
    return None
