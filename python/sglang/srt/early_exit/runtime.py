# SPDX-License-Identifier: Apache-2.0
"""Per-request Qwen3.5 GPU routing with concurrent cache backfill.

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
from torch import nn

from sglang.srt.early_exit.batch import (
    prepare_execution_actions,
    row_backend,
    row_batch,
    row_positions,
    validate_positions,
)
from sglang.srt.early_exit.conditional import ConditionalRegions
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
        if saved.get("format") != "five_class_shared_mlp_v1":
            raise ValueError(
                "early exit requires a five_class_shared_mlp_v1 checkpoint"
            )
        self.source = saved["source"]  # Count of executed layers, one-based.
        self.widths = tuple(saved["skips"])
        if self.source != 32 or self.widths != (5, 6, 7, 11):
            raise ValueError("this runtime supports source32 and widths5/6/7/11")
        if config.num_hidden_layers != 64 or config.hidden_size != 5120:
            raise ValueError("early exit expects the calibrated 64-layer 27B model")
        self.net = (
            nn.Sequential(
                nn.Linear(config.hidden_size, saved["width"], dtype=torch.float32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(saved["width"], 5, dtype=torch.float32),
            )
            .cuda()
            .eval()
            .requires_grad_(False)
        )
        self.net.load_state_dict(
            {k.removeprefix("net."): v for k, v in saved["state_dict"].items()}
        )
        assert all(
            parameter.dtype == torch.float32 for parameter in self.net.parameters()
        )
        self.threshold = next(
            p["threshold"] for p in saved["policies"] if p["goal"] == 0.99
        )
        self.mode = os.environ.get("SGLANG_EARLY_EXIT_CACHE", "async")
        if self.mode not in ("async", "serial"):
            raise ValueError("SGLANG_EARLY_EXIT_CACHE must be async or serial")
        self.regions = ConditionalRegions()
        self.max_batch_size = 8
        self.action = torch.zeros(self.max_batch_size, dtype=torch.int32, device="cuda")
        self.execution_actions = torch.full(
            (self.max_batch_size + 1,), -1, dtype=torch.int32, device="cuda"
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
        self.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def from_environment(cls, config, pp_group):
        path = os.environ.get("SGLANG_EARLY_EXIT_CHECKPOINT")
        if not path:
            return None
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        if pp_group.world_size != 1 or get_attention_tp_size() != 1:
            raise ValueError("GPU early exit currently requires TP1/PP1")
        return cls(path, config)

    def select(self, hidden, residual, valid):
        # Native training observes the complete BF16 residual sum at source32.
        value = (hidden + residual).float()
        value = value * torch.rsqrt(
            value.square().mean(-1, keepdim=True).clamp_min(1e-12)
        )
        probabilities = self.net(value).softmax(-1)
        confidence, index = probabilities.max(-1)
        selected = torch.where(confidence >= self.threshold, self.actions[index], 0)
        size = hidden.shape[0]
        chosen = torch.where(self.force[:size] >= 0, self.force[:size], selected)
        self.action[:size].copy_(torch.where(valid, chosen, -1))
        matches = (self.action[:size, None] == self.actions[None, :]).to(torch.int64)
        self.histogram.add_(matches.sum(0))
        self.row_histogram[:size].add_(matches)
        prepare_execution_actions(self.action[:size], self.execution_actions)

    def validate_decode(self, batch, num_tokens, positions):
        if not self.building_graph:
            raise RuntimeError(
                "early exit decode requires a captured CUDA graph; eager fallback is unsupported"
            )
        if (
            not 1 <= batch.batch_size <= self.max_batch_size
            or num_tokens != batch.batch_size
        ):
            raise ValueError(
                "per-request GPU early exit supports decode batch sizes 1 through 8"
            )
        if batch.spec_info is not None:
            raise ValueError(
                "per-request early exit does not support speculative decoding"
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
        if size not in self.states:
            self.states[size] = [
                (torch.empty_like(hidden), torch.empty_like(residual)) for _ in range(2)
            ]
        main_state, repair_state = self.states[size]
        for state in (main_state, repair_state):
            state[0].copy_(hidden)
            state[1].copy_(residual)
        rows = [row_batch(batch, index) for index in range(size)]
        position_rows = [row_positions(positions, index) for index in range(size)]
        main_backends, repair_backends = [], []
        for index in range(size):
            for repair, target in ((False, main_backends), (True, repair_backends)):
                scratch = self.scratch.setdefault((size, index, repair), {})
                target.append(row_backend(backend, index, scratch))
        current = torch.cuda.current_stream()
        self.repair_stream.wait_stream(current)
        asynchronous = self.mode == "async" and torch.cuda.is_current_stream_capturing()
        # Each captured body contains only one request. A skipped body launches
        # no layer kernels. Prefix and tail remain ordinary batched operations.
        # Static row expansion trades middle-layer GEMM batching for independent
        # GPU decisions; no action or compacted request count returns to Python.
        with torch.cuda.stream(self.repair_stream):
            for index in range(size):
                with forward_context(
                    ForwardContext(attn_backend=repair_backends[index])
                ):
                    self._segments(
                        model,
                        tuple(value[index : index + 1] for value in repair_state),
                        position_rows[index],
                        rows[index],
                        row=index,
                        repair=True,
                    )
        if not asynchronous:
            current.wait_stream(self.repair_stream)
        for index in range(size):
            with forward_context(ForwardContext(attn_backend=main_backends[index])):
                self._segments(
                    model,
                    tuple(value[index : index + 1] for value in main_state),
                    position_rows[index],
                    rows[index],
                    row=index,
                    repair=False,
                )
        if size == self.max_batch_size:
            self._batched_middle(model, main_state, positions, batch)
        hidden, residual = main_state
        for layer_idx in range(self.source + self.widths[-1], model.end_layer):
            hidden, residual = model.layers[layer_idx](
                positions=positions,
                hidden_states=hidden,
                residual=residual,
                forward_batch=batch,
            )
        current.wait_stream(self.repair_stream)
        return hidden, residual

    def _batched_middle(self, model, state, positions, batch):
        # All eight real actions must be zero. This exclusive IF body preserves
        # full-model GEMM shapes; row bodies have action -1 in this case.
        marker = batch.batch_size * 8
        self.regions.marker(2 * marker)
        hidden, residual = state
        for layer_idx in range(self.source, self.source + self.widths[-1]):
            hidden, residual = model.layers[layer_idx](
                positions=positions,
                hidden_states=hidden,
                residual=residual,
                forward_batch=batch,
            )
        state[0].copy_(hidden)
        state[1].copy_(residual)
        self.executed[:4].add_(batch.batch_size)
        self.row_executed[: batch.batch_size, :4].add_(1)
        self.batched_main_executed.add_(1)
        self.regions.marker(2 * marker + 1)

    def _segments(self, model, state, positions, batch, *, row, repair):
        begin = self.source
        for group, width in enumerate(self.widths):
            region = group + (4 if repair else 0)
            marker = row * 8 + region
            self.regions.marker(2 * marker)
            hidden, residual = state
            for layer_idx in range(begin, self.source + width):
                hidden, residual = model.layers[layer_idx](
                    positions=positions,
                    hidden_states=hidden,
                    residual=residual,
                    forward_batch=batch,
                )
            # Every consumer outside the conditional body reads stable state.
            # A bypass therefore retains the preceding boundary activation.
            state[0].copy_(hidden)
            state[1].copy_(residual)
            self.executed[region].add_(1)
            self.row_executed[row, region].add_(1)
            self.regions.marker(2 * marker + 1)
            begin = self.source + width

    def convert(self, graph):
        widths = [*self.widths, *(-w for w in self.widths)] * self.capture_batch_size
        indices = [index for index in range(self.capture_batch_size) for _ in range(8)]
        if self.capture_batch_size == self.max_batch_size:
            widths.append(1)
            indices.append(self.max_batch_size)
        moved = self.regions.convert(
            graph,
            self.execution_actions,
            widths,
            action_indices=indices,
        )
        self.converted.append(moved)
        logger.info(
            "Early-exit CUDA IF regions: %s; cache=%s; threshold=%s; batch=%s",
            moved,
            self.mode,
            self.threshold,
            self.capture_batch_size,
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
