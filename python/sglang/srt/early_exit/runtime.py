# SPDX-License-Identifier: Apache-2.0
"""Batch-one Qwen3.5 GPU routing with concurrent cache backfill.

The main stream bypasses a prefix of the marked middle layers. The repair
stream evaluates that prefix from the saved boundary activation, writing its
KV and GDN state. CUDA IF nodes select complementary regions. A device event
joins repair before the decode graph completes, so the next token sees every
cache update. Python runs this topology only during warmup and graph capture.
"""

import copy
import hashlib
import logging
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

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
        self.action = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.actions = torch.tensor(saved["actions"], dtype=torch.int32, device="cuda")
        self.force = torch.tensor(
            [int(os.environ.get("SGLANG_EARLY_EXIT_FORCE", "-1"))],
            dtype=torch.int32,
            device="cuda",
        )
        self.histogram = torch.zeros(5, dtype=torch.int64, device="cuda")
        self.executed = torch.zeros(8, dtype=torch.int64, device="cuda")
        self.repair_stream = torch.cuda.Stream()
        self.main_state = None
        self.repair_state = None
        self.scratch = {}
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

    def select(self, hidden, residual):
        # Native training observes the complete BF16 residual sum at source32.
        value = (hidden + residual).float()
        value = value * torch.rsqrt(
            value.square().mean(-1, keepdim=True).clamp_min(1e-12)
        )
        probabilities = self.net(value).softmax(-1)
        confidence, index = probabilities.max(-1)
        selected = torch.where(confidence >= self.threshold, self.actions[index], 0)
        self.action.copy_(torch.where(self.force >= 0, self.force, selected))
        self.histogram.add_((self.action == self.actions).to(torch.int64))

    def _repair_backend(self):
        # Decode attention uses shared per-backend logits scratch. Repair and
        # the suffix can run different attention layers concurrently, so each
        # stream needs its own scratch; KV/state pools stay shared per layer.
        backend = get_attn_backend()
        full = backend.full_attn_backend
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        if not isinstance(full, TritonAttnBackend):
            raise ValueError("early exit backfill requires --attention-backend triton")
        metadata = full.forward_metadata
        key = metadata.attn_logits.shape, metadata.attn_logits.dtype
        if key not in self.scratch:
            self.scratch[key] = (
                torch.empty_like(metadata.attn_logits),
                torch.empty_like(metadata.attn_lse),
            )
        logits, lse = self.scratch[key]
        repair_full = copy.copy(full)
        repair_full.forward_metadata = replace(
            metadata, attn_logits=logits, attn_lse=lse
        )
        repair_backend = copy.copy(backend)
        repair_backend.full_attn_backend = repair_full
        return repair_backend

    def forward_middle(self, model, hidden, residual, positions, batch):
        if not self.building_graph:
            raise RuntimeError(
                "early exit decode requires a captured CUDA graph; eager fallback is unsupported"
            )
        if batch.batch_size != 1 or hidden.shape[0] != 1:
            raise ValueError(
                "GPU early exit requires --max-running-requests 1 and graph bs1"
            )
        self.select(hidden, residual)
        if self.main_state is None:
            self.main_state = torch.empty_like(hidden), torch.empty_like(residual)
            self.repair_state = torch.empty_like(hidden), torch.empty_like(residual)
        for state in (self.main_state, self.repair_state):
            state[0].copy_(hidden)
            state[1].copy_(residual)
        backend = self._repair_backend()
        current = torch.cuda.current_stream()
        self.repair_stream.wait_stream(current)
        asynchronous = self.mode == "async" and torch.cuda.is_current_stream_capturing()
        # Warmup evaluates both regions to compile every kernel. Serialize it
        # because conditional nodes have not yet replaced the marked bodies.
        with torch.cuda.stream(self.repair_stream), forward_context(
            ForwardContext(attn_backend=backend)
        ):
            self._segments(model, self.repair_state, positions, batch, repair=True)
        if not asynchronous:
            current.wait_stream(self.repair_stream)
        self._segments(model, self.main_state, positions, batch, repair=False)
        hidden, residual = self.main_state
        for layer_idx in range(self.source + self.widths[-1], model.end_layer):
            hidden, residual = model.layers[layer_idx](
                positions=positions,
                hidden_states=hidden,
                residual=residual,
                forward_batch=batch,
            )
        current.wait_stream(self.repair_stream)
        return hidden, residual

    def _segments(self, model, state, positions, batch, *, repair):
        begin = self.source
        for group, width in enumerate(self.widths):
            region = group + (4 if repair else 0)
            self.regions.marker(2 * region)
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
            self.regions.marker(2 * region + 1)
            begin = self.source + width

    def convert(self, graph):
        moved = self.regions.convert(
            graph, self.action, [*self.widths, *(-w for w in self.widths)]
        )
        self.converted.append(moved)
        logger.info(
            "Early-exit CUDA IF regions: %s; cache=%s; threshold=%s",
            moved,
            self.mode,
            self.threshold,
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
