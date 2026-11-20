# SPDX-License-Identifier: Apache-2.0
"""A permanently reduced decode model; prefill retains all original layers.

The skipped interval never runs during decode and its caches are never read
again. Changing width requires a new model instance and freshly captured graphs.
"""

from contextlib import contextmanager

import torch

from sglang.srt.early_exit.batch import validate_positions
from sglang.srt.model_executor.forward_context import get_attn_backend


class FixedSkipRuntime:
    source = 32
    routing = "fixed_skip"
    lazy = None
    max_batch_size = 8

    def __init__(self, width, config):
        if width not in (0, 5, 7, 11):
            raise ValueError("Fixed decode skip must be 0, 5, 7 or 11")
        if config.num_hidden_layers != 64 or config.hidden_size != 5120:
            raise ValueError("Fixed decode skipping requires the 64-layer 27B model")
        self._width = width
        self.building_graph = False
        self.converted = []
        self.histogram = torch.zeros(5, dtype=torch.int64, device="cuda")
        self.row_histogram = torch.zeros((8, 5), dtype=torch.int64, device="cuda")
        self.layer_tokens = torch.zeros(64, dtype=torch.int64, device="cuda")

    @property
    def width(self):
        return self._width

    def validate_decode(self, batch, num_tokens, positions):
        if not self.building_graph:
            raise RuntimeError("Fixed skipping requires captured CUDA graph decode")
        if not 1 <= batch.batch_size <= 8 or num_tokens != batch.batch_size:
            raise ValueError("Fixed skipping supports decode batches 1 through 8")
        if batch.spec_info is not None:
            raise ValueError("Fixed skipping does not support speculative decoding")
        for value in (positions, batch.positions, getattr(batch, "mrope_positions", None)):
            if value is not None:
                validate_positions(value, num_tokens)

    def forward_middle(self, model, hidden, residual, positions, batch):
        self.validate_decode(batch, hidden.shape[0], positions)
        size = batch.batch_size
        backend = get_attn_backend()
        valid = backend.linear_attn_backend.forward_metadata.mamba_cache_indices[:size] >= 0
        column = (0, 5, 6, 7, 11).index(self.width)
        self.histogram[column].add_(valid.sum())
        self.row_histogram[:size, column].add_(valid)
        for index in range(self.source + self.width, model.end_layer):
            hidden, residual = model.layers[index](
                positions=positions, hidden_states=hidden, residual=residual,
                forward_batch=batch,
            )
            self.layer_tokens[index].add_(valid.sum())
        return hidden, residual

    def reset_metrics(self):
        self.histogram.zero_()
        self.row_histogram.zero_()
        self.layer_tokens.zero_()

    def metrics(self, size, steps):
        expected = [0] * (self.source + self.width) + [size * steps] * (64 - self.source - self.width)
        actual = self.layer_tokens.cpu().tolist()
        if actual != expected:
            raise ValueError("Fixed skip executed a skipped layer or omitted a suffix layer")
        histogram = self.histogram.cpu().tolist()
        if histogram[(0, 5, 6, 7, 11).index(self.width)] != size * steps:
            raise ValueError("Fixed skip did not apply to every decode token")
        return dict(routing=self.routing, fixed_skip_width=self.width,
                    decode_layer_tokens_from_boundary=actual,
                    skipped_layer_tokens=self.width * size * steps,
                    repaired_layer_tokens=0)

    def convert(self, graph):
        # This is a static reduced graph; there are no conditional regions.
        self.converted.append(0)

    @contextmanager
    def graph_build(self):
        self.building_graph = True
        try:
            yield
        finally:
            self.building_graph = False
