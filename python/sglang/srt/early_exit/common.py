# SPDX-License-Identifier: Apache-2.0
"""One GPU-selected action for the whole decode batch, without row grouping."""

import torch

from sglang.srt.early_exit.batch import repair_backend
from sglang.srt.early_exit.policy import select_action
from sglang.srt.early_exit.runtime import EarlyExitRuntime
from sglang.srt.model_executor.forward_context import (
    ForwardContext, forward_context, get_attn_backend,
)


def common_action(proposed):
    valid = proposed >= 0
    minimum = torch.where(valid, proposed, 11).amin()
    return torch.where(valid.any(), minimum, -1)


class BatchWideRuntime(EarlyExitRuntime):
    def __init__(self, checkpoint, config):
        super().__init__(checkpoint, config)
        if self.mode == "lazy":
            raise ValueError("batch_min uses serial or async cache repair")
        self.routing = "batch_min"
        self.common_action = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.proposal_histogram = torch.zeros_like(self.histogram)
        self.common_step_histogram = torch.zeros_like(self.histogram)

    def select(self, hidden, residual, valid):
        value = (hidden + residual).float()
        value *= torch.rsqrt(value.square().mean(-1, keepdim=True).clamp_min(1e-12))
        selected = select_action(self.net(value), self.actions, self.threshold, self.checkpoint_format)
        size = hidden.shape[0]
        chosen = torch.where(self.force[:size] >= 0, self.force[:size], selected)
        proposed = torch.where(valid, chosen, -1)
        self.proposed_action[:size].copy_(proposed)
        self.common_action.copy_(common_action(proposed).reshape(1))
        self.action[:size].copy_(torch.where(valid, self.common_action[0], -1))
        matches = (self.action[:size, None] == self.actions).to(torch.int64)
        self.histogram.add_(matches.sum(0))
        self.row_histogram[:size].add_(matches)
        self.proposal_histogram.add_((proposed[:, None] == self.actions).sum(0))
        self.common_step_histogram.add_((self.common_action == self.actions).to(torch.int64))

    def forward_middle(self, model, hidden, residual, positions, batch):
        self.validate_decode(batch, hidden.shape[0], positions)
        size = batch.batch_size
        backend = get_attn_backend()
        valid = backend.linear_attn_backend.forward_metadata.mamba_cache_indices[:size] >= 0
        self.select(hidden, residual, valid)
        self.capture_batch_size = size
        if size not in self.states:
            self.states[size] = [
                (torch.empty_like(hidden), torch.empty_like(residual)) for _ in range(2)
            ]
        main, repair = self.states[size]
        for state in (main, repair):
            state[0].copy_(hidden)
            state[1].copy_(residual)
        repair_attn = repair_backend(backend, self.scratch)
        current = torch.cuda.current_stream()
        self.repair_stream.wait_stream(current)
        asynchronous = self.mode == "async" and torch.cuda.is_current_stream_capturing()
        with torch.cuda.stream(self.repair_stream), forward_context(ForwardContext(attn_backend=repair_attn)):
            self._whole_segments(model, repair, positions, batch, valid, repair=True)
        if not asynchronous:
            current.wait_stream(self.repair_stream)
        self._whole_segments(model, main, positions, batch, valid, repair=False)
        hidden, residual = main
        for index in range(self.source + self.widths[-1], model.end_layer):
            hidden, residual = model.layers[index](
                positions=positions, hidden_states=hidden, residual=residual,
                forward_batch=batch,
            )
        current.wait_stream(self.repair_stream)
        return hidden, residual

    def _whole_segments(self, model, state, positions, batch, valid, *, repair):
        begin = self.source
        for segment, width in enumerate(self.widths):
            region = segment + (4 if repair else 0)
            self.regions.marker(2 * region)
            hidden, residual = state
            for index in range(begin, self.source + width):
                hidden, residual = model.layers[index](
                    positions=positions, hidden_states=hidden, residual=residual,
                    forward_batch=batch,
                )
            state[0].copy_(hidden)
            state[1].copy_(residual)
            self.executed[region].add_(valid.sum())
            self.row_executed[:batch.batch_size, region].add_(valid)
            self.batch_executed[region].add_(1)
            self.regions.marker(2 * region + 1)
            begin = self.source + width

    def convert(self, graph):
        moved = self.regions.convert(
            graph, self.common_action, [*self.widths, *(-w for w in self.widths)]
        )
        self.converted.append(moved)
