"""Inspect the real runtime's captured layer shapes and IF action mapping.

The graph recorder replaces CUDA scheduling, while forward_middle, select,
_segments and convert construct their real dispatch plan on CPU tensors.
GPU branch execution and numerical parity remain batch_smoke's responsibility.
"""

import contextlib
import types
import unittest
from unittest.mock import patch

import torch

from experiments.perfectblend.tests.test_batch_positions import (
    RUNTIME,
    Stream,
    batch,
    runtime,
)


class Plan:
    def __init__(self):
        self.open = None
        self.layers = {}

    def marker(self, code):
        if code % 2 == 0:
            if self.open is not None:
                raise AssertionError("nested conditional marker")
            self.open = code // 2
            self.layers[self.open] = []
        else:
            if self.open != code // 2:
                raise AssertionError("conditional marker is not paired")
            self.open = None

    def convert(self, graph, action, widths, action_indices):
        if set(self.layers) != set(range(len(widths))):
            raise AssertionError("IF predicates do not match captured markers")
        self.selected = []
        for group, (width, index) in enumerate(zip(widths, action_indices)):
            selected = int(action[index])
            if selected >= 0 and (
                selected < width if width > 0 else selected >= -width
            ):
                self.selected.append(group)
        return [len(self.layers[group]) for group in range(len(widths))]


def dispatch(actions, *, valid=None):
    size, plan = len(actions), Plan()
    positions = torch.arange(3 * size).view(3, size)
    request_batch = batch(size, positions)
    valid = [value >= 0 for value in actions] if valid is None else valid
    backend = types.SimpleNamespace(
        linear_attn_backend=types.SimpleNamespace(
            forward_metadata=types.SimpleNamespace(
                mamba_cache_indices=torch.tensor(
                    [row if enabled else -1 for row, enabled in enumerate(valid)]
                )
            )
        )
    )

    class Decoder:
        def __call__(self, *, positions, hidden_states, residual, forward_batch):
            if plan.open is not None:
                plan.layers[plan.open].append(hidden_states.shape[0])
            return hidden_states + 1, residual + 1

    model = types.SimpleNamespace(layers=[Decoder() for _ in range(44)], end_layer=44)
    instance = runtime()
    instance.regions = plan
    instance.converted = []
    instance.action = torch.zeros(8, dtype=torch.int32)
    instance.execution_actions = torch.full((9,), -1, dtype=torch.int32)
    instance.batched_main_executed = torch.zeros((), dtype=torch.int64)
    instance.actions = torch.tensor([0, 5, 6, 7, 11], dtype=torch.int32)
    instance.force = torch.full((8,), -1, dtype=torch.int32)
    instance.force[:size].copy_(torch.tensor(actions, dtype=torch.int32))
    instance.histogram = torch.zeros(5, dtype=torch.int64)
    instance.row_histogram = torch.zeros(8, 5, dtype=torch.int64)
    instance.net = lambda value: torch.zeros(value.shape[0], 5)
    instance.threshold = 0.99
    instance.select = types.MethodType(RUNTIME.EarlyExitRuntime.select, instance)
    with (
        patch.object(RUNTIME, "row_backend", return_value=backend),
        patch.object(RUNTIME, "get_attn_backend", return_value=backend),
        patch.object(torch.cuda, "current_stream", return_value=Stream()),
        patch.object(
            torch.cuda, "stream", side_effect=lambda _: contextlib.nullcontext()
        ),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
    ):
        instance.forward_middle(
            model,
            torch.ones(size, 4),
            torch.ones(size, 4),
            positions,
            request_batch,
        )
    instance.convert(None)
    return instance, plan


class BatchRoutingTest(unittest.TestCase):
    def test_batched_no_skip_uses_full_matrix_shapes(self):
        instance, plan = dispatch([0] * 8)
        self.assertEqual(plan.selected, [64])
        self.assertEqual(plan.layers[64], [8] * 11)
        self.assertEqual(instance.action.tolist(), [0] * 8)
        self.assertEqual(instance.histogram.tolist(), [8, 0, 0, 0, 0])
        self.assertEqual(instance.row_histogram.tolist(), [[1, 0, 0, 0, 0]] * 8)

    def test_learned_no_skip_keeps_source_action_counts(self):
        instance, plan = dispatch([-1] * 8, valid=[True] * 8)
        self.assertEqual(plan.selected, [64])
        self.assertEqual(instance.action.tolist(), [0] * 8)
        self.assertEqual(instance.histogram.tolist(), [8, 0, 0, 0, 0])

    def test_padding_disables_batched_path_and_both_padded_middle_bodies(self):
        for values in ([0] * 7 + [-1], [-1] * 8):
            with self.subTest(actions=values):
                instance, plan = dispatch(values)
                self.assertNotIn(64, plan.selected)
                for row, action in enumerate(values):
                    if action < 0:
                        self.assertFalse(
                            set(range(row * 8, row * 8 + 8)) & set(plan.selected)
                        )
                        self.assertEqual(instance.row_histogram[row].tolist(), [0] * 5)

    def test_mixed_actions_keep_independent_single_request_regions(self):
        values = [0, 5, 6, 7, 11, 0, 6, 11]
        instance, plan = dispatch(values)
        self.assertNotIn(64, plan.selected)
        for row, action in enumerate(values):
            for group, width in enumerate((5, 6, 7, 11)):
                self.assertEqual(row * 8 + group in plan.selected, action < width)
                self.assertEqual(row * 8 + 4 + group in plan.selected, action >= width)
            self.assertEqual(int(instance.row_histogram[row].sum()), 1)
        self.assertTrue(
            all(size == 1 for group in plan.selected for size in plan.layers[group])
        )
        self.assertEqual(instance.action.tolist(), values)

    def test_small_capture_sizes_retain_the_verified_row_path(self):
        for size in (1, 2, 4):
            with self.subTest(size=size):
                _, plan = dispatch([0] * size)
                self.assertEqual(len(plan.layers), size * 8)
                self.assertEqual(len(plan.selected), size * 4)
                self.assertTrue(
                    all(
                        value == 1
                        for group in plan.selected
                        for value in plan.layers[group]
                    )
                )


if __name__ == "__main__":
    unittest.main()
