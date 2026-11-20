"""CPU checks of the actual split dispatch and RoPE/MRoPE row packing."""

import contextlib
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def load_runtime():
    package = types.ModuleType("sglang")
    package.__path__ = [str(Path(__file__).resolve().parents[3] / "python/sglang")]
    with patch.dict(sys.modules, {"sglang": package}):
        return importlib.import_module("sglang.srt.early_exit.runtime")


RUNTIME = load_runtime()


class Stream:
    def wait_stream(self, _):
        pass


def batch(size, mrope=None):
    return types.SimpleNamespace(
        batch_size=size,
        input_ids=torch.arange(size),
        req_pool_indices=torch.arange(size) + 10,
        seq_lens=torch.arange(size) + 5,
        out_cache_loc=torch.arange(size) + 100,
        positions=torch.arange(size) + 4,
        mrope_positions=mrope,
        spec_info=None,
        forward_mode=types.SimpleNamespace(is_decode=lambda: True),
    )


def runtime():
    result = RUNTIME.EarlyExitRuntime.__new__(RUNTIME.EarlyExitRuntime)
    result.source, result.widths = 32, (5, 6, 7, 11)
    result.max_batch_size, result.building_graph, result.mode = 8, True, "serial"
    result.states, result.scratch = {}, {}
    result.lazy, result.oracle = None, False
    result.repair_stream = Stream()
    result.regions = types.SimpleNamespace(marker=lambda _: None)
    result.executed = torch.zeros(8, dtype=torch.int64)
    result.row_executed = torch.zeros(8, 8, dtype=torch.int64)
    result.batched_main_executed = torch.zeros((), dtype=torch.int64)
    result.batch_executed = torch.zeros(8, dtype=torch.int64)
    result.group_executed = torch.zeros(2, 8, dtype=torch.int64)
    result.group_action_histogram = torch.zeros(2, 5, dtype=torch.int64)
    result.group_size_histogram = torch.zeros(2, 9, dtype=torch.int64)
    result.bucket_histogram = torch.zeros(2, 9, dtype=torch.int64)
    result.group_actions = torch.full((2,), -1, dtype=torch.int32)
    result.group_counts = torch.zeros(2, dtype=torch.int64)
    result.group_capacities = torch.zeros(2, dtype=torch.int64)
    result.group_indices = torch.full((2, 8), -1, dtype=torch.int64)
    result.branch_actions = torch.full((48,), -1, dtype=torch.int32)
    result.routing = "split_skip"
    result.action = torch.zeros(8, dtype=torch.int32)
    result.proposed_action = torch.full((8,), -1, dtype=torch.int32)
    result.actions = torch.tensor([0, 5, 6, 7, 11], dtype=torch.int32)
    result.force = torch.full((8,), -1, dtype=torch.int32)
    result.histogram = torch.zeros(5, dtype=torch.int64)
    result.row_histogram = torch.zeros(8, 5, dtype=torch.int64)
    result.net = lambda value: torch.zeros(value.shape[0], 5)
    result.threshold = 0.99
    return result


def cuda_mocks(backend):
    stack = contextlib.ExitStack()
    stack.enter_context(patch.object(RUNTIME, "packed_backend", return_value=backend))
    stack.enter_context(patch.object(RUNTIME, "bucket_backend", return_value=backend))
    stack.enter_context(patch.object(RUNTIME, "repair_backend", return_value=backend))
    stack.enter_context(patch.object(RUNTIME, "get_attn_backend", return_value=backend))
    stack.enter_context(
        patch.object(torch.cuda, "current_stream", return_value=Stream())
    )
    stack.enter_context(
        patch.object(
            torch.cuda, "stream", side_effect=lambda _: contextlib.nullcontext()
        )
    )
    stack.enter_context(
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False)
    )
    return stack


class BatchPositionTest(unittest.TestCase):
    def dispatch(self, size, multimodal):
        backing = torch.arange(3 * 16).view(3, 16)
        positions = backing[:, :size] if multimodal else backing[0, :size]
        request_batch = batch(size, positions if multimodal else None)
        request_batch.positions = positions if not multimodal else backing[0, :size]
        instance = runtime()
        instance.force[:size].copy_(
            torch.tensor([0 if row % 2 else 11 for row in range(size)])
        )
        observed = []
        owner = self

        class Decoder:
            def __init__(self, layer):
                self.layer = layer

            def __call__(self, *, positions, hidden_states, residual, forward_batch):
                count = forward_batch.batch_size
                owner.assertEqual(tuple(hidden_states.shape), (count, 4))
                owner.assertEqual(positions.shape[-1], count)
                owner.assertEqual(positions.stride(-1), 1)
                if self.layer >= 43:
                    owner.assertIs(forward_batch, request_batch)
                    owner.assertIs(positions, original_positions)
                else:
                    # Original request ids are an independent oracle for the
                    # matching position and cache-slot row after GPU packing.
                    reqs = forward_batch.req_pool_indices
                    real = reqs >= 10
                    original = (reqs - 10).clamp_min(0)
                    expected = original_positions.index_select(-1, original)
                    shape = [1] * expected.ndim
                    shape[-1] = count
                    expected = torch.where(real.reshape(shape), expected, 0)
                    owner.assertTrue(torch.equal(positions, expected))
                    if multimodal:
                        owner.assertTrue(
                            torch.equal(forward_batch.mrope_positions, positions)
                        )
                    owner.assertTrue(
                        torch.equal(
                            forward_batch.out_cache_loc[real], original[real] + 100
                        )
                    )
                    observed.append(positions.clone())
                return hidden_states + 1, residual + 1

        model = types.SimpleNamespace(
            layers=[Decoder(layer) for layer in range(44)], end_layer=44
        )
        backend = types.SimpleNamespace(
            linear_attn_backend=types.SimpleNamespace(
                forward_metadata=types.SimpleNamespace(
                    mamba_cache_indices=torch.arange(size)
                )
            )
        )
        original_positions = positions
        with cuda_mocks(backend):
            instance.forward_middle(
                model,
                torch.zeros(size, 4),
                torch.zeros(size, 4),
                positions,
                request_batch,
            )
            first = [value.clone() for value in observed]
            observed.clear()
            backing.add_(1000)
            instance.forward_middle(
                model,
                torch.zeros(size, 4),
                torch.zeros(size, 4),
                positions,
                request_batch,
            )
        self.assertEqual(len(first), 33 * len(RUNTIME.capacities(size)))
        self.assertTrue(
            any(
                not torch.equal(before, after) for before, after in zip(first, observed)
            )
        )

    def test_mrope_middle_positions(self):
        for size in (1, 2, 3, 4, 8):
            with self.subTest(size=size):
                self.dispatch(size, multimodal=True)

    def test_rope_middle_positions(self):
        for size in (1, 2, 3, 4, 8):
            with self.subTest(size=size):
                self.dispatch(size, multimodal=False)

    def test_position_validation_precedes_decoder_dispatch(self):
        instance, request_batch = runtime(), batch(8)
        bad = (
            torch.zeros(1, 8, dtype=torch.int64),
            torch.zeros(0, 8, dtype=torch.int64),
            torch.zeros(3, 7, dtype=torch.int64),
            torch.zeros(8, 3, dtype=torch.int64).t(),
        )
        for positions in bad:
            with self.subTest(shape=positions.shape, stride=positions.stride()):
                with self.assertRaisesRegex(ValueError, "position"):
                    instance.validate_decode(request_batch, 8, positions)

    def test_position_metadata_is_validated_before_prefix_cache_writes(self):
        instance, request_batch = runtime(), batch(8)
        valid = torch.arange(8)
        request_batch.mrope_positions = torch.zeros(1, 8, dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "position"):
            instance.validate_decode(request_batch, 8, valid)
        request_batch.mrope_positions = torch.zeros(3, 8, dtype=torch.int64)
        request_batch.positions = torch.arange(16)[::2]
        with self.assertRaisesRegex(ValueError, "position"):
            instance.validate_decode(request_batch, 8, valid)


if __name__ == "__main__":
    unittest.main()
