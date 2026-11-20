"""CPU regression at the per-row runtime dispatch, including MRoPE strides.

CUDA scheduling and decoder arithmetic are replaced for this test. The real
forward_middle, _segments and row_batch paths still dispatch every main/repair
layer, so slicing the wrong axis fails at the same call site as the GPU kernel.
"""

import contextlib
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def load_runtime():
    # Import only the runtime and its pure-Python dependencies. Importing the
    # public serving frontend otherwise requires CUDA-unrelated optional deps.
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
        req_pool_indices=torch.arange(size),
        seq_lens=torch.arange(size) + 5,
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
    result.repair_stream = Stream()
    result.regions = types.SimpleNamespace(marker=lambda _: None)
    result.executed = torch.zeros(8, dtype=torch.int64)
    result.row_executed = torch.zeros(8, 8, dtype=torch.int64)
    result.batched_main_executed = torch.zeros((), dtype=torch.int64)
    result.select = lambda *args: None
    return result


class BatchPositionTest(unittest.TestCase):
    def dispatch(self, size, multimodal):
        # Graph slots can have a larger row stride than the captured batch.
        # Give each MRoPE axis different values, including in text-only tests.
        backing = torch.arange(3 * 16).view(3, 16)
        positions = backing[:, :size] if multimodal else backing[0, :size]
        request_batch = batch(size, positions if multimodal else None)
        observed = []
        owner = self

        class Decoder:
            def __init__(self, layer):
                self.layer = layer

            def __call__(self, *, positions, hidden_states, residual, forward_batch):
                if self.layer < 43 and forward_batch is not request_batch:
                    row = int(forward_batch.req_pool_indices[0])
                    expected = (
                        backing[:, row : row + 1]
                        if multimodal
                        else backing[0, row : row + 1]
                    )
                    owner.assertEqual(tuple(positions.shape), tuple(expected.shape))
                    owner.assertTrue(torch.equal(positions, expected))
                    owner.assertEqual(positions.stride(-1), 1)
                    if multimodal:
                        owner.assertEqual(positions.stride(0), backing.stride(0))
                        # This is the exact t/h/w addressing used by the real
                        # Triton kernel: base + axis * positions.stride(0).
                        loaded = torch.as_strided(
                            positions, (3, 1), (positions.stride(0), 1)
                        )
                        owner.assertTrue(torch.equal(loaded, expected))
                        owner.assertTrue(
                            torch.equal(forward_batch.mrope_positions, expected)
                        )
                        owner.assertEqual(
                            forward_batch.mrope_positions.data_ptr(),
                            positions.data_ptr(),
                        )
                    owner.assertEqual(tuple(forward_batch.positions.shape), (1,))
                    observed.append(positions)
                else:
                    owner.assertEqual(positions.shape[-1], size)
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
        with (
            patch.object(RUNTIME, "row_backend", return_value=backend),
            patch.object(RUNTIME, "get_attn_backend", return_value=backend),
            patch.object(torch.cuda, "current_stream", return_value=Stream()),
            patch.object(
                torch.cuda, "stream", side_effect=lambda _: contextlib.nullcontext()
            ),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        ):
            instance = runtime()
            instance.forward_middle(
                model,
                torch.zeros(size, 4),
                torch.zeros(size, 4),
                positions,
                request_batch,
            )
        self.assertEqual(len(observed), size * 11 * 2)
        # Capture must retain views of graph inputs, not a stale copied tensor.
        before = observed[-1].clone()
        backing.add_(1000)
        self.assertTrue(torch.equal(observed[-1], before + 1000))

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
