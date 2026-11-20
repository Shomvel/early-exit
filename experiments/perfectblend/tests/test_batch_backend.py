"""Check live whole-batch metadata aliases and stream-private repair scratch."""

import sys
import types
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch

from experiments.perfectblend.tests.test_batch_positions import RUNTIME, batch


@dataclass
class FullMetadata:
    attn_logits: torch.Tensor
    attn_lse: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    num_kv_splits: torch.Tensor


@dataclass
class LinearMetadata:
    mamba_cache_indices: torch.Tensor
    query_start_loc: torch.Tensor
    is_target_verify: bool = False
    has_mamba_track_mask: bool = False


class FullBackend:
    use_mla = False
    use_sliding_window_kv_pool = False
    sliding_window_size = None

    def _fill_kv_indptr_and_indices(self, size, lengths, requests, output):
        self.kv_indptr[0] = 0
        self.kv_indptr[1:] = lengths.cumsum(0)
        count = 0
        for row, length in zip(requests, lengths):
            output[count : count + length] = row * 100 + torch.arange(length)
            count += length

    def get_num_kv_splits(self, output, lengths):
        output.copy_(lengths + lengths.numel())


class LinearBackend:
    pass


class BatchBackendTest(unittest.TestCase):
    def test_packed_metadata_rebuilds_ragged_kv_and_preserves_gdn_ownership(self):
        full, linear = FullBackend(), LinearBackend()
        full.token_to_kv_pool = object()
        full.forward_metadata = FullMetadata(
            torch.zeros(4, 2, 8, 16),
            torch.zeros(4, 2, 8),
            torch.zeros(5, dtype=torch.int32),
            torch.empty(200, dtype=torch.int32),
            torch.ones(4, dtype=torch.int32),
        )
        linear.forward_metadata = LinearMetadata(
            torch.tensor([20, 21, 22, -1]), torch.arange(5)
        )
        original = types.SimpleNamespace(
            full_attn_backend=full, linear_attn_backend=linear
        )
        inputs = batch(4)
        indices = torch.tensor([0, 2, -1, -1])
        scratch = {}
        modules = {
            "sglang.srt.early_exit.masked_kv": types.SimpleNamespace(
                MaskedKVPool=lambda pool: pool
            )
        }
        with patch.dict(sys.modules, modules):
            packed = RUNTIME.packed_batch(inputs, indices)
            grouped = RUNTIME.packed_backend(original, packed, indices, scratch)
            metadata = grouped.full_attn_backend.forward_metadata
            self.assertEqual(metadata.kv_indptr.tolist(), [0, 5, 12, 13, 14])
            self.assertEqual(
                metadata.kv_indices[:12].tolist(),
                list(range(1000, 1005)) + list(range(1200, 1207)),
            )
            self.assertEqual(
                grouped.linear_attn_backend.forward_metadata.mamba_cache_indices.tolist(),
                [20, 22, -1, -1],
            )
            self.assertEqual(
                grouped.linear_attn_backend.forward_metadata.query_start_loc.tolist(),
                [0, 1, 2, 3, 4],
            )
            bucket = RUNTIME.bucket_batch(packed, 2)
            smaller = RUNTIME.bucket_backend(grouped, bucket, {})
            self.assertEqual(
                smaller.full_attn_backend.forward_metadata.kv_indptr.tolist(),
                [0, 5, 12],
            )
            self.assertEqual(
                smaller.full_attn_backend.forward_metadata.num_kv_splits.tolist(),
                [7, 9],
            )
            self.assertEqual(
                smaller.linear_attn_backend.forward_metadata.mamba_cache_indices.tolist(),
                [20, 22],
            )
            self.assertIsNot(metadata.attn_logits, full.forward_metadata.attn_logits)
            self.assertEqual(full.forward_metadata.kv_indptr.tolist(), [0] * 5)
            inputs.seq_lens.add_(2)
            repeated = RUNTIME.packed_backend(
                original, RUNTIME.packed_batch(inputs, indices), indices, scratch
            )
            self.assertIs(
                repeated.full_attn_backend.forward_metadata.kv_indices,
                metadata.kv_indices,
            )
            self.assertEqual(metadata.kv_indptr.tolist(), [0, 7, 16, 17, 18])

    def test_private_scratch_retains_full_batch_and_live_metadata(self):
        modules = {
            "sglang.srt.layers.attention.triton_backend": types.SimpleNamespace(
                TritonAttnBackend=FullBackend
            ),
            "sglang.srt.layers.attention.linear.gdn_backend": types.SimpleNamespace(
                GDNAttnBackend=LinearBackend
            ),
        }
        scratch = {}
        for size in (1, 2, 4, 8):
            with self.subTest(size=size), patch.dict(sys.modules, modules):
                full, linear = FullBackend(), LinearBackend()
                full.forward_metadata = FullMetadata(
                    torch.zeros(size, 4, 5, 8),
                    torch.zeros(size, 4, 5),
                    torch.arange(size + 1) * 7,
                    torch.arange(7 * size),
                    torch.ones(size),
                )
                linear.forward_metadata = LinearMetadata(
                    torch.arange(size), torch.arange(size + 1)
                )
                original = types.SimpleNamespace(
                    full_attn_backend=full,
                    linear_attn_backend=linear,
                    attn_backend_list=[full, linear],
                )
                repaired = RUNTIME.repair_backend(original, scratch)
                self.assertIsNot(repaired.full_attn_backend, full)
                self.assertIsNot(repaired.linear_attn_backend, linear)
                self.assertEqual(
                    repaired.attn_backend_list,
                    [repaired.full_attn_backend, repaired.linear_attn_backend],
                )
                before, after = (
                    full.forward_metadata,
                    repaired.full_attn_backend.forward_metadata,
                )
                for field in ("attn_logits", "attn_lse"):
                    self.assertEqual(
                        getattr(before, field).shape, getattr(after, field).shape
                    )
                    self.assertNotEqual(
                        getattr(before, field).data_ptr(),
                        getattr(after, field).data_ptr(),
                    )
                for field in ("kv_indptr", "kv_indices", "num_kv_splits"):
                    self.assertIs(getattr(before, field), getattr(after, field))
                other = repaired.linear_attn_backend.forward_metadata
                for field in ("mamba_cache_indices", "query_start_loc"):
                    self.assertIs(
                        getattr(linear.forward_metadata, field), getattr(other, field)
                    )
                linear.forward_metadata.mamba_cache_indices[-1] = -1
                self.assertEqual(int(other.mamba_cache_indices[-1]), -1)
                repeated = RUNTIME.repair_backend(original, scratch)
                self.assertIs(
                    repeated.full_attn_backend.forward_metadata.attn_logits,
                    after.attn_logits,
                )


if __name__ == "__main__":
    unittest.main()
