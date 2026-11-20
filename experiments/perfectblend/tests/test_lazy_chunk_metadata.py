"""Exercise FLA's real chunk call chain with preallocated graph metadata."""

import ast
import types
import unittest
from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import Mock

import torch


ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt/layers/attention/fla"


def function(file, name, namespace):
    node = next(n for n in ast.parse((ROOT / file).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / file), "exec"), namespace)
    return namespace[name]


class LazyChunkMetadataTest(unittest.TestCase):
    def test_recurrent_chunk_uses_preallocated_offsets_and_preserves_fallback(self):
        calls = []
        class Kernel:
            def __getitem__(self, _):
                return lambda **kwargs: calls.append(kwargs)

        fallback = torch.tensor([0, 1], dtype=torch.int64)
        prepare = Mock(return_value=fallback)
        execute = function("chunk_delta_h.py", "chunk_gated_delta_rule_fwd_h", dict(
            torch=torch, Optional=Optional, Tuple=Tuple, CHUNK_SIZE=64,
            triton=types.SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y),
            prepare_chunk_indices=Mock(side_effect=AssertionError("host indices")),
            prepare_chunk_offsets=prepare,
            chunk_gated_delta_rule_fwd_kernel_h_blockdim64=Kernel(),
        ))
        for length in (1, 16, 32, 64):
            qkv = torch.zeros(1, 64, 2, 4)
            lengths = torch.tensor([0, length], dtype=torch.int32)
            indices = torch.zeros(1, 2, dtype=torch.int64)
            offsets = torch.tensor([0, 1], dtype=torch.int64)
            execute(qkv, qkv, qkv, cu_seqlens=lengths,
                    chunk_indices=indices, chunk_offsets=offsets)
            prepare.assert_not_called()
            self.assertIs(calls[-1]["chunk_offsets"], offsets)
            self.assertIs(calls[-1]["cu_seqlens"], lengths)
        execute(qkv, qkv, qkv, cu_seqlens=lengths, chunk_indices=indices)
        prepare.assert_called_once_with(lengths, 64)
        self.assertIs(calls[-1]["chunk_offsets"], fallback)

    def test_chunk_pipeline_forwards_offsets_to_recurrent_kernel(self):
        recurrent = Mock(return_value=(torch.zeros(1), torch.zeros(1)))
        execute = function("chunk.py", "chunk_gated_delta_rule_fwd", dict(
            torch=torch, Optional=Optional, CHUNK_SIZE=64, SUPPRESS_LEVEL=0,
            chunk_local_cumsum=lambda value, **_: value,
            chunk_gated_delta_rule_fwd_intra=Mock(return_value=(None, None, None)),
            chunk_gated_delta_rule_fwd_h=recurrent,
            chunk_fwd_o=Mock(return_value=torch.zeros(1)),
        ))
        value = torch.zeros(1, 16, 2, 4)
        offsets = torch.tensor([0, 1])
        execute(value, value, value, value, value, 0.5, value, value,
                cu_seqlens=torch.tensor([0, 3]), chunk_indices=torch.zeros(1, 2),
                chunk_offsets=offsets)
        self.assertIs(recurrent.call_args.kwargs["chunk_offsets"], offsets)
