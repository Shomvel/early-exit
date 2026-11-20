"""Exercise the real replay adapter with the GDN projection's strided inputs."""

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


source = Path(__file__).resolve().parents[3] / "python/sglang/srt/early_exit/lazy.py"
node = next(
    item for item in ast.parse(source.read_text()).body
    if isinstance(item, ast.ClassDef) and item.name == "GraphPrefillKernel"
)
namespace = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)


class LazyKernelLayoutTest(unittest.TestCase):
    def test_strided_qkv_and_gates_obey_low_level_chunk_contract(self):
        def normalize(value):
            shape = value.shape
            flat = value.view(-1, shape[-1])
            self.assertTrue(flat.is_contiguous())
            return (flat / (flat.square().sum(-1, keepdim=True) + 1e-6).sqrt()).view(shape)

        def chunk(**kwargs):
            for name in ("q", "k", "v", "g", "beta"):
                self.assertTrue(kwargs[name].is_contiguous(), name)
            self.assertTrue(torch.equal(kwargs["v"], inputs[2]))
            self.assertTrue(torch.equal(kwargs["g"], inputs[3]))
            self.assertTrue(torch.equal(kwargs["beta"], inputs[4]))
            self.assertIs(kwargs["initial_state"], state)
            self.assertIs(kwargs["cu_seqlens"], lengths)
            return None, kwargs["v"], None, None, state, None

        modules = {
            "sglang.srt.layers.attention.fla.chunk": types.SimpleNamespace(
                chunk_gated_delta_rule_fwd=chunk
            ),
            "sglang.srt.layers.attention.fla.l2norm": types.SimpleNamespace(
                l2norm_fwd=normalize
            ),
        }
        for capacity in (16, 1, 32, 64):
            # causal_conv1d returns [channels, tokens]; GDN transposes it,
            # splits Q/K/V along channels, then views each as [1,T,H,D].
            packed = torch.randn(36, capacity).transpose(0, 1)
            inputs = [part.view(1, capacity, 3, 4) for part in packed.split(12, -1)]
            inputs += [torch.randn(1, capacity, 6)[:, :, ::2] for _ in range(2)]
            state = torch.randn(2, 3, 4, 4)
            lengths = torch.tensor([0, capacity], dtype=torch.int32)
            with patch.dict(sys.modules, modules):
                output, _, updated = namespace["GraphPrefillKernel"](None).extend(
                    *inputs,
                    ssm_states=state,
                    cache_indices=torch.tensor([1]),
                    query_start_loc=lengths,
                )
            self.assertTrue(torch.equal(output, inputs[2]))
            self.assertIs(updated, state)
