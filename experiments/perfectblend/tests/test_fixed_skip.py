"""Fixed decode never enters removed layers, even across repeated tokens."""

import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from experiments.perfectblend import fixed_skip_run
from experiments.perfectblend.tests.test_batch_positions import batch

package = types.ModuleType('sglang')
package.__path__ = [str(Path(__file__).resolve().parents[3] / 'python/sglang')]
with patch.dict(sys.modules, {'sglang': package}):
    FIXED = importlib.import_module('sglang.srt.early_exit.fixed')


class FixedSkipTest(unittest.TestCase):
    def test_jsonl_preserves_unicode_separators_in_generated_answers(self):
        answers = [dict(text='answer\u2028math\u0085next\u2029line', correct=True),
                   dict(text='second answer', correct=False)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'quality.jsonl'
            path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in answers))
            self.assertEqual(fixed_skip_run.read_rows(path), answers)

    def test_decode_call_order_padding_and_repeated_tokens(self):
        for width in (0, 5, 7, 11):
            runtime = FIXED.FixedSkipRuntime.__new__(FIXED.FixedSkipRuntime)
            runtime._width = width
            runtime.building_graph = True
            runtime.histogram = torch.zeros(5, dtype=torch.int64)
            runtime.row_histogram = torch.zeros((8, 5), dtype=torch.int64)
            runtime.layer_tokens = torch.zeros(64, dtype=torch.int64)
            calls = []
            caches = [[] for _ in range(64)]

            def decoder(index):
                def execute(*, hidden_states, residual, **kw):
                    calls.append(index)
                    caches[index].append(True)
                    return hidden_states + index, residual + 1
                return execute

            model = types.SimpleNamespace(layers=[decoder(i) for i in range(64)], end_layer=64)
            backend = types.SimpleNamespace(linear_attn_backend=types.SimpleNamespace(
                forward_metadata=types.SimpleNamespace(mamba_cache_indices=torch.tensor([1, 2, 3, -1]))))
            request = batch(4)
            runtime.reset_metrics()
            with patch.object(FIXED, 'get_attn_backend', return_value=backend):
                for _ in range(3):
                    hidden, residual = runtime.forward_middle(model, torch.zeros(4, 2), torch.zeros(4, 2), request.positions, request)
                    self.assertTrue((hidden == sum(range(32 + width, 64))).all())
                    self.assertTrue((residual == 32 - width).all())
            self.assertEqual(calls, list(range(32 + width, 64)) * 3)
            self.assertTrue(all(not cache for cache in caches[:32 + width]))
            self.assertEqual(runtime.row_histogram.sum(-1).tolist(), [3, 3, 3, 0, 0, 0, 0, 0])
            self.assertEqual(runtime.metrics(3, 3)['repaired_layer_tokens'], 0)
            runtime.layer_tokens[32 + width] -= 1
            with self.assertRaisesRegex(ValueError, 'omitted a suffix layer'):
                runtime.metrics(3, 3)

    def test_fixed_width_and_model_contracts(self):
        config = types.SimpleNamespace(num_hidden_layers=64, hidden_size=5120)
        with self.assertRaisesRegex(ValueError, '0, 5, 7 or 11'):
            FIXED.FixedSkipRuntime(6, config)
        config.num_hidden_layers = 63
        with self.assertRaisesRegex(ValueError, '64-layer'):
            FIXED.FixedSkipRuntime(5, config)


if __name__ == '__main__':
    unittest.main()
