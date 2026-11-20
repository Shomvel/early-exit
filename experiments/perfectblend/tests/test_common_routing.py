"""Whole-batch zero veto, common width, padding and execution accounting."""

import importlib
import types
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.perfectblend.tests.test_batch_positions import runtime
from experiments.perfectblend.tests.test_batch_protocol import load

package = types.ModuleType("sglang")
package.__path__ = [str(Path(__file__).resolve().parents[3] / "python/sglang")]
with patch.dict(sys.modules, {"sglang": package}):
    COMMON = importlib.import_module("sglang.srt.early_exit.common")
PROTOCOL = load("batch_evaluate")


class CommonRoutingTest(unittest.TestCase):
    def test_any_zero_vetoes_all_rows_and_padding_is_ignored(self):
        generator = torch.Generator().manual_seed(27)
        for size in range(1, 9):
            for _ in range(40):
                proposals = torch.tensor([0, 5, 6, 7, 11])[torch.randint(5, (size,), generator=generator)]
                for padded in (proposals, torch.cat((proposals, torch.tensor([-1, -1])))):
                    self.assertEqual(int(COMMON.common_action(padded)), min(proposals.tolist()))
        self.assertEqual(int(COMMON.common_action(torch.tensor([-1, -1]))), -1)
        self.assertEqual(int(COMMON.common_action(torch.tensor([11, 0, 11]))), 0)
        self.assertEqual(int(COMMON.common_action(torch.tensor([11, 7, -1]))), 7)

    def test_actual_selector_applies_common_action_and_retains_proposals(self):
        instance = runtime()
        instance.__class__ = COMMON.BatchWideRuntime
        instance.common_action = torch.zeros(1, dtype=torch.int32)
        instance.proposal_histogram = torch.zeros(5, dtype=torch.int64)
        instance.common_step_histogram = torch.zeros(5, dtype=torch.int64)
        for proposals, expected in (([0, 11, 11, -1], [0, 0, 0, -1]), ([11, 7, 6, -1], [6, 6, 6, -1])):
            instance.force[:4].copy_(torch.tensor(proposals))
            instance.select(torch.ones(4, 4), torch.zeros(4, 4), torch.tensor([True, True, True, False]))
            self.assertEqual(instance.proposed_action[:4].tolist(), proposals)
            self.assertEqual(instance.action[:4].tolist(), expected)

    def test_common_execution_metrics_reject_zero_veto_violation(self):
        common = [1, 1, 0, 0, 0]
        execution = [1, 2, 2, 2, 1, 0, 0, 0]
        snapshot = dict(
            routing="batch_min", histogram=[2, 2, 0, 0, 0],
            common_step_histogram=common, proposal_histogram=[1, 1, 0, 0, 2],
            row_histogram=[common] * 2 + [[0] * 5] * 6,
            batch_executed=execution, executed=[n * 2 for n in execution],
            row_executed=[execution] * 2 + [[0] * 8] * 6,
            last_proposed_actions=[0, 11], last_applied_actions=[0, 0],
        )
        self.assertEqual(PROTOCOL.batch_routing_metrics(snapshot, 2, 2)["routing"], "batch_min")
        snapshot["last_applied_actions"] = [0, 11]
        with self.assertRaisesRegex(ValueError, "minimum valid proposal"):
            PROTOCOL.batch_routing_metrics(snapshot, 2, 2)
