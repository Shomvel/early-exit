"""Compare deployed action selection with an independent per-row oracle."""

import unittest

import torch

from experiments.perfectblend.tests.test_batch_positions import RUNTIME, runtime


class ClassifierPolicyTest(unittest.TestCase):
    def test_linear_and_mlp_checkpoint_loading_preserves_logits(self):
        for depth, checkpoint_format, outputs in (
            (0, RUNTIME.FOUR_WIDTH_BCE, 4),
            (1, RUNTIME.FOUR_WIDTH_BCE, 4),
            (1, RUNTIME.FIVE_CLASS, 5),
        ):
            layers = [] if depth == 0 else [
                torch.nn.Linear(8, 3), torch.nn.ReLU(), torch.nn.Dropout(0.1)
            ]
            layers.append(torch.nn.Linear(3 if depth else 8, outputs))
            original = torch.nn.Sequential(*layers).eval()
            saved = dict(
                format=checkpoint_format, depth=depth, width=3 if depth else 0,
                state_dict={"net." + k: v for k, v in original.state_dict().items()},
            )
            restored = RUNTIME.load_classifier(saved, 8)
            inputs = torch.randn(7, 8)
            self.assertTrue(torch.equal(original(inputs), restored(inputs)))
            self.assertTrue(all(p.dtype == torch.float32 and not p.requires_grad for p in restored.parameters()))

    def test_bce_matches_independent_width_decisions(self):
        generator = torch.Generator().manual_seed(27)
        logits = torch.randn(1000, 4, generator=generator) * 4
        threshold = 0.9796755098104477
        scores = logits.sigmoid().tolist()
        expected = [
            max([0] + [w for w, p in zip((5, 6, 7, 11), row) if p >= threshold])
            for row in scores
        ]
        actual = RUNTIME.select_action(
            logits, runtime().actions, threshold, RUNTIME.FOUR_WIDTH_BCE
        )
        self.assertEqual(actual.tolist(), expected)

    def test_nonmonotone_scores_and_threshold_equality(self):
        logits = torch.tensor([[0.0, -1.0, -1.0, -1.0], [-1.0, 1.0, -1.0, 1.0], [-1.0] * 4])
        actual = RUNTIME.select_action(
            logits, runtime().actions, 0.5, RUNTIME.FOUR_WIDTH_BCE
        )
        self.assertEqual(actual.tolist(), [5, 11, 0])

    def test_legacy_argmax_policy_and_valid_row_mask(self):
        instance = runtime()
        logits = torch.tensor([[12.0, 0, 0, 0, 0], [0, 0, 0, 0, 12.0], [0.0] * 5])
        instance.net = lambda _: logits
        instance.select(torch.ones(3, 4), torch.zeros(3, 4), torch.tensor([True, False, True]))
        self.assertEqual(instance.proposed_action[:3].tolist(), [0, -1, 0])
        self.assertEqual(
            RUNTIME.select_action(logits, instance.actions, 0.99, RUNTIME.FIVE_CLASS).tolist(),
            [0, 11, 0],
        )

    def test_bce_actions_reach_split_routing(self):
        instance = runtime()
        instance.checkpoint_format = RUNTIME.FOUR_WIDTH_BCE
        instance.net = lambda _: torch.tensor([[-10.0] * 4, [10.0, -10, -10, -10], [10.0] * 4])
        instance.select(torch.ones(3, 4), torch.zeros(3, 4), torch.ones(3, dtype=torch.bool))
        self.assertEqual(instance.proposed_action[:3].tolist(), [0, 5, 11])
        self.assertEqual(instance.action[:3].tolist(), [0, 5, 5])
