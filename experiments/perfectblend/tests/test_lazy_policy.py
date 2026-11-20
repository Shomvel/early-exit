"""Deferred-work accounting oracle: re-entry, suffixes, and multi-chunk repair."""

import unittest

from experiments.perfectblend.tests.test_batch_protocol import load

POLICY = load("lazy_smoke")


class LazyPolicyTest(unittest.TestCase):
    def test_equal_or_wider_skips_keep_the_whole_debt(self):
        rows, debt = POLICY.expected([[5], [6], [7], [11], [11]], 16)
        self.assertEqual(rows, [[0, 40, 0, 0, 0, 5, 0, 0]])
        self.assertEqual(debt, 40)

    def test_positive_reentry_flushes_old_debt_then_enqueues_current(self):
        rows, debt = POLICY.expected([[5], [11], [7]], 16)
        self.assertEqual(rows, [[1, 23, 16, 4, 64, 3, 5, 0]])
        self.assertEqual(debt, 7)

    def test_long_holes_require_multiple_causal_chunks(self):
        rows, debt = POLICY.expected([[11]] * 35 + [[0]], 16)
        self.assertEqual(rows, [[1, 385, 385, 12, 192, 35, 140, 0]])
        self.assertEqual(debt, 0)

    def test_common_positive_width_and_independent_requests(self):
        rows, debt = POLICY.expected([[11, 0], [11, 5], [11, 0], [0, 0]], 16)
        self.assertEqual(rows[0][0], 2)
        self.assertEqual(rows[1][0], 1)
        self.assertEqual(rows[0][1:3], [27, 27])
        self.assertEqual(rows[1][1:3], [5, 5])
        self.assertEqual(debt, 0)
