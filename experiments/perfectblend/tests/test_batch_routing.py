"""Exercise the actual GPU branch construction with CPU tensor decisions.

These checks inspect the captured shapes and IF plan, not CUDA numerical parity.
"""

import itertools
import types
import unittest

import torch

from experiments.perfectblend.tests.test_batch_positions import (
    RUNTIME,
    batch,
    cuda_mocks,
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

    def convert(self, graph, action, widths, action_indices=None):
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


def dispatch(actions):
    size, plan = len(actions), Plan()
    positions = torch.arange(3 * size).view(3, size)
    request_batch = batch(size, positions)
    backend = types.SimpleNamespace(
        linear_attn_backend=types.SimpleNamespace(
            forward_metadata=types.SimpleNamespace(
                mamba_cache_indices=torch.tensor(
                    [row if value >= 0 else -1 for row, value in enumerate(actions)]
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
    instance.regions, instance.converted = plan, []
    instance.force[:size].copy_(torch.tensor(actions, dtype=torch.int32))
    with cuda_mocks(backend):
        instance.forward_middle(
            model, torch.ones(size, 4), torch.ones(size, 4), positions, request_batch
        )
    instance.convert(None)
    return instance, plan


class BatchRoutingTest(unittest.TestCase):
    def test_empty_groups_disable_every_body_and_preserve_padding(self):
        for size in (1, 2, 3, 4, 8):
            instance, plan = dispatch([-1] * size)
            self.assertEqual(plan.selected, [])
            self.assertEqual(instance.group_counts.tolist(), [0, 0])
            self.assertEqual(instance.group_capacities.tolist(), [0, 0])
            self.assertEqual(instance.action[:size].tolist(), [-1] * size)

    def test_two_groups_use_a_single_capacity_and_complementary_segments(self):
        for actions in (
            [0] * 8,
            [11] * 8,
            [0, 5, 0, 6, 7, 11, -1, -1],
            [0, 0, 0, 0, 11, 11, 11, 11],
            [6, 7, 11],
            [0, 11],
        ):
            instance, plan = dispatch(actions)
            size = len(actions)
            bucket_sizes = RUNTIME.capacities(size)
            width = min((value for value in actions if value > 0), default=-1)
            counts = [actions.count(0), sum(value > 0 for value in actions)]
            expected = []
            for pass_index, (group, repair) in enumerate(
                ((0, False), (1, False), (1, True))
            ):
                count = counts[group]
                if not count:
                    continue
                cap = min(value for value in bucket_sizes if value >= count)
                bucket = bucket_sizes.index(cap)
                chosen = width if group else 0
                for segment, end in enumerate((5, 6, 7, 11)):
                    if chosen >= end if repair else chosen < end:
                        marker = (pass_index * len(bucket_sizes) + bucket) * 4 + segment
                        expected.append(marker)
                        self.assertTrue(
                            all(shape == cap for shape in plan.layers[marker])
                        )
            self.assertEqual(plan.selected, sorted(expected))
            self.assertEqual(len(plan.selected), 4 * sum(count > 0 for count in counts))
            self.assertEqual(instance.proposed_action[:size].tolist(), actions)
            self.assertEqual(
                instance.action[:size].tolist(),
                [width if value > 0 else value for value in actions],
            )
            self.assertEqual(instance.group_counts.tolist(), counts)

    def test_partition_and_bucket_selection_cover_all_eight_row_patterns(self):
        for size in (1, 2, 3, 4, 8):
            for values in itertools.product((-1, 0, 11), repeat=size):
                proposed = torch.tensor(values, dtype=torch.int32)
                applied = torch.empty_like(proposed)
                actions = torch.empty(2, dtype=torch.int32)
                counts, buckets = torch.empty(2, dtype=torch.int64), torch.empty(
                    2, dtype=torch.int64
                )
                indices = torch.empty(2, size, dtype=torch.int64)
                RUNTIME.partition(proposed, applied, actions, counts, indices, buckets)
                for group in range(2):
                    expected = [
                        row
                        for row, value in enumerate(values)
                        if (value == 0 if group == 0 else value > 0)
                    ]
                    self.assertEqual(
                        indices[group].tolist(),
                        expected + [-1] * (size - len(expected)),
                    )
                    self.assertEqual(int(counts[group]), len(expected))
                    capacity = (
                        min(
                            (
                                cap
                                for cap in RUNTIME.capacities(size)
                                if cap >= len(expected)
                            ),
                            default=0,
                        )
                        if expected
                        else 0
                    )
                    self.assertEqual(int(buckets[group]), capacity)
                self.assertEqual(applied.tolist(), list(values))

    def test_mixed_positive_actions_share_shortest_positive_jump(self):
        instance, plan = dispatch([0, 5, 6, 7, 11, 0, 6, 11])
        self.assertEqual(instance.action.tolist(), [0, 5, 5, 5, 5, 0, 5, 5])
        self.assertEqual(instance.group_actions.tolist(), [0, 5])
        self.assertEqual(instance.group_capacities.tolist(), [2, 8])
        self.assertEqual(len(plan.layers), 48)
        self.assertEqual(len(plan.selected), 8)

    def test_packing_and_merging_restore_original_rows_including_padding(self):
        proposed = torch.tensor([11, 0, -1, 6, 0, 7, -1, 0], dtype=torch.int32)
        original = torch.arange(8 * 4).view(8, 4)
        instance, _ = dispatch(proposed.tolist())
        merged = original.clone()
        for group in range(2):
            packed = RUNTIME.gather_rows(original, instance.group_indices[group])
            members = proposed == 0 if group == 0 else proposed > 0
            merged = RUNTIME.merge_rows(packed + 100 * (group + 1), merged, members)
        expected = (
            original
            + torch.where(proposed == 0, 100, torch.where(proposed > 0, 200, 0))[
                :, None
            ]
        )
        self.assertTrue(torch.equal(merged, expected))


if __name__ == "__main__":
    unittest.main()
