"""CPU contracts for whole-batch smoke verification and benchmark accounting."""

import copy
import importlib.util
import types
import unittest
from pathlib import Path

import torch


def load(name):
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/early_exit"
        / f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SMOKE = load("batch_smoke")
EVALUATE = load("batch_evaluate")
WIDTHS = (0, 5, 6, 7, 11)


def summary(size, steps=16):
    capture = next(value for value in (1, 2, 4, 8) if value >= size)
    schedule = SMOKE.action_schedule(size)
    events = []
    rows, row_exec = [[0] * 5 for _ in range(8)], [[0] * 8 for _ in range(8)]
    groups, group_exec = [[0] * 5 for _ in range(2)], [[0] * 8 for _ in range(2)]
    sizes, buckets = [[0] * 9 for _ in range(2)], [[0] * 9 for _ in range(2)]
    for step in range(steps):
        proposed = schedule[step % len(schedule)] + [-1] * (capture - size)
        expected = SMOKE.split_oracle(proposed)
        events.append(dict(proposed=proposed, **expected))
        for row, action in enumerate(expected["applied"][:size]):
            rows[row][WIDTHS.index(action)] += 1
            for segment, width in enumerate(WIDTHS[1:]):
                row_exec[row][segment if action < width else segment + 4] += 1
        for group, action in enumerate(expected["group_actions"]):
            sizes[group][expected["group_counts"][group]] += 1
            buckets[group][expected["group_capacities"][group]] += 1
            if action < 0:
                continue
            groups[group][WIDTHS.index(action)] += 1
            for segment, width in enumerate(WIDTHS[1:]):
                group_exec[group][segment if action < width else segment + 4] += 1
    return dict(
        captured_batch_size=capture,
        capture_batch_sizes=[1, 2, 4, 8],
        routing="split_skip",
        routing_steps=events,
        converted_regions=[[1] * (12 * (index + 1)) for index in range(4)],
        row_histogram=rows,
        row_executed=row_exec,
        histogram=[sum(values) for values in zip(*rows)],
        executed=[sum(values) for values in zip(*row_exec)],
        batch_executed=[sum(values) for values in zip(*group_exec)],
        group_executed=group_exec,
        group_action_histogram=groups,
        group_size_histogram=sizes,
        bucket_histogram=buckets,
        batched_main_replays=groups[0][0],
        padding_steps=[
            dict(
                req_pool_indices=[0] * (capture - size),
                out_cache_loc=[0] * (capture - size),
                mamba_cache_indices=[-1] * (capture - size),
            )
            for _ in range(steps if capture > size else 0)
        ],
        padding_canary=dict(passed=True, exact=True, max_abs=0),
    )


def metric_snapshot(observation):
    event = observation["routing_steps"][-1]
    return dict(
        **{
            key: observation[key]
            for key in (
                "routing",
                "row_histogram",
                "row_executed",
                "histogram",
                "executed",
                "batch_executed",
                "group_executed",
                "group_action_histogram",
                "group_size_histogram",
                "bucket_histogram",
                "batched_main_replays",
            )
        },
        last_proposed_actions=event["proposed"],
        last_applied_actions=event["applied"],
        last_group_actions=event["group_actions"],
        last_group_counts=event["group_counts"],
        last_group_indices=event["group_indices"],
        last_group_capacities=event["group_capacities"],
    )


class BatchSmokeProtocolTest(unittest.TestCase):
    def test_every_batch_covers_actions_partition_sizes_and_minimal_buckets(self):
        for size in (1, 2, 3, 4, 8):
            with self.subTest(size=size):
                schedule = SMOKE.action_schedule(size)
                self.assertTrue(all(len(values) == size for values in schedule))
                self.assertEqual(
                    {
                        min(value for value in values if value > 0)
                        for values in schedule
                        if any(value > 0 for value in values)
                    },
                    {5, 6, 7, 11},
                )
                self.assertEqual(
                    {values.count(0) for values in schedule}, set(range(size + 1))
                )
                observed = summary(size)
                result = SMOKE.route_checks(observed, size, 16, schedule)
                self.assertTrue(result["passed"], result)
                self.assertEqual(
                    sum(observed["batch_executed"]),
                    4 * sum(map(sum, observed["group_action_histogram"])),
                )
                if size > 1:
                    self.assertGreater(result["split_steps"], 0)
                    self.assertGreater(result["proposed_diversity_steps"], 0)

    def test_old_actions_wrong_partition_bucket_and_repeated_launches_are_rejected(
        self,
    ):
        observed = summary(8)
        cases = []
        for field, changed in (
            ("applied", [11] * 8),
            ("group_actions", [0, 11]),
            ("group_counts", [4, 4]),
            ("group_indices", [[-1] * 8, list(reversed(range(8)))]),
            ("group_capacities", [8, 8]),
        ):
            case = copy.deepcopy(observed)
            case["routing_steps"][1][field] = changed
            cases.append(case)
        for field, changed in (
            ("routing", "batch_min"),
            ("batch_executed", [64] * 8),
            ("converted_regions", [[1] * 8] * 4),
            ("batched_main_replays", 0),
            ("group_size_histogram", [[0] * 9] * 2),
            ("bucket_histogram", [[0] * 9] * 2),
        ):
            case = copy.deepcopy(observed)
            case[field] = changed
            cases.append(case)
        for index, case in enumerate(cases):
            with self.subTest(index=index):
                self.assertFalse(
                    SMOKE.route_checks(case, 8, 16, SMOKE.action_schedule(8))["passed"]
                )

    def test_padding_sentinels_canary_and_counters_are_enforced(self):
        observed = summary(3)
        self.assertTrue(SMOKE.padding_checks(observed, 3, 16)["passed"])
        self.assertTrue(SMOKE.route_checks(observed, 3, 16)["passed"])
        for field, changed in (
            ("req_pool_indices", [4]),
            ("out_cache_loc", [4]),
            ("mamba_cache_indices", [0]),
        ):
            broken = copy.deepcopy(observed)
            broken["padding_steps"][0][field] = changed
            self.assertFalse(SMOKE.padding_checks(broken, 3, 16)["passed"])
        broken = copy.deepcopy(observed)
        broken["padding_canary"] = dict(passed=False, exact=False, max_abs=1)
        self.assertFalse(SMOKE.padding_checks(broken, 3, 16)["passed"])
        for field in ("row_executed", "row_histogram"):
            broken = copy.deepcopy(observed)
            broken[field][3][0] = 1
            self.assertFalse(SMOKE.route_checks(broken, 3, 16)["passed"])
        broken = copy.deepcopy(observed)
        broken["routing_steps"][0]["proposed"][3] = 0
        self.assertFalse(SMOKE.route_checks(broken, 3, 16)["passed"])

    def test_cache_canary_snapshots_own_storage_and_detect_real_state_writes(self):
        keys = torch.arange(8 * 2 * 3).reshape(8, 2, 3)
        values = keys + 100
        caches = {
            layer: types.SimpleNamespace(
                conv=[torch.zeros(4, 2, 3)], temporal=torch.zeros(4, 2, 3)
            )
            for layer in range(32, 43)
        }
        raw = types.SimpleNamespace(
            req_to_token_pool=types.SimpleNamespace(
                req_to_token=torch.tensor([[0, 0], [2, 3], [4, 5]]),
                mamba2_layer_cache=lambda layer: caches[layer],
            ),
            token_to_kv_pool=types.SimpleNamespace(
                get_key_buffer=lambda _: keys, get_value_buffer=lambda _: values
            ),
            attn_backend=types.SimpleNamespace(full_attn_layers={35, 39}),
        )
        canary = types.SimpleNamespace(req_pool_idx=2, mamba_pool_idx=3)
        before = SMOKE.cache_snapshot(raw, canary, 2)
        keys[2].add_(7)
        caches[32].temporal[1].add_(8)
        self.assertTrue(
            SMOKE.comparisons(before, SMOKE.cache_snapshot(raw, canary, 2), exact=True)[
                "passed"
            ]
        )
        caches[32].temporal[3].add_(1)
        self.assertFalse(
            SMOKE.comparisons(before, SMOKE.cache_snapshot(raw, canary, 2), exact=True)[
                "passed"
            ]
        )


class BatchEvaluatorProtocolTest(unittest.TestCase):
    def test_group_metrics_distinguish_real_rows_bucket_padding_and_launches(self):
        for size in (1, 2, 3, 4, 8):
            snapshot = metric_snapshot(summary(size))
            metrics = EVALUATE.batch_routing_metrics(snapshot, size, 16)
            group_steps = sum(map(sum, snapshot["group_action_histogram"]))
            self.assertEqual(sum(snapshot["histogram"]), size * 16)
            self.assertEqual(
                metrics["conditional_middle_region_launches"], 4 * group_steps
            )
            self.assertEqual(metrics["nonempty_group_replays"], group_steps)
            self.assertEqual(
                metrics["batched_no_skip_replays"],
                snapshot["group_action_histogram"][0][0],
            )
            self.assertEqual(metrics["conditional_group_real_row_slots"], size * 16)
            self.assertGreaterEqual(
                metrics["conditional_group_capacity_slots"], size * 16
            )
            self.assertEqual(
                metrics["conditional_group_padding_slots"],
                metrics["conditional_group_capacity_slots"] - size * 16,
            )
            for group in range(2):
                self.assertEqual(sum(metrics["group_size_histogram"][group]), 16)
                self.assertEqual(sum(metrics["group_bucket_histogram"][group]), 16)

    def test_legacy_protocol_and_bad_partition_or_physical_counts_are_rejected(self):
        for field, changed in (
            ("routing", "per_request"),
            ("routing", "batch_min"),
            ("batch_executed", [64] * 8),
            ("group_executed", [[64] * 8] * 2),
            ("group_action_histogram", [[16, 0, 0, 0, 0]] * 2),
            ("group_size_histogram", [[16] + [0] * 8] * 2),
            ("bucket_histogram", [[0] * 8 + [16]] * 2),
            ("batched_main_replays", 0),
            ("last_proposed_actions", [0, 5]),
            ("last_proposed_actions", [0, 5, 6, -1, -1]),
            ("last_proposed_actions", [0, 5, 6, 0]),
            ("last_applied_actions", [0, 5, 6, -1]),
            ("last_group_actions", [0, 11]),
            ("last_group_counts", [1, 1]),
            ("last_group_indices", [[0, -1, -1, -1], [1, 2, -1, -1]]),
            ("last_group_capacities", [1, 4]),
        ):
            snapshot = metric_snapshot(summary(3))
            snapshot[field] = changed
            with self.subTest(field=field, changed=changed), self.assertRaises(
                ValueError
            ):
                EVALUATE.batch_routing_metrics(snapshot, 3, 16)

    def test_completion_waste_fixed_replay_and_choice_scoring_remain_distinct(self):
        account = EVALUATE.DecodeAccounting([4, 3], {9})
        for offset, tokens in enumerate(((9, 1), (4, 2), (5, 3))):
            account.advance(tokens, elapsed=offset + 1)
        self.assertTrue(account.done)
        self.assertEqual(account.generated, [[9], [1, 2, 3]])
        self.assertEqual(account.effective_tokens + account.completed_row_waste, 6)
        replay = EVALUATE.DecodeAccounting([3, 3], {9}, ignore_eos=True)
        for step in range(3):
            replay.advance([9, 9], elapsed=step + 1)
        self.assertEqual(replay.effective_tokens, 6)
        choices = EVALUATE.DecodeAccounting([1, 1], {9}, choice_ids=[10, 11, 12, 13])
        choices.advance([100, 101], elapsed=1, scores=[[0, 3, 1, 2], [4, 0, 1, 2]])
        self.assertEqual(choices.generated, [[11], [10]])
        self.assertEqual(choices.reasons, ["choice", "choice"])

    def test_elapsed_time_is_summed_once_per_batch(self):
        row = dict(
            requests=2,
            prompt_tokens=20,
            effective_generated_tokens=32,
            completed_row_waste_tokens=0,
            computed_real_row_tokens=32,
            graph_padding_decode_slots=0,
            conditional_middle_region_launches=64,
            nonempty_group_replays=16,
            conditional_group_real_row_slots=32,
            conditional_group_capacity_slots=40,
            conditional_group_padding_slots=8,
            group_action_histogram={
                name: dict(zip(map(str, WIDTHS), [4, 3, 3, 3, 3]))
                for name in ("no_skip", "skip")
            },
            group_size_histogram=[[4, 6, 6] + [0] * 6] * 2,
            group_bucket_histogram=[[4, 6, 6] + [0] * 6] * 2,
            batched_no_skip_replays=4,
            prefill_seconds=1,
            prefill_cuda_ms=1000,
            decode_seconds=2,
            decode_cuda_ms=2000,
            total_seconds=3,
            routing_histogram_including_completed_row_waste={
                "0": 8,
                "5": 6,
                "6": 6,
                "7": 6,
                "11": 6,
            },
        )
        result = EVALUATE.timing_summary([row, copy.deepcopy(row)])
        self.assertEqual(result["decode_seconds"], 4)
        self.assertEqual(result["conditional_middle_region_launches"], 128)
        self.assertEqual(result["conditional_group_padding_slots"], 16)
        self.assertEqual(result["group_size_histogram"], [[8, 12, 12] + [0] * 6] * 2)
        self.assertEqual(result["effective_decode_tokens_per_second"], 16)


if __name__ == "__main__":
    unittest.main()
