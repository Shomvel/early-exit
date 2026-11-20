# SPDX-License-Identifier: Apache-2.0
"""A GPU-only branch test that checks both values and skipped kernel effects."""

import json
import os
from pathlib import Path

import torch

from sglang.srt.early_exit.batch import prepare_execution_actions
from sglang.srt.early_exit.conditional import ConditionalRegions


def parallel_probe(regions):
    action = torch.zeros(1, dtype=torch.int32, device="cuda")
    states = [torch.ones(1024, device="cuda") for _ in range(2)]
    executed = torch.zeros(8, dtype=torch.int32, device="cuda")
    pdl_executed = torch.zeros_like(executed)
    repair = torch.cuda.Stream()

    def segments(offset):
        for i in range(4):
            region = offset * 4 + i
            regions.marker(2 * region)
            states[offset].add_(i + 1)
            executed[region].add_(1)
            regions.pdl_increment(pdl_executed[region : region + 1])
            regions.pdl_increment(pdl_executed[region : region + 1])
            regions.marker(2 * region + 1)

    def forward():
        current = torch.cuda.current_stream()
        repair.wait_stream(current)
        with torch.cuda.stream(repair):
            segments(1)
        segments(0)
        current.wait_stream(repair)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        forward()
    moved = regions.convert(graph, action, [5, 6, 7, 11, -5, -6, -7, -11])
    graph.instantiate()
    results = []
    for width in (0, 11, 5, 6, 7, 11, 0):
        for state in states:
            state.fill_(1)
        executed.zero_()
        pdl_executed.zero_()
        action.fill_(width)
        graph.replay()
        torch.cuda.synchronize()
        expected = [int(width < k) for k in (5, 6, 7, 11)] + [
            int(width >= k) for k in (5, 6, 7, 11)
        ]
        assert executed.cpu().tolist() == expected
        assert pdl_executed.cpu().tolist() == [2 * x for x in expected]
        for stream_id, state in enumerate(states):
            value = 1 + sum(
                (i + 1) * e
                for i, e in enumerate(expected[stream_id * 4 : stream_id * 4 + 4])
            )
            assert torch.equal(state, torch.full_like(state, value))
        results.append(
            dict(action=width, executed=expected, values=[float(s[0]) for s in states])
        )
    return dict(passed=True, pdl_verified=True, moved_nodes=moved, cases=results)


def batch_probe(regions):
    """Verify independent rows and an exclusive, padding-free batched body."""
    batch_size = 8
    widths = (5, 6, 7, 11)
    action = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    execution_actions = torch.full(
        (batch_size + 1,), -1, dtype=torch.int32, device="cuda"
    )
    states = [torch.ones(batch_size, 1024, device="cuda") for _ in range(2)]
    executed = torch.zeros((batch_size, 8), dtype=torch.int32, device="cuda")
    pdl_executed = torch.zeros_like(executed)
    row_launches = torch.zeros(2, dtype=torch.int32, device="cuda")
    batched_launches = torch.zeros(1, dtype=torch.int32, device="cuda")
    repair = torch.cuda.Stream()

    def segments(stream_id):
        for row in range(batch_size):
            for group in range(4):
                region = stream_id * 4 + group
                marker = row * 8 + region
                regions.marker(2 * marker)
                states[stream_id][row].add_(group + 1)
                executed[row, region].add_(1)
                row_launches[stream_id].add_(1)
                counter = pdl_executed[row, region : region + 1]
                regions.pdl_increment(counter)
                regions.pdl_increment(counter)
                regions.marker(2 * marker + 1)

    def forward():
        prepare_execution_actions(action, execution_actions)
        current = torch.cuda.current_stream()
        repair.wait_stream(current)
        with torch.cuda.stream(repair):
            segments(1)
        segments(0)
        marker = batch_size * 8
        regions.marker(2 * marker)
        states[0].add_(sum(range(1, 5)))
        executed[:, :4].add_(1)
        batched_launches.add_(1)
        for row in range(batch_size):
            for group in range(4):
                counter = pdl_executed[row, group : group + 1]
                regions.pdl_increment(counter)
                regions.pdl_increment(counter)
        regions.marker(2 * marker + 1)
        current.wait_stream(repair)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        forward()
    moved = regions.convert(
        graph,
        execution_actions,
        [*widths, *(-width for width in widths)] * batch_size + [1],
        action_indices=[row for row in range(batch_size) for _ in range(8)]
        + [batch_size],
    )
    graph.instantiate()
    results = []
    for values in (
        (0, 11, 5, -1, -1, 6, 0, 7),
        (11, 5, -1, 0, 6, -1, 7, 11),
        (7, 0, 6, 5, 0, 11, 5, 6),
        (0,) * 8,
        (0,) * 7 + (-1,),
        (-1,) * 8,
        (0,) * 8,
    ):
        for state in states:
            state.fill_(1)
        executed.zero_()
        pdl_executed.zero_()
        row_launches.zero_()
        batched_launches.zero_()
        action.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        expected = [
            [int(value >= 0 and value < width) for width in widths]
            + [int(value >= width) for width in widths]
            for value in values
        ]
        assert executed.cpu().tolist() == expected
        assert pdl_executed.cpu().tolist() == [
            [2 * count for count in row] for row in expected
        ]
        batched = all(value == 0 for value in values)
        assert int(batched_launches[0]) == int(batched)
        expected_row_launches = [
            0 if batched else sum(sum(row[start : start + 4]) for row in expected)
            for start in (0, 4)
        ]
        assert row_launches.cpu().tolist() == expected_row_launches
        state_values = []
        for stream_id, state in enumerate(states):
            expected_values = [
                1 + sum((group + 1) * row[stream_id * 4 + group] for group in range(4))
                for row in expected
            ]
            expected_tensor = torch.tensor(
                expected_values, device="cuda", dtype=state.dtype
            )[:, None].expand_as(state)
            assert torch.equal(state, expected_tensor)
            state_values.append(expected_values)
        results.append(
            dict(
                actions=values,
                executed=expected,
                values=state_values,
                batched_main=int(batched),
                row_launches=expected_row_launches,
            )
        )
    return dict(
        passed=True,
        pdl_verified=True,
        padded_rows_verified=True,
        exclusive_batched_no_skip_verified=True,
        moved_nodes=moved,
        cases=results,
    )


def main():
    out = Path(os.environ["RUN_DIR"])
    torch.cuda.set_device(0)
    regions = ConditionalRegions()
    action = torch.zeros(1, dtype=torch.int32, device="cuda")
    state = torch.ones(1024, device="cuda")
    executed = torch.zeros(4, dtype=torch.int32, device="cuda")

    # Each conditional region mutates the same persistent state. A skipped
    # region must not execute its increment or counter write.
    def forward():
        for i in range(4):
            regions.marker(2 * i)
            state.add_(i + 1)
            executed[i].add_(1)
            regions.marker(2 * i + 1)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        forward()
    moved = regions.convert(graph, action, [5, 6, 7, 11])
    graph.instantiate()
    values = []
    for width in (0, 5, 6, 7, 11, 0, 11, 5):
        state.fill_(1)
        executed.zero_()
        action.fill_(width)
        graph.replay()
        torch.cuda.synchronize()
        expected = [int(width < k) for k in (5, 6, 7, 11)]
        assert executed.cpu().tolist() == expected
        assert torch.equal(
            state,
            torch.full_like(
                state, 1 + sum((i + 1) * e for i, e in enumerate(expected))
            ),
        )
        values.append(dict(action=width, executed=expected, value=float(state[0])))
    # No replay-time .item()/D2H controls the branches. Readback above checks
    # the completed outputs only; timed replay below has no host predicate.
    timing = {}
    for width in (0, 11):
        action.fill_(width)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        for _ in range(2000):
            graph.replay()
        end.record()
        end.synchronize()
        timing[str(width)] = start.elapsed_time(end) * 1000 / 2000
    parallel = parallel_probe(regions)
    batch = batch_probe(regions)
    summary = dict(
        passed=True,
        moved_nodes=moved,
        cases=values,
        microseconds=timing,
        parallel=parallel,
        batch=batch,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )
    (out / "conditional-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PB_CONDITIONAL_PASS " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
