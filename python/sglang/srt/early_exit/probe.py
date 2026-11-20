# SPDX-License-Identifier: Apache-2.0
"""A GPU-only branch test that checks both values and skipped kernel effects."""

import json
import os
from pathlib import Path

import torch

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
    summary = dict(
        passed=True,
        moved_nodes=moved,
        cases=values,
        microseconds=timing,
        parallel=parallel,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )
    (out / "conditional-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PB_CONDITIONAL_PASS " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
