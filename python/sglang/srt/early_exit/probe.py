# SPDX-License-Identifier: Apache-2.0
"""A GPU-only branch test that checks both values and skipped kernel effects."""

import json
import os
from pathlib import Path

import torch

from sglang.srt.early_exit.batch_smoke import action_schedule, split_oracle
from sglang.srt.early_exit.conditional import ConditionalRegions
from sglang.srt.early_exit.split import (
    branch_actions,
    capacities,
    gather_rows,
    merge_rows,
    partition,
)


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


def masked_store_probe():
    """Replay the actual Triton store into native BF16 NHD KV allocations."""
    from types import SimpleNamespace

    from sglang.srt.early_exit.masked_kv import MaskedKVPool
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    pool = MHATokenToKVPool(
        size=16,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=96,
        v_head_dim=128,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    # The runtime wraps HybridLinearKVPool; exercise its full_kv_pool contract
    # while using the native allocator and actual masked Triton implementation.
    wrapped = MaskedKVPool(
        SimpleNamespace(full_kv_pool=pool, get_kv_buffer=pool.get_kv_buffer)
    )
    layer = SimpleNamespace(layer_id=0)
    key, value = pool.get_kv_buffer(0)
    locs = [torch.zeros(4, dtype=torch.int64, device="cuda") for _ in range(2)]
    keys = [
        torch.full((4, 2, 96), 3 + group, dtype=torch.bfloat16, device="cuda")
        for group in range(2)
    ]
    values = [
        torch.full((4, 2, 128), 7 + group, dtype=torch.bfloat16, device="cuda")
        for group in range(2)
    ]
    repair, capture = torch.cuda.Stream(), torch.cuda.Stream()

    def forward():
        current = torch.cuda.current_stream()
        repair.wait_stream(current)
        with torch.cuda.stream(repair):
            wrapped.set_kv_buffer(layer, locs[1], keys[1], values[1])
        wrapped.set_kv_buffer(layer, locs[0], keys[0], values[0])
        current.wait_stream(repair)

    with torch.cuda.stream(capture):
        forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture):
        forward()
    results = []
    for locations in (
        ((0, 1, 0, 4), (2, 0, 7, 0)),
        ((0, 0, 0, 0), (0, 0, 0, 0)),
        ((3, 5, 8, 9), (10, 12, 0, 14)),
        ((0, 1, 0, 4), (2, 0, 7, 0)),
    ):
        key.fill_(-13)
        value.fill_(-17)
        for group, rows in enumerate(locations):
            locs[group].copy_(torch.tensor(rows, dtype=torch.int64, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        expected_key, expected_value = torch.full_like(key, -13), torch.full_like(
            value, -17
        )
        for group, rows in enumerate(locations):
            for row, slot in enumerate(rows):
                if slot > 0:
                    expected_key[slot].copy_(keys[group][row])
                    expected_value[slot].copy_(values[group][row])
        assert torch.equal(key, expected_key)
        assert torch.equal(value, expected_value)
        assert torch.equal(key[0], torch.full_like(key[0], -13))
        assert torch.equal(value[0], torch.full_like(value[0], -17))
        results.append(
            dict(
                locations=locations,
                real_slots=[slot for rows in locations for slot in rows if slot > 0],
                padding_slot_unchanged=True,
                exact=True,
            )
        )
    return dict(
        passed=True,
        native_pool="MHATokenToKVPool",
        dtype="bfloat16",
        layout="nhd",
        key_head_dim=96,
        value_head_dim=128,
        concurrent_streams=2,
        cases=results,
    )


def batch_probe(regions):
    """Actually partition, pack, bucket, execute and merge on the GPU graph."""
    size, widths = 8, (5, 6, 7, 11)
    bucket_sizes = capacities(size)
    branch_count = 12 * len(bucket_sizes)
    proposed = torch.zeros(size, dtype=torch.int32, device="cuda")
    applied = torch.zeros_like(proposed)
    group_actions = torch.full((2,), -1, dtype=torch.int32, device="cuda")
    counts = torch.zeros(2, dtype=torch.int64, device="cuda")
    buckets = torch.zeros_like(counts)
    indices = torch.full((2, size), -1, dtype=torch.int64, device="cuda")
    predicates = torch.full((branch_count,), -1, dtype=torch.int32, device="cuda")
    source = torch.arange(size * 1024, dtype=torch.float32, device="cuda").reshape(
        size, 1024
    )
    states = [torch.empty_like(source) for _ in range(3)]
    output = torch.empty_like(source)
    row_executed = torch.zeros((size, 8), dtype=torch.int32, device="cuda")
    group_executed = torch.zeros((2, 8), dtype=torch.int32, device="cuda")
    launched = torch.zeros(branch_count, dtype=torch.int32, device="cuda")
    pdl_executed = torch.zeros_like(launched)
    repair = torch.cuda.Stream()

    def segments(pass_index):
        group = 1 if pass_index == 2 else pass_index
        for bucket_index, capacity in enumerate(bucket_sizes):
            for segment in range(4):
                marker = (pass_index * len(bucket_sizes) + bucket_index) * 4 + segment
                region = segment + (4 if pass_index == 2 else 0)
                regions.marker(2 * marker)
                valid = (indices[group, :capacity] >= 0).to(torch.int32)
                states[pass_index][:capacity].add_(valid[:, None] * (segment + 1))
                members = proposed == 0 if group == 0 else proposed > 0
                row_executed[:, region].add_(members)
                group_executed[group, region].add_(1)
                launched[marker].add_(1)
                counter = pdl_executed[marker : marker + 1]
                regions.pdl_increment(counter)
                regions.pdl_increment(counter)
                regions.marker(2 * marker + 1)

    def forward():
        partition(proposed, applied, group_actions, counts, indices, buckets)
        branch_actions(group_actions, buckets, predicates, widths, size)
        for pass_index, state in enumerate(states):
            state.copy_(
                gather_rows(source, indices[1 if pass_index == 2 else pass_index])
            )
        current = torch.cuda.current_stream()
        repair.wait_stream(current)
        with torch.cuda.stream(repair):
            segments(2)
        segments(0)
        segments(1)
        merged = merge_rows(states[0], source, proposed == 0)
        output.copy_(merge_rows(states[1], merged, proposed > 0))
        current.wait_stream(repair)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        forward()
    moved = regions.convert(
        graph, predicates, [1] * branch_count, list(range(branch_count))
    )
    assert len(moved) == branch_count
    graph.instantiate()
    results = []
    cases = action_schedule(size) + [
        [0, 11, 5, -1, -1, 6, 0, 7],
        [7, 6, 11, 7, 11, -1, -1, -1],
        [11, 7, 11, 7, -1, -1, -1, -1],
        [0] * 7 + [-1],
        [-1] * size,
        [5, 7, 11, -1, -1, -1, -1, -1],
        [0] * size,
    ]
    for proposals in cases:
        row_executed.zero_()
        group_executed.zero_()
        launched.zero_()
        pdl_executed.zero_()
        proposed.copy_(torch.tensor(proposals, dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        expected = split_oracle(proposals)
        assert proposed.cpu().tolist() == proposals
        for name, actual in (
            ("applied", applied),
            ("group_actions", group_actions),
            ("group_counts", counts),
            ("group_indices", indices),
            ("group_capacities", buckets),
        ):
            assert actual.cpu().tolist() == expected[name], (name, actual, expected)
        expected_launches = [0] * branch_count
        expected_rows = [[0] * 8 for _ in range(size)]
        expected_groups = [[0] * 8 for _ in range(2)]
        expected_output = source.clone()
        for pass_index in range(3):
            group = 1 if pass_index == 2 else pass_index
            action = expected["group_actions"][group]
            members = expected["group_indices"][group][
                : expected["group_counts"][group]
            ]
            expected_state = torch.zeros_like(source)
            if action >= 0:
                capacity = expected["group_capacities"][group]
                bucket_index = bucket_sizes.index(capacity)
                enabled = [
                    int(action >= width if pass_index == 2 else action < width)
                    for width in widths
                ]
                increment = sum(
                    (segment + 1) * value for segment, value in enumerate(enabled)
                )
                for rank, row in enumerate(members):
                    expected_state[rank].copy_(source[row] + increment)
                    if pass_index != 2:
                        expected_output[row].copy_(expected_state[rank])
                    for segment, value in enumerate(enabled):
                        expected_rows[row][
                            segment + (4 if pass_index == 2 else 0)
                        ] += value
                for segment, value in enumerate(enabled):
                    expected_groups[group][
                        segment + (4 if pass_index == 2 else 0)
                    ] += value
                    expected_launches[
                        (pass_index * len(bucket_sizes) + bucket_index) * 4 + segment
                    ] = value
            assert torch.equal(states[pass_index], expected_state)
        assert torch.equal(output, expected_output)
        assert row_executed.cpu().tolist() == expected_rows
        assert group_executed.cpu().tolist() == expected_groups
        assert launched.cpu().tolist() == expected_launches
        assert pdl_executed.cpu().tolist() == [2 * value for value in expected_launches]
        assert sum(expected_launches) == 4 * sum(
            count > 0 for count in expected["group_counts"]
        )
        results.append(
            dict(
                proposed=proposals,
                **expected,
                selected_bodies=sum(expected_launches),
                launched=expected_launches
            )
        )
    masked = masked_store_probe()
    return dict(
        passed=True,
        routing="split_skip",
        pdl_verified=True,
        padded_rows_verified=True,
        all_padding_skips_all_bodies=True,
        capacities=list(bucket_sizes),
        moved_nodes=moved,
        cases=results,
        masked_kv_store=masked,
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
