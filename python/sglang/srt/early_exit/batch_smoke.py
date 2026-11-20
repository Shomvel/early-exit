# SPDX-License-Identifier: Apache-2.0
"""Real ModelRunner differential test of GPU no-skip/skip partitioning.

Compare token logits, KV slots and GDN state against full execution when all
proposals are zero. Check GPU packing and minimal capacity buckets per step,
then compare serial/async repair on identical full-model input trajectories.
No benchmark claim.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

FULL_PARITY_ATOL = 0.02
WIDTHS = (0, 5, 6, 7, 11)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cache_snapshot(raw, request, length):
    slots = raw.req_to_token_pool.req_to_token[request.req_pool_idx, :length].long()
    result = {}
    for layer in range(32, 43):
        if layer in raw.attn_backend.full_attn_layers:
            result[f"k{layer}"] = (
                raw.token_to_kv_pool.get_key_buffer(layer)[slots].cpu().clone()
            )
            result[f"v{layer}"] = (
                raw.token_to_kv_pool.get_value_buffer(layer)[slots].cpu().clone()
            )
        else:
            cache = raw.req_to_token_pool.mamba2_layer_cache(layer)
            result[f"conv{layer}"] = cache.conv[0][request.mamba_pool_idx].cpu().clone()
            result[f"ssm{layer}"] = cache.temporal[request.mamba_pool_idx].cpu().clone()
    return result


def run_batch(runner, prompts, *, steps, forced_tokens=None, actions=None):
    import torch

    from sglang.bench_one_batch import prepare_synthetic_inputs_for_latency_test
    from sglang.srt.early_exit.runtime import get_runtime

    size = len(prompts)
    raw = runner.torch_runner
    runtime = get_runtime(raw.model)
    capture_size = min(
        value for value in raw.server_args.cuda_graph_bs if value >= size
    )
    runner.clear()
    inputs = [ids[:-1] for ids in prompts]
    # A real, separately allocated request remains resident while graph padding
    # decodes the others. This canary checks that original and packed padding
    # cannot mutate another real request's KV/GDN state.
    canary_input = inputs[0] if size < capture_size else None
    all_inputs = inputs + ([canary_input] if canary_input is not None else [])
    all_reqs = prepare_synthetic_inputs_for_latency_test(
        len(all_inputs), len(prompts[0]) - 1, all_inputs
    )
    reqs = all_reqs[:size]
    current = torch.tensor(
        [ids[-1] for ids in prompts], dtype=torch.int64, device="cuda"
    )
    torch.cuda.synchronize()
    begin = time.perf_counter()
    _, _, batch = runner.extend(all_reqs)
    torch.cuda.synchronize()
    prefill = time.perf_counter() - begin
    canary_before = None
    if canary_input is not None:
        canary_before = cache_snapshot(raw, all_reqs[-1], len(canary_input))
        batch.filter_batch(keep_indices=list(range(size)))
    if runtime:
        for name in (
            "histogram",
            "executed",
            "row_histogram",
            "row_executed",
            "batch_executed",
            "group_executed",
            "group_action_histogram",
            "group_size_histogram",
            "bucket_histogram",
            "batched_main_executed",
        ):
            getattr(runtime, name).zero_()
        runtime.force.fill_(-1)
    generated, logits, routing_steps, padding_steps = [], [], [], []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    torch.cuda.synchronize()
    begin = time.perf_counter()
    start.record()
    for step in range(steps):
        if runtime and actions is not None:
            selected = actions[step % len(actions)]
            if len(selected) != size:
                raise ValueError(
                    "Every forced step must specify one action per request"
                )
            runtime.force[:size].copy_(
                torch.tensor(selected, dtype=torch.int32, device="cuda")
            )
        current, scores = runner.decode(current, batch)
        generated.append(current.cpu().tolist())
        logits.append(scores.detach().clone())
        if runtime:
            routing_steps.append(
                dict(
                    proposed=runtime.proposed_action[:capture_size].cpu().tolist(),
                    applied=runtime.action[:capture_size].cpu().tolist(),
                    group_actions=runtime.group_actions.cpu().tolist(),
                    group_counts=runtime.group_counts.cpu().tolist(),
                    group_indices=runtime.group_indices[:, :capture_size]
                    .cpu()
                    .tolist(),
                    group_capacities=runtime.group_capacities.cpu().tolist(),
                )
            )
        if size < capture_size:
            registry = raw.graph_runner.buffer_registry
            padding_steps.append(
                dict(
                    req_pool_indices=registry.get_slot("req_pool_indices")
                    .buffer[size:capture_size]
                    .cpu()
                    .tolist(),
                    out_cache_loc=registry.get_slot("out_cache_loc")
                    .buffer[size:capture_size]
                    .cpu()
                    .tolist(),
                    mamba_cache_indices=raw.attn_backend.linear_attn_backend.forward_metadata.mamba_cache_indices[
                        size:capture_size
                    ]
                    .cpu()
                    .tolist(),
                )
            )
        if forced_tokens is not None:
            current = torch.tensor(
                forced_tokens[step], dtype=torch.int64, device="cuda"
            )
    end.record()
    end.synchronize()
    elapsed = time.perf_counter() - begin
    state = {
        str(row): cache_snapshot(raw, request, len(ids) - 1 + steps)
        for row, (request, ids) in enumerate(zip(reqs, prompts))
    }
    canary_check = (
        comparisons(
            canary_before,
            cache_snapshot(raw, all_reqs[-1], len(canary_input)),
            exact=True,
        )
        if canary_before is not None
        else dict(passed=True, exercised=False)
    )
    summary = dict(
        generated=generated,
        batch_size=size,
        captured_batch_size=capture_size,
        prompt_tokens=[len(ids) for ids in prompts],
        steps=steps,
        prefill_seconds=prefill,
        decode_seconds=elapsed,
        decode_cuda_ms=start.elapsed_time(end),
        histogram=runtime.histogram.cpu().tolist() if runtime else None,
        executed=runtime.executed.cpu().tolist() if runtime else None,
        row_histogram=runtime.row_histogram.cpu().tolist() if runtime else None,
        row_executed=runtime.row_executed.cpu().tolist() if runtime else None,
        batch_executed=runtime.batch_executed.cpu().tolist() if runtime else None,
        batched_main_replays=(
            int(runtime.batched_main_executed.cpu()) if runtime else 0
        ),
        routing=runtime.routing if runtime else None,
        converted_regions=runtime.converted if runtime else None,
        capture_batch_sizes=list(raw.server_args.cuda_graph_bs),
        group_executed=runtime.group_executed.cpu().tolist() if runtime else None,
        group_action_histogram=(
            runtime.group_action_histogram.cpu().tolist() if runtime else None
        ),
        group_size_histogram=(
            runtime.group_size_histogram.cpu().tolist() if runtime else None
        ),
        bucket_histogram=runtime.bucket_histogram.cpu().tolist() if runtime else None,
        routing_steps=routing_steps,
        padding_steps=padding_steps,
        padding_canary=canary_check,
    )
    return summary, torch.stack(logits).cpu(), state


def split_oracle(proposed):
    """Independent Python partition/bucket oracle; never used to route replay."""
    if not proposed or any(value not in (-1, *WIDTHS) for value in proposed):
        raise ValueError("Split proposals must be -1/0/5/6/7/11")
    size = len(proposed)
    allowed = sorted({value for value in (1, 2, 4, 8, size) if value <= size})
    members = [
        [row for row, value in enumerate(proposed) if value == 0],
        [row for row, value in enumerate(proposed) if value > 0],
    ]
    positive = [value for value in proposed if value > 0]
    width = min(positive, default=-1)
    counts = list(map(len, members))
    return dict(
        applied=[width if value > 0 else value for value in proposed],
        group_actions=[0 if counts[0] else -1, width],
        group_counts=counts,
        group_indices=[rows + [-1] * (size - len(rows)) for rows in members],
        group_capacities=[
            min(value for value in allowed if value >= count) if count else 0
            for count in counts
        ],
    )


def action_schedule(size):
    # All zero / all positive plus every nontrivial partition size. Diverse
    # positive proposals exercise their common minimum without a zero veto.
    schedule = [[0] * size]
    for index in range(1, len(WIDTHS)):
        schedule.append(
            [WIDTHS[index + row % (len(WIDTHS) - index)] for row in range(size)]
        )
    for count in range(1, size):
        index = 1 + (count - 1) % 4
        positive_rank, values = 0, []
        for row in range(size):
            if (row + count) % size < count:
                values.append(0)
            else:
                values.append(WIDTHS[index + positive_rank % (len(WIDTHS) - index)])
                positive_rank += 1
        schedule.append(values)
    return schedule


def worker(args):
    import torch

    from sglang.srt.early_exit.runtime import get_runtime
    from sglang.srt.early_exit.smoke import load_runner

    output = Path(os.environ["RUN_DIR"])
    runner, tokenizer = load_runner(args.model, max_batch_size=args.max_batch_size)
    original = runner.torch_runner.forward

    def checked(batch, *positional, **keywords):
        result = original(batch, *positional, **keywords)
        if batch.forward_mode.is_decode() and not result.can_run_graph:
            raise AssertionError("Batched early-exit validation requires graph replay")
        return result

    runner.torch_runner.forward = checked
    runtime = get_runtime(runner.torch_runner.model)
    assert (runtime is not None) == (args.worker != "full")
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"Explain why {17 + row} times {19 + row} equals {(17 + row) * (19 + row)}, then count to {20 + 3 * row}.",
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        for row in range(args.max_batch_size)
    ]
    results, tensors = {}, {}
    with torch.inference_mode():
        for size in args.batch_sizes:
            chosen = prompts[:size]
            forced = None
            if args.worker != "full":
                forced = json.loads((output / "batch-full.json").read_text())[
                    str(size)
                ]["reference"]["generated"]
            run_batch(runner, chosen, steps=2, actions=[[0] * size])
            reference, scores, state = run_batch(
                runner,
                chosen,
                steps=args.steps,
                forced_tokens=forced,
                actions=[[0] * size],
            )
            results[str(size)] = dict(reference=reference)
            tensors[str(size)] = dict(reference_logits=scores, reference_state=state)
            if runtime:
                mixed, scores, state = run_batch(
                    runner,
                    chosen,
                    steps=args.steps,
                    forced_tokens=forced,
                    actions=action_schedule(size),
                )
                results[str(size)]["mixed"] = mixed
                tensors[str(size)].update(mixed_logits=scores, mixed_state=state)
                learned, _, _ = run_batch(runner, chosen, steps=args.steps)
                results[str(size)]["learned"] = learned
    save(output / ("batch-" + args.worker + ".json"), results)
    torch.save(tensors, output / ("batch-" + args.worker + "-tensors.pt"))
    print(
        "PB_BATCH_WORKER_PASS "
        + json.dumps(dict(mode=args.worker, sizes=args.batch_sizes)),
        flush=True,
    )


def comparisons(left, right, exact=False):
    import torch

    if isinstance(left, dict):
        assert left.keys() == right.keys()
        values = {key: comparisons(left[key], right[key], exact=exact) for key in left}
        return dict(
            passed=all(value["passed"] for value in values.values()),
            exact=all(value["exact"] for value in values.values()),
            max_abs=(
                max(value["max_abs"] for value in values.values())
                if all(value["max_abs"] is not None for value in values.values())
                else None
            ),
            tensors=values,
        )
    assert left.shape == right.shape
    reference, candidate = left.float(), right.float()
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    difference = (reference - candidate).abs()
    identical = torch.equal(left, right)
    maximum = float(difference.max()) if finite else None
    return dict(
        passed=finite and (identical if exact else maximum <= FULL_PARITY_ATOL),
        exact=identical,
        finite=finite,
        max_abs=maximum,
        mean_abs=float(difference.mean()) if finite else None,
        relative_l2=(
            float(difference.norm() / reference.norm().clamp_min(1e-12))
            if finite
            else None
        ),
        shape=list(left.shape),
        atol=0 if exact else FULL_PARITY_ATOL,
        rtol=0,
    )


def padding_checks(summary, size, steps):
    padding = summary["captured_batch_size"] - size
    observations = summary["padding_steps"]
    errors = []
    if len(observations) != (steps if padding else 0):
        errors.append("padding observation count differs")
    for offset, observed in enumerate(observations):
        for name, expected in (
            ("req_pool_indices", [0] * padding),
            ("out_cache_loc", [0] * padding),
            ("mamba_cache_indices", [-1] * padding),
        ):
            if observed[name] != expected:
                errors.append(dict(step=offset, field=name, actual=observed[name]))
    canary = summary["padding_canary"]
    if not canary["passed"] or (padding and not canary.get("exact", False)):
        errors.append("resident real-request KV/GDN canary changed")
    return dict(passed=not errors, errors=errors, padded_rows=padding, canary=canary)


def route_checks(summary, size, steps, schedule=None):
    """Check GPU partitions, minimal buckets and per-row/group launch counts."""
    errors = []

    def equal(name, actual, expected):
        if actual != expected:
            errors.append(dict(field=name, actual=actual, expected=expected))

    equal("routing", summary["routing"], "split_skip")
    equal("trace_steps", len(summary["routing_steps"]), steps)
    expected_regions = [
        12 * len({cap for cap in (1, 2, 4, 8, captured) if cap <= captured})
        for captured in summary["capture_batch_sizes"]
    ]
    equal(
        "regions_per_graph",
        sorted(len(value) for value in summary["converted_regions"]),
        sorted(expected_regions),
    )
    capture = summary["captured_batch_size"]
    row_histograms = [[0] * 5 for _ in range(8)]
    row_executions = [[0] * 8 for _ in range(8)]
    group_histograms = [[0] * 5 for _ in range(2)]
    size_histograms = [[0] * 9 for _ in range(2)]
    bucket_histograms = [[0] * 9 for _ in range(2)]
    group_executions = [[0] * 8 for _ in range(2)]
    diversity = split_steps = positive_steps = 0
    for offset, event in enumerate(summary["routing_steps"]):
        proposed = event["proposed"]
        equal(f"step{offset}.proposal_rows", len(proposed), capture)
        equal(
            f"step{offset}.proposal_padding", proposed[size:], [-1] * (capture - size)
        )
        if schedule is not None:
            equal(
                f"step{offset}.proposed",
                proposed[:size],
                schedule[offset % len(schedule)],
            )
        if len(proposed) != capture or any(
            value not in WIDTHS for value in proposed[:size]
        ):
            errors.append(dict(field=f"step{offset}.proposal_domain", actual=proposed))
            continue
        expected = split_oracle(proposed)
        for name, value in expected.items():
            equal(f"step{offset}.{name}", event[name], value)
        for row, action in enumerate(expected["applied"][:size]):
            row_histograms[row][WIDTHS.index(action)] += 1
            segments = [int(action < width) for width in WIDTHS[1:]] + [
                int(action >= width) for width in WIDTHS[1:]
            ]
            row_executions[row] = [a + b for a, b in zip(row_executions[row], segments)]
        for group, action in enumerate(expected["group_actions"]):
            size_histograms[group][expected["group_counts"][group]] += 1
            bucket_histograms[group][expected["group_capacities"][group]] += 1
            if action >= 0:
                group_histograms[group][WIDTHS.index(action)] += 1
                segments = [int(action < width) for width in WIDTHS[1:]] + [
                    int(action >= width) for width in WIDTHS[1:]
                ]
                group_executions[group] = [
                    a + b for a, b in zip(group_executions[group], segments)
                ]
        diversity += int(len(set(proposed[:size])) > 1)
        split_steps += int(all(expected["group_counts"]))
        positive_steps += int(expected["group_counts"][1] > 0)
    batch_execution = [sum(values) for values in zip(*group_executions)]
    equal("row_histogram", summary["row_histogram"], row_histograms)
    equal("row_executed", summary["row_executed"], row_executions)
    equal(
        "histogram",
        summary["histogram"],
        [sum(values) for values in zip(*row_histograms)],
    )
    equal(
        "executed",
        summary["executed"],
        [sum(values) for values in zip(*row_executions)],
    )
    equal("group_action_histogram", summary["group_action_histogram"], group_histograms)
    equal("group_size_histogram", summary["group_size_histogram"], size_histograms)
    equal("bucket_histogram", summary["bucket_histogram"], bucket_histograms)
    equal("group_executed", summary["group_executed"], group_executions)
    equal("batch_executed", summary["batch_executed"], batch_execution)
    equal(
        "physical_region_count",
        sum(summary["batch_executed"]),
        4 * sum(map(sum, group_histograms)),
    )
    equal(
        "batched_main_replays", summary["batched_main_replays"], group_histograms[0][0]
    )
    if schedule is not None and len(schedule) > 1:
        equal(
            "forced_jump_coverage",
            [
                WIDTHS[index]
                for index, count in enumerate(
                    [sum(values) for values in zip(*group_histograms)]
                )
                if count
            ],
            list(WIDTHS),
        )
        if size > 1 and (not diversity or not split_steps):
            errors.append(
                dict(
                    field="mixed_partition_coverage",
                    diversity=diversity,
                    split_steps=split_steps,
                )
            )
    return dict(
        passed=not errors,
        errors=errors,
        group_action_counts=group_histograms,
        proposed_diversity_steps=diversity,
        split_steps=split_steps,
        positive_skip_steps=positive_steps,
    )


def main(args):
    import torch

    from sglang.srt.early_exit.conditional import ConditionalRegions
    from sglang.srt.early_exit.probe import batch_probe

    output = Path(os.environ["RUN_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.yaml").exists():
        raise ValueError("Use a new managed RUN_DIR")
    save(
        output / "config.yaml",
        dict(
            model=str(args.model),
            classifier=str(args.classifier),
            batch_sizes=args.batch_sizes,
            capture_max_batch=args.max_batch_size,
            steps=args.steps,
            source=32,
            skips=[5, 6, 7, 11],
            tp=1,
            pp=1,
            attention_backend="triton",
            decision="split_skip: GPU partition into zero and positive proposals, positive-group minimum jump, smallest active 1/2/4/8 capacity per group; padding excluded",
            full_vs_no_skip_atol=FULL_PARITY_ATOL,
            full_vs_no_skip_rtol=0,
        ),
    )
    probe = batch_probe(ConditionalRegions())
    save(output / "batch-conditional-summary.json", probe)
    for mode in ("full", "serial", "async"):
        environment = dict(os.environ)
        for name in (
            "SGLANG_EARLY_EXIT_CHECKPOINT",
            "SGLANG_EARLY_EXIT_FORCE",
            "SGLANG_EARLY_EXIT_CACHE",
        ):
            environment.pop(name, None)
        if mode != "full":
            environment.update(
                SGLANG_EARLY_EXIT_CHECKPOINT=str(args.classifier),
                SGLANG_EARLY_EXIT_CACHE=mode,
                SGLANG_EARLY_EXIT_ROUTING="split_skip",
            )
        command = [
            sys.executable,
            "-u",
            "-m",
            "sglang.srt.early_exit.batch_smoke",
            *sys.argv[1:],
            "--worker",
            mode,
        ]
        subprocess.run(command, env=environment, check=True)
    records = {
        mode: json.loads((output / ("batch-" + mode + ".json")).read_text())
        for mode in ("full", "serial", "async")
    }
    values = {
        mode: torch.load(
            output / ("batch-" + mode + "-tensors.pt"),
            map_location="cpu",
            weights_only=True,
        )
        for mode in records
    }
    checks = {}
    for size in args.batch_sizes:
        key = str(size)
        check = {}
        for mode in ("serial", "async"):
            check[mode + "_no_skip_logits"] = comparisons(
                values["full"][key]["reference_logits"],
                values[mode][key]["reference_logits"],
            )
            check[mode + "_no_skip_cache"] = comparisons(
                values["full"][key]["reference_state"],
                values[mode][key]["reference_state"],
            )
            check[mode + "_no_skip_tokens"] = dict(
                passed=records["full"][key]["reference"]["generated"]
                == records[mode][key]["reference"]["generated"]
            )
            for name, schedule in (
                ("reference", [[0] * size]),
                ("mixed", action_schedule(size)),
                ("learned", None),
            ):
                check[mode + "_" + name + "_routes"] = route_checks(
                    records[mode][key][name],
                    size,
                    args.steps,
                    schedule,
                )
                check[mode + "_" + name + "_padding"] = padding_checks(
                    records[mode][key][name], size, args.steps
                )
        check["full_padding"] = padding_checks(
            records["full"][key]["reference"], size, args.steps
        )
        check["serial_async_logits"] = comparisons(
            values["serial"][key]["mixed_logits"],
            values["async"][key]["mixed_logits"],
            exact=True,
        )
        check["serial_async_cache"] = comparisons(
            values["serial"][key]["mixed_state"],
            values["async"][key]["mixed_state"],
            exact=True,
        )
        checks[key] = check
    passed = all(
        value["passed"] for check in checks.values() for value in check.values()
    )
    summary = dict(
        passed=passed,
        batch_sizes=args.batch_sizes,
        checks=checks,
        records=records,
        conditional_probe=probe,
        classifier_sha256=hashlib.sha256(args.classifier.read_bytes()).hexdigest(),
        routing="split_skip",
        scope="GPU no-skip/skip partition, positive-group minimum width, minimal active graph buckets, physical launch counts and real-cache padding isolation; throughput requires a separate paired benchmark.",
    )
    save(output / "batch-smoke-summary.json", summary)
    compact = {
        size: {
            name: {
                key: value[key]
                for key in ("passed", "exact", "max_abs")
                if key in value
            }
            for name, value in check.items()
        }
        for size, check in checks.items()
    }
    print(
        ("PB_BATCH_SMOKE_PASS " if passed else "PB_BATCH_SMOKE_FAIL ")
        + json.dumps(dict(passed=passed, checks=compact)),
        flush=True,
    )
    if not passed:
        raise AssertionError(
            "Batched route, cache or numerical parity failed; inspect batch-smoke-summary.json"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 3, 4, 8])
    parser.add_argument("--max-batch-size", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--worker", choices=("full", "serial", "async"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.steps < max(
        len(action_schedule(size)) for size in args.batch_sizes
    ) or not all(1 <= size <= args.max_batch_size for size in args.batch_sizes):
        parser.error(
            "steps must cover all five actions and partition sizes, and batches fit the captured maximum"
        )
    for name in ("model", "classifier"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


if __name__ == "__main__":
    arguments = parse_args()
    (worker if arguments.worker else main)(arguments)
