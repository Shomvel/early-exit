# SPDX-License-Identifier: Apache-2.0
"""Real batched ModelRunner differential test for independent GPU jump actions.

Compare every row's token logits, KV slots and GDN state against full execution
when actions are zero, then compare serial/async repair with heterogeneous
forced actions and fixed full-model input trajectories. No benchmark claim.
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


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_batch(runner, prompts, *, steps, forced_tokens=None, actions=None):
    import torch

    from sglang.bench_one_batch import prepare_synthetic_inputs_for_latency_test
    from sglang.srt.early_exit.runtime import get_runtime

    size = len(prompts)
    raw = runner.torch_runner
    runtime = get_runtime(raw.model)
    runner.clear()
    reqs = prepare_synthetic_inputs_for_latency_test(
        size, len(prompts[0]) - 1, [ids[:-1] for ids in prompts]
    )
    current = torch.tensor(
        [ids[-1] for ids in prompts], dtype=torch.int64, device="cuda"
    )
    torch.cuda.synchronize()
    begin = time.perf_counter()
    _, _, batch = runner.extend(reqs)
    torch.cuda.synchronize()
    prefill = time.perf_counter() - begin
    if runtime:
        runtime.histogram.zero_()
        runtime.executed.zero_()
        runtime.row_histogram.zero_()
        runtime.row_executed.zero_()
        runtime.batched_main_executed.zero_()
        runtime.force.fill_(-1)
    generated, logits = [], []
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
        if forced_tokens is not None:
            current = torch.tensor(
                forced_tokens[step], dtype=torch.int64, device="cuda"
            )
    end.record()
    end.synchronize()
    elapsed = time.perf_counter() - begin
    state = {}
    for row, (request, ids) in enumerate(zip(reqs, prompts)):
        slots = raw.req_to_token_pool.req_to_token[
            request.req_pool_idx, : len(ids) - 1 + steps
        ].long()
        row_state = {}
        for layer in range(32, 43):
            if layer in raw.attn_backend.full_attn_layers:
                row_state[f"k{layer}"] = raw.token_to_kv_pool.get_key_buffer(layer)[
                    slots
                ].cpu()
                row_state[f"v{layer}"] = raw.token_to_kv_pool.get_value_buffer(layer)[
                    slots
                ].cpu()
            else:
                cache = raw.req_to_token_pool.mamba2_layer_cache(layer)
                row_state[f"conv{layer}"] = cache.conv[0][request.mamba_pool_idx].cpu()
                row_state[f"ssm{layer}"] = cache.temporal[request.mamba_pool_idx].cpu()
        state[str(row)] = row_state
    summary = dict(
        generated=generated,
        batch_size=size,
        prompt_tokens=[len(ids) for ids in prompts],
        steps=steps,
        prefill_seconds=prefill,
        decode_seconds=elapsed,
        decode_cuda_ms=start.elapsed_time(end),
        histogram=runtime.histogram.cpu().tolist() if runtime else None,
        executed=runtime.executed.cpu().tolist() if runtime else None,
        row_histogram=runtime.row_histogram.cpu().tolist() if runtime else None,
        row_executed=runtime.row_executed.cpu().tolist() if runtime else None,
        batched_main_replays=(
            int(runtime.batched_main_executed.cpu()) if runtime else 0
        ),
    )
    return summary, torch.stack(logits).cpu(), state


def action_schedule(size):
    widths = (0, 11, 5, 6, 7)
    return [
        [widths[(row + step) % len(widths)] for row in range(size)] for step in range(5)
    ]


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


def route_checks(summary, size, steps, schedule=None):
    """Check each segment's exact count, not just main/repair complementarity."""
    widths = (0, 5, 6, 7, 11)
    errors = []

    def equal(name, actual, expected):
        if actual != expected:
            errors.append(dict(field=name, actual=actual, expected=expected))

    histograms, executions = summary["row_histogram"], summary["row_executed"]
    equal("counter_rows", [len(histograms), len(executions)], [8, 8])
    for row, (histogram, executed) in enumerate(zip(histograms, executions)):
        equal(f"row{row}.tokens", sum(histogram), steps if row < size else 0)
        if row >= size:
            equal(f"row{row}.padding_histogram", histogram, [0] * 5)
            equal(f"row{row}.padding_execution", executed, [0] * 8)
            continue
        if schedule is not None:
            expected = [
                sum(
                    schedule[step % len(schedule)][row] == width
                    for step in range(steps)
                )
                for width in widths
            ]
            equal(f"row{row}.histogram", histogram, expected)
        expected_main = [
            sum(count for count, action in zip(histogram, widths) if action < boundary)
            for boundary in widths[1:]
        ]
        equal(
            f"row{row}.execution",
            executed,
            expected_main + [steps - count for count in expected_main],
        )
    equal(
        "histogram",
        summary["histogram"],
        [sum(row[index] for row in histograms) for index in range(5)],
    )
    equal(
        "executed",
        summary["executed"],
        [sum(row[index] for row in executions) for index in range(8)],
    )
    batched = summary["batched_main_replays"]
    if schedule is not None:
        equal(
            "batched_main_replays",
            batched,
            sum(
                size == 8
                and all(action == 0 for action in schedule[step % len(schedule)])
                for step in range(steps)
            ),
        )
    elif not 0 <= batched <= (min(row[0] for row in histograms) if size == 8 else 0):
        errors.append(dict(field="batched_main_replays", actual=batched))
    return dict(passed=not errors, errors=errors)


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
            decision="one GPU action per request; static one-request conditional middle bodies; fully real bs8 with all actions zero uses an exclusive batched middle body",
            full_vs_row_atol=FULL_PARITY_ATOL,
            full_vs_row_rtol=0,
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
        scope="Batch correctness, heterogeneous actions and padded-row isolation; throughput requires a separate paired benchmark.",
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
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--max-batch-size", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--worker", choices=("full", "serial", "async"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.steps < 1 or not all(
        1 <= size <= args.max_batch_size for size in args.batch_sizes
    ):
        parser.error(
            "steps must be positive and batch-sizes within the captured maximum"
        )
    for name in ("model", "classifier"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


if __name__ == "__main__":
    arguments = parse_args()
    (worker if arguments.worker else main)(arguments)
