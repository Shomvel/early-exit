# SPDX-License-Identifier: Apache-2.0
"""Full-model and independently scheduled re-entry prefill controls."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def expected(schedule, capacity):
    """CPU scheduling oracle, used only after execution to verify GPU counters."""
    pending = [[] for _ in schedule[0]]
    stats = [[0] * 8 for _ in pending]
    for proposed in schedule:
        positive = [width for width in proposed if width > 0]
        width = min(positive, default=0)
        for row, proposal in enumerate(proposed):
            action = width if proposal > 0 else proposal
            previous = pending[row]
            if previous and action < max(previous):
                stats[row][0] += 1
                stats[row][2] += sum(previous)
                counts = [
                    sum(value >= end for value in previous) for end in (5, 6, 7, 11)
                ]
                stats[row][3] += sum(
                    (count + capacity - 1) // capacity for count in counts
                )
                stats[row][4] = stats[row][3] * capacity
                stats[row][6] += sum(counts)
                previous.clear()
            stats[row][1] += max(0, action)
            stats[row][5] += int(action > 0)
            if action > 0:
                previous.append(action)
    return stats, sum(map(sum, pending))


def schedule(size):
    values = [[0] * size]
    base = (11, 11, 0, 5, 11, 7, 0)
    values += [
        [base[(step + row) % len(base)] for row in range(size)] for step in range(7)
    ]
    values += [[11] * size for _ in range(35)]  # exercises >2 WHILE chunks
    values += [[0] * size]
    values += [
        [5 if row % 2 else 11 for row in range(size)],
        [7] * size,
        [6] * size,
        [0] * size,
    ]
    return values


def run(runner, prompts, actions, tokens=None, canary=False):
    import torch

    from sglang.bench_one_batch import prepare_synthetic_inputs_for_latency_test
    from sglang.srt.early_exit.batch_smoke import cache_snapshot, comparisons
    from sglang.srt.early_exit.runtime import get_runtime

    raw = runner.torch_runner
    runtime = get_runtime(raw.model)
    runner.clear()
    if runtime and runtime.lazy:
        runtime.lazy.reset()
    size = len(prompts)
    inputs = [ids[:-1] for ids in prompts]
    reqs = prepare_synthetic_inputs_for_latency_test(
        size + int(canary), len(inputs[0]), inputs + ([inputs[0]] if canary else [])
    )
    _, _, batch = runner.extend(reqs)
    before = cache_snapshot(raw, reqs[-1], len(inputs[0])) if canary else None
    if canary:
        batch.filter_batch(keep_indices=list(range(size)))
    current = torch.tensor(
        [ids[-1] for ids in prompts], device="cuda", dtype=torch.int64
    )
    scores, generated = [], []
    if runtime:
        runtime.histogram.zero_()
    for step, action in enumerate(actions):
        if runtime:
            runtime.force[:size].copy_(
                torch.tensor(action, device="cuda", dtype=torch.int32)
            )
        current, logits = runner.decode(current, batch)
        scores.append(logits.cpu().clone())
        generated.append(current.cpu().tolist())
        if tokens is not None:
            current = torch.tensor(tokens[step], device="cuda", dtype=torch.int64)
    torch.cuda.synchronize()
    states = {
        str(row): cache_snapshot(raw, request, len(prompts[row]) - 1 + len(actions))
        for row, request in enumerate(reqs[:size])
    }
    canary_result = (
        comparisons(before, cache_snapshot(raw, reqs[-1], len(inputs[0])), exact=True)
        if canary
        else {"passed": True, "exercised": False}
    )
    assert canary_result["passed"], canary_result
    metrics = runtime.lazy.metrics(batch) if runtime and runtime.lazy else None
    return (
        {"generated": generated, "canary": canary_result, "metrics": metrics},
        torch.stack(scores),
        states,
    )


def worker(args):
    import torch

    from sglang.srt.early_exit.batch_smoke import comparisons
    from sglang.srt.early_exit.runtime import get_runtime
    from sglang.srt.early_exit.smoke import load_runner

    torch.set_num_threads(4)
    out = Path(os.environ["RUN_DIR"])
    runner, tokenizer = load_runner(args.model, max_batch_size=8)
    raw = runner.torch_runner
    runtime = get_runtime(raw.model)
    base = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Explain why 17 times 19 equals 323, then count from one to twenty.",
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    result = {
        "worker": args.worker,
        "cases": [],
        "converted": runtime.converted if runtime else None,
    }
    with torch.inference_mode():
        for size in (1, 2, 3, 4, 8):
            prompts = [base] * size
            path = out / f"lazy-full-bs{size}.pt"
            control_actions = [[0] * size for _ in range(8)]
            if args.worker == "full":
                control, logits, cache = run(
                    runner, prompts, control_actions, canary=size == 3
                )
                long, _, _ = run(runner, prompts, [[0] * size for _ in schedule(size)])
                torch.save(
                    {
                        "control": control,
                        "logits": logits,
                        "cache": cache,
                        "tokens": long["generated"],
                    },
                    path,
                )
                result["cases"].append({"size": size, "canary": control["canary"]})
                continue
            reference = torch.load(path, map_location="cpu", weights_only=True)
            control, logits, cache = run(
                runner,
                prompts,
                control_actions,
                tokens=reference["control"]["generated"],
                canary=size == 3,
            )
            logit_check = comparisons(logits, reference["logits"], exact=False)
            cache_check = comparisons(cache, reference["cache"], exact=False)
            assert logit_check["passed"] and cache_check["passed"], (
                logit_check,
                cache_check,
            )
            mixed, logits, cache = run(
                runner,
                prompts,
                schedule(size),
                tokens=reference["tokens"],
                canary=size == 3,
            )
            want_stats, want_debt = expected(schedule(size), runtime.lazy.capacity)
            assert mixed["metrics"]["row_stats"] == want_stats, (
                mixed["metrics"],
                want_stats,
            )
            assert mixed["metrics"]["pending_layer_tokens"] == want_debt == 0
            graph = raw.graph_runner
            raw.graph_runner = None
            runtime.oracle = True
            try:
                oracle, oracle_logits, oracle_cache = run(
                    runner,
                    prompts,
                    schedule(size),
                    tokens=reference["tokens"],
                    canary=size == 3,
                )
            finally:
                raw.graph_runner = graph
                runtime.oracle = False
            eager_logits = comparisons(logits, oracle_logits, exact=False)
            eager_cache = comparisons(cache, oracle_cache, exact=False)
            assert eager_logits["passed"] and eager_cache["passed"], (
                eager_logits,
                eager_cache,
            )
            assert oracle["metrics"]["row_stats"] == want_stats
            deferred, _, _ = run(
                runner, prompts, [[11] * size] * 10, tokens=reference["tokens"][:10]
            )
            assert deferred["metrics"]["replayed_layer_tokens"] == 0
            assert deferred["metrics"]["pending_layer_tokens"] == 110 * size
            result["cases"].append(
                dict(
                    size=size,
                    no_skip_logits=logit_check,
                    no_skip_cache=cache_check,
                    oracle_logits=eager_logits,
                    oracle_cache=eager_cache,
                    mixed=mixed,
                    deferred=deferred,
                )
            )
            print("PB_LAZY_SMOKE_CASE " + json.dumps(result["cases"][-1]), flush=True)
    save(out / f"lazy-smoke-{args.worker}.json", result)


def main(args):
    from sglang.srt.early_exit.batch_evaluate import protocol

    p = protocol()
    out = Path(os.environ["RUN_DIR"])
    p.verify_model(args)
    save(
        out / "config.yaml",
        dict(
            model=str(args.model),
            classifier=str(args.classifier),
            chunk=args.chunk,
            source32=True,
            widths=[0, 5, 6, 7, 11],
            test="full and eager causal-prefill oracle",
        ),
    )
    for mode in ("full", "lazy"):
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("SGLANG_EARLY_EXIT_")
            or k == "SGLANG_EARLY_EXIT_SOURCE_ROOT"
        }
        if mode == "lazy":
            env.update(
                SGLANG_EARLY_EXIT_CHECKPOINT=str(args.classifier),
                SGLANG_EARLY_EXIT_CACHE="lazy",
                SGLANG_EARLY_EXIT_CHUNK=str(args.chunk),
            )
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                __spec__.name,
                *sys.argv[1:],
                "--worker",
                mode,
            ],
            env=env,
            check=True,
        )
    summary = dict(
        passed=True,
        chunk_capacity=args.chunk,
        workers={
            mode: json.loads((out / f"lazy-smoke-{mode}.json").read_text())
            for mode in ("full", "lazy")
        },
        source=p.execution_identity(args),
    )
    save(out / "lazy-smoke-summary.json", summary)
    print(
        "PB_LAZY_SMOKE_PASS " + json.dumps({"passed": True, "chunk": args.chunk}),
        flush=True,
    )


if __name__ == "__main__":
    from sglang.srt.early_exit.batch_evaluate import protocol

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--worker", choices=("full", "lazy"))
    parser.add_argument(
        "--weight-inventory",
        type=Path,
        default=protocol().source_root()
        / "experiments/perfectblend/references/model-weight-inventory.json",
    )
    args = parser.parse_args()
    (worker if args.worker else main)(args)
