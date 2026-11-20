# SPDX-License-Identifier: Apache-2.0
"""Same-GPU full / async / deferred-prefill benchmark pilot."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sglang.srt.early_exit import batch_evaluate as b

MODES = ("full", "async", "lazy")


def collect(args, out):
    p = b.protocol()
    selected, _, replay = b.selected_questions(args)
    expected = {
        (size, row["evaluation_index"]) for size in args.batch_sizes for row in selected
    }
    receipts, quality, timing = {}, {}, {}
    for mode in MODES:
        root = out / mode
        paths = b.output_paths(root, "full" if mode == "full" else "early_exit")
        qpath, tpath, rpath = paths
        receipt = json.loads(rpath.read_text())
        assert receipt["quality_sha256"] == p.sha(qpath) and receipt[
            "timing_sha256"
        ] == p.sha(tpath)
        quality[mode], timing[mode] = p.read_jsonl(qpath), p.read_jsonl(tpath)
        assert len(quality[mode]) == len(expected)
        assert {
            (r["requested_batch_size"], r["evaluation_index"]) for r in quality[mode]
        } == expected
        assert receipt["quality_count"] == len(quality[mode]) and receipt[
            "batch_count"
        ] == len(timing[mode])
        assert receipt["cache_policy"] == (None if mode == "full" else mode)
        assert receipt["runtime_present"] == (mode != "full")
        assert receipt["graph_execution"]["decode_real_row_slots"] >= sum(
            r["computed_real_row_tokens"] for r in timing[mode]
        )
        if mode != "full":
            sizes = receipt["server_args"]["cuda_graph_bs"]
            counts = [
                (
                    (8 * len([c for c in (1, 2, 4, 8) if c <= size]) + 4 * size)
                    if mode == "lazy"
                    else 12 * len([c for c in (1, 2, 4, 8) if c <= size])
                )
                for size in sizes
            ]
            assert sorted(map(len, receipt["converted_regions"])) == sorted(counts)
        receipts[mode] = receipt
    for key in (
        "gpu_uuid",
        "gpu",
        "torch",
        "cuda",
        "server_args",
        "identity",
        "trajectory_sha256",
    ):
        assert all(receipts[m][key] == receipts["full"][key] for m in MODES), key
    results = {}
    for size in args.batch_sizes:
        fixed = {}
        for mode in MODES:
            rows = [
                r
                for r in timing[mode]
                if r["requested_batch_size"] == size and r["kind"] == "fixed_replay"
            ]
            assert set(i for r in rows for i in r["evaluation_indices"]) == replay
            tokens = sum(r["effective_generated_tokens"] for r in rows)
            wall = sum(r["decode_seconds"] for r in rows)
            cuda = sum(r["decode_cuda_ms"] for r in rows) / 1000
            fixed[mode] = dict(
                tokens=tokens,
                decode_seconds=wall,
                decode_cuda_seconds=cuda,
                tokens_per_second=tokens / wall,
            )
            if mode == "lazy":
                for key in (
                    "skipped_layer_tokens",
                    "replayed_layer_tokens",
                    "pending_layer_tokens",
                    "reentry_events",
                    "prefill_chunks",
                    "prefill_capacity_slots",
                    "replayed_segment_tokens",
                ):
                    fixed[mode][key] = sum(r["lazy_cache"][key] for r in rows)
                assert (
                    fixed[mode]["skipped_layer_tokens"]
                    == fixed[mode]["replayed_layer_tokens"]
                    + fixed[mode]["pending_layer_tokens"]
                )
                fixed[mode]["mean_replayed_chunk_length"] = fixed[mode][
                    "replayed_segment_tokens"
                ] / max(1, fixed[mode]["prefill_chunks"])
                fixed[mode]["capacity_utilization"] = fixed[mode][
                    "replayed_segment_tokens"
                ] / max(1, fixed[mode]["prefill_capacity_slots"])
        for mode in ("async", "lazy"):
            fixed[mode]["speedup_vs_full"] = (
                fixed["full"]["decode_seconds"] / fixed[mode]["decode_seconds"]
            )
        fixed["lazy"]["speedup_vs_async"] = (
            fixed["async"]["decode_seconds"] / fixed["lazy"]["decode_seconds"]
        )
        grades = {}
        for task in b.TASKS:
            grades[task] = {}
            for mode in MODES:
                rows = [
                    r
                    for r in quality[mode]
                    if r["requested_batch_size"] == size and r["benchmark"] == task
                ]
                times = [
                    r
                    for r in timing[mode]
                    if r["requested_batch_size"] == size
                    and r["benchmark"] == task
                    and r["kind"] == "quality"
                ]
                grades[task][mode] = dict(
                    count=len(rows),
                    correct=sum(r["result"]["correct"] for r in rows),
                    accuracy=sum(r["result"]["correct"] for r in rows) / len(rows),
                    truncated=sum(r["result"]["truncated"] for r in rows),
                    effective_tokens=sum(
                        r["effective_generated_tokens"] for r in times
                    ),
                    completed_row_waste=sum(
                        r["completed_row_waste_tokens"] for r in times
                    ),
                    decode_seconds=sum(r["decode_seconds"] for r in times),
                    total_seconds=sum(r["total_seconds"] for r in times),
                )
        results[str(size)] = dict(fixed128=fixed, quality=grades)
    summary = dict(
        passed=True,
        scope="pilot; original test splits, not a new full-benchmark score",
        chunk_capacity=args.chunk,
        selected_indices=[r["evaluation_index"] for r in selected],
        replay_indices=sorted(replay),
        matched_tokens=args.matched_tokens,
        results=results,
        worker_receipts=receipts,
        source={
            name: p.sha(Path(__file__).with_name(name))
            for name in (
                "lazy.py",
                "lazy_smoke.py",
                "lazy_evaluate.py",
                "conditional.py",
                "conditional.cu",
                "runtime.py",
                "policy.py",
            )
        },
    )
    b.save(out / "lazy-evaluation-summary.json", summary)
    print("PB_LAZY_EVAL_PASS " + json.dumps(results), flush=True)


def main(args):
    out = Path(os.environ["RUN_DIR"])
    selected, _, replay = b.selected_questions(args)
    b.save(
        out / "config.yaml",
        dict(
            model=str(args.model),
            classifier=str(args.classifier),
            data=str(args.data),
            modes=MODES,
            batch_sizes=args.batch_sizes,
            chunk=args.chunk,
            examples_per_task=args.examples_per_task,
            matched_tokens=args.matched_tokens,
            selected_indices=[r["evaluation_index"] for r in selected],
            replay_indices=sorted(replay),
            completion_policy="static requests with finished-row waste counted",
        ),
    )
    for mode in MODES:
        root = out / mode
        root.mkdir()
        if mode != "full":
            shutil.copyfile(
                out / "full/batch-reference-trajectories.json",
                root / "batch-reference-trajectories.json",
            )
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("SGLANG_EARLY_EXIT_")
            or k == "SGLANG_EARLY_EXIT_SOURCE_ROOT"
        }
        env["RUN_DIR"] = str(root)
        if mode != "full":
            env.update(
                SGLANG_EARLY_EXIT_CHECKPOINT=str(args.classifier),
                SGLANG_EARLY_EXIT_CACHE=mode,
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
    collect(args, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "classifier", "data"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--weight-inventory",
        type=Path,
        default=b.protocol().source_root()
        / "experiments/perfectblend/references/model-weight-inventory.json",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--examples-per-task", type=int, default=8)
    parser.add_argument("--replay-examples-per-task", type=int, default=8)
    parser.add_argument("--matched-tokens", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--max-total-tokens", type=int, default=131072)
    parser.add_argument("--worker", choices=MODES)
    args = parser.parse_args()
    if args.worker:
        args.worker_mode = "full" if args.worker == "full" else "early_exit"
        args.cache_policy = "async" if args.worker == "full" else args.worker
        b.worker(args)
    else:
        main(args)
