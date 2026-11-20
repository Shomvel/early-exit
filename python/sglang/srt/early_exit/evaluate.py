# SPDX-License-Identifier: Apache-2.0
"""Same-GPU paired SGLang ModelRunner GSM8K/MATH/MMLU evaluation."""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED = {"gsm8k": 1319, "math": 5000, "mmlu": 14042}
BENCHMARK_SHA256 = "a04901c438d78667d050805b26fad1d4092f6e70ed0bde2b2f666ef9aa72ddfe"
SEED = 20261120
_MATH_EQUIVALENCE = None


def source_root():
    configured = os.environ.get("SGLANG_EARLY_EXIT_SOURCE_ROOT")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parents[4]
    )


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle]


def read_benchmarks(data):
    path = Path(data) / "benchmarks.jsonl"
    if sha(path) != BENCHMARK_SHA256:
        raise ValueError("benchmarks.jsonl differs from the pinned evaluation dataset")
    rows = read_jsonl(path)
    tests, development = [], defaultdict(list)
    for row in rows:
        if row["split"] == "test":
            tests.append(dict(row, evaluation_index=len(tests)))
        elif row["benchmark"] == "mmlu" and row["split"] == "dev":
            development[row["subject"]].append(row)
        else:
            raise ValueError("Unexpected benchmark split")
    if dict(Counter(row["benchmark"] for row in tests)) != EXPECTED:
        raise ValueError("Full benchmark counts differ from the pinned protocol")
    subjects = {row["subject"] for row in tests if row["benchmark"] == "mmlu"}
    if (
        set(development) != subjects
        or len(subjects) != 57
        or any(len(v) != 5 for v in development.values())
    ):
        raise ValueError("MMLU requires exactly five dev questions for all 57 subjects")
    return tests, development


def select_rows(tests, *, workers, smoke=False, limit_per_task=None):
    rows = list(tests)
    random.Random(SEED).shuffle(rows)
    limit = 2 if smoke else limit_per_task
    if limit is not None:
        counts = Counter()
        selected = []
        for row in rows:
            if counts[row["benchmark"]] < limit:
                counts[row["benchmark"]] += 1
                selected.append(row)
        rows = selected
    if workers > len(rows):
        raise ValueError("There must be at least one selected question per worker")
    return rows


def matched_indices(rows, count):
    result = set()
    for benchmark in ("gsm8k", "math"):
        result.update(
            row["evaluation_index"]
            for row in sorted(
                (r for r in rows if r["benchmark"] == benchmark),
                key=lambda r: r["evaluation_index"],
            )[:count]
        )
    return result


def mmlu_example(row, with_answer=False):
    text = row["question"] + "\n"
    text += "".join(
        f"{label}. {choice}\n" for label, choice in zip("ABCD", row["choices"])
    )
    text += "Answer:"
    return text + (" " + "ABCD"[row["answer"]] + "\n\n" if with_answer else "")


def prompt(tokenizer, row, development):
    if row["benchmark"] == "mmlu":
        subject = row["subject"]
        header = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
        examples = development[subject]
        if len(examples) != 5:
            raise ValueError("MMLU prompt requires exactly five dev examples")
        return tokenizer.encode(
            header
            + "".join(mmlu_example(x, True) for x in examples)
            + mmlu_example(row),
            add_special_tokens=False,
        )
    question = row.get("question", row.get("problem"))
    question += (
        "\nPlease reason step by step, and put your final answer within \\boxed{}."
    )
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
        raise ValueError("The chat tokenizer must return one list of token IDs")
    return ids


def boxed(text):
    for match in reversed(list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))):
        depth = 1
        for index in range(match.end(), len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            if depth == 0:
                return text[match.end() : index]
    return None


def math_equivalent(left, right):
    global _MATH_EQUIVALENCE
    if _MATH_EQUIVALENCE is None:
        path = source_root() / "experiments/perfectblend/native/math_equivalence.py"
        if not path.is_file():
            raise ValueError(
                "MATH evaluation requires the source checkout's attributed math_equivalence.py"
            )
        spec = importlib.util.spec_from_file_location(
            "early_exit_math_equivalence", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MATH_EQUIVALENCE = module.is_equiv
    return _MATH_EQUIVALENCE(left, right)


def grade(row, text):
    if row["benchmark"] == "mmlu":
        reference = "ABCD"[row["answer"]]
        return dict(correct=text == reference, prediction=text, reference=reference)
    final = text.split("</think>")[-1]
    prediction = boxed(final)
    if row["benchmark"] == "math":
        reference = boxed(row["solution"])
        correct = (
            prediction is not None
            and reference is not None
            and math_equivalent(prediction, reference)
        )
    else:
        numbers = re.findall(
            r"[-+]?\d[\d,]*(?:\.\d+)?", prediction if prediction is not None else final
        )
        prediction = numbers[-1].replace(",", "") if numbers else None
        reference = row["answer"].split("####")[-1].strip().replace(",", "")
        try:
            correct = (
                prediction is not None
                and abs(float(prediction) - float(reference)) < 1e-8
            )
        except ValueError:
            correct = False
    return dict(correct=bool(correct), prediction=prediction, reference=reference)


def percentiles(values):
    ordered = sorted(values)
    result = {}
    for label, quantile in (("p50", 0.5), ("p95", 0.95)):
        offset = (len(ordered) - 1) * quantile
        lower = int(offset)
        upper = min(lower + 1, len(ordered) - 1)
        result[label] = ordered[lower] + (ordered[upper] - ordered[lower]) * (
            offset - lower
        )
    return result


def summarize(records):
    """Summarize already paired, unique per-question records."""
    if len({row["evaluation_index"] for row in records}) != len(records):
        raise ValueError("Duplicate paired evaluation indices")
    result = {}
    for benchmark in EXPECTED:
        group = [row for row in records if row["benchmark"] == benchmark]
        if not group:
            raise ValueError(f"No completed questions for {benchmark}")
        modes = {}
        for mode in ("full", "early_exit"):
            values = [row["results"][mode] for row in group]
            totals = {
                key: sum(row[key] for row in values)
                for key in (
                    "generated_tokens",
                    "prefill_seconds",
                    "decode_seconds",
                    "decode_cuda_ms",
                    "total_seconds",
                    "skipped_blocks_total",
                )
            }
            correct = sum(row["correct"] for row in values)
            modes[mode] = dict(
                count=len(values),
                correct=correct,
                accuracy=correct / len(values),
                truncated=sum(row["truncated"] for row in values),
                **totals,
                decode_tokens_per_second=totals["generated_tokens"]
                / totals["decode_seconds"],
                decode_cuda_tokens_per_second=totals["generated_tokens"]
                * 1000
                / totals["decode_cuda_ms"],
                mean_skipped_blocks=totals["skipped_blocks_total"]
                / totals["generated_tokens"],
                percentiles={
                    key: percentiles([row[key] for row in values])
                    for key in (
                        "generated_tokens",
                        "prefill_seconds",
                        "decode_seconds",
                        "total_seconds",
                        "time_to_first_token_seconds",
                    )
                },
            )
        modes["comparison"] = dict(
            accuracy_delta=modes["early_exit"]["accuracy"] - modes["full"]["accuracy"],
            end_to_end_speedup=modes["full"]["total_seconds"]
            / modes["early_exit"]["total_seconds"],
            decode_throughput_speedup=modes["early_exit"]["decode_tokens_per_second"]
            / modes["full"]["decode_tokens_per_second"],
            decode_cuda_throughput_speedup=modes["early_exit"][
                "decode_cuda_tokens_per_second"
            ]
            / modes["full"]["decode_cuda_tokens_per_second"],
            lost_correct=sum(
                row["results"]["full"]["correct"]
                and not row["results"]["early_exit"]["correct"]
                for row in group
            ),
            gained_correct=sum(
                not row["results"]["full"]["correct"]
                and row["results"]["early_exit"]["correct"]
                for row in group
            ),
        )
        subjects = sorted(
            {row["subject"] for row in group if row["subject"] is not None}
        )
        if subjects:
            modes["by_subject"] = {}
            for subject in subjects:
                subset = [row for row in group if row["subject"] == subject]
                modes["by_subject"][subject] = dict(
                    count=len(subset),
                    **{
                        mode
                        + "_accuracy": sum(
                            row["results"][mode]["correct"] for row in subset
                        )
                        / len(subset)
                        for mode in ("full", "early_exit")
                    },
                )
            if benchmark == "mmlu":
                modes["macro_accuracy"] = {
                    mode: sum(
                        value[mode + "_accuracy"]
                        for value in modes["by_subject"].values()
                    )
                    / len(subjects)
                    for mode in ("full", "early_exit")
                }
                modes["macro_subject_count"] = len(subjects)
        matched = [row["matched_tokens"] for row in group if row["matched_tokens"]]
        if matched:
            for row in matched:
                if (
                    row["full"]["generated_tokens"]
                    != row["early_exit"]["generated_tokens"]
                ):
                    raise ValueError("Matched replay token counts differ")
            full = sum(row["full"]["decode_seconds"] for row in matched)
            candidate = sum(row["early_exit"]["decode_seconds"] for row in matched)
            modes["matched_token_decode"] = dict(
                examples=len(matched),
                tokens=sum(row["full"]["generated_tokens"] for row in matched),
                full_seconds=full,
                early_exit_seconds=candidate,
                speedup=full / candidate,
                cuda_speedup=sum(row["full"]["decode_cuda_ms"] for row in matched)
                / sum(row["early_exit"]["decode_cuda_ms"] for row in matched),
            )
        result[benchmark] = modes
    return result


def emit(tag, **values):
    print("PB_SGLANG_" + tag + " " + json.dumps(values, allow_nan=False), flush=True)


def verify_model(args):
    inventory = json.loads(args.weight_inventory.read_text())
    indexed = set(
        json.loads((args.model / "model.safetensors.index.json").read_text())[
            "weight_map"
        ].values()
    )
    if indexed != {row["file"] for row in inventory["files"]}:
        raise ValueError("Model index and weight inventory differ")
    for row in inventory["files"]:
        if Path(row["file"]).is_absolute() or ".." in Path(row["file"]).parts:
            raise ValueError("The model inventory must reference local shard files")
        status = (args.model / row["file"]).stat()
        if (status.st_size, status.st_mtime_ns) != (row["bytes"], row["mtime_ns"]):
            raise ValueError(
                "Model storage stamp changed; rehash with fingerprint_model.py before running"
            )
    for key, name in (
        ("config_sha256", "config.json"),
        ("index_sha256", "model.safetensors.index.json"),
    ):
        if key in inventory and sha(args.model / name) != inventory[key]:
            raise ValueError("Model metadata changed: " + name)


def generate(
    runner, tokenizer, ids, *, limit, choice_ids=None, forced=None, ignore_eos=False
):
    import torch

    from sglang.bench_one_batch import prepare_synthetic_inputs_for_latency_test
    from sglang.srt.early_exit.runtime import get_runtime

    if len(ids) < 2:
        raise ValueError("Evaluation prompts must contain at least two tokens")
    maximum = (
        len(forced) if forced is not None else (1 if choice_ids is not None else limit)
    )
    raw = runner.torch_runner
    if maximum < 1 or len(ids) + maximum > raw.model_config.context_len:
        raise ValueError(
            "Prompt and output reservation exceed the configured context; prompts are never truncated"
        )
    runner.clear()
    runtime = get_runtime(raw.model)
    reqs = prepare_synthetic_inputs_for_latency_test(1, len(ids) - 1, [ids[:-1]])
    current = torch.tensor([ids[-1]], dtype=torch.int64, device="cuda")
    choices = (
        torch.tensor(choice_ids, dtype=torch.int64, device="cuda")
        if choice_ids is not None
        else None
    )
    eos_ids = tokenizer.eos_token_id
    eos = set(eos_ids if isinstance(eos_ids, (tuple, list)) else [eos_ids]) - {None}
    vocabulary = tokenizer.get_vocab()
    eos.update(
        vocabulary[token]
        for token in ("<|im_end|>", "<|endoftext|>")
        if token in vocabulary
    )
    torch.cuda.synchronize()
    tick = time.perf_counter()
    _, _, batch = runner.extend(reqs)
    torch.cuda.synchronize()
    prefill = time.perf_counter() - tick
    if runtime:
        runtime.histogram.zero_()
        runtime.executed.zero_()
    generated, choice_scores = [], None
    first = None
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    torch.cuda.synchronize()
    tick = time.perf_counter()
    start.record()
    for offset in range(maximum):
        current, logits = runner.decode(current, batch)
        if choices is not None:
            choice_scores = logits[0, choices].float().cpu().tolist()
            choice = max(range(4), key=lambda index: choice_scores[index])
            token = choice_ids[choice]
        else:
            token = int(current[0])
        generated.append(token)
        if offset == 0:
            first = time.perf_counter() - tick
        if choices is not None or (forced is None and not ignore_eos and token in eos):
            break
        if forced is not None and offset + 1 < maximum:
            current = torch.tensor([forced[offset]], dtype=torch.int64, device="cuda")
    end.record()
    end.synchronize()
    elapsed = time.perf_counter() - tick
    histogram = (
        runtime.histogram.cpu().tolist() if runtime else [len(generated), 0, 0, 0, 0]
    )
    executed = runtime.executed.cpu().tolist() if runtime else None
    if sum(histogram) != len(generated):
        raise ValueError("Routing histogram does not match executed decode steps")
    if executed is not None and (
        len(executed) != 8
        or any(executed[i] + executed[i + 4] != len(generated) for i in range(4))
    ):
        raise ValueError(
            "Main/repair conditional execution counts do not cover each decode step"
        )
    widths = (0, 5, 6, 7, 11)
    return dict(
        text=(
            "ABCD"[choice]
            if choices is not None
            else tokenizer.decode(generated, skip_special_tokens=True)
        ),
        generated_ids=generated,
        generated_tokens=len(generated),
        prompt_tokens=len(ids),
        prefill_seconds=prefill,
        decode_seconds=elapsed,
        decode_cuda_ms=start.elapsed_time(end),
        total_seconds=prefill + elapsed,
        time_to_first_token_seconds=prefill + first,
        truncated=forced is None
        and choices is None
        and not ignore_eos
        and len(generated) == maximum
        and generated[-1] not in eos,
        skipped_blocks_total=sum(
            width * count for width, count in zip(widths, histogram)
        ),
        skip_counts={str(width): count for width, count in zip(widths, histogram)},
        choice_scores=choice_scores,
        executed=executed,
    )


def paths(out, mode, rank):
    return out / f"sglang-{mode}-{rank}.jsonl", out / f"worker-{mode}-{rank}.json"


def worker(args):
    import torch

    from sglang.srt.early_exit.runtime import get_runtime
    from sglang.srt.early_exit.smoke import load_runner

    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    out = Path(os.environ["RUN_DIR"])
    tests, development = read_benchmarks(args.data)
    selected = select_rows(
        tests,
        workers=args.workers,
        smoke=args.smoke,
        limit_per_task=args.limit_per_task,
    )
    shard = selected[args.rank :: args.workers]
    matched = matched_indices(selected, args.matched_examples_per_task)
    verify_model(args)
    if args.worker_mode == "full" and os.environ.get("SGLANG_EARLY_EXIT_CHECKPOINT"):
        raise ValueError(
            "The baseline subprocess must not have a classifier environment variable"
        )
    runner, tokenizer = load_runner(args.model)
    raw = runner.torch_runner
    original_forward = raw.forward
    graph_counts = dict(decode_steps=0)

    def checked_forward(batch, *positional, **keywords):
        output = original_forward(batch, *positional, **keywords)
        if batch.forward_mode.is_decode():
            if not output.can_run_graph:
                raise ValueError(
                    "Every measured decode must use the captured batch-one CUDA graph"
                )
            graph_counts["decode_steps"] += 1
        return output

    raw.forward = checked_forward
    runtime = get_runtime(runner.torch_runner.model)
    if args.worker_mode == "full" and runtime is not None:
        raise ValueError("Full-model baseline must have no EarlyExitRuntime")
    if args.worker_mode == "early_exit":
        if (
            runtime is None
            or runtime.mode != args.cache_policy
            or not runtime.converted
        ):
            raise ValueError(
                "Early-exit evaluation requires its converted CUDA graph and requested cache policy"
            )
        if runtime.checkpoint_sha256 != sha(args.classifier):
            raise ValueError("Loaded classifier differs from the configured checkpoint")
        if any(
            len(graph) != 8 or any(count <= 0 for count in graph)
            for graph in runtime.converted
        ):
            raise ValueError("Early-exit graph has an empty conditional region")
        runtime.force.fill_(-1)
    choice_ids = [
        tokenizer.encode(" " + label, add_special_tokens=False) for label in "ABCD"
    ]
    if not all(len(ids) == 1 for ids in choice_ids):
        raise ValueError("MMLU A/B/C/D must each be one token with a leading space")
    choice_ids = [ids[0] for ids in choice_ids]
    warm = prompt(tokenizer, shard[0], development)
    with torch.inference_mode():
        for _ in range(2):
            generate(runner, tokenizer, warm, limit=16)
        full_replays = {}
        if args.worker_mode == "early_exit":
            for row in read_jsonl(paths(out, "full", args.rank)[0]):
                if row["matched_tokens"]:
                    full_replays[row["evaluation_index"]] = row["matched_tokens"][
                        "forced_ids"
                    ]
        path, receipt = paths(out, args.worker_mode, args.rank)
        with path.open("x") as handle:
            for done, row in enumerate(shard, 1):
                ids = prompt(tokenizer, row, development)
                maximum = (
                    1
                    if row["benchmark"] == "mmlu"
                    else (2048 if row["benchmark"] == "gsm8k" else 4096)
                )
                result = generate(
                    runner,
                    tokenizer,
                    ids,
                    limit=maximum,
                    choice_ids=choice_ids if row["benchmark"] == "mmlu" else None,
                )
                result.update(grade(row, result["text"]))
                replay = {}
                if row["evaluation_index"] in matched:
                    length = args.matched_tokens
                    if args.worker_mode == "full":
                        forced = result["generated_ids"][:length]
                        if len(forced) < length:
                            forced = generate(
                                runner, tokenizer, ids, limit=length, ignore_eos=True
                            )["generated_ids"]
                    else:
                        forced = full_replays[row["evaluation_index"]]
                    measured = generate(
                        runner, tokenizer, ids, limit=len(forced), forced=forced
                    )
                    replay = {
                        key: measured[key]
                        for key in (
                            "generated_tokens",
                            "prefill_seconds",
                            "decode_seconds",
                            "decode_cuda_ms",
                            "total_seconds",
                            "skipped_blocks_total",
                        )
                    }
                    replay["forced_ids"] = forced
                record = dict(
                    benchmark=row["benchmark"],
                    evaluation_index=row["evaluation_index"],
                    source_file=row["source_file"],
                    index=row["index"],
                    subject=row.get("subject", row.get("type")),
                    level=row.get("level"),
                    prompt_sha256=hashlib.sha256(
                        json.dumps(ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                    result=result,
                    matched_tokens=replay,
                )
                handle.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
                handle.flush()
                if done == 1 or done % 10 == 0:
                    emit(
                        "EVAL",
                        mode=args.worker_mode,
                        rank=args.rank,
                        done=done,
                        total=len(shard),
                        benchmark=row["benchmark"],
                        correct=result["correct"],
                        generated_tokens=result["generated_tokens"],
                    )
    evidence = dict(
        mode=args.worker_mode,
        rank=args.rank,
        count=len(shard),
        prediction_sha256=sha(path),
        gpu=torch.cuda.get_device_name(0),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        source_file_sha256=sha(__file__),
        runtime_source_sha256={
            name: sha(Path(__file__).with_name(name))
            for name in ("runtime.py", "conditional.py", "conditional.cu", "smoke.py")
        },
        model_config_sha256=sha(args.model / "config.json"),
        model_index_sha256=sha(args.model / "model.safetensors.index.json"),
        weight_inventory_sha256=sha(args.weight_inventory),
        benchmark_sha256=BENCHMARK_SHA256,
        classifier_sha256=sha(args.classifier),
        tokenizer_sha256={
            name: sha(args.model / name)
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "chat_template.jinja",
            )
            if (args.model / name).is_file()
        },
        math_grader_sha256=sha(
            source_root() / "experiments/perfectblend/native/math_equivalence.py"
        ),
        runtime_present=runtime is not None,
        cache_policy=runtime.mode if runtime else None,
        threshold=runtime.threshold if runtime else None,
        converted_regions=runtime.converted if runtime else None,
        graph_execution=graph_counts,
        server_args={
            key: getattr(runner.torch_runner.server_args, key)
            for key in (
                "dtype",
                "tp_size",
                "pp_size",
                "attention_backend",
                "linear_attn_decode_backend",
                "linear_attn_prefill_backend",
                "context_length",
                "cuda_graph_bs",
                "disable_cuda_graph",
                "mem_fraction_static",
                "max_total_tokens",
                "max_mamba_cache_size",
            )
        },
    )
    save(receipt, evidence)
    emit(
        "WORKER_PASS",
        mode=args.worker_mode,
        rank=args.rank,
        count=len(shard),
        prediction_sha256=evidence["prediction_sha256"],
    )


def aggregate(args, out, selected):
    paired, receipts = [], []
    for rank in range(args.workers):
        data = {}
        worker_receipts = {}
        for mode in ("full", "early_exit"):
            path, receipt_path = paths(out, mode, rank)
            receipt = json.loads(receipt_path.read_text())
            if sha(path) != receipt["prediction_sha256"]:
                raise ValueError(
                    "Prediction file SHA256 differs from its worker receipt"
                )
            rows = read_jsonl(path)
            if receipt["count"] != len(rows) or len(
                {r["evaluation_index"] for r in rows}
            ) != len(rows):
                raise ValueError("Missing or duplicate worker predictions")
            if receipt["runtime_present"] != (mode == "early_exit"):
                raise ValueError("Worker receipt has an unexpected runtime path")
            if receipt["classifier_sha256"] != sha(args.classifier):
                raise ValueError("Classifier changed during evaluation")
            if receipt["graph_execution"]["decode_steps"] < sum(
                row["result"]["generated_tokens"] for row in rows
            ):
                raise ValueError(
                    "Worker CUDA-graph receipt omits measured decode steps"
                )
            data[mode] = {row["evaluation_index"]: row for row in rows}
            worker_receipts[mode] = receipt
            receipts.append(receipt)
        if data["full"].keys() != data["early_exit"].keys():
            raise ValueError("Full/early-exit question identities differ")
        for key in (
            "gpu_uuid",
            "model_config_sha256",
            "model_index_sha256",
            "weight_inventory_sha256",
            "benchmark_sha256",
            "classifier_sha256",
            "source_file_sha256",
            "server_args",
            "tokenizer_sha256",
            "runtime_source_sha256",
            "math_grader_sha256",
            "torch",
            "cuda",
        ):
            if worker_receipts["full"][key] != worker_receipts["early_exit"][key]:
                raise ValueError("Full/early-exit execution identity differs: " + key)
        output = out / f"predictions-{rank}.jsonl"
        with output.open("x") as handle:
            for index, full in data["full"].items():
                candidate = data["early_exit"][index]
                for key in (
                    "benchmark",
                    "source_file",
                    "index",
                    "subject",
                    "level",
                    "prompt_sha256",
                ):
                    if full[key] != candidate[key]:
                        raise ValueError("Paired question metadata differs")
                matched = {}
                if full["matched_tokens"] or candidate["matched_tokens"]:
                    if (
                        full["matched_tokens"]["forced_ids"]
                        != candidate["matched_tokens"]["forced_ids"]
                    ):
                        raise ValueError(
                            "Matched replay input token trajectories differ"
                        )
                    matched = {
                        mode: {
                            key: value
                            for key, value in row["matched_tokens"].items()
                            if key != "forced_ids"
                        }
                        for mode, row in (("full", full), ("early_exit", candidate))
                    }
                record = {
                    key: full[key]
                    for key in (
                        "benchmark",
                        "evaluation_index",
                        "source_file",
                        "index",
                        "subject",
                        "level",
                        "prompt_sha256",
                    )
                }
                record.update(
                    results={"full": full["result"], "early_exit": candidate["result"]},
                    matched_tokens=matched,
                )
                handle.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
                paired.append(record)
    if {row["evaluation_index"] for row in paired} != {
        row["evaluation_index"] for row in selected
    } or len(paired) != len(selected):
        raise ValueError(
            "Paired predictions do not cover the exact selected evaluation set"
        )
    counts = dict(Counter(row["benchmark"] for row in paired))
    complete = counts == EXPECTED and not args.smoke
    if not args.smoke and args.limit_per_task is None and not complete:
        raise ValueError(
            "A full evaluation must complete all three benchmark test sets"
        )
    metrics = summarize(paired)
    if complete and metrics["mmlu"]["macro_subject_count"] != 57:
        raise ValueError("Complete MMLU evaluation must cover all 57 subjects")
    summary = dict(
        passed=True,
        complete=complete,
        complete_benchmarks=complete,
        counts=counts,
        benchmarks=metrics,
        workers=args.workers,
        classifier_sha256=sha(args.classifier),
        worker_receipts=receipts,
        paired_prediction_sha256={
            f"predictions-{rank}.jsonl": sha(out / f"predictions-{rank}.jsonl")
            for rank in range(args.workers)
        },
        settings=dict(
            engine="SGLang ModelRunner",
            baseline="independent process without EarlyExitRuntime",
            dtype="BF16",
            graph_batch_size=1,
            concurrency=1,
            tp=1,
            pp=1,
            context_length=receipts[0]["server_args"]["context_length"],
            attention_backend="triton",
            gdn_backend="triton",
            cache_policy=args.cache_policy,
            policy="five-class cal99",
            mode_order="all full shards, then early-exit shards on the same respective GPU",
            warmup_sequences=2,
            warmup_tokens=16,
            sampling="greedy",
            thinking=False,
            mmlu_shots=5,
            gsm8k_max_new_tokens=2048,
            math_max_new_tokens=4096,
            matched_tokens=args.matched_tokens,
            matched_examples_per_task=args.matched_examples_per_task,
        ),
        timing_scope="ModelRunner prefill of all but the final prompt token, plus synchronized decode starting with that final token and including token readback; excludes model load, graph capture, warmup, tokenization, HTTP and server scheduler.",
        matched_protocol="Up to matched-examples-per-task selected GSM8K and MATH questions by original evaluation index. Exactly matched-tokens decode steps use the full-model trajectory; after EOS, baseline continuation supplies the remaining fixed inputs. Both modes replay the same stored token IDs.",
        cache_semantics="Skipped-path middle activations bypass work on the main stream; all skipped KV/GDN states are backfilled each token, then joined before the next token. This differs from native capture's one-step frozen-cache counterfactual.",
        limits="Complete full/early-exit quality and speed require complete_benchmarks=true. Fixed process order can retain thermal/load drift. Matched replay uses identical full-model token inputs, ignoring EOS to reach the fixed length when necessary; it is separate from task generation.",
        official_comparison="Primary baseline: the same checkpoint1479 fully executed. Inspected official Qwen3.8-27B card does not list these three original benchmark scores.",
    )
    save(out / "evaluation-summary.json", summary)
    emit(
        "EVAL_PASS",
        complete_benchmarks=complete,
        counts=counts,
        summary_sha256=sha(out / "evaluation-summary.json"),
    )


def main(args):
    import torch

    out = Path(os.environ["RUN_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    if any(
        (out / name).exists() for name in ("config.yaml", "evaluation-summary.json")
    ) or list(out.glob("sglang-*.jsonl")):
        raise ValueError(
            "RUN_DIR already has an experiment config; use a new immutable run"
        )
    tests, _ = read_benchmarks(args.data)
    selected = select_rows(
        tests,
        workers=args.workers,
        smoke=args.smoke,
        limit_per_task=args.limit_per_task,
    )
    verify_model(args)
    if not args.classifier.is_file() or torch.cuda.device_count() < args.workers:
        raise ValueError(
            "A classifier checkpoint and at least one visible GPU per worker are required"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        visible.split(",")[: args.workers]
        if visible
        else [str(index) for index in range(args.workers)]
    )
    save(
        out / "config.yaml",
        dict(
            argv=sys.argv,
            model=str(args.model),
            data=str(args.data),
            classifier=str(args.classifier),
            classifier_sha256=sha(args.classifier),
            weight_inventory_sha256=sha(args.weight_inventory),
            benchmark_sha256=BENCHMARK_SHA256,
            policy=args.policy,
            cache_policy=args.cache_policy,
            workers=args.workers,
            selected_counts=dict(Counter(row["benchmark"] for row in selected)),
            smoke=args.smoke,
            limit_per_task=args.limit_per_task,
        ),
    )
    for mode in ("full", "early_exit"):
        children = []
        try:
            for rank, device in enumerate(devices):
                source = str(Path(__file__).resolve().parents[3])
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=device,
                    PYTHONDONTWRITEBYTECODE="1",
                    PYTHONPATH=source
                    + (
                        os.pathsep + os.environ["PYTHONPATH"]
                        if os.environ.get("PYTHONPATH")
                        else ""
                    ),
                )
                for key in (
                    "SGLANG_EARLY_EXIT_CHECKPOINT",
                    "SGLANG_EARLY_EXIT_CACHE",
                    "SGLANG_EARLY_EXIT_FORCE",
                ):
                    env.pop(key, None)
                if mode == "early_exit":
                    env.update(
                        SGLANG_EARLY_EXIT_CHECKPOINT=str(args.classifier),
                        SGLANG_EARLY_EXIT_CACHE=args.cache_policy,
                        SGLANG_EARLY_EXIT_FORCE="-1",
                    )
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "sglang.srt.early_exit.evaluate",
                    *sys.argv[1:],
                    "--worker-mode",
                    mode,
                    "--rank",
                    str(rank),
                ]
                children.append(subprocess.Popen(command, env=env))
            for child in children:
                if child.wait():
                    raise RuntimeError(
                        "SGLang evaluation worker failed; inspect preserved partial JSONL and worker logs"
                    )
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                if child.poll() is None:
                    child.wait(timeout=60)
    aggregate(args, out, selected)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--policy", choices=("multiclass",), default="multiclass")
    parser.add_argument("--cache-policy", choices=("async", "serial"), default="async")
    parser.add_argument(
        "--weight-inventory",
        type=Path,
        default=source_root()
        / "experiments/perfectblend/references/model-weight-inventory.json",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--matched-examples-per-task", type=int, default=8)
    parser.add_argument("--matched-tokens", type=int, default=128)
    parser.add_argument(
        "--worker-mode", choices=("full", "early_exit"), help=argparse.SUPPRESS
    )
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("workers", "matched_tokens"):
        if getattr(args, name) < 1:
            parser.error(name.replace("_", "-") + " must be positive")
    if args.limit_per_task is not None and args.limit_per_task < 1:
        parser.error("limit-per-task must be positive")
    if args.matched_examples_per_task < 0:
        parser.error("matched-examples-per-task must be nonnegative")
    if args.smoke and args.limit_per_task is not None:
        parser.error("Use either --smoke or --limit-per-task")
    if args.worker_mode is not None and (
        args.rank is None or not 0 <= args.rank < args.workers
    ):
        parser.error("Internal workers require a valid rank")
    if args.worker_mode is None and args.rank is not None:
        parser.error("rank is an internal worker option")
    for name in ("model", "data", "classifier", "weight_inventory"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


if __name__ == "__main__":
    args = parse_args()
    (worker if args.worker_mode else main)(args)
