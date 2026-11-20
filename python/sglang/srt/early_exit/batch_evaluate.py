# SPDX-License-Identifier: Apache-2.0
"""Paired, one-GPU pilot of real benchmark requests at batch sizes 1/2/4/8.

Static microbatches retain completed requests until the last request finishes.
Those decode slots are reported as completed-row waste. Batch elapsed time is
recorded once; per-question completion latency is never summed as GPU time.
The separate fixed-token replay uses one full-model batch-one trajectory per
question at every batch size and in both modes.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

TASKS = ("gsm8k", "math", "mmlu")
WIDTHS = (0, 5, 6, 7, 11)
_PROTOCOL = None


def protocol():
    global _PROTOCOL
    if _PROTOCOL is None:
        from sglang.srt.early_exit import evaluate

        _PROTOCOL = evaluate
    return _PROTOCOL


def save(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


class DecodeAccounting:
    """Track effective outputs separately from still-executed finished rows."""

    def __init__(self, limits, eos_tokens, *, ignore_eos=False, choice_ids=None):
        if not limits or any(limit < 1 for limit in limits):
            raise ValueError("Every real request requires a positive output limit")
        self.limits = list(limits)
        self.eos_tokens = set(eos_tokens)
        self.ignore_eos = ignore_eos
        self.choice_ids = choice_ids
        self.generated = [[] for _ in limits]
        self.choice_scores = [None for _ in limits]
        self.choices = [None for _ in limits]
        self.completion_seconds = [None for _ in limits]
        self.reasons = [None for _ in limits]
        self.steps = 0

    def advance(self, tokens, *, elapsed, scores=None):
        if len(tokens) != len(self.limits) or self.done:
            raise ValueError(
                "Decode output rows differ from the active static microbatch"
            )
        for row, token in enumerate(tokens):
            if self.reasons[row] is not None:
                continue
            if self.choice_ids is not None:
                values = scores[row]
                choice = max(range(4), key=lambda index: values[index])
                token = self.choice_ids[choice]
                self.choices[row], self.choice_scores[row] = choice, values
            self.generated[row].append(int(token))
            reason = None
            if self.choice_ids is not None:
                reason = "choice"
            elif not self.ignore_eos and token in self.eos_tokens:
                reason = "eos"
            elif len(self.generated[row]) == self.limits[row]:
                reason = "limit"
            if reason is not None:
                self.reasons[row] = reason
                self.completion_seconds[row] = elapsed
        self.steps += 1

    @property
    def done(self):
        return all(reason is not None for reason in self.reasons)

    @property
    def effective_tokens(self):
        return sum(map(len, self.generated))

    @property
    def completed_row_waste(self):
        return self.steps * len(self.limits) - self.effective_tokens


def question_batches(rows, batch_size):
    """Keep task limits homogeneous while allowing different prompt lengths."""
    for task in TASKS:
        group = [row for row in rows if row["benchmark"] == task]
        for begin in range(0, len(group), batch_size):
            yield group[begin : begin + batch_size]


def maximum(row):
    return {"gsm8k": 2048, "math": 4096, "mmlu": 1}[row["benchmark"]]


def metadata(row, ids):
    return dict(
        benchmark=row["benchmark"],
        evaluation_index=row["evaluation_index"],
        source_file=row["source_file"],
        index=row["index"],
        subject=row.get("subject", row.get("type")),
        level=row.get("level"),
        prompt_sha256=fingerprint(ids),
    )


def eos_tokens(tokenizer):
    value = tokenizer.eos_token_id
    result = set(value if isinstance(value, (list, tuple)) else [value]) - {None}
    vocabulary = tokenizer.get_vocab()
    result.update(
        vocabulary[name]
        for name in ("<|im_end|>", "<|endoftext|>")
        if name in vocabulary
    )
    return result


def batch_routing_metrics(snapshot, size, steps):
    """Validate applied row routes, group launches and minimal capacity use."""
    if not 1 <= size <= 8 or steps < 1:
        raise ValueError("Batch route metrics require real rows and decode steps")
    if snapshot["routing"] == "batch_min":
        return common_routing_metrics(snapshot, size, steps)
    if snapshot["routing"] != "split_skip":
        raise ValueError("Batch pilot requires split_skip routing")
    rows, executions = snapshot["row_histogram"], snapshot["row_executed"]
    if len(rows) != 8 or len(executions) != 8:
        raise ValueError("Routing receipt omits captured/padded rows")
    groups = snapshot["group_action_histogram"]
    sizes, buckets = snapshot["group_size_histogram"], snapshot["bucket_histogram"]
    if len(groups) != 2 or len(sizes) != 2 or len(buckets) != 2:
        raise ValueError("Split routing receipt requires two groups")

    def segments(histogram):
        main = [
            sum(count for count, action in zip(histogram, WIDTHS) if action < width)
            for width in WIDTHS[1:]
        ]
        total = sum(histogram)
        return main + [total - count for count in main]

    def valid_counts(values, length):
        return len(values) == length and all(
            isinstance(value, int) and value >= 0 for value in values
        )

    for row, (histogram, execution) in enumerate(zip(rows, executions)):
        if (
            not valid_counts(histogram, 5)
            or sum(histogram) != (steps if row < size else 0)
            or execution != segments(histogram)
        ):
            raise ValueError("Applied row histogram or main/repair executions differ")
    if snapshot["histogram"] != [sum(values) for values in zip(*rows)] or snapshot[
        "executed"
    ] != [sum(values) for values in zip(*executions)]:
        raise ValueError("Aggregate route counters differ from real row executions")
    capture = next(value for value in (1, 2, 4, 8) if value >= size)
    capacities = [value for value in (1, 2, 4, 8) if value <= capture]
    group_exec = []
    for group, (actions, counts, capacities_hist) in enumerate(
        zip(groups, sizes, buckets)
    ):
        if (
            not valid_counts(actions, 5)
            or not valid_counts(counts, 9)
            or not valid_counts(capacities_hist, 9)
        ):
            raise ValueError("Split histograms require nonnegative integer counts")
        if (
            sum(counts) != steps
            or sum(capacities_hist) != steps
            or any(counts[size + 1 :])
        ):
            raise ValueError("Group size/bucket histograms differ from decode steps")
        if sum(actions) != steps - counts[0] or (
            any(actions[1:]) if group == 0 else actions[0] != 0
        ):
            raise ValueError("Group action histogram differs from its nonempty steps")
        real_slots = sum(count * index for index, count in enumerate(counts))
        wanted = (
            snapshot["histogram"][0] if group == 0 else sum(snapshot["histogram"][1:])
        )
        if real_slots != wanted:
            raise ValueError("Group sizes differ from actual zero/positive row actions")
        expected_buckets = [0] * 9
        for count, frequency in enumerate(counts):
            if not frequency:
                continue
            capacity = (
                min(value for value in capacities if value >= count) if count else 0
            )
            expected_buckets[capacity] += frequency
        if capacities_hist != expected_buckets:
            raise ValueError("Group bucket histogram differs from minimal capacities")
        group_exec.append(segments(actions))
    launched = [sum(values) for values in zip(*group_exec)]
    if (
        snapshot["group_executed"] != group_exec
        or snapshot["batch_executed"] != launched
        or sum(launched) != 4 * sum(map(sum, groups))
    ):
        raise ValueError("Physical IF launches differ from nonempty group actions")
    if snapshot["batched_main_replays"] != groups[0][0]:
        raise ValueError(
            "No-skip group replays differ from zero-action group histogram"
        )
    proposed = snapshot["last_proposed_actions"]
    if (
        len(proposed) != capture
        or any(value not in WIDTHS for value in proposed[:size])
        or proposed[size:] != [-1] * (capture - size)
    ):
        raise ValueError("Last proposal trace has invalid real or padded rows")
    members = [
        [row for row, value in enumerate(proposed) if value == 0],
        [row for row, value in enumerate(proposed) if value > 0],
    ]
    counts = list(map(len, members))
    minimum = min((value for value in proposed if value > 0), default=-1)
    expected = dict(
        last_applied_actions=[minimum if value > 0 else value for value in proposed],
        last_group_actions=[0 if counts[0] else -1, minimum],
        last_group_counts=counts,
        last_group_indices=[
            indices + [-1] * (capture - len(indices)) for indices in members
        ],
        last_group_capacities=[
            min(value for value in capacities if value >= count) if count else 0
            for count in counts
        ],
    )
    if any(snapshot[key] != value for key, value in expected.items()):
        raise ValueError(
            "Last GPU partition/action/bucket differs from independent oracle"
        )
    capacity_slots = sum(
        index * count for histogram in buckets for index, count in enumerate(histogram)
    )
    return dict(
        routing="split_skip",
        group_action_histogram={
            name: dict(zip(map(str, WIDTHS), histogram))
            for name, histogram in zip(("no_skip", "skip"), groups)
        },
        group_size_histogram=sizes,
        group_bucket_histogram=buckets,
        group_middle_region_launches=group_exec,
        batch_middle_region_launches=launched,
        conditional_middle_region_launches=sum(launched),
        nonempty_group_replays=sum(map(sum, groups)),
        conditional_group_real_row_slots=size * steps,
        conditional_group_capacity_slots=capacity_slots,
        conditional_group_padding_slots=capacity_slots - size * steps,
        batched_no_skip_replays=groups[0][0],
        last_proposed_actions=proposed,
        **expected,
    )


def common_routing_metrics(snapshot, size, steps):
    if not {"common_step_histogram", "proposal_histogram"}.issubset(snapshot):
        raise ValueError("Common-batch receipt omits action counters")
    histogram = snapshot["histogram"]
    rows = snapshot["row_histogram"]
    common = snapshot["common_step_histogram"]
    if (
        sum(common) != steps
        or sum(snapshot["proposal_histogram"]) != size * steps
        or histogram != [count * size for count in common]
        or any(row != (common if index < size else [0] * 5) for index, row in enumerate(rows))
    ):
        raise ValueError("Common-batch actions differ across rows or decode steps")
    expected = [
        sum(count for count, action in zip(common, WIDTHS) if action < width)
        for width in WIDTHS[1:]
    ]
    expected += [steps - count for count in expected]
    if snapshot["batch_executed"] != expected or snapshot["executed"] != [n * size for n in expected]:
        raise ValueError("Common-batch main/repair regions do not complement each other")
    if any(row != (expected if index < size else [0] * 8) for index, row in enumerate(snapshot["row_executed"])):
        raise ValueError("Common-batch row executions differ")
    proposed = snapshot["last_proposed_actions"]
    real = [value for value in proposed if value >= 0]
    action = min(real) if real else -1
    applied = [action if value >= 0 else -1 for value in proposed]
    if len(real) != size or snapshot["last_applied_actions"] != applied:
        raise ValueError("Common-batch action must be the minimum valid proposal")
    return dict(
        routing="batch_min", common_step_histogram=dict(zip(map(str, WIDTHS), common)),
        proposal_histogram=dict(zip(map(str, WIDTHS), snapshot["proposal_histogram"])),
        batch_middle_region_launches=expected,
        conditional_middle_region_launches=sum(expected),
        last_proposed_actions=proposed, last_applied_actions=applied,
    )


def generate_batch(
    runner,
    tokenizer,
    prompts,
    *,
    limits,
    eos,
    choice_ids=None,
    forced=None,
    ignore_eos=False,
):
    import torch

    from sglang.bench_one_batch import prepare_synthetic_inputs_for_latency_test
    from sglang.srt.early_exit.runtime import get_runtime

    size = len(prompts)
    if not size or size != len(limits) or any(len(ids) < 2 for ids in prompts):
        raise ValueError(
            "A microbatch requires matching prompts/limits and nonempty prefill"
        )
    if forced is not None:
        if len(forced) != size or any(
            len(tokens) != limits[row] for row, tokens in enumerate(forced)
        ):
            raise ValueError(
                "Fixed replay requires one complete forced trajectory per row"
            )
        if choice_ids is not None:
            raise ValueError(
                "Choice scoring and fixed-token replay are distinct protocols"
            )
    steps = max(limits)
    raw = runner.torch_runner
    # Completed rows still receive decode slots until this reservation ends.
    if any(len(ids) + steps > raw.model_config.context_len for ids in prompts):
        raise ValueError(
            "Static-batch reservation exceeds a request context; prompts are never truncated"
        )
    reserved = sum(len(ids) - 1 + steps for ids in prompts)
    if reserved > raw.max_total_num_tokens:
        raise ValueError("Static-batch reservation exceeds the configured token pool")
    runner.clear()
    requests = prepare_synthetic_inputs_for_latency_test(
        size, len(prompts[0]) - 1, [ids[:-1] for ids in prompts]
    )
    current = torch.tensor(
        [ids[-1] for ids in prompts], dtype=torch.int64, device="cuda"
    )
    choices = (
        torch.tensor(choice_ids, dtype=torch.int64, device="cuda")
        if choice_ids is not None
        else None
    )
    runtime = get_runtime(raw.model)
    torch.cuda.synchronize()
    prefill_start, prefill_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    tick = time.perf_counter()
    prefill_start.record()
    _, _, batch = runner.extend(requests)
    prefill_end.record()
    prefill_end.synchronize()
    prefill = time.perf_counter() - tick
    if runtime and runtime.routing == "fixed_skip":
        runtime.reset_metrics()
    elif runtime:
        if runtime.lazy is not None:
            runtime.lazy.reset()
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
        if runtime.routing == "batch_min":
            runtime.common_step_histogram.zero_()
            runtime.proposal_histogram.zero_()
        runtime.force.fill_(-1)
    accounting = DecodeAccounting(
        limits, eos, ignore_eos=ignore_eos or forced is not None, choice_ids=choice_ids
    )
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    torch.cuda.synchronize()
    tick = time.perf_counter()
    start.record()
    first = None
    while not accounting.done:
        current, logits = runner.decode(current, batch)
        tokens = current.cpu().tolist()
        scores = (
            logits[:, choices].float().cpu().tolist() if choices is not None else None
        )
        elapsed = time.perf_counter() - tick
        if first is None:
            first = elapsed
        offset = accounting.steps
        accounting.advance(tokens, scores=scores, elapsed=elapsed)
        if forced is not None and not accounting.done:
            current = torch.tensor(
                [tokens[offset] for tokens in forced], dtype=torch.int64, device="cuda"
            )
    end.record()
    end.synchronize()
    decode = time.perf_counter() - tick
    computed_tokens = size * accounting.steps
    histogram = (
        runtime.histogram.cpu().tolist() if runtime else [computed_tokens, 0, 0, 0, 0]
    )
    row_histograms = (
        runtime.row_histogram[:size].cpu().tolist()
        if runtime
        else [[accounting.steps, 0, 0, 0, 0] for _ in prompts]
    )
    executions = (
        runtime.row_executed[:size].cpu().tolist()
        if runtime and runtime.routing != "fixed_skip" else None
    )
    if sum(histogram) != computed_tokens or any(
        sum(row) != accounting.steps for row in row_histograms
    ):
        raise ValueError("Routing counters omit real decode slots")
    if (
        executions is not None
        and runtime.lazy is None
        and any(
            any(row[index] + row[index + 4] != accounting.steps for index in range(4))
            for row in executions
        )
    ):
        raise ValueError("Main and repair bodies do not cover every real decode row")
    capture_size = min(
        value for value in raw.server_args.cuda_graph_bs if value >= size
    )
    routing_metrics = runtime.metrics(size, accounting.steps) if runtime and runtime.routing == "fixed_skip" else (
        dict(routing=runtime.routing, lazy_cache=runtime.lazy.metrics(batch))
        if runtime is not None and runtime.lazy is not None
        else (
            batch_routing_metrics(
                dict(
                    routing=runtime.routing,
                    histogram=histogram,
                    executed=runtime.executed.cpu().tolist(),
                    row_histogram=runtime.row_histogram.cpu().tolist(),
                    row_executed=runtime.row_executed.cpu().tolist(),
                    batch_executed=runtime.batch_executed.cpu().tolist(),
                    batched_main_replays=int(runtime.batched_main_executed.cpu()),
                    last_proposed_actions=runtime.proposed_action[:capture_size]
                    .cpu()
                    .tolist(),
                    last_applied_actions=runtime.action[:capture_size].cpu().tolist(),
                    group_executed=runtime.group_executed.cpu().tolist(),
                    group_action_histogram=runtime.group_action_histogram.cpu().tolist(),
                    group_size_histogram=runtime.group_size_histogram.cpu().tolist(),
                    bucket_histogram=runtime.bucket_histogram.cpu().tolist(),
                    last_group_actions=runtime.group_actions.cpu().tolist(),
                    last_group_counts=runtime.group_counts.cpu().tolist(),
                    last_group_indices=runtime.group_indices[:, :capture_size]
                    .cpu()
                    .tolist(),
                    last_group_capacities=runtime.group_capacities.cpu().tolist(),
                    common_step_histogram=(runtime.common_step_histogram.cpu().tolist() if runtime.routing == "batch_min" else None),
                    proposal_histogram=(runtime.proposal_histogram.cpu().tolist() if runtime.routing == "batch_min" else None),
                ),
                size,
                accounting.steps,
            )
            if runtime
            else dict(
                routing="full",
                group_action_histogram={
                    name: dict.fromkeys(map(str, WIDTHS), 0)
                    for name in ("no_skip", "skip")
                },
                group_size_histogram=[[0] * 9 for _ in range(2)],
                group_bucket_histogram=[[0] * 9 for _ in range(2)],
                group_middle_region_launches=[[0] * 8 for _ in range(2)],
                batch_middle_region_launches=[0] * 8,
                conditional_middle_region_launches=0,
                nonempty_group_replays=0,
                conditional_group_real_row_slots=0,
                conditional_group_capacity_slots=0,
                conditional_group_padding_slots=0,
                batched_no_skip_replays=0,
            )
        )
    )
    records = []
    for row, generated in enumerate(accounting.generated):
        choice = accounting.choices[row]
        text = (
            "ABCD"[choice]
            if choice is not None
            else tokenizer.decode(generated, skip_special_tokens=True)
        )
        records.append(
            dict(
                text=text,
                generated_ids=generated,
                generated_tokens=len(generated),
                prompt_tokens=len(prompts[row]),
                finish_reason=accounting.reasons[row],
                truncated=forced is None
                and not ignore_eos
                and choice is None
                and accounting.reasons[row] == "limit"
                and generated[-1] not in eos,
                logical_completion_seconds=prefill + accounting.completion_seconds[row],
                time_to_first_token_seconds=prefill + first,
                completed_row_waste_tokens=accounting.steps - len(generated),
                choice_scores=accounting.choice_scores[row],
                routing_histogram_including_completed_row_waste={
                    str(width): count
                    for width, count in zip(WIDTHS, row_histograms[row])
                },
            )
        )
    measured = dict(
        requests=size,
        prompt_tokens=sum(map(len, prompts)),
        decode_steps=accounting.steps,
        effective_generated_tokens=accounting.effective_tokens,
        completed_row_waste_tokens=accounting.completed_row_waste,
        computed_real_row_tokens=computed_tokens,
        captured_batch_size=capture_size,
        graph_padding_decode_slots=(capture_size - size) * accounting.steps,
        **routing_metrics,
        prefill_seconds=prefill,
        prefill_cuda_ms=prefill_start.elapsed_time(prefill_end),
        decode_seconds=decode,
        decode_cuda_ms=start.elapsed_time(end),
        total_seconds=prefill + decode,
        time_to_first_token_seconds=prefill + first,
        routing_histogram_including_completed_row_waste=dict(
            zip(map(str, WIDTHS), histogram)
        ),
        reserved_token_slots=reserved,
    )
    return records, measured


def selected_questions(args):
    p = protocol()
    tests, development = p.read_benchmarks(args.data)
    selected = p.select_rows(tests, workers=1, limit_per_task=args.examples_per_task)
    if dict(Counter(row["benchmark"] for row in selected)) != {
        task: args.examples_per_task for task in TASKS
    }:
        raise ValueError("Requested pilot exceeds a benchmark test set")
    replay = p.matched_indices(selected, args.replay_examples_per_task)
    return selected, development, replay


def execution_identity(args):
    p = protocol()
    identity = p.execution_identity(args)
    identity.update(
        batch_evaluator_sha256=p.sha(__file__),
        batch_metadata_sha256=p.sha(Path(__file__).with_name("batch.py")),
        batch_grouping_sha256=p.sha(Path(__file__).with_name("split.py")),
        batch_masked_kv_sha256=p.sha(Path(__file__).with_name("masked_kv.py")),
        protocol_evaluator_sha256=p.sha(Path(__file__).with_name("evaluate.py")),
    )
    return identity


def output_paths(out, mode):
    return (
        out / f"batch-quality-{mode}.jsonl",
        out / f"batch-timing-{mode}.jsonl",
        out / f"batch-worker-{mode}.json",
    )


def worker(args):
    import torch

    from sglang.srt.early_exit.runtime import get_runtime
    from sglang.srt.early_exit.smoke import load_runner

    p = protocol()
    torch.set_num_threads(4)
    if torch.cuda.device_count() != 1:
        raise ValueError(
            "Paired batch pilot requires exactly one allocated visible GPU"
        )
    torch.cuda.set_device(0)
    out = Path(os.environ["RUN_DIR"])
    selected, development, replay_indices = selected_questions(args)
    p.verify_model(args)
    if args.worker_mode == "full" and os.environ.get("SGLANG_EARLY_EXIT_CHECKPOINT"):
        raise ValueError("Full baseline must start without the classifier environment")
    runner, tokenizer = load_runner(
        args.model,
        max_batch_size=max(args.batch_sizes),
        max_total_tokens=args.max_total_tokens,
    )
    raw = runner.torch_runner
    original_forward = raw.forward
    graph_counts = dict(decode_steps=0, decode_real_row_slots=0)

    def checked_forward(batch, *positional, **keywords):
        result = original_forward(batch, *positional, **keywords)
        if batch.forward_mode.is_decode():
            if not result.can_run_graph:
                raise ValueError(
                    "Every measured batch decode must replay a captured CUDA graph"
                )
            graph_counts["decode_steps"] += 1
            graph_counts["decode_real_row_slots"] += batch.batch_size
        return result

    raw.forward = checked_forward
    runtime = get_runtime(raw.model)
    if (runtime is not None) != (args.worker_mode == "early_exit"):
        raise ValueError("Worker has the wrong full/early-exit model path")
    if runtime is not None and (
        runtime.mode != args.cache_policy
        or runtime.routing != "split_skip"
        or runtime.checkpoint_sha256 != p.sha(args.classifier)
        or not runtime.converted
    ):
        raise ValueError(
            "Early-exit runtime identity, cache policy or conditional graphs differ"
        )
    option_ids = [
        tokenizer.encode(" " + label, add_special_tokens=False) for label in "ABCD"
    ]
    if any(len(value) != 1 for value in option_ids):
        raise ValueError("Each leading-space MMLU answer must be one token")
    option_ids = [value[0] for value in option_ids]
    eos = eos_tokens(tokenizer)
    inputs = {
        row["evaluation_index"]: p.prompt(tokenizer, row, development)
        for row in selected
    }
    trajectory_path = out / "batch-reference-trajectories.json"
    with torch.inference_mode():
        warm_prompts = [
            inputs[selected[index % len(selected)]["evaluation_index"]]
            for index in range(max(args.batch_sizes))
        ]
        for _ in range(2):
            generate_batch(
                runner,
                tokenizer,
                warm_prompts,
                limits=[16] * len(warm_prompts),
                eos=eos,
            )
        if args.worker_mode == "full":
            trajectories = {}
            for row in selected:
                index = row["evaluation_index"]
                if index in replay_indices:
                    generated, _ = generate_batch(
                        runner,
                        tokenizer,
                        [inputs[index]],
                        limits=[args.matched_tokens],
                        eos=eos,
                        ignore_eos=True,
                    )
                    trajectories[str(index)] = dict(
                        forced_ids=generated[0]["generated_ids"],
                        prompt_sha256=fingerprint(inputs[index]),
                    )
            save(
                trajectory_path,
                dict(
                    protocol="full-model batch-one greedy continuation ignoring EOS",
                    matched_tokens=args.matched_tokens,
                    trajectories=trajectories,
                ),
            )
        reference = json.loads(trajectory_path.read_text())
        trajectories = reference["trajectories"]
        if (
            reference["matched_tokens"] != args.matched_tokens
            or set(map(int, trajectories)) != replay_indices
        ):
            raise ValueError("Reference replay questions or lengths differ")
        quality_path, timing_path, receipt_path = output_paths(out, args.worker_mode)
        quality_count, batch_count = 0, 0
        with quality_path.open("x") as quality, timing_path.open("x") as timing:
            for size in args.batch_sizes:
                for kind, rows in (
                    ("quality", selected),
                    (
                        "fixed_replay",
                        [
                            row
                            for row in selected
                            if row["evaluation_index"] in replay_indices
                        ],
                    ),
                ):
                    for batch_index, rows_in_batch in enumerate(
                        question_batches(rows, size)
                    ):
                        task = rows_in_batch[0]["benchmark"]
                        ids = [inputs[row["evaluation_index"]] for row in rows_in_batch]
                        forced = None
                        if kind == "fixed_replay":
                            forced = []
                            for row, tokens in zip(rows_in_batch, ids):
                                entry = trajectories[str(row["evaluation_index"])]
                                if (
                                    entry["prompt_sha256"] != fingerprint(tokens)
                                    or len(entry["forced_ids"]) != args.matched_tokens
                                ):
                                    raise ValueError(
                                        "Fixed replay prompt or input trajectory differs"
                                    )
                                forced.append(entry["forced_ids"])
                        limits = (
                            [maximum(row) for row in rows_in_batch]
                            if kind == "quality"
                            else [args.matched_tokens] * len(rows_in_batch)
                        )
                        choices = (
                            option_ids if task == "mmlu" and kind == "quality" else None
                        )
                        results, measured = generate_batch(
                            runner,
                            tokenizer,
                            ids,
                            limits=limits,
                            eos=eos,
                            choice_ids=choices,
                            forced=forced,
                        )
                        batch_id = f"bs{size}-{kind}-{batch_index}"
                        if kind == "quality":
                            for row, tokens, result in zip(rows_in_batch, ids, results):
                                result.update(p.grade(row, result["text"]))
                                record = dict(
                                    **metadata(row, tokens),
                                    requested_batch_size=size,
                                    batch_id=batch_id,
                                    result=result,
                                )
                                quality.write(
                                    json.dumps(
                                        record, ensure_ascii=False, allow_nan=False
                                    )
                                    + "\n"
                                )
                                quality_count += 1
                            quality.flush()
                        measured.update(
                            batch_id=batch_id,
                            mode=args.worker_mode,
                            kind=kind,
                            benchmark=task,
                            requested_batch_size=size,
                            evaluation_indices=[
                                row["evaluation_index"] for row in rows_in_batch
                            ],
                            prompt_sha256=[fingerprint(value) for value in ids],
                            forced_trajectory_sha256=(
                                fingerprint(forced) if forced is not None else None
                            ),
                        )
                        timing.write(
                            json.dumps(measured, ensure_ascii=False, allow_nan=False)
                            + "\n"
                        )
                        timing.flush()
                        batch_count += 1
                        print(
                            "PB_BATCH_EVAL "
                            + json.dumps(
                                dict(
                                    mode=args.worker_mode,
                                    batch_size=size,
                                    kind=kind,
                                    batch=batch_index,
                                    benchmark=task,
                                    requests=len(rows_in_batch),
                                    effective_tokens=measured[
                                        "effective_generated_tokens"
                                    ],
                                    completed_row_waste=measured[
                                        "completed_row_waste_tokens"
                                    ],
                                )
                            ),
                            flush=True,
                        )
    receipt = dict(
        mode=args.worker_mode,
        quality_count=quality_count,
        batch_count=batch_count,
        quality_sha256=p.sha(quality_path),
        timing_sha256=p.sha(timing_path),
        trajectory_sha256=p.sha(trajectory_path),
        identity=execution_identity(args),
        gpu=torch.cuda.get_device_name(0),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        runtime_present=runtime is not None,
        routing=runtime.routing if runtime else "full",
        cache_policy=runtime.mode if runtime else None,
        graph_execution=graph_counts,
        threshold=runtime.threshold if runtime else None,
        classifier_format=runtime.checkpoint_format if runtime else None,
        converted_regions=runtime.converted if runtime else None,
        server_args={
            key: getattr(raw.server_args, key)
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
        memory=dict(
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_reserved_bytes=torch.cuda.memory_reserved(),
        ),
    )
    save(receipt_path, receipt)
    print(
        "PB_BATCH_EVAL_WORKER_PASS "
        + json.dumps(
            dict(
                mode=args.worker_mode, quality_count=quality_count, batches=batch_count
            )
        ),
        flush=True,
    )


def timing_summary(rows):
    totals = {
        key: sum(row[key] for row in rows)
        for key in (
            "requests",
            "prompt_tokens",
            "effective_generated_tokens",
            "completed_row_waste_tokens",
            "computed_real_row_tokens",
            "graph_padding_decode_slots",
            "conditional_middle_region_launches",
            "nonempty_group_replays",
            "conditional_group_real_row_slots",
            "conditional_group_capacity_slots",
            "conditional_group_padding_slots",
            "batched_no_skip_replays",
            "prefill_seconds",
            "prefill_cuda_ms",
            "decode_seconds",
            "decode_cuda_ms",
            "total_seconds",
        )
    }
    if not rows or totals["decode_seconds"] <= 0 or totals["decode_cuda_ms"] <= 0:
        raise ValueError("A timing summary requires completed measured batches")
    histogram = {
        str(width): sum(
            row["routing_histogram_including_completed_row_waste"][str(width)]
            for row in rows
        )
        for width in WIDTHS
    }
    return dict(
        batches=len(rows),
        **totals,
        routing_histogram_including_completed_row_waste=histogram,
        group_action_histogram={
            name: {
                str(width): sum(
                    row["group_action_histogram"][name][str(width)] for row in rows
                )
                for width in WIDTHS
            }
            for name in ("no_skip", "skip")
        },
        group_size_histogram=[
            [
                sum(row["group_size_histogram"][group][count] for row in rows)
                for count in range(9)
            ]
            for group in range(2)
        ],
        group_bucket_histogram=[
            [
                sum(row["group_bucket_histogram"][group][count] for row in rows)
                for count in range(9)
            ]
            for group in range(2)
        ],
        mean_main_skipped_layers_including_completed_row_waste=sum(
            width * histogram[str(width)] for width in WIDTHS
        )
        / totals["computed_real_row_tokens"],
        effective_decode_tokens_per_second=totals["effective_generated_tokens"]
        / totals["decode_seconds"],
        effective_decode_cuda_tokens_per_second=1000
        * totals["effective_generated_tokens"]
        / totals["decode_cuda_ms"],
        requests_per_second=totals["requests"] / totals["total_seconds"],
        real_row_token_efficiency=totals["effective_generated_tokens"]
        / totals["computed_real_row_tokens"],
    )


def aggregate(args, out):
    p = protocol()
    selected, _, replay_indices = selected_questions(args)
    indices = {row["evaluation_index"] for row in selected}
    quality, timings, receipts = {}, {}, {}
    for mode in ("full", "early_exit"):
        quality_path, timing_path, receipt_path = output_paths(out, mode)
        receipt = json.loads(receipt_path.read_text())
        if receipt["quality_sha256"] != p.sha(quality_path) or receipt[
            "timing_sha256"
        ] != p.sha(timing_path):
            raise ValueError("Batch evaluator artifact differs from its receipt")
        quality[mode], timings[mode] = p.read_jsonl(quality_path), p.read_jsonl(
            timing_path
        )
        if receipt["quality_count"] != len(quality[mode]) or receipt[
            "batch_count"
        ] != len(timings[mode]):
            raise ValueError("Batch evaluator receipt counts differ")
        if receipt["runtime_present"] != (mode == "early_exit"):
            raise ValueError("Full/early-exit receipt paths differ")
        if receipt["routing"] != ("split_skip" if mode == "early_exit" else "full"):
            raise ValueError("Pilot receipt belongs to a different routing policy")
        if mode == "early_exit":
            expected_regions = [
                12 * len([cap for cap in (1, 2, 4, 8) if cap <= size])
                for size in receipt["server_args"]["cuda_graph_bs"]
            ]
            if sorted(map(len, receipt["converted_regions"])) != sorted(
                expected_regions
            ):
                raise ValueError(
                    "Split runtime IF regions differ from captured capacities"
                )
        if any(row["routing"] != receipt["routing"] for row in timings[mode]):
            raise ValueError("Batch timing routing differs from the worker receipt")
        if receipt["graph_execution"]["decode_real_row_slots"] < sum(
            row["computed_real_row_tokens"] for row in timings[mode]
        ):
            raise ValueError("Worker graph receipt omits measured real decode slots")
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
        if receipts["full"][key] != receipts["early_exit"][key]:
            raise ValueError("Paired execution identity differs: " + key)
    for mode in quality:
        expected = {(size, index) for size in args.batch_sizes for index in indices}
        actual = [
            (row["requested_batch_size"], row["evaluation_index"])
            for row in quality[mode]
        ]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(
                "Quality predictions do not cover each selected question exactly once per batch size"
            )
    baseline = {
        row["evaluation_index"]: row
        for row in quality["full"]
        if row["requested_batch_size"] == 1
    }
    results = {}
    for size in args.batch_sizes:
        case = {}
        for task in TASKS:
            modes = {
                mode: {
                    row["evaluation_index"]: row
                    for row in values
                    if row["requested_batch_size"] == size and row["benchmark"] == task
                }
                for mode, values in quality.items()
            }
            if modes["full"].keys() != modes["early_exit"].keys():
                raise ValueError("Paired question identities differ")
            for index in modes["full"]:
                for key in (
                    "prompt_sha256",
                    "benchmark",
                    "source_file",
                    "index",
                    "subject",
                    "level",
                    "batch_id",
                ):
                    if modes["full"][index][key] != modes["early_exit"][index][key]:
                        raise ValueError("Paired question metadata differs: " + key)
            quality_metrics = {}
            for mode, rows in modes.items():
                values = list(rows.values())
                quality_metrics[mode] = dict(
                    count=len(values),
                    correct=sum(row["result"]["correct"] for row in values),
                    accuracy=sum(row["result"]["correct"] for row in values)
                    / len(values),
                    generated_tokens=sum(
                        row["result"]["generated_tokens"] for row in values
                    ),
                    truncated=sum(row["result"]["truncated"] for row in values),
                    token_trajectory_changes_vs_full_batch1=sum(
                        row["result"]["generated_ids"]
                        != baseline[index]["result"]["generated_ids"]
                        for index, row in rows.items()
                    ),
                    prediction_changes_vs_full_batch1=sum(
                        row["result"]["prediction"]
                        != baseline[index]["result"]["prediction"]
                        for index, row in rows.items()
                    ),
                )
            quality_metrics.update(
                accuracy_delta=quality_metrics["early_exit"]["accuracy"]
                - quality_metrics["full"]["accuracy"],
                lost_correct=sum(
                    row["result"]["correct"]
                    and not modes["early_exit"][index]["result"]["correct"]
                    for index, row in modes["full"].items()
                ),
                gained_correct=sum(
                    not row["result"]["correct"]
                    and modes["early_exit"][index]["result"]["correct"]
                    for index, row in modes["full"].items()
                ),
            )
            task_metrics = dict(quality=quality_metrics)
            for kind in ("quality", "fixed_replay"):
                if kind == "fixed_replay" and task == "mmlu":
                    continue
                grouped = {
                    mode: [
                        row
                        for row in values
                        if row["requested_batch_size"] == size
                        and row["benchmark"] == task
                        and row["kind"] == kind
                    ]
                    for mode, values in timings.items()
                }
                mapped = {
                    mode: {row["batch_id"]: row for row in values}
                    for mode, values in grouped.items()
                }
                if mapped["full"].keys() != mapped["early_exit"].keys():
                    raise ValueError("Paired timing batches differ")
                covered = [
                    index
                    for row in grouped["full"]
                    for index in row["evaluation_indices"]
                ]
                wanted = (
                    set(modes["full"])
                    if kind == "quality"
                    else set(modes["full"]) & replay_indices
                )
                if len(covered) != len(wanted) or set(covered) != wanted:
                    raise ValueError(
                        "Timing batches omit or duplicate selected questions"
                    )
                for batch_id, left in mapped["full"].items():
                    right = mapped["early_exit"][batch_id]
                    for key in (
                        "evaluation_indices",
                        "prompt_sha256",
                        "forced_trajectory_sha256",
                        "requests",
                    ):
                        if left[key] != right[key]:
                            raise ValueError("Paired timing inputs differ: " + key)
                    if kind == "fixed_replay" and (
                        left["effective_generated_tokens"]
                        != left["requests"] * args.matched_tokens
                        or right["effective_generated_tokens"]
                        != left["effective_generated_tokens"]
                    ):
                        raise ValueError(
                            "Fixed replay must decode identical request/token counts"
                        )
                summary = {
                    mode: timing_summary(values) for mode, values in grouped.items()
                }
                summary.update(
                    end_to_end_speedup=summary["full"]["total_seconds"]
                    / summary["early_exit"]["total_seconds"],
                    effective_decode_throughput_speedup=summary["early_exit"][
                        "effective_decode_tokens_per_second"
                    ]
                    / summary["full"]["effective_decode_tokens_per_second"],
                    decode_cuda_speedup=summary["full"]["decode_cuda_ms"]
                    / summary["early_exit"]["decode_cuda_ms"],
                )
                task_metrics[
                    "generation_timing" if kind == "quality" else "fixed_token_replay"
                ] = summary
            case[task] = task_metrics
        results[str(size)] = case
    for size, case in results.items():
        for task, metrics in case.items():
            for kind in ("generation_timing", "fixed_token_replay"):
                if kind not in metrics:
                    continue
                for mode in ("full", "early_exit"):
                    measured = metrics[kind][mode]
                    baseline_timing = results["1"][task][kind][mode]
                    measured["effective_decode_throughput_vs_batch1"] = (
                        measured["effective_decode_tokens_per_second"]
                        / baseline_timing["effective_decode_tokens_per_second"]
                    )
                    measured["requests_per_second_vs_batch1"] = (
                        measured["requests_per_second"]
                        / baseline_timing["requests_per_second"]
                    )
    summary = dict(
        passed=True,
        complete_benchmarks=False,
        batch_sizes=args.batch_sizes,
        selected_counts=dict(Counter(row["benchmark"] for row in selected)),
        results=results,
        worker_receipts=receipts,
        timing_scope="One elapsed time per static microbatch, including output token/choice readback; excludes model load, graph capture, warmup, tokenization, HTTP and scheduler. Completed real request rows keep decoding and are reported as waste. Per-request logical_completion_seconds marks EOS/limit observation inside the decode loop; results are flushed after the microbatch and these latencies are never summed as device time.",
        replay_protocol="Every requested batch size and both modes replay the same stored batch-one full-model greedy trajectory for each selected GSM8K/MATH question, ignoring EOS to reach the fixed length.",
        quality_protocol="Pinned evaluator prompts and grading; greedy non-thinking chat for GSM8K/MATH with 2048/4096 limits, five-shot MMLU choice scoring. This selected pilot complements the independent full benchmark run.",
        runtime_scope="split_skip: GPU partition puts zero proposals in the no-skip group and positive proposals in the skip group. The skip group shares its minimum positive width. Device counts select each group's smallest 1/2/4/8 capacity; only that bucket executes. Packed middle capacity padding, group launches and real-row actions are reported separately. Original padding is excluded from selection and counters.",
        limits="Small selected pilot, fixed full-before-early-exit order, static completion waste and batch-dependent BF16 numerics. Throughput improvements require measured results; batch>1 support alone establishes no speedup.",
    )
    save(out / "batch-evaluation-summary.json", summary)
    print(
        "PB_BATCH_EVAL_PASS "
        + json.dumps(
            dict(
                batch_sizes=args.batch_sizes,
                counts=summary["selected_counts"],
                summary_sha256=p.sha(out / "batch-evaluation-summary.json"),
            )
        ),
        flush=True,
    )


def main(args):
    out = Path(os.environ["RUN_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.yaml").exists() or list(out.glob("batch-quality-*.jsonl")):
        raise ValueError("Use a new immutable managed RUN_DIR")
    selected, _, replay = selected_questions(args)
    save(
        out / "config.yaml",
        dict(
            model=str(args.model),
            data=str(args.data),
            classifier=str(args.classifier),
            identity=execution_identity(args),
            batch_sizes=args.batch_sizes,
            examples_per_task=args.examples_per_task,
            replay_examples_per_task=args.replay_examples_per_task,
            matched_tokens=args.matched_tokens,
            selected_indices=[row["evaluation_index"] for row in selected],
            replay_indices=sorted(replay),
            max_total_tokens=args.max_total_tokens,
            cache_policy=args.cache_policy,
            routing="split_skip",
            tp=1,
            pp=1,
            completion_policy="Static microbatch; finished request rows keep computing and are counted as waste",
        ),
    )
    for mode in ("full", "early_exit"):
        environment = dict(os.environ)
        for name in (
            "SGLANG_EARLY_EXIT_CHECKPOINT",
            "SGLANG_EARLY_EXIT_CACHE",
            "SGLANG_EARLY_EXIT_FORCE",
        ):
            environment.pop(name, None)
        if mode == "early_exit":
            environment.update(
                SGLANG_EARLY_EXIT_CHECKPOINT=str(args.classifier),
                SGLANG_EARLY_EXIT_CACHE=args.cache_policy,
                SGLANG_EARLY_EXIT_ROUTING="split_skip",
                SGLANG_EARLY_EXIT_FORCE="-1",
            )
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "sglang.srt.early_exit.batch_evaluate",
                *sys.argv[1:],
                "--worker-mode",
                mode,
            ],
            env=environment,
            check=True,
        )
    aggregate(args, out)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument(
        "--weight-inventory",
        type=Path,
        default=protocol().source_root()
        / "experiments/perfectblend/references/model-weight-inventory.json",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--examples-per-task", type=int, default=16)
    parser.add_argument("--replay-examples-per-task", type=int, default=8)
    parser.add_argument("--matched-tokens", type=int, default=128)
    parser.add_argument("--max-total-tokens", type=int)
    parser.add_argument("--cache-policy", choices=("async", "serial"), default="async")
    parser.add_argument(
        "--worker-mode", choices=("full", "early_exit"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    args.batch_sizes = sorted(set(args.batch_sizes))
    if (
        not args.batch_sizes
        or args.batch_sizes[0] != 1
        or any(size not in (1, 2, 4, 8) for size in args.batch_sizes)
    ):
        parser.error("batch-sizes must include 1 and use only 1, 2, 4, 8")
    if (
        not 1 <= args.replay_examples_per_task <= args.examples_per_task
        or args.matched_tokens < 1
    ):
        parser.error(
            "Require positive replay count <= examples-per-task and positive matched-tokens"
        )
    if args.max_total_tokens is None:
        args.max_total_tokens = max(18000, max(args.batch_sizes) * 16384)
    if args.max_total_tokens < 1:
        parser.error("max-total-tokens must be positive")
    for name in ("model", "data", "classifier", "weight_inventory"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


if __name__ == "__main__":
    arguments = parse_args()
    (worker if arguments.worker_mode else main)(arguments)
