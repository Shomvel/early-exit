# SPDX-License-Identifier: Apache-2.0
"""Real SGLang ModelRunner differential test for asynchronous cache repair."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from sglang.srt.early_exit.runtime import get_runtime


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_runner(model, max_batch_size=1, max_total_tokens=18000):
    from sglang.bench_one_batch import load_model
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.server_args import PortArgs, ServerArgs

    args = ServerArgs(
        model_path=str(model),
        dtype="bfloat16",
        tp_size=1,
        attention_backend="triton",
        linear_attn_decode_backend="triton",
        linear_attn_prefill_backend="triton",
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        max_running_requests=max_batch_size,
        cuda_graph_bs=[size for size in (1, 2, 4, 8) if size <= max_batch_size],
        cuda_graph_max_bs=max_batch_size,
        context_length=16384,
        mem_fraction_static=0.75,
        max_total_tokens=max_total_tokens,
        max_mamba_cache_size=max(4, max_batch_size + 2),
        chunked_prefill_size=-1,
        log_level="info",
    )
    _set_envs_and_config(args)
    initialize_moe_config(args)
    initialize_fp8_gemm_config(args)
    initialize_fp4_gemm_config(args)
    runner, tokenizer = load_model(args, PortArgs.init_new(args), 0, 0)
    assert runner.torch_runner.graph_runner is not None
    return runner, tokenizer


@torch.inference_mode()
def run_sequence(
    runner, ids, *, forced_tokens=None, actions=None, steps=32, profile=None
):
    from sglang.bench_one_batch import prepare_synthetic_inputs_for_latency_test

    runner.clear()
    raw = runner.torch_runner
    runtime = get_runtime(raw.model)
    if runtime:
        runtime.histogram.zero_()
        runtime.executed.zero_()
    reqs = prepare_synthetic_inputs_for_latency_test(1, len(ids) - 1, [ids[:-1]])
    torch.cuda.synchronize()
    tick = time.perf_counter()
    _, _, batch = runner.extend(reqs)
    torch.cuda.synchronize()
    prefill = time.perf_counter() - tick
    current = torch.tensor([ids[-1]], dtype=torch.int64, device="cuda")
    generated, logits = [], []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    profiler = None
    if profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        profiler.start()
    start.record()
    tick = time.perf_counter()
    for step in range(steps):
        if runtime and actions is not None:
            runtime.force.fill_(actions[step % len(actions)])
        current, scores = runner.decode(current, batch)
        # Output readback is the usual token generation boundary. It never
        # supplies a jump decision to Python or chooses an executed layer.
        generated.append(int(current[0]))
        logits.append(scores.detach().clone())
        if forced_tokens is not None:
            current = torch.tensor(
                [forced_tokens[step]], dtype=torch.int64, device="cuda"
            )
    end.record()
    end.synchronize()
    wall = time.perf_counter() - tick
    if profiler:
        profiler.stop()
        profiler.export_chrome_trace(str(profile))
    slots = raw.req_to_token_pool.req_to_token[
        reqs[0].req_pool_idx, : len(ids) - 1 + steps
    ].long()
    kv = raw.token_to_kv_pool
    state = {}
    for layer in range(32, 43):
        if layer in raw.attn_backend.full_attn_layers:
            state[f"k{layer}"] = kv.get_key_buffer(layer)[slots].cpu()
            state[f"v{layer}"] = kv.get_value_buffer(layer)[slots].cpu()
        else:
            cache = raw.req_to_token_pool.mamba2_layer_cache(layer)
            index = reqs[0].mamba_pool_idx
            state[f"conv{layer}"] = cache.conv[0][index].cpu()
            state[f"ssm{layer}"] = cache.temporal[index].cpu()
    result = dict(
        generated=generated,
        prefill_seconds=prefill,
        decode_seconds=wall,
        decode_cuda_ms=start.elapsed_time(end),
        steps=steps,
        histogram=runtime.histogram.cpu().tolist() if runtime else None,
        executed=runtime.executed.cpu().tolist() if runtime else None,
        converted=runtime.converted if runtime else None,
    )
    return result, torch.cat(logits).cpu(), state


def worker(args):
    out = Path(os.environ["RUN_DIR"])
    runner, tokenizer = load_runner(args.model)
    prompt = tokenizer.apply_chat_template(
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
    forced = None
    if args.worker not in ("full", "diagnostic"):
        forced = json.loads((out / "full.json").read_text())["reference"]["generated"]
    if args.worker == "diagnostic":
        result, _, _ = run_sequence(runner, prompt, actions=[0], steps=1)
        print("PB_SGLANG_DIAGNOSTIC " + json.dumps(result), flush=True)
        return
    # Exercise prefill, sampling and decode before starting any measurements.
    run_sequence(runner, prompt, forced_tokens=forced, actions=[0], steps=32)
    reference, reference_logits, reference_state = run_sequence(
        runner, prompt, forced_tokens=forced, actions=[0], steps=32
    )
    results = dict(reference=reference)
    tensors = dict(reference_logits=reference_logits, reference_state=reference_state)
    if args.worker != "full":
        mixed, mixed_logits, mixed_state = run_sequence(
            runner,
            prompt,
            forced_tokens=forced,
            actions=[0, 11, 5, 0, 7, 6, 11, 0],
            steps=32,
        )
        results["mixed"] = mixed
        tensors.update(mixed_logits=mixed_logits, mixed_state=mixed_state)
        # A separate profiler pass establishes that the captured repair and
        # suffix kernels occupy different streams; exclude it from timing.
        profiled, _, _ = run_sequence(
            runner,
            prompt,
            forced_tokens=forced,
            actions=[11],
            steps=8,
            profile=out / (args.worker + "-trace.json"),
        )
        results["profiled"] = profiled
        runtime = get_runtime(runner.torch_runner.model)
        runtime.force.fill_(-1)
        learned, _, _ = run_sequence(runner, prompt, steps=32)
        results["learned"] = learned
    save(out / (args.worker + ".json"), results)
    torch.save(tensors, out / (args.worker + "-tensors.pt"))
    print(
        "PB_SGLANG_WORKER " + json.dumps(dict(mode=args.worker, **results)), flush=True
    )


def compare_tensors(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        values = [compare_tensors(left[k], right[k]) for k in left]
        return dict(
            exact=all(x["exact"] for x in values),
            max_abs=max(x["max_abs"] for x in values),
        )
    return dict(
        exact=torch.equal(left, right),
        max_abs=float((left.float() - right.float()).abs().max()),
    )


def trace_overlap(path):
    trace = json.loads(Path(path).read_text())
    points, streams = [], set()
    for event in trace["traceEvents"]:
        stream = event.get("args", {}).get("stream")
        if event.get("cat") == "kernel" and stream is not None:
            streams.add(stream)
            points.append((event["ts"], 1, stream))
            points.append((event["ts"] + event["dur"], -1, stream))
    active, overlap, last = {}, 0.0, None
    for timestamp, delta, stream in sorted(points):
        if last is not None and sum(count > 0 for count in active.values()) > 1:
            overlap += timestamp - last
        active[stream] = active.get(stream, 0) + delta
        last = timestamp
    return dict(
        kernel_streams=sorted(streams),
        cross_stream_overlap_microseconds=overlap,
        trace_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )


def main(args):
    out = Path(os.environ["RUN_DIR"])
    save(
        out / "config.yaml",
        dict(
            model=str(args.model),
            classifier=str(args.classifier),
            source=32,
            widths=[5, 6, 7, 11],
            batch_size=1,
            tp_size=1,
            dtype="bfloat16",
            classifier_dtype="float32",
            attention_backend="triton",
            modes=["full", "serial", "async"],
            steps=32,
            diagnose=args.diagnose,
        ),
    )
    from sglang.srt.early_exit.kernel_probe import main as kernel_probe
    from sglang.srt.early_exit.probe import main as probe

    probe()
    if args.diagnose:
        os.environ["SGLANG_EARLY_EXIT_SANITIZER"] = "1"
    diagnostics = kernel_probe()
    for case, result in diagnostics["cases"].items():
        if not result["passed"]:
            print("PB_KERNEL_FAILURE " + json.dumps(result), flush=True)
            for suffix in ("-sanitizer.log", ".stderr"):
                path = out / (case + suffix)
                if path.exists():
                    print(path.name + "\n" + path.read_text()[-16000:], flush=True)
    if args.diagnose:
        sanitizer = shutil.which("compute-sanitizer")
        if sanitizer is None:
            raise RuntimeError("compute-sanitizer is required for --diagnose")
        env = dict(os.environ)
        env["SGLANG_EARLY_EXIT_CHECKPOINT"] = str(args.classifier)
        env["SGLANG_EARLY_EXIT_CACHE"] = "serial"
        env["SGLANG_EARLY_EXIT_DEBUG_DOT"] = str(out / "conditional-graph.dot")
        subprocess.run(
            [
                sanitizer,
                "--tool",
                "memcheck",
                "--target-processes",
                "all",
                "--error-exitcode",
                "91",
                sys.executable,
                "-u",
                "-m",
                __spec__.name,
                "--model",
                str(args.model),
                "--worker",
                "diagnostic",
            ],
            env=env,
            check=True,
        )
        assert diagnostics["passed"], diagnostics
        return
    assert diagnostics["passed"], diagnostics
    for mode in ("full", "serial", "async"):
        env = dict(os.environ)
        if mode == "full":
            env.pop("SGLANG_EARLY_EXIT_CHECKPOINT", None)
        else:
            env["SGLANG_EARLY_EXIT_CHECKPOINT"] = str(args.classifier)
            env["SGLANG_EARLY_EXIT_CACHE"] = mode
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                __spec__.name,
                "--model",
                str(args.model),
                "--worker",
                mode,
            ],
            env=env,
            check=True,
        )
    records = {
        mode: json.loads((out / (mode + ".json")).read_text())
        for mode in ("full", "serial", "async")
    }
    data = {
        mode: torch.load(
            out / (mode + "-tensors.pt"), map_location="cpu", weights_only=True
        )
        for mode in records
    }
    comparisons = {}
    for mode in ("serial", "async"):
        comparisons[mode + "_no_skip_logits"] = compare_tensors(
            data["full"]["reference_logits"], data[mode]["reference_logits"]
        )
        assert (
            records["full"]["reference"]["generated"]
            == records[mode]["reference"]["generated"]
        )
        assert comparisons[mode + "_no_skip_logits"]["max_abs"] < 0.02
    for key in ("mixed_logits", "mixed_state"):
        comparisons["serial_async_" + key] = compare_tensors(
            data["serial"][key], data["async"][key]
        )
        assert comparisons["serial_async_" + key]["exact"], comparisons
    for mode in ("serial", "async"):
        executed = records[mode]["mixed"]["executed"]
        assert all(executed[i] + executed[i + 4] == 32 for i in range(4)), executed
    summary = dict(
        passed=True,
        comparisons=comparisons,
        records=records,
        profiles={
            mode: trace_overlap(out / (mode + "-trace.json"))
            for mode in ("serial", "async")
        },
        async_vs_serial_cuda_speedup=records["serial"]["mixed"]["decode_cuda_ms"]
        / records["async"]["mixed"]["decode_cuda_ms"],
        classifier_sha256=hashlib.sha256(args.classifier.read_bytes()).hexdigest(),
    )
    save(out / "smoke-summary.json", summary)
    print("PB_SGLANG_SMOKE_PASS " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--worker", choices=("full", "serial", "async", "diagnostic"))
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()
    (worker if args.worker else main)(args)
