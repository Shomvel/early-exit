# SPDX-License-Identifier: Apache-2.0
"""Isolated real-operator CUDA conditional-graph regression probes, without a model.

Run with RUN_DIR set. Each case starts a new process so a CUDA illegal
instruction cannot poison the remaining cases. Progress is flushed before
each CUDA operation and synchronization boundary.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

CASES = (
    "linear_bf16",
    "gemma_fused_default",
    "gemma_fused_sgl_pdl_off",
    "gemma_fused_sgl_pdl_on",
    "triton_tiny",
)
HIDDEN = 5120
SEED = 20261120


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def emit(tag, **values):
    print("PB_KERNEL_" + tag + " " + json.dumps(values, allow_nan=False), flush=True)


class Progress:
    def __init__(self, output, case):
        self.output = output
        self.case = case
        self.history = []
        self.current = "starting"

    def mark(self, stage):
        self.current = stage
        self.history.append(stage)
        value = dict(case=self.case, stage=stage, history=self.history)
        save(self.output / (self.case + "-progress.json"), value)
        emit("STAGE", case=self.case, stage=stage)

    def sync(self, torch, stage):
        self.mark(stage + ":synchronize")
        torch.cuda.synchronize()
        self.mark(stage + ":complete")


def build_operation(torch, case):
    """Return stable mutable buffers and one real operator invocation."""
    x = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device="cuda")
    metadata = dict(input_shape=list(x.shape), input_dtype=str(x.dtype))
    if case == "linear_bf16":
        weight = (
            torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16, device="cuda")
            / HIDDEN**0.5
        )
        output = torch.full_like(x, -7)

        def operation():
            output.copy_(torch.nn.functional.linear(x, weight))

        return (
            dict(input=x, output=output),
            operation,
            dict(
                metadata,
                operator="torch.nn.functional.linear",
                weight_shape=list(weight.shape),
                weight_dtype=str(weight.dtype),
                backend="PyTorch CUDA BF16 dense linear (cuBLAS)",
            ),
        )
    if case.startswith("gemma_fused"):
        import sgl_kernel
        import sgl_kernel.elementwise as elementwise

        residual = torch.randn_like(x)
        weight = torch.randn(HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1
        eps = 1e-6
        if case == "gemma_fused_default":

            def operation():
                sgl_kernel.gemma_fused_add_rmsnorm(x, residual, weight, eps)

            dispatch = "sgl_kernel public wrapper used by GemmaRMSNorm; default dispatch and PDL"
            pdl = "default"
        else:
            pdl = case.endswith("_on")

            def operation():
                torch.ops.sgl_kernel.gemma_fused_add_rmsnorm.default(
                    x, residual, weight, eps, pdl
                )

            dispatch = "direct registered sgl_kernel CUDA operator, bypassing optional FlashInfer wrapper dispatch"
        return (
            dict(input=x, residual=residual),
            operation,
            dict(
                metadata,
                operator="gemma_fused_add_rmsnorm",
                dispatch=dispatch,
                enable_pdl=pdl,
                eps=eps,
                weight_dtype=str(weight.dtype),
                elementwise_sha256=digest(elementwise.__file__),
                wrapper_has_flashinfer=bool(
                    getattr(elementwise, "_has_flashinfer", False)
                ),
            ),
        )
    if case == "triton_tiny":
        # Triton's JIT resolves language symbols from the function's module
        # globals, including when the kernel is created by this lazy factory.
        global tl
        import triton
        import triton.language as tl

        @triton.jit
        def tiny(source, destination, count: tl.constexpr, block: tl.constexpr):
            offsets = tl.program_id(0) * block + tl.arange(0, block)
            value = tl.load(source + offsets, offsets < count, other=0).to(tl.float32)
            tl.store(destination + offsets, value * 0.5 + 1.0, offsets < count)

        output = torch.full_like(x, -7)

        def operation():
            tiny[(triton.cdiv(HIDDEN, 256),)](x, output, HIDDEN, 256)

        return (
            dict(input=x, output=output),
            operation,
            dict(
                metadata,
                operator="Triton BF16 load -> FP32 affine -> BF16 store",
                triton=triton.__version__,
                grid=[triton.cdiv(HIDDEN, 256)],
                block=256,
            ),
        )
    raise ValueError(case)


def snapshot(buffers):
    return {name: value.detach().cpu().clone() for name, value in buffers.items()}


def compare(torch, actual, expected):
    results = {}
    for name, reference in expected.items():
        value = actual[name]
        results[name] = dict(
            exact=torch.equal(value, reference),
            max_abs=float((value.float() - reference.float()).abs().max()),
            finite=bool(torch.isfinite(value).all()),
            sha256=hashlib.sha256(
                value.view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        )
    return results


def require_exact(values):
    if not all(value["exact"] and value["finite"] for value in values.values()):
        raise AssertionError("Operator output differs: " + json.dumps(values))


def worker(args, output):
    progress = Progress(output, args.worker)
    try:
        progress.mark("import_torch")
        import torch

        from sglang.srt.early_exit.conditional import ConditionalRegions

        progress.mark("initialize_cuda")
        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        torch.manual_seed(SEED)
        regions = ConditionalRegions()
        with torch.inference_mode():
            progress.mark("allocate_operator_inputs")
            buffers, operation, metadata = build_operation(torch, args.worker)
            initial = {name: value.clone() for name, value in buffers.items()}
            action = torch.zeros(1, dtype=torch.int32, device="cuda")
            executed = torch.zeros(1, dtype=torch.int32, device="cuda")
            progress.sync(torch, "allocation")

            def reset(width):
                for name, value in buffers.items():
                    value.copy_(initial[name])
                action.fill_(width)
                executed.zero_()

            def marked():
                regions.marker(0)
                operation()
                executed.add_(1)
                regions.marker(1)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            progress.mark("warmup")
            with torch.cuda.stream(stream):
                for _ in range(args.warmup):
                    reset(0)
                    operation()
            progress.sync(torch, "warmup")
            reset(0)
            progress.sync(torch, "eager_reset")
            progress.mark("eager_operator")
            operation()
            progress.sync(torch, "eager_operator")
            expected = snapshot(buffers)
            initial_cpu = snapshot(initial)
            if not all(
                bool(torch.isfinite(value).all()) for value in expected.values()
            ):
                raise AssertionError("Eager operator produced non-finite output")

            captures = {}
            moved = None
            for kind in ("original", "conditional"):
                reset(0)
                progress.sync(torch, kind + "_before_capture")
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                progress.mark(kind + "_capture")
                with torch.cuda.graph(graph, stream=stream):
                    marked()
                progress.sync(torch, kind + "_after_capture")
                if kind == "conditional":
                    progress.mark("conditional_convert")
                    moved = regions.convert(graph, action, [11])
                    progress.sync(torch, "conditional_convert")
                    if len(moved) != 1 or moved[0] < 1:
                        raise AssertionError(
                            "Conditional conversion moved no operator nodes"
                        )
                progress.mark(kind + "_instantiate")
                graph.instantiate()
                progress.sync(torch, kind + "_instantiate")
                results = []
                for ordinal, width in enumerate((0, 11, 0, 11)):
                    label = f"{kind}_action{width}_repeat{ordinal}"
                    reset(width)
                    progress.sync(torch, label + "_reset")
                    progress.mark(label + "_replay")
                    graph.replay()
                    progress.sync(torch, label + "_replay")
                    should_execute = kind == "original" or width == 0
                    counter = int(executed.cpu()[0])
                    if counter != int(should_execute):
                        raise AssertionError(
                            f"Wrong execution counter for {label}: {counter}"
                        )
                    observed = snapshot(buffers)
                    comparisons = compare(
                        torch, observed, expected if should_execute else initial_cpu
                    )
                    require_exact(comparisons)
                    results.append(
                        dict(
                            action=width,
                            executed=counter,
                            compared_to=(
                                "eager" if should_execute else "initial_buffers"
                            ),
                            tensors=comparisons,
                        )
                    )
                    progress.mark(label + "_verified")
                captures[kind] = dict(passed=True, cases=results)
                # Keep both graph objects and their allocator pools alive until
                # every replay is checked; no operator buffer is reallocated.
                captures[kind]["_graph"] = graph
            for value in captures.values():
                value.pop("_graph")
            summary = dict(
                passed=True,
                case=args.worker,
                metadata=metadata,
                captures=captures,
                moved_nodes=moved,
                hidden_size=HIDDEN,
                seed=SEED,
                warmup=args.warmup,
                gpu=torch.cuda.get_device_name(0),
                gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                torch=str(torch.__version__),
                cuda=torch.version.cuda,
                source_sha256=digest(__file__),
                conditional_py_sha256=digest(
                    Path(__file__).with_name("conditional.py")
                ),
                conditional_cuda_sha256=digest(
                    Path(__file__).with_name("conditional.cu")
                ),
                stages=progress.history,
            )
            save(output / (args.worker + "-summary.json"), summary)
            emit("PASS", case=args.worker, moved_nodes=moved)
    except Exception as error:
        # An illegal instruction poisons the CUDA context. Persist CPU-only
        # failure evidence and let this process exit; do not issue CUDA cleanup.
        failure = dict(
            passed=False,
            case=args.worker,
            stage=progress.current,
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
            stages=progress.history,
            source_sha256=digest(__file__),
        )
        save(output / (args.worker + "-summary.json"), failure)
        emit("FAIL", case=args.worker, stage=progress.current, error=str(error))
        raise SystemExit(1) from None


def orchestrate(args, output):
    summaries = {}
    source = str(Path(__file__).resolve().parents[3])
    for case in args.cases:
        summary_path = output / (case + "-summary.json")
        if summary_path.exists() or (output / (case + "-progress.json")).exists():
            raise ValueError("Probe outputs already exist; use a new RUN_DIR")
        command = [
            sys.executable,
            "-u",
            "-m",
            "sglang.srt.early_exit.kernel_probe",
            "--worker",
            case,
            "--warmup",
            str(args.warmup),
        ]
        if os.environ.get("SGLANG_EARLY_EXIT_SANITIZER") == "1":
            sanitizer = shutil.which("compute-sanitizer")
            if sanitizer is None:
                summaries[case] = dict(
                    passed=False,
                    case=case,
                    stage="sanitizer_lookup",
                    error="SGLANG_EARLY_EXIT_SANITIZER=1 requires compute-sanitizer",
                )
                continue
            command = [
                sanitizer,
                "--tool",
                "memcheck",
                "--error-exitcode",
                "86",
                "--log-file",
                str(output / (case + "-sanitizer.log")),
                *command,
            ]
        env = dict(
            os.environ,
            RUN_DIR=str(output),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=source
            + (
                os.pathsep + os.environ["PYTHONPATH"]
                if os.environ.get("PYTHONPATH")
                else ""
            ),
        )
        emit("DISPATCH", case=case)
        with (output / (case + ".stdout")).open("x") as stdout, (
            output / (case + ".stderr")
        ).open("x") as stderr:
            try:
                completed = subprocess.run(
                    command, env=env, stdout=stdout, stderr=stderr, timeout=args.timeout
                )
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                exit_code = "timeout"
        if summary_path.exists():
            value = json.loads(summary_path.read_text())
        else:
            progress_path = output / (case + "-progress.json")
            progress = (
                json.loads(progress_path.read_text()) if progress_path.exists() else {}
            )
            value = dict(
                passed=False,
                case=case,
                stage=progress.get("stage", "before_worker_progress"),
                error="Worker did not write a result",
                progress=progress,
            )
        value["process_exit_code"] = exit_code
        value["passed"] = value["passed"] and exit_code == 0
        summaries[case] = value
        emit(
            "RESULT",
            case=case,
            passed=value["passed"],
            exit_code=exit_code,
            stage=value.get("stage"),
        )
    summary = dict(
        passed=all(value["passed"] for value in summaries.values()),
        cases=summaries,
        hidden_size=HIDDEN,
        independent_processes=True,
        source_sha256=digest(__file__),
        scope="Real operators at batch1/hidden5120 with default runtime PDL and explicit SGL norm PDL variants. Exact eager/original/conditional comparison, actions0/11, no model weights or performance claim.",
    )
    save(output / "kernel-probe-summary.json", summary)
    emit(
        "SUMMARY",
        passed=summary["passed"],
        cases={key: value["passed"] for key, value in summaries.items()},
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--timeout", type=int, default=180, help="Maximum seconds per isolated case"
    )
    parser.add_argument("--worker", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup < 1 or args.timeout < 1:
        parser.error("warmup and timeout must be positive")
    if len(set(args.cases)) != len(args.cases):
        parser.error("cases must be unique")
    return args


def main(args=None, output=None):
    """Return diagnostics without preventing a caller's subsequent sanitizer run."""
    options = args or argparse.Namespace(
        cases=list(CASES), warmup=5, timeout=180, worker=None
    )
    directory = Path(output or os.environ["RUN_DIR"]).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return (worker if options.worker else orchestrate)(options, directory)


if __name__ == "__main__":
    main(parse_args())
