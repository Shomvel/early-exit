"""Plan and run this experiment through an external YiCloud training_run.py."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PREFIX = Path("experiments/perfectblend")
SUMMARY_FILES = ("selection.json", "summary.json", "evaluation-summary.json", "smoke-summary.json")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_new(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def git(repo, *arguments):
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def clean_source(repo, expected=None):
    repo = Path(repo).expanduser().resolve()
    if Path(git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise ValueError("--repo must name the Git checkout root")
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Source must be committed and clean, including untracked files")
    submodules = subprocess.check_output(["git", "-C", str(repo), "submodule", "status", "--recursive"], text=True)
    if any(line[:1] != " " for line in submodules.splitlines()):
        raise ValueError("Initialize recursive submodules at their recorded commits")
    head = git(repo, "rev-parse", "HEAD")
    if expected is not None and head != expected:
        raise ValueError("Prepared source_ref differs from the current clean HEAD")
    return repo, head


def external_path(value, repo, name):
    if value is None:
        raise ValueError(f"{name} is required for the selected stage")
    path = Path(value).expanduser().resolve()
    if path == repo or repo in path.parents:
        raise ValueError(f"{name} must be outside the source checkout")
    return path


def topology(payload):
    tasks = payload.get("TaskSpecs")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("payload.TaskSpecs must contain at least one task")
    replicas = total = 0
    for task in tasks.values():
        count, gpus = int(task.get("Replicas", 1)), int(task.get("GPU", 0))
        if count < 1 or gpus < 0:
            raise ValueError("Invalid replica/GPU count")
        if "@sha256:" not in task.get("Image", ""):
            raise ValueError("Each task Image must be pinned by digest")
        for key in task.get("Env", {}):
            if re.search(r"TOKEN|SECRET|PASSWORD|CREDENTIAL|ACCESS_KEY|API_KEY|PRIVATE_KEY", key, re.I):
                raise ValueError("Pass platform credentials to the external control tool, outside the payload")
        replicas += count
        total += count * gpus
    return replicas, total


def plan(args):
    repo, head = clean_source(args.repo)
    if not Path(args.runtime_python).is_absolute():
        raise ValueError("--runtime-python must be an absolute interpreter path in the pinned image")
    output = external_path(args.output, repo, "--output")
    external_path(args.runs_root, repo, "--runs-root")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,79}", args.run_name):
        raise ValueError("Use a unique semantic run name of at most 80 lowercase characters")
    payload = read(args.payload_file)
    replicas, total = topology(payload)
    if total != args.expected_gpus or total < 1:
        raise ValueError("--expected-gpus must equal the payload's total GPU count and be positive")
    payload["Name"] = args.run_name
    command = [args.runtime_python, "-u"]
    kind = args.workload_kind or ("module" if args.stage == "multiclass" else "training")
    if args.stage == "custom":
        if not args.entrypoint or not args.arguments_file:
            raise ValueError("custom requires --entrypoint and --arguments-file")
        entry = Path(args.entrypoint)
        if entry.is_absolute() or ".." in entry.parts or not (repo / entry).is_file():
            raise ValueError("--entrypoint must be an existing repository-relative Python script")
        if repo not in (repo / entry).resolve().parents:
            raise ValueError("--entrypoint must resolve inside the checkout")
        arguments = read(args.arguments_file)
        if not isinstance(arguments, list) or not all(isinstance(x, str) and x for x in arguments):
            raise ValueError("--arguments-file must contain a JSON argv array of nonempty strings")
        command += [entry.as_posix(), *arguments]
    else:
        if any((args.entrypoint, args.arguments_file)):
            raise ValueError("Use the custom stage for an explicit entrypoint and argv")
        if replicas != 1:
            raise ValueError("Built-in stages use one replica; native workers run on local GPUs")
        if args.stage == "multiclass":
            if total != 1 or kind != "module":
                raise ValueError("The captured-feature multiclass experiment requires one GPU and kind=module")
            capture = external_path(args.capture_run, repo, "--capture-run")
            command += [(PREFIX / "train_multiclass.py").as_posix(), "--capture", str(capture)]
        else:
            if kind != "training":
                raise ValueError("Full-model training, smoke, and evaluation use workload_kind=training")
            model = external_path(args.model, repo, "--model")
            data = external_path(args.data, repo, "--data")
            inventory = external_path(args.weight_inventory, repo, "--weight-inventory")
            if not inventory.is_file():
                raise ValueError("--weight-inventory must name the verified model inventory JSON")
            workers = args.workers or total
            if workers != total:
                raise ValueError("Native --workers must equal the local GPU count")
            common = ["--model", str(model), "--data", str(data), "--weight-inventory", str(inventory),
                      "--workers", str(workers)]
            if args.stage in ("smoke", "pilot", "capture"):
                command += [(PREFIX / "native/pipeline.py").as_posix(), "--stage", args.stage, *common]
                if args.stage == "capture":
                    command += ["--pilot", str(external_path(args.pilot_run, repo, "--pilot-run"))]
            else:
                command += [(PREFIX / "native/evaluate.py").as_posix(), *common,
                            "--classifier-run", str(external_path(args.classifier_run, repo, "--classifier-run"))]
                if args.stage == "eval-smoke":
                    command += ["--smoke"]
    if not (repo / command[2]).is_file():
        raise ValueError("Selected entrypoint does not exist in --repo")
    spec = dict(repo=str(repo), source_ref=head, label=args.run_name, run_name=args.run_name,
                workload_kind=kind, expected_total_gpus=total, payload=payload, command=command)
    if args.retry_of:
        spec["retry_of"] = args.retry_of
    write_new(output, spec)
    print(json.dumps(dict(spec=str(output), source_ref=head, command=command, workload_kind=kind,
                          gpus=total, prepared_run=str(Path(args.runs_root).expanduser().resolve() / args.run_name)), indent=2))


def tool_command(args):
    tool = args.training_run.expanduser().resolve()
    if not tool.is_file():
        raise ValueError("--training-run must name the complete external control tool's training_run.py")
    return [sys.executable, str(tool)]


def connection(args):
    result = []
    if args.direct:
        result.append("--direct")
    if args.credentials_file:
        result += ["--credentials-file", str(args.credentials_file.expanduser().resolve())]
    if args.endpoint:
        result += ["--endpoint", args.endpoint]
    return result


def delegate(args):
    cli = tool_command(args)
    if args.mode == "prepare":
        spec = read(args.spec)
        repo, _ = clean_source(spec["repo"], spec["source_ref"])
        runs_root = external_path(args.runs_root, repo, "--runs-root")
        if (runs_root / spec["run_name"]).exists():
            raise ValueError("Run already exists; submit/show/reconcile that run instead of preparing twice")
        subprocess.run(cli + ["launch", "--spec", str(args.spec.expanduser().resolve()),
                              "--runs-root", str(runs_root), *connection(args)], check=True)
    elif args.mode == "show":
        subprocess.run(cli + ["show", str(args.run.expanduser().resolve()), "--format", "json"], check=True)
    elif args.mode == "submit":
        run = str(args.run.expanduser().resolve())
        submit = cli + ["submit", "--run", run, *connection(args)]
        if args.execute:
            submit.append("--execute")
        try:
            subprocess.run(submit, check=True)
        finally:
            if args.execute:
                # A failed API response can still mean submission took effect. Let the
                # control tool reconcile its recorded identity before attempting a retry.
                subprocess.run(cli + ["reconcile", run, "--watch", "--execute",
                                      "--poll-interval", str(args.poll_interval), *connection(args)], check=True)
    else:
        command = cli + ["reconcile", str(args.run.expanduser().resolve()),
                         "--poll-interval", str(args.poll_interval), *connection(args)]
        if args.execute:
            command.append("--execute")
        if args.watch:
            command.append("--watch")
        subprocess.run(command, check=True)


def collect(args):
    run = args.run.expanduser().resolve()
    manifest = read(run / "run.json")
    phase = manifest["yicloud"].get("terminal_phase")
    if phase != "Succeeded" or manifest.get("state") != "verified":
        raise ValueError("Collect requires a successfully reconciled, verified run")
    replicas, _ = topology(manifest["yicloud"]["payload"])
    markers = [read(path) for path in sorted((run / "logs").glob("runtime-finished-replica-*.json"))]
    if len(markers) != replicas or any(marker.get("exit_code") != 0 for marker in markers):
        raise ValueError("Missing or unsuccessful runtime replica markers")
    stem = args.label or manifest["run_name"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,79}", stem):
        raise ValueError("Invalid receipt label")
    output = args.output_dir.expanduser().resolve()
    files = {}
    for name in SUMMARY_FILES:
        path = run / name
        if path.is_file():
            if path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError(f"Summary exceeds the small-receipt limit: {name}")
            files[name] = dict(sha256=sha(path), bytes=path.stat().st_size)
    if not files:
        raise ValueError("No completed experiment summary exists")
    targets = [output / (stem + "-" + name) for name in (*files, "final.json", "verified.json")]
    if any(path.exists() for path in targets):
        raise ValueError("Receipt output already exists; choose an unused output directory or label")
    output.mkdir(parents=True, exist_ok=True)
    for name in files:
        shutil.copyfile(run / name, output / (stem + "-" + name))
    write_new(output / (stem + "-final.json"), dict(run=str(run), uuid=manifest["uuid"],
              job_id=manifest["yicloud"]["job_id"], state=manifest["state"], phase=phase))
    write_new(output / (stem + "-verified.json"), dict(
        markers=[{key: value[key] for key in ("replica", "exit_code") if key in value} for value in markers],
        files=files, run_manifest_sha256=sha(run / "run.json")))
    print(json.dumps(dict(output=str(output), label=stem, files=files), indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    plan_parser = sub.add_parser("plan", help="Write an immutable raw launch spec; does not create a run or GPU job")
    plan_parser.add_argument("--repo", type=Path, required=True, help="Clean shared source checkout visible in the GPU container")
    plan_parser.add_argument("--runs-root", type=Path, required=True)
    plan_parser.add_argument("--payload-file", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True, help="New JSON file outside the source checkout")
    plan_parser.add_argument("--run-name", required=True)
    plan_parser.add_argument("--stage", choices=("smoke", "pilot", "capture", "eval-smoke", "eval", "multiclass", "custom"), required=True)
    plan_parser.add_argument("--expected-gpus", type=int, required=True)
    plan_parser.add_argument("--workload-kind", choices=("training", "module"))
    plan_parser.add_argument("--runtime-python", default="/opt/venv/bin/python3")
    plan_parser.add_argument("--workers", type=int)
    for name in ("model", "data", "weight-inventory", "pilot-run", "capture-run", "classifier-run", "arguments-file"):
        plan_parser.add_argument("--" + name, type=Path)
    plan_parser.add_argument("--entrypoint", help="Repository-relative Python script for stage=custom")
    plan_parser.add_argument("--retry-of", help="Prior terminal failed run UUID; every retry has a new run name")
    for name in ("prepare", "submit", "reconcile", "show"):
        child = sub.add_parser(name)
        child.add_argument("--training-run", type=Path, required=True)
        if name == "prepare":
            child.add_argument("--spec", type=Path, required=True)
            child.add_argument("--runs-root", type=Path, required=True)
        else:
            child.add_argument("--run", type=Path, required=True)
        if name != "show":
            child.add_argument("--direct", action="store_true")
            child.add_argument("--credentials-file", type=Path)
            child.add_argument("--endpoint")
        if name in ("submit", "reconcile"):
            child.add_argument("--execute", action="store_true")
            child.add_argument("--poll-interval", type=float, default=20)
        if name == "reconcile":
            child.add_argument("--watch", action="store_true")
    collect_parser = sub.add_parser("collect", help="Export small summary receipts from an already verified shared run")
    collect_parser.add_argument("--run", type=Path, required=True)
    collect_parser.add_argument("--output-dir", type=Path, required=True)
    collect_parser.add_argument("--label")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        if hasattr(args, "poll_interval") and args.poll_interval <= 0:
            raise ValueError("--poll-interval must be positive")
        if args.mode == "plan":
            plan(args)
        elif args.mode == "collect":
            collect(args)
        else:
            delegate(args)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
