"""Apply a hashed source overlay to the exact SGLang package in a pinned image.

This is for remote environments where shipping a complete upstream Git bundle
is expensive. The managed source checkout contains every modified file and
the expected upstream hashes. Runtime files live under RUN_DIR, keeping the
recorded checkout immutable. The image digest pins the remaining dependencies.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replica_runtime_root(run_directory, environment):
    replica = int(environment.get("KUBEBRAIN_REPLICA", "0"))
    total = int(environment.get("KUBEBRAIN_REPLICA_TOTAL", "1"))
    if total < 1 or not 0 <= replica < total:
        raise ValueError("Invalid replica identity for the runtime overlay")
    return (
        Path(run_directory)
        if total == 1
        else Path(run_directory) / f"runtime-replica-{replica}"
    )


def main(args):
    source = Path(__file__).resolve().parents[2]
    manifest = json.loads((source / "overlay-manifest.json").read_text())
    installed = Path(importlib.util.find_spec("sglang").origin).parent
    for relative, expected in manifest["base_sha256"].items():
        actual = digest(installed / relative)
        if actual != expected:
            raise ValueError(f"SGLang image source mismatch: {relative}: {actual}")
    runtime_root = replica_runtime_root(os.environ["RUN_DIR"], os.environ)
    runtime_root.mkdir(parents=True, exist_ok=True)
    output = runtime_root / "sglang-python"
    target = output / "sglang"
    shutil.copytree(
        installed, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    written = {}
    for relative, expected in manifest["overlay_sha256"].items():
        original = source / "python" / "sglang" / relative
        assert digest(original) == expected, relative
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        written[relative] = digest(destination)
    receipt = dict(
        base_sha256=manifest["base_sha256"],
        overlay_sha256=written,
        fork_source=manifest["fork_source"],
        installed_package=str(installed),
    )
    (output.parent / "runtime-overlay-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    env = dict(os.environ)
    env["SGLANG_EARLY_EXIT_SOURCE_ROOT"] = str(source)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(output), str(source), env.get("PYTHONPATH", "")]
    )
    print("PB_SGLANG_OVERLAY " + json.dumps(receipt), flush=True)
    os.execve(
        sys.executable, [sys.executable, "-u", "-m", args.module, *args.arguments], env
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    options, arguments = parser.parse_known_args()
    options.arguments = arguments
    main(options)
