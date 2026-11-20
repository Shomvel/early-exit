#!/usr/bin/env python3
"""Wait for a verified, immutable evaluation checkpoint before starting workers."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--source-run-uuid", required=True)
    parser.add_argument("--local-workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args, command = parser.parse_known_args()
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("An explicit evaluation command is required after --")
    import torch

    if torch.cuda.device_count() != args.local_workers:
        raise ValueError("Visible GPU count differs from the local worker topology")
    replica = int(os.environ.get("KUBEBRAIN_REPLICA", "0"))
    ready = Path(os.environ["RUN_DIR"]) / f"handoff-ready-replica-{replica}.json"
    temporary = ready.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            dict(
                replica=replica,
                run_uuid=os.environ["RUN_UUID"],
                gpus=torch.cuda.device_count(),
                handoff=str(args.handoff),
            )
        )
        + "\n"
    )
    temporary.replace(ready)
    print("PB_HANDOFF_READY " + ready.read_text().strip(), flush=True)
    deadline = time.monotonic() + args.timeout_seconds
    while not args.handoff.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError(
                "Evaluation handoff was not released before the deadline"
            )
        time.sleep(5)
    gate = json.loads(args.handoff.read_text())
    if gate["source_run_uuid"] != args.source_run_uuid:
        raise ValueError("Evaluation handoff identifies another source run")
    if gate["target_run_uuid"] != os.environ["RUN_UUID"]:
        raise ValueError("Evaluation handoff identifies another destination run")
    snapshot = Path(gate["resume_from"])
    manifest = snapshot / "resume-manifest.json"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != gate["manifest_sha256"]:
        raise ValueError("Evaluation handoff manifest fingerprint differs")
    print(
        "PB_HANDOFF_RELEASED "
        + json.dumps(
            dict(
                replica=replica,
                resume_from=str(snapshot),
                manifest_sha256=gate["manifest_sha256"],
            )
        ),
        flush=True,
    )
    os.execv(command[0], command + ["--resume-from", str(snapshot)])


if __name__ == "__main__":
    main()
