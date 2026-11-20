# SPDX-License-Identifier: Apache-2.0
"""Validate immutable partial-evaluation snapshots before reusing predictions."""

import ast
import hashlib
import json
import re
from pathlib import Path

FORMAT = "early_exit_evaluation_resume_v1"
SOURCE_UUID = "8273319d-8232-492c-a92d-9dbf64fd6913"
PROTOCOL_FUNCTIONS = (
    "read_benchmarks",
    "select_rows",
    "matched_indices",
    "mmlu_example",
    "prompt",
    "boxed",
    "math_equivalent",
    "grade",
    "generate",
    "verify_model",
)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def protocol_fingerprint(path):
    tree = ast.parse(Path(path).read_text())
    nodes = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    if any(name not in nodes for name in PROTOCOL_FUNCTIONS):
        raise ValueError("Evaluation protocol is missing a required function")
    return {
        name: hashlib.sha256(
            ast.dump(nodes[name], include_attributes=False).encode()
        ).hexdigest()
        for name in PROTOCOL_FUNCTIONS
    }


def load_resume(directory, *, current_identity, evaluator_path):
    """Return verified full predictions and their provenance; never trust partial lines."""
    directory = Path(directory).resolve()
    manifest_path = directory / "resume-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("format") != FORMAT
        or manifest.get("source_run_uuid") != SOURCE_UUID
    ):
        raise ValueError("Unsupported resume format or source run UUID")
    files = manifest["files"]
    required = {"run.json", "config.yaml", "source/evaluate.py"}
    required.update(
        "source/" + name for name in current_identity["runtime_source_sha256"]
    )
    if not required.issubset(files):
        raise ValueError("Resume snapshot omits source/configuration provenance")
    for relative, evidence in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Resume files must be snapshot-relative")
        path = directory / relative_path
        if path.is_symlink() or not path.is_file():
            raise ValueError("Resume evidence must be an ordinary immutable file")
        if path.stat().st_size != evidence["bytes"] or sha(path) != evidence["sha256"]:
            raise ValueError("Resume file fingerprint differs: " + relative)
    run = json.loads((directory / "run.json").read_text())
    if run["uuid"] != manifest["source_run_uuid"]:
        raise ValueError("Resume source run identity differs")
    source_identity = manifest["source_identity"]
    for key, expected in current_identity.items():
        if source_identity.get(key) != expected:
            raise ValueError("Resume execution identity differs: " + key)
    if sha(directory / "source/evaluate.py") != source_identity["evaluator_sha256"]:
        raise ValueError("Resume evaluator fingerprint differs")
    for name, expected in current_identity["runtime_source_sha256"].items():
        if sha(directory / "source" / name) != expected:
            raise ValueError("Resume runtime source fingerprint differs: " + name)
    if protocol_fingerprint(directory / "source/evaluate.py") != protocol_fingerprint(
        evaluator_path
    ):
        raise ValueError("Resume evaluator changes the generation or grading protocol")
    config = json.loads((directory / "config.yaml").read_text())
    for key in ("weight_inventory_sha256", "benchmark_sha256", "classifier_sha256"):
        if config[key] != current_identity[key]:
            raise ValueError("Resume config fingerprint differs: " + key)
    if config.get("smoke") or config.get("limit_per_task") is not None:
        raise ValueError("Only the complete original benchmark protocol can be resumed")
    rows = {}
    source_workers = config["workers"]
    prediction_files = {
        relative: info
        for relative, info in files.items()
        if re.fullmatch(r"sglang-full-\d+\.jsonl", relative)
    }
    if set(prediction_files) != {
        f"sglang-full-{rank}.jsonl" for rank in range(source_workers)
    }:
        raise ValueError(
            "Resume snapshot must include every original full worker, including empty files"
        )
    for relative, info in prediction_files.items():
        rank = int(re.fullmatch(r"sglang-full-(\d+)\.jsonl", relative).group(1))
        if info["mode"] != "full" or info["rank"] != rank:
            raise ValueError("Resume prediction metadata differs from its filename")
        raw = (directory / relative).read_text()
        if raw and not raw.endswith("\n"):
            raise ValueError("Resume contains an unfinished JSONL record")
        count = 0
        for line in raw.splitlines():
            row = json.loads(line)
            index = row["evaluation_index"]
            if not isinstance(index, int) or index < 0 or index in rows:
                raise ValueError(
                    "Resume predictions contain invalid or duplicate evaluation indices"
                )
            row["resumed_from"] = dict(
                source_run_uuid=manifest["source_run_uuid"],
                source_rank=rank,
                source_file=relative,
                source_file_sha256=info["sha256"],
                resume_manifest_sha256=sha(manifest_path),
                same_gpu_generation_timing=False,
            )
            rows[index] = row
            count += 1
        if count != info["count"]:
            raise ValueError("Resume prediction count differs: " + relative)
    if len(rows) != manifest["imported_count"]:
        raise ValueError("Resume imported count differs")
    return rows, dict(
        source_run_uuid=manifest["source_run_uuid"],
        manifest_sha256=sha(manifest_path),
        imported_count=len(rows),
        source_evaluator_sha256=source_identity["evaluator_sha256"],
    )
