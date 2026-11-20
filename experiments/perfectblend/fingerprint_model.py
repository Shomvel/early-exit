"""Hash a local HF safetensors model and verify byte identity before a run."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inventory(model):
    index = model / "model.safetensors.index.json"
    names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    records = []
    for name in names:
        path = model / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not name.endswith(".safetensors"):
            raise ValueError("The weight index must reference local safetensors files")
        before = path.stat()
        digest = sha(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Weight changed while hashing: {name}")
        records.append(dict(file=name, bytes=after.st_size, mtime_ns=after.st_mtime_ns, sha256=digest))
        print(json.dumps(dict(file=name, sha256=digest)), flush=True)
    return dict(model=str(model), index_sha256=sha(index), config_sha256=sha(model / "config.json"), files=records)


def compare(actual, expected):
    left = {row["file"]: (row["bytes"], row["sha256"]) for row in actual["files"]}
    right = {row["file"]: (row["bytes"], row["sha256"]) for row in expected["files"]}
    if left != right:
        raise ValueError("Model shard filenames, sizes, or SHA256 values differ from the reference")
    for key in ("index_sha256", "config_sha256"):
        if key in expected and actual[key] != expected[key]:
            raise ValueError(f"Model metadata differs: {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "verify"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, help="Required reference for verify")
    parser.add_argument("--reference", type=Path, help="Optional byte identity reference when creating a new storage stamp")
    parser.add_argument("--output", type=Path, help="Required new JSON file for create")
    args = parser.parse_args()
    if args.mode == "create" and not args.output:
        parser.error("create requires --output")
    if args.mode == "verify" and not args.inventory:
        parser.error("verify requires --inventory")
    if args.mode == "create" and args.output.exists():
        parser.error("Output already exists; choose a new inventory filename")
    actual = inventory(args.model.expanduser().resolve())
    reference = args.reference if args.mode == "create" else args.inventory
    if reference:
        compare(actual, json.loads(reference.read_text()))
    if args.mode == "create":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(actual, handle, indent=2)
            handle.write("\n")
    print(json.dumps(dict(status="verified", files=len(actual["files"]), inventory_sha256=sha(args.output if args.mode == "create" else args.inventory))))


if __name__ == "__main__":
    main()
