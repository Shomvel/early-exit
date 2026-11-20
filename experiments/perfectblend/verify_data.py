"""Verify pinned samples, split disjointness, benchmark counts, and optional downloads."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

REFERENCES = Path(__file__).resolve().parent / "references"
BENCHMARK_SHA256 = "a04901c438d78667d050805b26fad1d4092f6e70ed0bde2b2f666ef9aa72ddfe"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=REFERENCES)
    parser.add_argument("--include-downloads", action="store_true", help="Also hash the original parquet downloads")
    args = parser.parse_args()
    data, references = args.data.expanduser().resolve(), args.references.expanduser().resolve()
    manifest = read(references / "sample-manifest.json")
    assert read(data / "sample-manifest.json") == manifest, "sample-manifest.json differs"
    assert read(data / "benchmark-sources.json") == read(references / "benchmark-sources.json"), "Benchmark revisions differ"
    content_seen, rows_seen = set(), set()
    split_counts = {}
    for split in manifest["splits"]:
        path = data / (split["split"] + ".jsonl")
        assert sha(path) == split["sha256"] and path.stat().st_size == split["bytes"], str(path)
        count = 0
        sources = Counter()
        for line in path.open():
            row = json.loads(line)
            digest = hashlib.sha256(json.dumps(row["conversations"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            assert digest == row["content_sha256"], row["id"]
            assert digest not in content_seen and row["row_index"] not in rows_seen, row["id"]
            content_seen.add(digest)
            rows_seen.add(row["row_index"])
            count += 1
            sources[str(row["source"])] += 1
        assert count == split["count"] and dict(sources) == split["sources"], split["split"]
        split_counts[split["split"]] = count
    benchmark = data / "benchmarks.jsonl"
    assert sha(benchmark) == BENCHMARK_SHA256, "benchmarks.jsonl SHA256 differs"
    counts = Counter()
    subjects = set()
    for line in benchmark.open():
        row = json.loads(line)
        counts[row["benchmark"], row["split"]] += 1
        if row["benchmark"] == "mmlu":
            subjects.add(row["subject"])
    assert counts == {("gsm8k", "test"): 1319, ("math", "test"): 5000, ("mmlu", "test"): 14042, ("mmlu", "dev"): 285}, counts
    assert len(subjects) == 57, len(subjects)
    if args.include_downloads:
        inventories = [read(references / "open-perfectblend-inventory.json"), *read(references / "benchmark-sources.json")]
        for source in inventories:
            folder = data / source["repo"].replace("/", "--")
            for record in source["files"]:
                path = folder / record["file"]
                assert path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], str(path)
    print(json.dumps(dict(status="verified", splits=split_counts,
                          benchmark_counts={"/".join(key): value for key, value in counts.items()},
                          mmlu_subjects=len(subjects), downloads_hashed=args.include_downloads), indent=2))


if __name__ == "__main__":
    main()
