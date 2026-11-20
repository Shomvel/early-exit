"""Download pinned public inputs and build a reproducible conversation sample."""
import argparse
import concurrent.futures
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

DATA = None
SEED = 20261120
REPO = "mlabonne/open-perfectblend"
REV = "af60f3c18201652a83a93f46fcfee1b646ba3df7"
API = None


def emit(tag, **values):
    print(tag + " " + json.dumps(values, ensure_ascii=False), flush=True)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download(repo, revision=None, include=None):
    info = API.dataset_info(repo, revision=revision)
    revision = info.sha
    folder = DATA / repo.replace("/", "--")
    folder.mkdir(parents=True, exist_ok=True)
    files = [f.rfilename for f in info.siblings
             if f.rfilename.endswith(".parquet") and (include is None or include(f.rfilename))]
    assert files, repo
    emit("DOWNLOAD_START", repo=repo, revision=revision, files=len(files))
    def one(name):
        path = Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision,
                                   local_dir=folder, token=False))
        record = dict(file=name, bytes=path.stat().st_size, sha256=sha(path),
                      rows=pq.ParquetFile(path).metadata.num_rows)
        emit("DOWNLOADED", repo=repo, **record)
        return record
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        inventory = list(pool.map(one, files))
    meta = dict(repo=repo, revision=revision, files=inventory)
    (folder / "inventory.json").write_text(json.dumps(meta, indent=2) + "\n")
    return folder, meta


def normalize(text):
    return " ".join(re.findall(r"\w+", text.lower()))


def benchmark_data():
    specs = [
        ("gsm8k", "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", lambda p: p.startswith("main/") and "/test-" in p),
        ("math", "DigitalLearningGmbH/MATH-lighteval", "0530c78699ea5e8eb5530600900e1f328b48acad", lambda p: p.startswith("data/") and "/test-" in p),
        ("mmlu", "cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", lambda p: p.startswith("all/") and any(x in p for x in ("/test-", "/dev-"))),
    ]
    rows, meta = [], []
    for name, repo, revision, include in specs:
        folder, inventory = download(repo, revision, include=include)
        for file in inventory["files"]:
            split = "dev" if "/dev-" in file["file"] else "test"
            part = pq.read_table(folder / file["file"]).to_pylist()
            for idx, item in enumerate(part):
                rows.append(dict(benchmark=name, split=split, index=idx,
                                 source_file=file["file"], **item))
        meta.append(dict(benchmark=name, **inventory))
    counts = Counter((r["benchmark"], r["split"]) for r in rows)
    assert counts["gsm8k", "test"] == 1319, counts
    assert counts["math", "test"] == 5000, counts
    assert counts["mmlu", "test"] == 14042, counts
    assert counts["mmlu", "dev"] == 285, counts
    path = DATA / "benchmarks.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (DATA / "benchmark-sources.json").write_text(json.dumps(meta, indent=2) + "\n")
    emit("BENCHMARKS_READY", counts={f"{b}/{s}": n for (b,s),n in counts.items()}, sha256=sha(path))
    return rows


def sample(folder, inventory, benchmarks):
    # Shuffle a fixed pool of row indices, then rejection-sample eligible records.
    # This is uniform over eligible rows; no source or length preference is applied.
    total = sum(x["rows"] for x in inventory["files"])
    candidates = random.Random(SEED).sample(range(total), min(total, 60000))
    candidate_set = set(candidates)
    pool = {}
    offset = 0
    for file in inventory["files"]:
        parquet = pq.ParquetFile(folder / file["file"])
        selected = [i-offset for i in candidate_set if offset <= i < offset+file["rows"]]
        if selected:
            table = parquet.read()
            for i, row in zip(sorted(selected), table.take(sorted(selected)).to_pylist()):
                pool[offset+i] = (row, file["file"], i)
        offset += file["rows"]
    exact = set()
    anchors = {}
    for b in benchmarks:
        q = b.get("question", b.get("problem", ""))
        normalized = normalize(q)
        if len(normalized) < 30:
            continue
        exact.add(normalized)
        words = normalized.split()
        if len(words) >= 13:
            for i in {0, (len(words)-13)//2, len(words)-13}:
                anchors.setdefault(tuple(words[i:i+13]), b["benchmark"])
    selected, seen, exclusions = [], set(), Counter()
    for index in candidates:
        row, file, local_index = pool[index]
        conv = row["conversations"]
        if not conv or not any(m["from"] in ("gpt", "assistant") and m["value"].strip() for m in conv):
            exclusions["missing_assistant"] += 1
            continue
        questions = [normalize(m["value"]) for m in conv if m["from"] in ("human", "user")]
        overlap = False
        for q in questions:
            if q in exact:
                overlap = True
                break
            w = q.split()
            if any(tuple(w[i:i+13]) in anchors for i in range(max(0, len(w)-12))):
                overlap = True
                break
        if overlap:
            exclusions["benchmark_overlap"] += 1
            continue
        content_sha = hashlib.sha256(json.dumps(conv, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if content_sha in seen:
            exclusions["duplicate_conversation"] += 1
            continue
        seen.add(content_sha)
        selected.append(dict(id=f"opb-{index}", row_index=index, source_file=file,
                             row_in_file=local_index, content_sha256=content_sha, **row))
        if len(selected) == 12000:
            break
    assert len(selected) == 12000, (len(selected), exclusions)
    manifests = []
    for split, start, end in [("train",0,10000),("calibration",10000,11000),("heldout",11000,12000)]:
        path = DATA / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected[start:end]))
        manifests.append(dict(split=split, count=end-start, sha256=sha(path), bytes=path.stat().st_size,
                              sources=dict(Counter(str(r["source"]) for r in selected[start:end]))))
    manifest = dict(dataset=REPO, revision=REV, seed=SEED, sampling="uniform row-index rejection sampling",
                    dataset_rows=total, candidate_pool=60000, splits=manifests, exclusions=dict(exclusions),
                    decontamination="Normalized exact query and 13-word benchmark query anchors, checked across all user turns; no semantic paraphrase guarantee.")
    (DATA / "sample-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    emit("SAMPLE_READY", **manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Output/cache directory outside the source checkout")
    parser.add_argument("--endpoint", default="https://huggingface.co", help="HF origin or a compatible mirror, using identical pinned revisions")
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    DATA = args.data_dir.expanduser().resolve()
    API = HfApi(endpoint=args.endpoint, token=False)
    # Pass the endpoint explicitly to both metadata and content requests.
    original_hf_hub_download = hf_hub_download
    def hf_hub_download(*args_, **kwargs):
        return original_hf_hub_download(*args_, endpoint=args.endpoint, **kwargs)
    DATA.mkdir(parents=True, exist_ok=True)
    # The requested complete dataset is downloaded even though only a sample is trained.
    folder, inventory = download(REPO, REV)
    benchmarks = benchmark_data()
    sample(folder, inventory, benchmarks)
