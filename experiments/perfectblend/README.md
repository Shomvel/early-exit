# Open-PerfectBlend 27B early exit

This directory contains the data preparation, geometry selection, frozen-model feature capture, shared classifier training, calibration, native evaluation, and SGLang evaluation entrypoints for the experiment. The repository is [Shomvel/early-exit](https://github.com/Shomvel/early-exit), based on SGLang `v0.5.13.post1` (`85fd90072d1a9f2432842b03588f63b745e524e4`). The model used here is the prior Qwen3.8-27B SFT checkpoint 1479, a 64-block Qwen3.5-family hybrid full/GDN attention model with hidden size 5,120. All base weights remain frozen.

## Files

| Entry | Purpose |
|---|---|
| `prepare_data.py` | Download all six pinned Open-PerfectBlend shards and benchmark sources; sample 10k/1k/1k conversations |
| `verify_data.py` | Verify pinned JSONL hashes, conversation hashes, disjoint splits, benchmark counts and optional raw download hashes |
| `fingerprint_model.py` | Hash all indexed safetensors shards and stamp their shared-storage metadata |
| `native/pipeline.py` | Smoke, training-data pilot scan, cached counterfactual feature capture, binary head fitting and calibration |
| `train_multiclass.py` | Fit the five-action shared MLP from the same already captured features |
| `native/evaluate.py` | Native Transformers full/early-exit paired benchmark evaluator |
| `sglang_run.py`, `sglang_eval.py` | SGLang server orchestration and benchmark evaluation; see each script's `--help` |
| `managed.py` | Build raw specs and delegate prepare, submit, reconcile and status to an external `training_run.py` |
| `report.py` | Render a **completed native** benchmark comparison from successful run receipts; full counts are required |
| `references/` | Small manifests, training summaries, source provenance and verified historical status; no model or raw dataset |

Outputs, downloaded data, checkpoints and capture shards belong outside the source checkout. The local `.gitignore` excludes those artifact types. [Experiment details](../../docs/early_exit_experiment.md) describe the method and its limits. [Verified five-class results](references/five-class-report.md) compare the heads.

## Data preparation

Use Python 3.11 or later. The CPU downloader needs `requirements-data.txt`; it does not need a GPU or model dependencies. Install dependencies in the data-preparation environment before launching an offline GPU workload.

```bash
python3 -m pip install -r experiments/perfectblend/requirements-data.txt
python3 experiments/perfectblend/prepare_data.py --data-dir /mnt/work/early-exit/data
python3 experiments/perfectblend/verify_data.py --data /mnt/work/early-exit/data --include-downloads
```

The default endpoint is `https://huggingface.co`; `--endpoint` supports a compatible mirror with the same pinned revisions. No HF token is required for these public inputs. The script downloads the complete Open-PerfectBlend revision `af60f3c18201652a83a93f46fcfee1b646ba3df7` (1,420,909 rows), shuffles 60,000 row indices with seed 20261120, and rejection-samples 12,000 eligible unique conversations: 10,000 train, 1,000 calibration, 1,000 heldout. It excludes missing assistant messages, normalized exact benchmark queries, and three anchored 13-word spans per benchmark query across all user turns. The recorded sample excluded 546 missing-assistant rows and 132 overlaps. The exact recipe is fixed in source; changing it defines a new dataset version and needs new reference manifests.

GSM8K uses 1,319 test questions; original MATH uses 5,000 test questions; original MMLU uses 14,042 test questions and 285 dev questions across 57 subjects. Revisions and source parquet hashes are in `references/benchmark-sources.json`. `verify_data.py` checks the recorded recipe's exact data artifacts; it is not a generic validator for newly resampled datasets.

## Model and runtime

The original shared model directory is `/gpfs/shared-model-weight/qwen38-20261019/hf-model`. Its shard hashes are in `references/model-weight-inventory.json`. For a new storage copy, verify full SHA256 identity and write a new inventory outside the checkout:

```bash
python3 experiments/perfectblend/fingerprint_model.py create \
  --model /gpfs/shared-model-weight/qwen38-20261019/hf-model \
  --reference experiments/perfectblend/references/model-weight-inventory.json \
  --output /mnt/work/early-exit/model-inventory.json
```

The reference's mtime values identify the original GPFS copy. `create` compares shard names, sizes and SHA256, then records the new copy's mtimes. Workers check shard inventory completeness, sizes and mtimes before loading, plus config/index hashes if present. A full verification can be repeated with `fingerprint_model.py verify --model ... --inventory ...` before launch. Keep the model immutable between hashing and execution.

The native stages need CUDA/H200, PyTorch, NumPy, Transformers with `Qwen3_5DynamicCache`, and the compatible FLA/causal-conv GDN kernels. The original validated native image is pinned in `configs/payload.example.json`; it includes the model dependencies. `training_run.py` is an external YiCloud control tool and requires its **complete** supporting scripts directory. This repository does not vendor that tool or credentials. Use its current `launch --help` and site contract before preparing jobs.

SGLang core is pinned to `v0.5.13.post1`; the corresponding verified source/runtime image is `registry.lg.shzhisuan.com/library/sglang:v0.5.13.post1-cu129@sha256:163e3f6e921a03b2eaaaf8e2f4f9b7ddf98d7bef54e2deddfadadb3733ae14d4`. Use the SGLang image and runner for serving evaluation. The native trainer image and SGLang serving image are separate environments.

## Managed launch

Run these commands on a control host where both the clean source checkout and shared output paths are visible to the GPU container. The examples use configurable task paths; stage data and commit source changes first. Each run needs a unique name and one immutable source commit.

Copy `configs/payload.example.json` to an external task directory and set the actual project, quota, image digest, mounts, GPU count and runtime limit. The example has **one replica with four H200s**. Native workers each load a complete model on one GPU; they are independent benchmark/data shards, not tensor parallel ranks. Match `--expected-gpus` and `--workers` to the payload. The five-class captured-feature stage requires one GPU and `workload_kind=module`.

```bash
python3 experiments/perfectblend/managed.py plan \
  --repo /mnt/work/early-exit/source \
  --runs-root /mnt/work/early-exit/training_runs \
  --payload-file /mnt/work/early-exit/control/native-payload.json \
  --output /mnt/work/early-exit/control/pilot-r1-spec.json \
  --run-name pb-pilot-new-r1 --stage pilot --expected-gpus 4 --workers 4 \
  --model /gpfs/shared-model-weight/qwen38-20261019/hf-model \
  --data /mnt/work/early-exit/data \
  --weight-inventory /mnt/work/early-exit/model-inventory.json
```

The generated raw spec records HEAD, image digest, exact argv, workload kind and GPU topology. `plan` is local and does not prepare or submit a cloud run. Prepare once, inspect the tool's recorded plan/dry-run and UUID, then execute that prepared run:

```bash
python3 experiments/perfectblend/managed.py prepare \
  --training-run /mnt/work/early-exit/tool/scripts/training_run.py \
  --spec /mnt/work/early-exit/control/pilot-r1-spec.json \
  --runs-root /mnt/work/early-exit/training_runs --direct

python3 experiments/perfectblend/managed.py show \
  --training-run /mnt/work/early-exit/tool/scripts/training_run.py \
  --run /mnt/work/early-exit/training_runs/pb-pilot-new-r1

python3 experiments/perfectblend/managed.py submit \
  --training-run /mnt/work/early-exit/tool/scripts/training_run.py \
  --run /mnt/work/early-exit/training_runs/pb-pilot-new-r1 --direct --execute
```

`prepare` delegates `training_run.py launch` without `--execute`. `submit` without `--execute` is a submission dry-run. With `--execute`, it immediately follows submission with foreground `reconcile --watch --execute`, including after an uncertain API failure. Keep the control process alive until terminal reconciliation. `managed.py reconcile --training-run ... --run ... --watch --execute` resumes observation using the existing identity. Credentials enter the external tool's process environment or its `--credentials-file`; keep values out of payload/spec/source. Do not prepare the same run twice or retry an ambiguous CreateJob response under a new identity before reconciliation. A failed attempt stays archived; `--retry-of UUID` links a new run.

Built-in stages and extra inputs:

| `--stage` | Extra inputs | Outputs in the tool-provided `RUN_DIR` |
|---|---|---|
| `smoke` | model/data/inventory | Cache and no-skip correctness controls on eight conversations |
| `pilot` | model/data/inventory | `selection.json`, pilot states and source-specific small heads |
| `capture` | model/data/inventory, `--pilot-run /shared/pilot-run` | `capture-N.pt`, `classifier.pt`, `summary.json` |
| `multiclass` | `--capture-run /shared/capture-run` | `classifier-multiclass.pt`, `summary.json`, scores |
| `eval-smoke` | model/data/inventory, `--classifier-run /shared/capture-run` | Small execution check; no full benchmark claim |
| `eval` | same as eval-smoke | Paired predictions and complete `evaluation-summary.json` |
| `custom` | `--entrypoint path/in/repo.py --arguments-file /external/argv.json` | Defined by that script; explicit `--workload-kind training` or `module` |

The original five-class script deliberately requires the recorded four capture shards and baseline classifier SHA256. Its `SOURCE`, `SKIPS`, expected counts and label definition describe that experiment. A different geometry or new captures require a versioned script/config update, not silently reusing those constants.

For SGLang use `custom` and a repository-relative `experiments/perfectblend/sglang_run.py` (real-model differential smoke) or `experiments/perfectblend/sglang_eval.py` (benchmarks) entrypoint. Pass a JSON array of its arguments. The command is executed from the prepared immutable source checkout; it writes outputs under `RUN_DIR`. Use `workload_kind=training` for full-model smoke/evaluation.

The SGLang runtime uses CUDA conditional nodes for device-only decisions and overlaps a skipped middle prefix with the main decode suffix on a second stream. The background branch repairs both KV and GDN state before the next decode step. `--cache-policy serial` provides an identical-policy serial reference; the independent full-model process is the main quality/speed baseline. See the [runtime contract](../../python/sglang/srt/early_exit/README.md).

```bash
RUN_DIR=/shared/smoke python3 experiments/perfectblend/sglang_run.py \
  --model /shared/model --classifier /shared/classifier-multiclass.pt

RUN_DIR=/shared/eval-smoke python3 experiments/perfectblend/sglang_eval.py \
  --model /shared/model --data /shared/data \
  --classifier /shared/classifier-multiclass.pt --workers 2 --smoke
```

Omit `--smoke` for all 20,361 test questions. The evaluator requires the pinned benchmark SHA and model inventory; pass `--weight-inventory /shared/inventory.json` when using another inventory path. It checks graph execution and complementary repair counters at every request, preserves original predictions, and reports generation lengths, accuracy, latency and separate 128-token matched replay. `complete_benchmarks` is true only when all three full question counts are present. A successful smoke has `complete_benchmarks=false`.

The evaluator also supports four nodes with eight independent GPUs each: pass
`--workers 32 --local-workers 8` in a four-replica managed Job. Replica identity
comes from `KUBEBRAIN_REPLICA` and `KUBEBRAIN_REPLICA_TOTAL`. Each replica runs its
eight full-model workers before the corresponding early-exit workers; replica
zero waits for all 64 worker receipts and four completion markers before
aggregating. Runtime overlays and their receipts use separate
`runtime-replica-N` directories under the shared run. The concurrent real-launcher
regression is `python3 -m unittest discover -s experiments/perfectblend/tests -p test_launch_overlay.py`.

For this experiment's 8-to-32-GPU migration, `--resume-from /shared/snapshot`
accepts an immutable `early_exit_evaluation_resume_v1` snapshot bound to the
original run UUID. It checks every file's SHA256, configuration, model,
classifier, benchmark, tokenizer, runtime and evaluator protocol, then verifies
each imported question and prompt. Imported full predictions contribute to
accuracy and generation-length statistics. Free-generation speed comparisons
use newly computed same-GPU pairs; all fixed-token replays are measured again
on the new paired GPU. A snapshot contains the original run/config, five source
files, complete JSONL lines from all eight original full workers, and its
`resume-manifest.json`. See `evaluation_resume.py` for the schema. The recorded
migration retained 6,111 full predictions; it does not resume the user-stopped
native Transformers run.

The optional `launch_overlay.py --module sglang.srt.early_exit.smoke ...` launcher accepts an immutable source overlay with `overlay-manifest.json`. It checks the base source hashes against the pinned image, copies its package into `RUN_DIR`, and applies only hash-verified modifications. Standard full checkouts can use the direct entrypoints above.

The native scripts can also run directly in the verified image with explicit `RUN_DIR` and identical argv, but managed GPU execution should retain UUID, source/image and terminal replica receipts. Scripts write nonempty resolved `config.yaml` files at runtime; all 27B model weights stay frozen.

## Results and reporting

At the recorded cal99 threshold, the original sigmoid gate achieved heldout one-step action precision 98.6910%, coverage 31.5757% and mean skipped blocks 3.18512. The five-class head achieved 98.7718%, 31.6132% and 3.47745. These are **teacher-forced full-history one-step** counterfactual label metrics. The five-class cal99 accepted positions all chose width 11. They do not measure generation accuracy or end-to-end serving acceleration. Head-only H200 timings and class imbalance are documented in `references/five-class-report.md`.

The original native smoke/evaluation smoke succeeded. The native full benchmark run was stopped at the user's request and has no complete benchmark summary. Its partial predictions remain in the original shared run. `references/run-status.json` records the status; this package does not resume that stopped run.

The [verified SGLang runtime smoke](references/sglang-runtime-report.md) passed exact full/no-skip and serial/async cache/logit controls, and recorded actual GPU stream overlap. Its short control timings are separate from complete benchmark scores and speed estimates.

The [32-GPU batch-one evaluation](references/sglang-full-benchmark-report.md)
completed all 20,361 questions in each mode. GSM8K accuracy was 96.74%/95.91%,
MATH 89.30%/84.74% and MMLU 80.79%/80.81% for full/early exit. Fixed128 replay
speedups were 1.0063×/1.0098× for GSM8K/MATH; MATH end-to-end speedup was 0.8862×.
All 64 worker receipts and four runtime exits passed. `references/run-status.json`
records the immutable identities. Multi-request runtime and throughput
experiments retain independent sources and results.

The separate small-batch request evaluator uses real, different-length prompts:

```sh
PYTHONPATH=python RUN_DIR=/shared/batch-pilot python -m sglang.srt.early_exit.batch_evaluate \
  --model /shared/model --classifier /shared/classifier-multiclass.pt \
  --data /shared/data --batch-sizes 1 2 4 8 \
  --examples-per-task 16 --replay-examples-per-task 8 --matched-tokens 128
```

Run `sglang.srt.early_exit.batch_smoke` first to verify the runtime. The request
evaluator runs full and early exit in separate processes on the same GPU. It
handles per-request EOS and generation limits, scores MMLU choices, and records
each microbatch's elapsed time once. Effective tokens, finished-row work and
CUDA graph padding are separate counters. All batch sizes and modes replay the
same full batch-one 128-token trajectories for fixed-input timing. This entrypoint
is a throughput pilot and always reports `complete_benchmarks=false`. The full
20,361-question evaluation remains a separate run.

The historical [small-batch GPU verification](references/sglang-batch-runtime-report.md)
passed full/no-skip and serial/async mixed-route bitwise logits/cache controls at
batch1/2/3/4/8, with exact route, padding and exclusive batch-eight branch counts.
The real-request batch1/2 entrypoint E2E also succeeded. The completed
[per-request throughput pilot](references/sglang-per-request-batch-pilot-report.md)
records its independent source and all batch sizes. Its fixed128 EE/full
throughput ratios were 1.011/0.885/0.708/0.542 at batch 1/2/4/8.

The current runtime implements `split_skip`: after source 32, GPU partitioning
forms one no-skip batch and one common-width skip batch. Each group uses its
smallest capacity 1/2/4/8, with original request/cache ownership restored before
the shared suffix. The [split-group GPU smoke](references/sglang-split-smoke-report.md)
passed full/no-skip and serial/async mixed bitwise logits/cache checks for
batch 1/2/3/4/8, device partition/capacity checks and padding canaries. The completed
[split-group throughput pilot](references/sglang-split-batch-pilot-report.md)
measured fixed128 EE/full ratios of 0.959/0.904/0.871/0.879 at batch 1/2/4/8.
At batch8, full reached 478.70 token/s and grouped EE reached 420.55 token/s;
group-capacity utilization was 81.0%. Grouping improves the historical
per-request ratio at larger batches, while the current path remains slower
than full execution. Variable-length generation separately reports completed-row
work and output-length changes. The pilot retains an independent source/run
identity and does not replace the complete batch1 quality evaluation. See the
[runtime contract](../../python/sglang/srt/early_exit/README.md) for device-only
selection, cache backfill, padding stores and acceptance checks.

After a successful completed native run, collect small receipts and render the comparison:

```bash
python3 experiments/perfectblend/managed.py collect \
  --run /mnt/work/early-exit/training_runs/pb-eval-new-r1 \
  --output-dir /mnt/work/early-exit/receipts

python3 experiments/perfectblend/report.py \
  --artifacts /mnt/work/early-exit/receipts \
  --data-manifest /mnt/work/early-exit/data/sample-manifest.json \
  --pilot pb-pilot-new-r1 --capture pb-capture-new-r1 --evaluation pb-eval-new-r1 \
  --output-dir /mnt/work/early-exit/report
```

Collect pilot and capture receipts in the same directory first. Collection requires reconciled `Succeeded` state and every expected runtime replica exit 0. Rendering checks summary hashes, the trained classifier hash, selected geometry and exact full benchmark counts; it rejects stopped and smoke-only results. This renderer describes the native paired protocol; SGLang output uses its own evaluation summary.

The primary baseline is the **full execution of the same SFT checkpoint 1479**. The inspected official Qwen3.8-27B model card revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0` does not list GSM8K, original MATH or original MMLU scores. Official figures for other suites are not substituted into this comparison.

Historical validated sources are recorded in `references/source-provenance.json`. Packaging adds explicit path/inventory configuration and the portable control wrapper; a new packaged run must record its new commit rather than claim the original byte-identical source. `native/math_equivalence.py` preserves the [Hendrycks MATH normalization](https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py) used in the original experiment, with source attribution and string-literal cleanup.

## Experimental deferred prefill on re-entry

`SGLANG_EARLY_EXIT_CACHE=lazy` defers skipped KV/GDN work until a request's applied skip width decreases. Saved layer32 hidden/residual history is replayed with causal chunk prefill under GPU CUDA WHILE control. The experimental runner supports static request batches and resets pending state when clearing the pools. [Implementation and launch status](references/lazy-prefill-status.md) records the controls and current platform block; GPU correctness and performance are still unverified.

Use the managed `custom` stage with `--entrypoint experiments/perfectblend/sglang_lazy.py`, `--workload-kind training`, one GPU, and an external arguments-file containing the model, classifier and inventory paths. Its `--stage smoke` runs the full/no-skip, eager-oracle, long-hole and padding controls. After a passing smoke, `--stage evaluate --data /shared/data` runs the same-GPU full/async/lazy pilot at batch1/2/4/8 and writes `lazy-evaluation-summary.json`. Both accept `--chunk 16`; smoke writes `lazy-smoke-summary.json`. The existing completed benchmark scores belong to their recorded async runtimes.
