# Open-PerfectBlend early-exit experiment

The reproducible experiment lives in [`experiments/perfectblend`](../experiments/perfectblend/README.md). It uses 10,000 sampled Open-PerfectBlend conversations to fit a small classifier on a frozen 27B SFT checkpoint, with separate 1,000-conversation calibration and heldout splits. SGLang runtime changes, native training, data preparation and benchmark entrypoints are kept in the same repository.

## Geometry and labels

The model has 64 transformer blocks and hidden size 5,120. The pilot scans source block outputs 8, 9, 10, 11, 12, 16, 24, 32, 40 and 48, with skip widths 1–7, 11, 15 and 23 where `source + width < 64`. It uses only the first 640 training conversations: the first 512 for fitting, the next 128 for development. A small three-epoch head estimates saved blocks at 99% development safety precision; 95% breaks ties. Four widths are retained per source, then the source is selected by the same metric. This training-data pilot selected source **32** and widths **5, 6, 7, 11**.

Block numbers in experiment files are **one-based**. Source 32 means the hidden output after block 32. Widths 5, 6, 7 and 11 bypass blocks 33–37, 33–38, 33–39 and 33–43 respectively. The tail blocks, final norm and LM head still execute. The SGLang code translates this to zero-based layer indices at the runtime boundary.

Final labels are captured with native full-history caches. For each conversation the script chooses up to eight consecutive observed positions in the first 256 answer tokens, with a 1,024-token context cap. At every position it runs the full next-token computation and counterfactual bypasses from identical prior history. Each width's binary target is whether its top-1 next token equals the full model's top-1 token. These targets need not be monotonic in width. Realized position counts are 79,865 train, 7,993 calibration and 7,984 heldout.

The pilot uses static no-cache full-context counterfactuals to select geometry. Final training uses the separate cached capture described above. All four captures preserve raw teacher-forced inputs, per-width predictions and full-model targets, allowing the binary and five-class heads to share exactly the same evidence. Cache controls check trace boundaries, preserved/skipped cache state and disabled-gate parity.

## Heads and calibration

The binary `SharedGate` RMS-normalizes FP32 hidden values, concatenates a candidate-width one-hot vector and evaluates the **same** `Linear(5124,1024) → ReLU → Dropout(0.1) → Linear(1024,1)` for four candidates. Four sigmoid scores train with binary cross entropy. Execution chooses the longest width passing a common threshold, otherwise 0.

The five-class head processes hidden once: `Linear(5120,1024) → ReLU → Dropout(0.1) → Linear(1024,5)`. The softmax action order is `[0,5,6,7,11]`. Targets choose the longest actually safe width among the four binary labels, or 0 when none is safe. Six epochs of unweighted cross entropy use the same training positions. At inference, argmax 0 or confidence below threshold selects no bypass. Dropout is disabled for inference.

Both use FP32 RMS normalization `h * rsqrt(mean(h²).clamp_min(1e-12))`, AdamW learning rate `1e-4`, weight decay `0.03`, batch size 1,024, six epochs and seed 20261120. The 27B model is frozen. Calibration thresholds maximize mean saved blocks subject to at least 32 accepted calibration positions and target precision 95%/99%. Heldout positions do not tune thresholds; benchmark tasks do not tune geometry or policy.

| cal99 policy | Threshold | Calibration action precision | Heldout action precision | Heldout coverage | Mean skipped blocks |
|---|---:|---:|---:|---:|---:|
| Shared sigmoid | 0.9307399958968162 | 99.0009% | 98.6910% | 31.5757% | 3.18512 |
| Shared five-class softmax | 0.9885793690681457 | 99.0213% | 98.7718% | 31.6132% | 3.47745 |

The five-class cal99 policy accepts only width 11 in this run. Longest-safe labels are heavily skewed toward 11, and the explicit no-skip class has low recall. See the [full class counts, confusion matrix and head latency comparison](../experiments/perfectblend/references/five-class-report.md). Head-only CUDA timings and expected skipped-block fractions describe different quantities from full serving acceleration.

## Evaluation protocol

The main quality baseline is full execution of the same checkpoint 1479, with the same prompts, tokenizer, decoding, hardware and runtime. GSM8K contains 1,319 test questions; original MATH contains 5,000; original MMLU contains 14,042 test questions and its 285 dev examples for five-shot prompts. Pinned revisions, parquet hashes, sample hashes and model shard hashes are under `references/`.

The native evaluator pairs full and early-exit execution on the same GPU, alternating order per question. GSM8K and MATH use nonthinking greedy chat generation with limits 2,048/4,096 tokens and preserve predictions, references, correctness, generated length, truncation, prefill/decode timing and per-request skip statistics. MATH uses normalized boxed exact match. MMLU uses five dev examples for each subject and the A/B/C/D single-token next-word scores, with micro and 57-subject macro accuracy. Decode speed excludes prefill; total time includes it. A separate matched-token replay controls for output-length differences.

SGLang runners use the corresponding full-model ModelRunner as the performance baseline. Runtime, graph mode, model and policy digest, concurrency, token limits and the measured execution path must accompany reported speedups. Hybrid GDN recurrent state and full-attention KV behavior during skipped steps are part of the runtime correctness contract. Native one-step training labels do not by themselves verify a serving cache policy or repeated free-generation decisions.

The original native full benchmark run was intentionally stopped before completion. Small smoke runs establish execution and parity only. Its partial output does not provide the three complete task scores or a full speedup. SGLang evaluation results should be reported from their own completed receipts rather than from the historical native smoke summary.

The inspected official Qwen3.8-27B card revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0` omits these three original benchmark scores. The experiment therefore compares against its own full checkpoint execution. The official model card evidence is retained in `references/official-card.json`.

## Execution and provenance

The [README](../experiments/perfectblend/README.md) gives data, fingerprint, managed launch and collection commands. GPU runs require a clean immutable shared checkout, exact image digest, precise argv, a new run UUID and final reconciliation of Job and every expected replica. `managed.py` delegates this lifecycle to the separately installed YiCloud `training_run.py`; it does not create GPU jobs itself or carry credentials.

The repository starts from SGLang `v0.5.13.post1`, commit `85fd90072d1a9f2432842b03588f63b745e524e4`, matching the verified `v0.5.13.post1-cu129` runtime image. The native validated experiment source and five-class source commits and original checksums are in [`references/source-provenance.json`](../experiments/perfectblend/references/source-provenance.json). Packaging adds configurable paths, explicit inventories and reusable entrypoints. The historical result files refer to their historical validated sources; packaged runs record a new commit.
