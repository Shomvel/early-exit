# Verified concurrent cache backfill

Run `pb-sglang-smoke-1120-r6` completed successfully on an H200 using the
pinned SGLang CUDA12.9 image. The GPU job, its unique replica and lifecycle
controller reached `Succeeded`; runtime exit code was 0. Run UUID, immutable
overlay commit, image digest and result SHA256 values are in
[`run-status.json`](run-status.json).

The main decode path and skipped-layer cache repair use complementary CUDA
conditional branches. Decisions stay on device. Repair updates both full
attention KV and GDN convolution/recurrent state, then joins before the next
decode graph. Captured regions are extracted using CUDA graph clones, which
retain library launch data and PDL edge metadata. Five real-operator
regressions passed before the model loaded. The earlier r5 diagnostic
localized an illegal instruction in reconstructed cuBLAS NVJET kernel nodes;
r6 operator regressions used ordinary execution, without Compute Sanitizer.

Four differential checks were **bitwise exact**, each with maximum absolute
difference 0: full versus serial action0 logits; full versus async action0
logits; serial versus async mixed-action logits; and serial versus async
mixed-action KV/GDN state. Generated action0 tokens matched the full model.
All four segments had exactly one main-or-repair execution per decode step.

| 32-token control | CUDA elapsed, ms |
|---|---:|
| Full model | 489.227 |
| Serial repair, mixed actions | 493.265 |
| Async repair, same mixed actions | 480.470 |
| Serial repair, learned policy | 498.548 |
| Async repair, learned policy | 488.217 |

The mixed-action serial-to-async ratio was 1.02663×. The learned-policy
full-to-async ratio was 1.00207×. These are single short controls in separate
full/serial/async processes on the same GPU, with warmup excluded. They are
not benchmark speed estimates. Mixed actions use identical forced full-model
token inputs; the learned-policy control generates freely. Process order,
clock and load drift can affect these short timings.

A separate 8-step forced-width11 profiler pass recorded 31.077ms of
cross-stream kernel overlap for async and 0 for serial. Both traces and their
SHA256 values remain in the shared run. The sealed
[`smoke summary`](sglang-runtime-smoke-summary.json) includes every control,
route histogram, execution count, comparison and trace hash.

Cache repair executes the skipped layers, so both branches together perform
roughly the full model's weight traffic. Complete GSM8K/MATH/MMLU accuracy
and speed require a separately completed benchmark run.
