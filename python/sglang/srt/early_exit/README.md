# GPU early exit and concurrent cache repair

The experimental path targets the calibrated Qwen3.5-family 27B checkpoint:
64 layers, width 5120, a source after layer32, actions `[0,5,6,7,11]`. The
classifier stays FP32 and the frozen language model stays BF16. Currently
the runtime requires a Hopper GPU, CUDA12.9, PyTorch2.11, TP1/PP1,
the Triton attention/GDN backends, and a decode CUDA graph. The experimental
batch path supports ordinary decode with one to eight requests.

`SGLANG_EARLY_EXIT_CHECKPOINT=/shared/classifier-multiclass.pt` enables the
path. `SGLANG_EARLY_EXIT_CACHE=async` overlaps cache repair with the suffix;
`serial` evaluates the identical repair before continuing the main path.
`SGLANG_EARLY_EXIT_FORCE=-1` runs the calibrated policy. Setting it to an
action width is a control for differential testing. These variables are
experimental runtime controls rather than public serving API parameters.

`four_width_bce_mlp_v1` checkpoints load the recall sweep's single-token,
one-hidden-layer BCE head (1024 ReLU units, FP32). Its four sigmoid outputs
predict safety independently at widths 5/6/7/11. The policy selects the longest
width meeting the checkpoint's calibration threshold; scores are not summed.
The original `five_class_shared_mlp_v1` format retains argmax selection.
Worker receipts record the classifier format, threshold and checkpoint hash.

At the source boundary the GPU computes the action and saves the hidden
and residual tensors. The main branch bypasses the selected prefix of
layers33–43 and runs the remaining middle layers and layers 44–64. The
repair branch starts at the saved boundary and evaluates the skipped prefix,
updating its full-attention KV and GDN convolution/recurrent state. Main and
repair predicates are complementary for four fixed layer segments. This
executes each middle layer exactly once across the two branches.

At each decode step, `split_skip` partitions real rows on the GPU into a
no-skip group (proposal 0) and a skip group (positive proposals). The skip group
uses the shortest positive proposal as its common jump. Padding proposals are
`-1` and enter neither group. Empty groups execute no middle layers. Raw
proposals, applied row actions, group membership and counts remain on device.

Layers 1–32 use the original batch. GPU gathers pack each group's hidden state,
RoPE/MRoPE positions and request/cache indices in source order. Each nonempty
group selects the smallest captured capacity in 1/2/4/8 that fits its count;
CUDA IF nodes enable only that capacity's middle operators. Group outputs merge
back into source order before layers 44–64. There are twelve marked segments per
capacity: four no-skip main, four skip main and four skip repair. Each nonempty
group executes four complementary segments per step. Real-row executions,
physical region launches, group sizes and capacity padding have separate counters.

Full-attention ragged KV indices and GDN state indices follow the gathered
requests. Repair uses private full-attention scratch. Distinct groups can
execute the same layer concurrently; the packed BF16/NHD KV writer suppresses
padding stores into reserved slot 0, while GDN padding retains index -1. This
keeps their real cache writes disjoint. The GPU probe checks actual KV stores,
and the padded-batch smoke checks a separately allocated resident request as a
cache canary.

Common positive jumps can change policy semantics when positive widths differ.
Quality, proposal/applied histograms, group occupancy and fixed-input throughput
are evaluated independently using the same frozen head. Prefix/suffix batching
is shared; only the middle path splits. A split of 2/6 within batch 8 therefore
uses capacities 2/8, while a 4/4 split uses 4/4.

CUDA conditional graph nodes read the device action directly. Capture-time
markers delimit the regions; the CUDA helper clones the captured graph,
retains each complete layer region and embeds it as an IF body child graph.
This preserves captured library launch data and PDL dependency metadata. Replay
does not read the action on the CPU and does not traverse Python layer
wrappers. Persistent boundary buffers retain the input when a region is
bypassed. Unsupported eager decode fails before modifying caches.

The two streams share per-layer caches but use separate full-attention
scratch. A device event forks after the boundary copy and joins before the
decode graph finishes. The next token therefore observes every cache write.
This version overlaps within one token. Cross-token overlap would require
separate graph launches and per-layer completion dependencies before a token
can re-enter a repaired layer.

Cache repair evaluates transformer layers and adds roughly the full model's
arithmetic and weight traffic. Its possible benefit is a shorter dependency
chain. HBM and SM contention can erase the overlap benefit; the runtime
reports measured latency rather than inferring speed from skipped layers.
This cache policy differs from the native experiment's frozen skipped
caches. Its quality is evaluated independently using the same fixed head.

Run the direct CUDA branch probe, then the real model differential smoke:

```sh
RUN_DIR=/shared/probe python -m sglang.srt.early_exit.probe
RUN_DIR=/shared/kernel-probe python -m sglang.srt.early_exit.kernel_probe
RUN_DIR=/shared/smoke python -m sglang.srt.early_exit.smoke \
  --model /shared/model --classifier /shared/classifier-multiclass.pt
```

The smoke starts independent full/serial/async model processes on the same
GPU, excludes an initial warmup, checks disabled-route logits/tokens against
the full model, compares mixed-action serial/async logits and repaired caches,
and records complementary execution counters and a separate profiler trace.
Before loading the model it checks captured BF16 linear, Gemma normalization
and Triton kernels in isolated processes. `--diagnose` runs these operators
and a one-token model replay under Compute Sanitizer for fault localization.
Small CUDA probes establish graph behavior; complete benchmark claims require
the evaluator's full question counts and successful managed-run receipts.

For small-batch correctness, run:

```sh
RUN_DIR=/shared/batch-smoke python -m sglang.srt.early_exit.batch_smoke \
  --model /shared/model --classifier /shared/classifier-multiclass.pt \
  --batch-sizes 1 2 3 4 8 --max-batch-size 8 --steps 16
```

This entrypoint checks GPU partitioning, group-width selection, capacity
selection, conditional execution and padding in a synthetic graph, then starts
full/serial/async ModelRunner processes on the same GPU. Heterogeneous row
proposals exercise actions 0/5/6/7/11 and mixed group sizes while following the
full model's input trajectory. It verifies per-step membership and applied
actions, per-row, per-group and physical execution counts,
serial/async bitwise cache/logit equality, and graph padding isolation (batch3
uses graph4). Full/no-skip parity retains fixed absolute tolerance0.02 and
relative tolerance0, plus exact generated tokens. A failed control writes its
report and exits with an error. These timings are correctness diagnostics.
