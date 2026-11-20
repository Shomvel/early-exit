# GPU early exit and concurrent cache repair

The experimental path targets the calibrated Qwen3.5-family 27B checkpoint:
64 layers, width 5120, a source after layer32, actions `[0,5,6,7,11]`. The
classifier stays FP32 and the frozen language model stays BF16. Currently
the runtime requires a Hopper GPU, CUDA12.9, PyTorch2.11, TP1/PP1, batch1,
the Triton attention/GDN backends, and a decode CUDA graph for bs1.

`SGLANG_EARLY_EXIT_CHECKPOINT=/shared/classifier-multiclass.pt` enables the
path. `SGLANG_EARLY_EXIT_CACHE=async` overlaps cache repair with the suffix;
`serial` evaluates the identical repair before continuing the main path.
`SGLANG_EARLY_EXIT_FORCE=-1` runs the calibrated policy. Setting it to an
action width is a control for differential testing. These variables are
experimental runtime controls rather than public serving API parameters.

At the source boundary the GPU computes the action and saves the hidden
and residual tensors. The main branch bypasses the selected prefix of
layers33–43 and runs the remaining middle layers and layers44–64. The
repair branch starts at the saved boundary and evaluates the skipped prefix,
updating its full-attention KV and GDN convolution/recurrent state. Main and
repair predicates are complementary for four fixed layer segments. This
executes each middle layer exactly once across the two branches.

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
