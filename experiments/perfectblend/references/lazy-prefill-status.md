# Re-entry chunk prefill: implementation awaiting GPU validation

The opt-in `SGLANG_EARLY_EXIT_CACHE=lazy` path saves the layer32 hidden and residual tensors at each skipped token's KV slot. Per-request frontiers track the missing suffix of each layer segment. Equal or increasing applied skip widths accumulate history. A decreasing width flushes that request's missing prefixes in layer order before processing the current token; causal chunk prefill restores both full-attention KV and GDN convolution/recurrent state. Completed requests may retain unpaid history, which is reported separately.

GPU CUDA WHILE nodes repeat bounded prefill chunks while the device cursor precedes the missing suffix end. The default chunk capacity is16; 1/16/32/64 are supported. Ragged lengths and attention prefixes remain on device. Static FLA chunk indices avoid CPU length materialization inside captured graphs. Main no-skip/skip groups keep their existing GPU partition and minimum-capacity routing. The eager host-scheduled oracle is test-only and excluded from performance measurements.

This prototype supports static experiment request batches. Call `runtime.lazy.reset()` when clearing the request/cache pools; the supplied smoke and pilot runners do this. Continuous serving request lifecycle integration is still outside this prototype.

Prepared controls cover batch1/2/3/4/8, a resident padding canary, full/no-skip logits and caches, mixed action schedules, positive-width re-entry, a35-token hole requiring multiple WHILE iterations, and ten continuously skipped tokens with no cache repair. GPU execution is compared with an independently host-scheduled causal-prefill oracle at the existing0.02 absolute tolerance. These controls are implemented and have **not run on GPU**.

The pilot entrypoint runs full/async/lazy sequentially on the same GPU with frozen checkpoint1479, the shared five-class head, batch1/2/4/8,8 original test examples per task and identical128-token GSM8K/MATH replay trajectories. It records quality, truncation, prefill/decode time, completed-row waste, re-entry counts, replayed layer/token work, chunk utilization and debt when the static microbatch reservation ends. No new speed or quality result is available.

## Validation and launch evidence (2026-11-20)

- 25 CPU regressions passed, including the four new delayed-work accounting cases. Python compilation and diff checks passed.
- YiCloud skill synchronized at `b7aa8f6be408706b9a98b96c0de4060520d2e19c`.
- Smoke prepared UUID `960db3b5-4a2d-41ae-ad14-8d7e797e9bf2`; immutable initial GPU-source snapshot `ad010463dcdfe2c538264ba97f7b38cbb725d9e2`; manifest SHA `c1318b211426204bd2e29f24a7904b5a4473c1e69b02a550131667caed600864`.
- Prepare controller `j-1795160533-497519` succeeded, unique replica `twmqk`, process exit0, `PB_CONTROL_PREPARED` verified.
- Submit controller `j-1795161090-227680`, unique replica `jqrg8`, failed at the credential read preflight with process exit1. The failure happened **before** managed GPU submit. There is no new GPU JobId.
- Both local and existing remote credentials receive `Code101 AUTH_FAIL: cannot get resource scanstatus in API group /v1 in the project fdj/fdj-infra` from GetJob. ListJobs also fails on scan-status retrieval. ListJobReplicas and GetLogs remain readable and independently establish the controller outcomes.
- BP Dev SSH and Northjob API are reachable; the known model directories do not contain the frozen checkpoint path. No Northjob job was created and no alternate weights were evaluated.

The launch uses the [YiCloud reproducible-run workflow](https://github.com/china-qijizhifeng/yicloud-skill/blob/b7aa8f6be408706b9a98b96c0de4060520d2e19c/references/reproducible-training.md), which requires terminal reconciliation after both GetJob and every expected ListJobReplicas entry become terminal. The denied job read blocks that lifecycle; raw unmanaged GPU submission was not used.

Once the job-read permission is restored, create a fresh immutable snapshot and smoke attempt; the prepared r1 source precedes later evaluator and cleanup changes. Only after smoke passes should the new full/async/lazy pilot run. Keep prior formal batch1 and split-pilot scores attached to their original runtimes.
