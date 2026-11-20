# 最高 recall 分类头：整批共同跳层测速

整批共享最小建议跳宽，任一有效 token 建议 0 就整批不跳。padding 不参与决策；中段直接处理原始 batch，不拆组、不重排、不合并。使用即时异步补 KV/GDN，所跳层仍在修复 stream 计算；此次没有运行延迟重入 prefill。

**结果接近持平：batch1 提升 1.09%，batch2/4/8 分别变化 −0.07%/−0.69%/−1.08%。**

| Batch | 完整模型 token/s | 整批 EE token/s | EE/full | 共同跳层步数 | 平均跳层/token |
|---:|---:|---:|---:|---:|---:|
| 1 | 64.95 | 65.65 | 1.01085× | 45.77% | 3.9915 |
| 2 | 126.21 | 126.12 | 0.99928× | 23.96% | 1.8411 |
| 4 | 247.35 | 245.64 | 0.99313× | 8.33% | 0.5807 |
| 8 | 475.11 | 469.98 | 0.98922× | 1.04% | 0.0677 |

生成轨迹上的个体建议跳层比例约 45.7%。batch8 共同跳层仅 1.04%，平均跳 0.0677/64 层。整批规则消除了拆组开销；共同跳层机会随 batch 增大迅速减少，即时缓存修复也保留了所跳层的计算。

## 选择与验证

使用完整 recall sweep 中 .99 目标下 recall 最高的 `bce4_linear/longest`，seed0；阈值固定为 calibration 得到的 `0.9809505939483643`，没有按测速结果重新选择。原 heldout：覆盖率 56.56%、recall 57.15%、precision 98.60%，平均跳 5.0487 层。分类头保持 FP32，模型 BF16。

真实 GPU head 与独立原始权重 `F.linear` 逐位置动作一致；7,984 个 heldout 位置 accepted=4,516、TP=4,453、FP=63 均精确复现。宽度 7/11 的计数相对历史报告各相差 1；本轮完整原序与 heldout 重分块一致，未将差异归因于分块。阈值邻域和实际计数保存在 summary 的 policy 字段。

GPU smoke：batch1/2/3/4/8，full/no-skip logits、tokens、缓存一致；serial/async 混合动作 logits 与 KV/GDN 缓存精确一致；零动作否决、正宽度取最小值、padding 与常驻请求 canary 通过。36 项本地回归检查通过。

## 测速口径

一张 H200 上依次执行 full 和 async。GSM8K/MATH/MMLU 各 8 条，共 24 条真实请求；每条固定 128 个 decode token，每个 batch 大小重复 3 次。同一份完整模型 batch1 生成轨迹用于全部配置。每种模式每个 batch 大小测量 9,216 token，总计 73,728 个测量 token。

吞吐为三轮总 token / 总 decode 时间；包含逐步 token 读回，排除预热、模型加载、图捕获、tokenization、prefill、HTTP 和调度。CUDA event 计时给出相同趋势。没有结束行空算；所有 decode 强制验收为 CUDA graph 执行。此处没有自由生成准确率结论。

| Batch | Full 三轮 token/s | EE 三轮 token/s |
|---:|---|---|
| 1 | 64.961 / 64.966 / 64.918 | 65.628 / 65.670 / 65.663 |
| 2 | 126.212 / 126.229 / 126.180 | 126.129 / 126.207 / 126.014 |
| 4 | 247.318 / 247.365 / 247.352 | 245.752 / 245.702 / 245.481 |
| 8 | 475.079 / 475.216 / 475.021 | 470.277 / 469.971 / 469.704 |

## 可追溯输入与终态

- Run: `pb-high-recall-common-1120-r1`；UUID `6acbf0e5-e11e-4da7-b062-12859b78633c`。
- GPU job: `j-1795175658-709168`；作业和唯一副本均 Succeeded；runtime exit0，managed state=verified。
- GPU UUID: `96746877-b37a-f868-546c-93ae0c01b0b4`；PyTorch 2.11.0+cu129 / CUDA12.9。
- Runtime implementation: `b6656ffd`；完整运行源码 overlay: `47e53b0840dcf1683e50b4cb9404b40a9631aaba`。
- 镜像：`registry.lg.shzhisuan.com/library/sglang:v0.5.13.post1-cu129@sha256:163e3f6e921a03b2eaaaf8e2f4f9b7ddf98d7bef54e2deddfadadb3733ae14d4`。
- Summary SHA256: `1b4e5ed3f0276c851a37dcc0e89d0e6e040b7f4eecda853aadeca40ea67cf030`。
- Runner SHA256: `3b0bbab79620d2dad903238756a4ce34b6863452f76611d8ba8626b0af2b04ca`，源码为 `experiments/perfectblend/high_recall_run.py`。
- Classifier SHA256: `f90e35b8c785eb983362961839a65fa9e37d4d92ca791b0e07e9618c6156469e`。
- 原始 linear head SHA256: `7d191d43530ba899261f9b8885803a0763c52c733cad4c065518687c6ab7c8ae`。
- 输入轨迹 SHA256: `fe286799df2d09ce7cfca4dfbe685044b93ea101162079a689e5cefd13c2cfe6`。
- Full timing SHA256: `bca681738a6142e1649f543f2902f5e5f9873ac8b15cfc5cd928abcbe717f75b`。
- EE timing SHA256: `bf129d549a82c943342949e4e2f93c0a16eac1cb9aa6fbd52090433895e7866e`。
- Smoke SHA256: `0e45b34f41b58664c4736f25c810d60da4eb211543775f7eea94f8fedcae2615`。
- 原始运行目录：`/mnt/work/shom/early-exit-perfectblend-20261120/training_runs/pb-high-recall-common-1120-r1/`。`full/timing.jsonl`、`async/timing.jsonl` 与各自 `receipt.json` 保留每批原始记录及指纹；smoke 的完整 tensor 证据留在共享运行目录。

配套归档：`high-recall-common-summary.json`、`high-recall-common-smoke.json`、`high-recall-common-prepared.json`、`high-recall-common-final.json`、`high-recall-common-verified.json`。

此前 split/lazy 尝试的失败回执仍独立保留。本报告的速度只来自完成验收的整批运行，不混用其他作业结果。
