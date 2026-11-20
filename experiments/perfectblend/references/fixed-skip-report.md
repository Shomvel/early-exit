# 固定跳 5／7／11 层：速度与质量

**固定跳层确实省计算，但长生成质量损失显著。** 跳5／7／11层的 decode 吞吐分别提升约6.6%～6.9%／10.2%～10.6%／17.7%～18.2%。跳5／7层时 GSM8K 接近完整模型，MATH下降7.03／6.25个百分点；跳11层时两项生成任务准确率均降到一半以下。

| 配置 | GSM8K（正确/128） | MATH（正确/128） | MMLU（正确/128） |
|---|---:|---:|---:|
| 完整模型 | 95.31%（122） | 90.63%（116） | 77.34%（99） |
| 固定跳5层 | 96.09%（123） | 83.59%（107） | 82.81%（106） |
| 固定跳7层 | 96.09%（123） | 84.38%（108） | 82.03%（105） |
| 固定跳11层 | 46.09%（59） | 39.06%（50） | 84.38%（108） |

MMLU保留完整prompt prefill，仅对最后一个输入token使用跳层路径。这种单步选项打分与GSM8K/MATH的连续自由生成有明显差别；本样本的上升不外推为全量基准提升。

## 固定长度decode速度

单元格为token/s；括号为相对完整模型的吞吐比。

| Batch | 完整模型 | 跳5层 | 跳7层 | 跳11层 |
|---:|---:|---:|---:|---:|
| 1 | 65.04 | 69.30（1.0655×） | 71.69（1.1023×） | 76.58（1.1775×） |
| 2 | 126.31 | 134.75（1.0668×） | 139.40（1.1037×） | 148.97（1.1795×） |
| 4 | 247.61 | 264.33（1.0675×） | 273.62（1.1050×） | 292.44（1.1810×） |
| 8 | 475.71 | 508.33（1.0686×） | 526.30（1.1063×） | 562.31（1.1820×） |

三轮原始计时、CUDA event时间与逐批输入轨迹指纹保留在summary和原始timing.jsonl。四配置使用同一张H200，固定顺序full→跳5→跳7→跳11；没有随机交错，可能包含顺序导致的温度或负载漂移。跳0层单独用于路径验证。

## 配对质量与实际生成用时

丢失/新增正确表示相对于完整模型同题结果的变化。batch用时为128题在静态batch8下各批prefill+decode时间之和，包含结束行空算。

| 任务 | 配置 | 丢失/新增正确 | 截断数 | 有效生成token | 结束行空算token | batch用时（秒） |
|---|---|---:|---:|---:|---:|---:|
| gsm8k | 完整模型 | 0/0 | 1 | 46,861 | 38,883 | 181.69 |
| gsm8k | 跳5 | 0/1 | 1 | 43,530 | 38,342 | 161.54 |
| gsm8k | 跳7 | 1/2 | 1 | 42,003 | 37,165 | 151.14 |
| gsm8k | 跳11 | 64/1 | 20 | 80,617 | 149,335 | 412.39 |
| math | 完整模型 | 0/0 | 6 | 119,796 | 229,612 | 748.72 |
| math | 跳5 | 10/1 | 12 | 139,478 | 271,570 | 824.62 |
| math | 跳7 | 8/0 | 14 | 145,250 | 276,646 | 819.25 |
| math | 跳11 | 67/1 | 49 | 260,346 | 263,942 | 955.58 |
| mmlu | 完整模型 | 0/0 | 0 | 128 | 0 | 9.72 |
| mmlu | 跳5 | 1/8 | 0 | 128 | 0 | 8.66 |
| mmlu | 跳7 | 1/7 | 0 | 128 | 0 | 8.64 |
| mmlu | 跳11 | 1/10 | 0 | 128 | 0 | 8.62 |

MATH有效输出在跳5／7／11层时增加16.43%／21.25%／117.32%，batch用时增加约10.14%／9.42%／27.63%。跳5层的10道新增错题中6道被截断；跳7层为8道中7道；跳11层为67道中42道。损失包含更长解答导致的截断和已经给出最终答案但答错。

跳11层时GSM8K有效输出增加72.03%、batch用时增至2.27倍。每步运行更快，冗长生成和结束行空算使整批完成更慢。跳5／7层的GSM8K batch用时分别缩短约11.09%／16.81%。

这些结果支持把离线可跳比例与直接删层后的生成质量分开评估。对于需要保留完整模型质量的方案，后续可将跳层路径作为draft，由完整模型验证；本轮没有实现或测量speculative decoding的接受率、验证成本或最终加速。

## 计算与测评口径

这轮使用永久固定的 decode 路径。完整模型有 64 层；每个跳层配置在第 32 层后始终绕过接下来的 5／7／11 层，然后执行剩余层。prompt prefill 保持全部 64 层。没有分类器、条件分组、修复 stream 或 KV/GDN 补算；跳过层在本条请求的 decode 中始终不会重入。更换跳宽会重新加载模型并捕获图。

这是对“所有 token 都跳”的直接测量。之前离线的 92.7% 是 teacher-forced、单步 argmax 一致率；这次的 GSM8K/MATH 让各配置自由生成，测量误差随上下文变化后的最终答题结果。

速度和质量分别测量：

- **速度：**GSM8K、MATH、MMLU 各 8 条请求，共 24 条。由完整模型 batch1 生成每题固定 128-token 的参考轨迹；四个配置及 batch1/2/4/8 重放相同的逐步输入，每种配置每个 batch 重复三轮，共 9,216 个测量 token。吞吐按三轮 token 总数 / decode wall time 总和计算，同时保留 CUDA event 计时。
- **质量：**相同三项测试各 128 题，共 384 题，四个配置逐题配对。沿用固定种子 `20261120`、原 test split、原 prompt 和 grader。GSM8K/MATH 使用 greedy、关闭 thinking，最大生成长度 2048/4096；MMLU 使用每科 5-shot、一个 token 的 A/B/C/D 选项 logits 打分。
- **批次：**质量采用按任务分组的静态 batch8。已结束的行继续占用该批 decode 槽，直到该批最后一行结束；输出有效 token、结束行空算、截断与 batch wall time 分别记录。整批用时每批只计一次。
- **计时范围：**包含 token 读回与 GPU 执行计数；排除模型加载、图捕获、预热、tokenization、prefill、HTTP 与调度。质量的整批用时另含 prefill。固定跳层路径含逐层执行计数，实测数值包含这些计数的开销。

MATH 样本覆盖全部 7 类、5 个难度层级；MMLU 样本覆盖 48／57 科。128 题结果用于观察这几个固定配置的变化，完整模型对照来自同一组题；MMLU 的样本内变化不外推为全量基准提升。

## 路径验证

- 跳 0 层对照：batch1/3/8、各 8 步，完整 logits 和生成 token 与完整模型精确一致；batch3 同时覆盖 graph padding。
- 每个实测 decode 都检查 CUDA graph 执行，无 eager 回退。
- 每个批次检查所有有效 token 都使用指定跳宽，以及每个 suffix 层的执行计数。被跳层计数为 0、修复层 token 为 0。
- 39 项 CPU 回归通过，包括固定路径逐层调用顺序、多 token 不重入、padding 计数、非法配置与 Unicode JSONL 读取。

实现与运行入口：`python/sglang/srt/early_exit/fixed.py`、`experiments/perfectblend/fixed_skip_run.py`。完整 prefill 与固定 decode 路径的跳 0 层精确一致性是本轮直接 GPU 验证；不同跳宽的质量以答题成绩验收。

## 运行与证据

- Run：pb-fixed-skip-1120-r2；UUID `fb73bdb2-013a-4517-b7f9-4c9f52456146`。
- GPU job：`j-1795178291-540076`；作业和唯一副本Succeeded，runtime exit0，managed state=verified。
- GPU UUID：`6e102519-ad97-9132-624b-b5cb19b0e47b`；NVIDIA H200、PyTorch2.11.0+cu129、CUDA12.9、BF16、TP1/PP1。
- 实现源码：`9be2cde398047fb3216164fdcf54d9c5dbce2978`；冻结overlay：`6af377886c872744f3bb80b8630621c77e9abf6b`。
- 镜像：`registry.lg.shzhisuan.com/library/sglang:v0.5.13.post1-cu129@sha256:163e3f6e921a03b2eaaaf8e2f4f9b7ddf98d7bef54e2deddfadadb3733ae14d4`。
- Summary SHA256：`3a3460eaef410a563148cbe6835c217b99299b96de5718c309fe75c0eeb2cd16`。
- GPFS原始目录：`/mnt/work/shom/early-exit-perfectblend-20261120/training_runs/pb-fixed-skip-1120-r2/`。
- 各配置保留receipt.json、timing.jsonl、quality.jsonl、quality-timing.jsonl、trajectories.json、control.json；fixed0只保留控制结果和回执。quality含逐题生成token、答案文本、预测值和参考值。
- 本地导出：实验根目录artifacts/pb-fixed-skip-1120-r2-evidence/，30文件逐一校验SHA256；重新从原始记录运行fixed_skip_run.collect()与GPU summary完全一致。

配套归档：[汇总](fixed-skip-summary.json)、[30文件指纹清单](fixed-skip-evidence-manifest.json)、[跳0层对照](fixed-skip-zero-control.json)、[题目选择](fixed-skip-selection.json)、[prepared](fixed-skip-prepared.json)、[final](fixed-skip-final.json)、[verified](fixed-skip-verified.json)。

首个r1任务在完整模型阶段主动停止，以修正汇总时Unicode分隔符被splitlines()当作JSONL换行的隐患。补充回归后以新UUID的r2从头运行，r1保留Stopped回执；本报告数据全部来自r2，managed manifest记录retry_of。
