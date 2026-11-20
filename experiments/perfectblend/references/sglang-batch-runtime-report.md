# 多请求 early-exit 验收

2026-11-20，在同一张 H200 上依次运行 full、serial、async 三个独立进程。
冻结的 checkpoint1479、五分类头和 cal99 阈值与正式评测一致。
此次验证覆盖 batch 1/2/3/4/8，每个配置 16 个 decode 步；batch3 使用 graph4，检查 padding。

| 检查 | 结果 |
|---|---|
| Full 与强制不跳的 logits | 所有 batch 逐位一致 |
| Full 与强制不跳的各层 KV/GDN 状态 | 所有 batch 逐位一致 |
| Serial 与 async 混合动作的 logits 和状态 | 所有 batch 逐位一致 |
| 各请求动作直方图、main/repair 互补计数 | 全部通过 |
| Padding 请求的中段执行和动作计数 | 全部为零 |
| Batch8 整批不跳分支 | 16 步执行 16 次，逐请求分支关闭 |
| Batch8 混合动作 | 整批不跳分支为零，逐请求分支执行 |
| CUDA probe 的全 padding、混合、全不跳重入 | 状态、PDL、分支互斥及实际执行计数通过 |

Run UUID `86aadba5-14c4-4398-b939-cf443e8d042c`，GPU Job `j-1795150007-691586`。
源快照 `4adba5efe68d04426bad125f0e1c1a8a57f8557c`；汇总 SHA256
`2ad2ec6ebe1f3050578d44a84bcaa5d6c688559bf7780f43a6365f3d747d67ea`。
GPU、控制器及全部副本均 Succeeded，运行退出码 0。
完整原始产物位于
`/mnt/work/shom/early-exit-perfectblend-20261120/training_runs/pb-sglang-batch-smoke-1120-r3`。
小型摘要与生命周期证据见 [summary](sglang-batch-runtime-summary.json) 和
[verified](sglang-batch-runtime-verified.json)。

实现为每个请求保留 GPU action，前32层和第44层后的尾部保持批量计算，中段使用静态展开的逐请求 CUDA IF。
MRoPE `[3,T]` 沿 token 轴切片，并保留原位置通道 stride 与实时图输入引用。
Batch8 的全部真实请求都选择 action0 时，GPU 进入单个批量中段 IF，保留完整模型的 GEMM 形状。
任意正跳宽或 padding 都关闭该分支。原分类器动作、统计与实际分支谓词分别保留。
选择和分支执行均留在 GPU；修复 KV/GDN 的流在下一 token 前汇合。

首次多请求作业在错误 MRoPE 切片后触发 cuBLAS 错误；修复并回归后，第二次作业发现 batch8 逐行不跳路径有数值差。
原容差 `atol=0.02, rtol=0` 保持不变。增加同形状的批量不跳分支后，上表所有数值检查达到逐位一致。
静态位置、stride、真实 dispatch 和 marker/action 映射回归见
[位置证据](batch-position-cpu-validation.json) 和 [路由证据](batch-no-skip-cpu-validation.json)。

真实请求入口 E2E 另用 batch1/2，GSM8K、MATH、MMLU 各2题，已全部完成并验收。
每项 full/EE 各答对2题；完整模型 batch2 有效吞吐约103 token/s，相对 batch1 约1.59倍。
样本、长度、空算及完整结果见 [E2E summary](sglang-batch-eval-e2e-summary.json) 和
[E2E receipt](sglang-batch-eval-e2e-verified.json)。

这些运行验证了多请求路径和评测入口。混合中段的逐行计算、结束请求空算和跨请求的 BF16 数值变化会影响实际收益。
每项16题、batch1/2/4/8的吞吐试验及独立32卡完整 benchmark 继续运行；其准确率和速度以各自最终验收结果为准。
