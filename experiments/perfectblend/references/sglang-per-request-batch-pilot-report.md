# Batch 1/2/4/8 pilot 验收

1 H200、同卡顺序执行 full/async early exit。GPU、controller 与全部副本 Succeeded，运行退出0，summary SHA核验通过。每项16题质量样本，GSM8K/MATH各8条固定128-token轨迹；这是pilot，正式32卡结果独立统计。

固定轨迹合并统计，所有batch复用相同轨迹，2048有效token、16请求且无结束行空算：

| Batch | full token/s | early exit token/s | EE/full 吞吐 |
|---|---:|---:|---:|
| 1 | 66.79 | 67.52 | 1.011x |
| 2 | 129.43 | 114.57 | 0.885x |
| 4 | 254.59 | 180.38 | 0.708x |
| 8 | 488.44 | 264.56 | 0.542x |

扩大batch提高两模式吞吐；当前early exit的条件中段按请求执行，batch2/4/8吞吐落后full。前后段仍批量执行，所有决策和条件分支留在GPU。

质量生成的真实有效token吞吐、结束行空算和输出长度另列；每批wall time只计一次：

| Batch | full有效token/s | EE有效token/s | full空算/有效token | EE空算/有效token | EE行效率 |
|---|---:|---:|---:|---:|---:|
| 1 | 66.33 | 66.67 | 0/17288 | 0/20222 | 100.0% |
| 2 | 91.23 | 76.35 | 7048/17280 | 9559/20179 | 67.9% |
| 4 | 135.42 | 81.04 | 14817/17243 | 23913/20407 | 46.0% |
| 8 | 180.20 | 98.40 | 30929/18479 | 36723/20741 | 36.1% |

质量正确数（每任务16题）：

| Batch | GSM8K full/EE | MATH full/EE | MMLU full/EE |
|---|---:|---:|---:|
| 1 | 15/15 | 16/14 | 13/13 |
| 2 | 15/15 | 16/14 | 13/13 |
| 4 | 15/15 | 16/13 | 13/13 |
| 8 | 15/15 | 16/16 | 13/13 |

MATH的batch差异包含BF16路径变化。pilot分数不能用于选择batch或替代正式分数。MMLU全部动作不跳，E2E时间受prefill和固定full-before-EE顺序影响，不能用它报告跳层加速。

证据：
- GPU JobId `j-1795150409-791357`；run UUID `4d46df45-2661-4a81-b8fa-b75475178754`。
- 源码 `4adba5efe68d04426bad125f0e1c1a8a57f8557c`；GPU UUID `6c0f270b-7695-12c6-bd6f-a148d319cae4`。
- summary SHA256 `e68b842e03ace1cc1670b952d45f90d0b314b5488048e2fecb30096e0693b500`。
- 本地summary、终态日志和watch-verified.json含完整来源及SHA。
