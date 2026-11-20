# 跳层与不跳分为两个 batch：GPU 验收

`split_skip` 在第32层后按分类头建议把有效 token 分成两组：不跳组保留动作0；跳层组取所有正建议中的最短跳宽。GPU 稳定打包隐藏状态、位置、请求和 KV/GDN 索引，分别选择能容纳各组人数的最小1/2/4/8图容量，在第44层前恢复原始顺序。空组关闭全部中间层，原始 padding 进入任何组的数量为0。

每个容量有12个 IF 区域，分别覆盖不跳主路、跳层主路与跳层缓存补齐的各四段。GPU仅执行所选容量，全部决策保留在设备上。缓存补齐异步分支在下一 token 前汇合；它评估被跳过层，收益依赖缩短的关键路径与硬件争用。两组并行时，BF16/NHD KV writer抑制占位槽0写入，GDN padding 保留-1索引。

## 实测验收

1张H200，同checkpoint1479与冻结五分类头，独立full/serial/async进程。batch1/2/3/4/8各16步，batch3使用graph4。所有组人数、候选动作0/5/6/7/11、阈值策略、混合分组及padding均检查。

- full与强制不跳：所有batch的logits和KV/GDN状态逐位相同，最大绝对差0；原容差atol0.02/rtol0保留。
- serial与async混合动作：所有batch的logits和修复后KV/GDN状态逐位相同。
- 每步proposed/applied、分组actions/counts/有序indices、最小capacity与独立oracle一致；row/group/physical执行次数精确匹配。
- 48个IF的合成GPU图实际执行分组、gather、选桶、分支与merge；空组、全padding与PDL检查通过。
- native MHATokenToKVPool上的双流真实Triton masked store通过：真实slot写入正确，slot0哨兵保持。
- batch3→graph4的原生padding哨兵及另一个resident真实请求的KV/GDN canary通过。
- CPU共21项回归与Black/isort/Ruff F、compileall、git diff检查通过；包含6,681种分组pattern与真实dispatcher构图检查。

该run验证正确性，性能由独立真实请求pilot测量。

## 可追溯证据

- Run UUID：`62ffa9ed-8521-42c2-8a4d-58cb4e4fb1b0`
- GPU Job：`j-1795155158-558246`；controller：`j-1795155150-113938`；全部job/副本Succeeded，runtime exit0。
- 不可变source：`94bfec7df86c8cb63d30657a3f61c750cb38ca79`
- 完整summary SHA256：`101414b178be44e12fe2839fdc9d53ecd3bfbfa6747b7160cc3e7fd1c8a1f3ff`
- 完整summary：`/mnt/work/shom/early-exit-perfectblend-20261120/training_runs/pb-sglang-split-smoke-1120-r1/batch-smoke-summary.json`
- [压缩明细](sglang-split-smoke-summary.json)省略逐tensor与部分逐步记录，保留完整原件路径与指纹；[终态回执](sglang-split-smoke-verified.json)绑定原始summary。

同卡full/EE throughput pilot使用source`b3b8a0dde386992eb9294d3ef6e1f0644dfb0ded`。runtime、split、masked writer、batch helpers、smoke/probe与上述通过版本逐字相同，仅冻结更新后的批量评测入口。每任务16题、数学任务各8个fixed128轨迹、batch1/2/4/8；完整质量成绩见独立[batch1正式报告](sglang-full-benchmark-report.md)。
