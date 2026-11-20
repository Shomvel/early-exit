# 五分类共享 MLP 真实实验

复用同一份原生缓存反事实标签（10k 训练、1k 校准、1k 留出），五分类动作是 0/5/6/7/11。标签取最长安全宽度；直接未加权交叉熵训练六轮，校准阈值只使用 calibration split。

| 留出集策略 | 动作安全精度 | 覆盖率 | 平均省层 | block 比例 |
|---|---:|---:|---:|---:|
| 原共享sigmoid cal95 | 95.8029% (6300/6576) | 82.3647% | 8.78732 | 13.7302% |
| 五分类softmax cal95 | 95.4228% (6421/6729) | 84.2811% | 9.26240 | 14.4725% |
| 原共享sigmoid cal99 | 98.6910% (2488/2521) | 31.5757% | 3.18512 | 4.9768% |
| 五分类softmax cal99 | 98.7718% (2493/2524) | 31.6132% | 3.47745 | 5.4335% |

cal99 五分类阈值 `0.9885793690681457`；校准实测 99.0213%，留出实测 98.7718%。与原头相比，留出覆盖率接近，平均省层增加约 9.18%。实际精度与目标值分别保留。

原 sigmoid 头在完全相同 hidden 上复现了原校准/留出 TP、FP 和 accepted 计数；分类器保存后重载，留出 softmax 分数逐元素完全一致。

## 头部计算开销

| Batch | 原头 eager μs | 五分类 eager μs | 原头 CUDA graph μs | 五分类 CUDA graph μs |
|---:|---:|---:|---:|---:|
| 1 | 82.12 | 63.37 | 34.99 | 23.21 |
| 8 | 85.21 | 67.68 | 51.67 | 44.59 |
| 32 | 85.49 | 69.76 | 60.14 | 54.43 |
| 128 | 97.35 | 69.19 | 86.83 | 57.88 |
| 1024 | 306.53 | 123.54 | 292.42 | 115.88 |

H200，FP32 权重/归一化、BF16 captured hidden 输入、TF32 matmul 开启；每档 100 次 warmup，每组 200 次、五组取 median。CUDA events 测头部前向；不包含模型、CPU 取动作、路由、KV/GDN 更新或 HTTP。固定输入的 CUDA graph 只测 replay。原头与新头参数 5,249,025 / 5,249,029，线性 MAC 20,992,000 / 5,248,000，约降低 4 倍。batch1 实际 head latency 提升为 eager 1.30×、graph 1.51×，尚需 SGLang 端到端实测。

## 类别分布与限制

留出集最长安全动作计数：不跳192、跳5为20、跳6为40、跳7为333、跳11为7,399。原始五分类 accuracy 92.7480%，多数类直接预测11的基线为92.6728%。未加权CE在本次六轮训练中偏向11：预测不跳14、跳7为119、跳11为7,851，跳5/6均为0。原始不跳类召回1/192。cal99放行的2,524个位置全部选11。显式不跳类别已接入，但本轮证据说明它仍依赖置信度拒绝获得高精度。完整混淆矩阵见 summary。

标签是 teacher-forced 完整历史上的单步安全性。此实验没有运行27B自由生成或SGLang评测，不能从头部时延直接推导模型加速或任务准确率。

## 训练与可复现产物

六轮 loss：0.390531, 0.289267, 0.268499, 0.246877, 0.222869, 0.197963；累计 optimizer steps 468。训练参数为 width1024、ReLU、dropout.1、AdamW lr1e-4、weight decay.03、batch1024、seed20261120。

GPU `j-1795138496-291034` 与 submit controller `j-1795138431-665789` 均 Succeeded；唯一 replica `j-1795138496-291034-6kwrl` Succeeded，runtime replica0 exit0。UUID `880d9145-031b-446d-87ac-2fff509918c1`。

checkpoint：`/mnt/work/shom/early-exit-perfectblend-20261120/training_runs/pb-multiclass-1120-r1/classifier-multiclass.pt`；SHA256 `47b94a7b12155ad1a58a773fb2326cc2c28bc2129d25a72752a2a10374b46596`。

summary SHA256 `a06509656d5cae022c948de7aeb91f314a1fb4d657d00bec459f798b6c55b067`；源码commit `ed9aeac6bd191ec8eff191100045a88588404ce4`。完整来源、分片哈希和 checkpoint schema 见 `five-class-result-manifest.json`。

本仓库入口：`train_multiclass.py --capture /shared/capture-run`；通过 `managed.py` 的 `multiclass` stage 生成新运行。历史运行来自独立小型源码仓库；新训练使用新的 run name、clean source commit 和 UUID。
