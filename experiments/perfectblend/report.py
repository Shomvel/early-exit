"""Render completed, verified experimental results without changing any run."""
import argparse
import hashlib
import json
from pathlib import Path

ART=None


def read(name):
    return json.loads((ART/name).read_text())


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def require_success(stem):
    final=read(stem+'-final.json')
    verified=read(stem+'-verified.json')
    assert final['phase']=='Succeeded',final
    assert verified['markers'] and all(m['exit_code']==0 for m in verified['markers'])
    for name,info in verified['files'].items():
        assert sha(ART/(stem+'-'+name))==info['sha256'],name
    return final


def percentage(value):
    return '无接受样本' if value is None else f'{value*100:.2f}%'


def render(args):
    global ART
    ART=args.artifacts.expanduser().resolve()
    pilot=args.pilot
    capture=args.capture
    evaluation=args.evaluation
    runs={stem:require_success(stem) for stem in (pilot,capture,evaluation)}
    selection=read(pilot+'-selection.json')
    trained=read(capture+'-summary.json')
    evaluated=read(evaluation+'-evaluation-summary.json')
    manifest=json.loads(args.data_manifest.read_text())
    assert trained['train_conversations']==10000
    assert trained['source']==selection['source'] and trained['skips']==selection['skips']
    assert evaluated['classifier_sha256']==trained['classifier_sha256']
    assert evaluated['counts']==dict(gsm8k=1319,math=5000,mmlu=14042)
    source,skips=trained['source'],trained['skips']
    lines=['# Open-PerfectBlend 27B native early-exit 实测', '',
           f"输入为第 {source} 个 block 的输出；候选跳过宽度为 {', '.join(map(str,skips))}。编号从 1 开始，宽度 k 对应跳过 block {source+1} 至 ({source}+k)。保留最后的原生 block、norm 和 LM head。",
           '', '## 准确率与实际加速', '',
           '| 数据集 | 题数 | 完整模型 | Early exit | 变化（百分点） | 端到端加速 | Decode tokens/s 加速 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,title in [('gsm8k','GSM8K'),('math','MATH'),('mmlu','MMLU')]:
        row=evaluated['benchmarks'][name]
        full,candidate,comparison=row['full'],row['early_exit'],row['comparison']
        lines.append(f"| {title} | {full['count']:,} | {percentage(full['accuracy'])} | {percentage(candidate['accuracy'])} | {comparison['accuracy_delta']*100:+.2f} | {comparison['end_to_end_speedup']:.3f}× | {comparison['decode_throughput_speedup']:.3f}× |")
    lines+=['', '主对照为同一 SFT checkpoint 1479 的完整执行。主策略在 benchmark 前固定为 cal99；小于 1× 表示当前执行方式更慢。速度按配对样本的累计实测时间计算。', '',
            '## 分类器及新数据校准', '',
            '| 目标精度 | 阈值 | 校准集实际精度 | 留出集实际精度 | 留出覆盖率 | 留出平均跳层 |',
            '|---|---:|---:|---:|---:|---:|']
    for p in trained['policies']:
        held=p['heldout'];cal=p['calibration']
        lines.append(f"| {percentage(p['goal'])} | {p['threshold']:.8f} | {percentage(cal['precision'])} | {percentage(held['precision'])} | {percentage(held['coverage'])} | {held['mean_skipped_blocks']:.3f} |")
    lines+=['',f"训练使用 {trained['train_conversations']:,} 条会话，独立校准与留出各 1,000 条。实际观测位置数：`{json.dumps(trained['positions'])}`。六个 epoch，最后一轮累计 optimizer step 为 {trained['curves'][-1]['optimizer_steps']}；27B 权重冻结。",'',
            '输入层和跳层宽度在新训练数据的 640 条 pilot 会话上选择。最终 MLP 每条选取答案前 256 token 内最多八个连续位置，上下文上限 1,024 token，用原生缓存和完整历史上的单步 top-1 反事实标签训练；连续跳层的误差传播由上表自由生成结果检验。', '',
            '## 延迟、输出长度和截断', '',
            '| 数据集 | 路径 | 生成 token | 截断题数 | Prefill 总秒 | Decode 总秒 | 总秒 | Decode token/s | 平均跳过 block |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('gsm8k','math','mmlu'):
        row=evaluated['benchmarks'][name]
        for mode in ('full','early_exit'):
            p=row[mode]
            lines.append(f"| {name} | {mode} | {p['generated_tokens']:,} | {p['truncated']} | {p['prefill_seconds']:.2f} | {p['decode_seconds']:.2f} | {p['total_seconds']:.2f} | {p['decode_tokens_per_second']:.2f} | {p['mean_skipped_blocks']:.3f} |")
        if 'matched_token_decode' in row:
            p=row['matched_token_decode']
            lines.append(f"\n{name} 固定相同 token 重放：{p['examples']} 题、{p['tokens']} token，decode 加速 {p['speedup']:.3f}×。\n")
    macro=evaluated['benchmarks']['mmlu'].get('macro_accuracy')
    if macro:lines+=['',f"MMLU 57 科 macro accuracy：完整模型 {percentage(macro['full'])}，early exit {percentage(macro['early_exit'])}。"]
    lines+=['', '## 数据与评测设置', '',
            f"Open-PerfectBlend 固定版本 `{manifest['revision']}`，共 {manifest['dataset_rows']:,} 条，六个分片完整下载。随机种子 {manifest['seed']}，均匀行采样并排除 benchmark query 精确匹配和 13 词片段命中；采样期间排除 {manifest['exclusions']['benchmark_overlap']} 条命中记录。三份 split 内容指纹互斥。", '',
            'H200、BF16、batch size 1、native Transformers SDPA/FLA GDN。每题两种路径在同一 GPU 串行运行，顺序交替。GSM8K/MATH 采用 nonthinking greedy 聊天生成，最大输出分别 2,048/4,096 token；MATH 使用 Hendrycks 规范化 boxed exact match。MMLU 使用原 dev 五样例 completion prompt、A/B/C/D 单 token 下一词分数。', '',
            'MMLU 主要耗时为完整 prefill；跳层发生在 decode。原始逐题文件同时保留输出、参考答案、正确性、长度和时间；summary 含每科结果、lost/gained correct 及延迟 p50/p95。', '',
            '## 官方成绩对照', '',
            '已核对的 Qwen3.8-27B 官方模型卡 revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0` 未列出 GSM8K、原版 MATH 和原版 MMLU 的成绩。官方发布权重与本次 SFT checkpoint 的文件布局不同，布局不足以判断 tensor 是否相同。本次同权重完整模型实测构成可比基线。', '',
            '## 产物与运行验收', '',
            f"分类器：`{runs[capture]['run']}/classifier.pt`，SHA256 `{trained['classifier_sha256']}`。逐题结果：`{runs[evaluation]['run']}/predictions-0.jsonl` 至 `predictions-{evaluated['workers']-1}.jsonl`。", '',
            '以下运行均已完成 Job 和预期 replica 的终态核对，并取得 runtime exit code 0：', '',
            '| 阶段 | UUID | JobId |', '|---|---|---|']
    for stem,run in runs.items():lines.append(f"| {stem} | `{run['uuid']}` | `{run['job_id']}` |")
    lines+=['', f"回执目录 `{ART}` 的 `*-verified.json` 绑定 summary SHA256。数据版本与样本哈希见 `{args.data_manifest.resolve()}` 及同目录的 `benchmark-sources.json`；模型分片指纹由运行的 `config.yaml` 引用。执行入口见仓库 `experiments/perfectblend/README.md`。", '']
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report=args.output_dir/'report.md';report.write_text('\n'.join(lines))
    (args.output_dir/'results.json').write_text(json.dumps(dict(runs=runs,sampling=manifest,training=trained,evaluation=evaluated),ensure_ascii=False,indent=2)+'\n')
    print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--artifacts',type=Path,required=True,help='Directory of collected *-final/verified/summary receipts')
    p.add_argument('--data-manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--pilot',required=True,help='Full run label, e.g. pb-pilot-1120-r2')
    p.add_argument('--capture',required=True,help='Full run label')
    p.add_argument('--evaluation',required=True,help='Full completed evaluation run label')
    render(p.parse_args())
