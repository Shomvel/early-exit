"""Same-GPU quality and fixed-trajectory speed of permanently reduced decode."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path('/mnt/work/shom/early-exit-perfectblend-20261120')
MODEL = Path('/gpfs/shared-model-weight/qwen38-20261019/hf-model')
MODES = ('full', 'fixed5', 'fixed7', 'fixed11')
BATCHES = (1, 2, 4, 8)
TASKS = ('gsm8k', 'math', 'mmlu')
QUALITY_COUNT = 128
REPEATS = 3
TOKENS = 128


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def worker(mode):
    import torch
    from sglang.srt.early_exit import batch_evaluate as b
    from sglang.srt.early_exit.runtime import get_runtime
    from sglang.srt.early_exit.smoke import load_runner

    torch.set_num_threads(4)
    assert torch.cuda.device_count() == 1
    out = Path(os.environ['RUN_DIR'])
    p = b.protocol()
    args = SimpleNamespace(data=ROOT/'data', model=MODEL, classifier=None,
                           weight_inventory=p.source_root()/'experiments/perfectblend/references/model-weight-inventory.json')
    p.verify_model(args)
    tests, development = p.read_benchmarks(args.data)
    selected = p.select_rows(tests, workers=1, limit_per_task=QUALITY_COUNT)
    speed_rows = p.select_rows(tests, workers=1, limit_per_task=8)
    runner, tokenizer = load_runner(MODEL, max_batch_size=8, max_total_tokens=131072)
    raw = runner.torch_runner
    runtime = get_runtime(raw.model)
    assert (runtime is None) == (mode == 'full')
    if runtime:
        assert runtime.routing == 'fixed_skip' and runtime.width == int(mode[5:])
        assert len(runtime.converted) == 4
    graph_counts = dict(decode_steps=0, decode_real_row_slots=0)
    original = raw.forward

    def checked(batch, *a, **kw):
        result = original(batch, *a, **kw)
        if batch.forward_mode.is_decode():
            assert result.can_run_graph, 'Eager fallback invalidates this experiment'
            graph_counts['decode_steps'] += 1
            graph_counts['decode_real_row_slots'] += batch.batch_size
        return result

    raw.forward = checked
    eos = b.eos_tokens(tokenizer)
    inputs = {r['evaluation_index']: p.prompt(tokenizer, r, development) for r in selected}
    assert all(r['evaluation_index'] in inputs for r in speed_rows)
    option_ids = [tokenizer.encode(' '+label, add_special_tokens=False) for label in 'ABCD']
    assert all(len(v) == 1 for v in option_ids)
    option_ids = [v[0] for v in option_ids]

    def generate(rows, **kw):
        return b.generate_batch(runner, tokenizer, [inputs[r['evaluation_index']] for r in rows], eos=eos, **kw)

    with torch.inference_mode():
        # Warm full prefill, decode and sampling before collecting any timings.
        for _ in range(2):
            generate(speed_rows[:8], limits=[16]*8, ignore_eos=True)
        # Exact logits establish fixed0 parity, including a padded batch of 3.
        controls = {}
        original_decode = runner.decode
        for size in (1, 3, 8):
            digests = []

            def observe(*a, **kw):
                current, logits = original_decode(*a, **kw)
                assert not torch.isnan(logits).any()
                assert torch.isfinite(logits.amax(-1)).all()
                digests.append(hashlib.sha256(logits.detach().float().cpu().numpy().tobytes()).hexdigest())
                return current, logits

            runner.decode = observe
            generated, measured = generate(speed_rows[:size], limits=[8]*size, ignore_eos=True)
            controls[str(size)] = dict(logits_sha256=digests,
                                       generated_ids=[r['generated_ids'] for r in generated],
                                       captured_batch_size=measured['captured_batch_size'],
                                       routing=measured['routing'])
            runner.decode = original_decode
        save(out/'control.json', controls)
        if mode == 'fixed0':
            full = json.loads((out.parent/'full/control.json').read_text())
            for size in controls:
                for key in ('logits_sha256', 'generated_ids', 'captured_batch_size'):
                    assert controls[size][key] == full[size][key], (size, key)
            save(out/'receipt.json', dict(mode=mode, passed=True, exact_full_logits_parity=True,
                                          control_sha256=sha(out/'control.json')))
            print('PB_FIXED_CONTROL_PASS '+json.dumps(dict(mode=mode, exact_full_logits_parity=True)), flush=True)
            return

        trajectory_path = out/'trajectories.json'
        if mode == 'full':
            trajectories = {}
            for row in speed_rows:
                index = row['evaluation_index']
                generated, _ = generate([row], limits=[TOKENS], ignore_eos=True)
                trajectories[str(index)] = dict(forced_ids=generated[0]['generated_ids'],
                                                prompt_sha256=b.fingerprint(inputs[index]))
            save(trajectory_path, trajectories)
        trajectories = json.loads(trajectory_path.read_text())
        with (out/'timing.jsonl').open('x') as handle:
            for repeat in range(REPEATS):
                for size in BATCHES:
                    for rows in b.question_batches(speed_rows, size):
                        assert len(rows) == size
                        forced = [trajectories[str(r['evaluation_index'])]['forced_ids'] for r in rows]
                        prompts = [inputs[r['evaluation_index']] for r in rows]
                        for row, ids in zip(rows, prompts):
                            assert trajectories[str(row['evaluation_index'])]['prompt_sha256'] == b.fingerprint(ids)
                        _, measured = generate(rows, limits=[TOKENS]*size, forced=forced)
                        measured.update(mode=mode, repeat=repeat, requested_batch_size=size,
                                        evaluation_indices=[r['evaluation_index'] for r in rows],
                                        prompt_sha256=[b.fingerprint(v) for v in prompts],
                                        forced_trajectory_sha256=b.fingerprint(forced))
                        assert measured['effective_generated_tokens'] == TOKENS*size
                        assert measured['completed_row_waste_tokens'] == 0
                        handle.write(json.dumps(measured)+'\n'); handle.flush()
                    print('PB_FIXED_SPEED_PROGRESS '+json.dumps(dict(mode=mode, repeat=repeat, batch_size=size)), flush=True)

        with (out/'quality.jsonl').open('x') as quality, (out/'quality-timing.jsonl').open('x') as timing:
            for task in TASKS:
                task_rows = [r for r in selected if r['benchmark'] == task]
                assert len(task_rows) == QUALITY_COUNT
                for batch_index, rows in enumerate(b.question_batches(task_rows, 8)):
                    generated, measured = generate(rows, limits=[b.maximum(r) for r in rows],
                                                    choice_ids=option_ids if task == 'mmlu' else None)
                    for row, result in zip(rows, generated):
                        result.update(p.grade(row, result['text']))
                        result.update(mode=mode, benchmark=task, evaluation_index=row['evaluation_index'],
                                      prompt_sha256=b.fingerprint(inputs[row['evaluation_index']]))
                        quality.write(json.dumps(result, ensure_ascii=False)+'\n')
                    quality.flush()
                    measured.update(mode=mode, benchmark=task, batch_index=batch_index,
                                    evaluation_indices=[r['evaluation_index'] for r in rows])
                    timing.write(json.dumps(measured)+'\n'); timing.flush()
                    print('PB_FIXED_QUALITY_PROGRESS '+json.dumps(dict(mode=mode, benchmark=task, completed=(batch_index+1)*8)), flush=True)

    receipt = dict(mode=mode, passed=True, gpu=torch.cuda.get_device_name(0),
                   gpu_uuid=str(torch.cuda.get_device_properties(0).uuid), torch=str(torch.__version__),
                   cuda=torch.version.cuda, identity=p.execution_identity(args), graph_execution=graph_counts,
                   trajectory_sha256=sha(trajectory_path), quality_sha256=sha(out/'quality.jsonl'),
                   timing_sha256=sha(out/'timing.jsonl'), quality_timing_sha256=sha(out/'quality-timing.jsonl'),
                   control_sha256=sha(out/'control.json'), fixed_skip_width=runtime.width if runtime else 0,
                   server_args={k:getattr(raw.server_args,k) for k in ('dtype','tp_size','pp_size','attention_backend','linear_attn_decode_backend','linear_attn_prefill_backend','cuda_graph_bs','max_total_tokens')})
    save(out/'receipt.json', receipt)
    print('PB_FIXED_WORKER_PASS '+json.dumps(dict(mode=mode, passed=True)), flush=True)


def collect(out):
    receipts = {m:json.loads((out/m/'receipt.json').read_text()) for m in MODES}
    for key in ('gpu_uuid','gpu','torch','cuda','identity','trajectory_sha256','server_args'):
        assert all(receipts[m][key] == receipts['full'][key] for m in MODES), key
    timing, quality, quality_timing = {}, {}, {}
    for mode in MODES:
        for filename, digest in (('timing.jsonl','timing_sha256'), ('quality.jsonl','quality_sha256'), ('quality-timing.jsonl','quality_timing_sha256')):
            assert sha(out/mode/filename) == receipts[mode][digest]
        timing[mode] = read_rows(out/mode/'timing.jsonl')
        records = read_rows(out/mode/'quality.jsonl')
        quality[mode] = {r['evaluation_index']: r for r in records}
        assert len(quality[mode]) == len(records) == 3*QUALITY_COUNT
        quality_timing[mode] = read_rows(out/mode/'quality-timing.jsonl')
        assert quality[mode].keys() == quality['full'].keys()
        for index, row in quality[mode].items():
            assert row['prompt_sha256'] == quality['full'][index]['prompt_sha256']
            assert row['reference'] == quality['full'][index]['reference']
    speed = {}
    for size in BATCHES:
        comparison = {}
        for mode in MODES:
            repeats = []
            for repeat in range(REPEATS):
                rows = [r for r in timing[mode] if r['repeat'] == repeat and r['requested_batch_size'] == size]
                full = [r for r in timing['full'] if r['repeat'] == repeat and r['requested_batch_size'] == size]
                assert len(rows) == len(full) == 24//size
                for row, reference in zip(rows, full):
                    for key in ('evaluation_indices','prompt_sha256','forced_trajectory_sha256'):
                        assert row[key] == reference[key], key
                seconds = sum(r['decode_seconds'] for r in rows)
                tokens = sum(r['effective_generated_tokens'] for r in rows)
                assert tokens == 24*TOKENS
                repeats.append(dict(tokens=tokens, decode_seconds=seconds,
                                    cuda_ms=sum(r['decode_cuda_ms'] for r in rows), tokens_per_second=tokens/seconds))
            comparison[mode] = dict(repeats=repeats, tokens_per_second=sum(r['tokens'] for r in repeats)/sum(r['decode_seconds'] for r in repeats))
        for mode in MODES:
            comparison[mode]['speedup_vs_full'] = comparison[mode]['tokens_per_second']/comparison['full']['tokens_per_second']
        speed[str(size)] = comparison
    scores = {}
    for task in TASKS:
        group = {}
        for mode in MODES:
            rows = [r for r in quality[mode].values() if r['benchmark'] == task]
            assert len(rows) == QUALITY_COUNT
            measured = [r for r in quality_timing[mode] if r['benchmark'] == task]
            lost = [r['evaluation_index'] for r in rows if quality['full'][r['evaluation_index']]['correct'] and not r['correct']]
            gained = [r['evaluation_index'] for r in rows if not quality['full'][r['evaluation_index']]['correct'] and r['correct']]
            group[mode] = dict(count=len(rows), correct=sum(r['correct'] for r in rows),
                               accuracy=sum(r['correct'] for r in rows)/len(rows),
                               lost_correct_indices=lost, gained_correct_indices=gained,
                               truncated=sum(r['truncated'] for r in rows),
                               generated_tokens=sum(r['generated_tokens'] for r in rows),
                               completed_row_waste_tokens=sum(r['completed_row_waste_tokens'] for r in rows),
                               total_batch_seconds=sum(r['total_seconds'] for r in measured))
        scores[task] = group
    control = json.loads((out/'fixed0/receipt.json').read_text())
    assert control['passed'] and control['exact_full_logits_parity']
    return dict(passed=True, speed=speed, quality=scores, receipts=receipts, control=control,
                scope='Full prefill; static decode skip after layer32; no classifier or cache repair; greedy quality; fixed full-model trajectories for speed')


def main():
    out = Path(os.environ['RUN_DIR'])
    source = Path(__file__).resolve().parents[2]
    save(out/'config.yaml', dict(model=str(MODEL), source_layer=32, skipped_widths=[5,7,11],
                                prefill='all64 layers', cache_repair=False, classifier=False,
                                quality_examples_per_task=QUALITY_COUNT, quality_batch_size=8,
                                quality_limits=dict(gsm8k=2048, math=4096, mmlu=1),
                                speed_examples_per_task=8, matched_tokens=TOKENS,
                                speed_batches=BATCHES, speed_repeats=REPEATS))
    for mode in ('full','fixed0','fixed5','fixed7','fixed11'):
        sub = out/mode; sub.mkdir()
        env = {k:v for k,v in os.environ.items() if not k.startswith('SGLANG_EARLY_EXIT_')}
        env['RUN_DIR'] = str(sub)
        if mode != 'full':
            env['SGLANG_EARLY_EXIT_FIXED_SKIP'] = mode[5:]
            if mode != 'fixed0':
                shutil.copyfile(out/'full/trajectories.json', sub/'trajectories.json')
        argv = [sys.executable,'-u',str(source/'experiments/perfectblend/launch_overlay.py'),
                '--module','experiments.perfectblend.fixed_skip_run','--worker',mode]
        print('PB_FIXED_STAGE '+json.dumps(dict(mode=mode, argv=argv)), flush=True)
        subprocess.run(argv, env=env, check=True)
    result = collect(out)
    save(out/'summary.json', result)
    print('PB_FIXED_PASS '+json.dumps(dict(passed=True, speed=result['speed'], quality=result['quality'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=(*MODES, 'fixed0'))
    args = parser.parse_args()
    worker(args.worker) if args.worker else main()
