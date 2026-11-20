"""Verify recall-head parity, then measure three same-GPU fixed replay repeats."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path('/mnt/work/shom/early-exit-perfectblend-20261120')
MODEL='/gpfs/shared-model-weight/qwen38-20261019/hf-model'
CLASSIFIER=ROOT/'high-recall/bce4-linear-longest-cal99-1120-r2/classifier-bce4-longest-cal99.pt'
CHECKPOINT_SHA='f90e35b8c785eb983362961839a65fa9e37d4d92ca791b0e07e9618c6156469e'
MODES=('full','async')

def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify():
    import numpy as np
    import torch
    from torch import nn
    from sglang.srt.early_exit.policy import select_action, load_classifier
    out=Path(os.environ['RUN_DIR'])
    assert sha(CLASSIFIER)==CHECKPOINT_SHA
    saved=torch.load(CLASSIFIER,map_location='cpu',weights_only=True)
    net=load_classifier(saved,5120).cuda()
    hidden,truth,mask_all=[],[],[]
    expected_shards=json.loads((ROOT/'training_runs/pb-recall-1120-r1/summary.json').read_text())['data']['shards']
    for record in expected_shards:
        path=Path(record['path']);assert sha(path)==record['sha256']
        part=torch.load(path,map_location='cpu',weights_only=True)
        split=[s for s,c in zip(part['split'],part['counts']) for _ in range(c)]
        mask=torch.tensor([s=='heldout' for s in split])
        hidden.append(part['hidden']);truth.append(part['labels']);mask_all.extend(mask.tolist())
    hidden=torch.cat(hidden).cuda();truth=torch.cat(truth).bool()
    assert hidden.shape==(95842,5120)
    heldout_mask=torch.tensor(mask_all);truth=truth[heldout_mask]
    actions=torch.tensor([0,5,6,7,11],device='cuda',dtype=torch.int32)
    threshold=saved['policies'][0]['threshold'];choices=[];full_scores=[]
    with torch.inference_mode():
        for x in hidden.split(4096):
            x=x.float();x=x*torch.rsqrt(x.square().mean(-1,keepdim=True).clamp_min(1e-12))
            logits=net(x);full_scores.append(logits.sigmoid().cpu())
            choices.append(select_action(logits,actions,threshold,checkpoint_format=saved['format']).cpu())
    choice=torch.cat(choices)[heldout_mask]
    original_path=Path(saved['provenance']['source_checkpoint'])
    assert sha(original_path)==saved['provenance']['source_sha256']
    original=torch.load(original_path,map_location='cpu',weights_only=True)
    weight=original['state_dict']['net.0.weight'].cuda();bias=original['state_dict']['net.0.bias'].cuda()
    reference=[]
    with torch.inference_mode():
        for x in hidden.split(4096):
            x=x.float();x=x*torch.rsqrt(x.square().mean(-1,keepdim=True).clamp_min(1e-12))
            scores=torch.nn.functional.linear(x,weight,bias).sigmoid().cpu().numpy()
            reference.append(np.where(scores>=threshold,np.array([5,6,7,11]),0).max(-1))
    reference=np.concatenate(reference)[np.asarray(mask_all)]
    assert np.array_equal(choice.numpy(),reference),'runtime differs from independent original-head F.linear oracle'
    full_scores=torch.cat(full_scores)[heldout_mask];alternate=[]
    with torch.inference_mode():
        for x in hidden[heldout_mask.cuda()].split(4096):
            x=x.float();x=x*torch.rsqrt(x.square().mean(-1,keepdim=True).clamp_min(1e-12))
            alternate.append(select_action(net(x),actions,threshold,checkpoint_format=saved['format']).cpu())
    alternate=torch.cat(alternate);different=(choice!=alternate).nonzero().flatten()
    distances=(full_scores.double()-threshold).abs();near=torch.argsort(distances.min(-1).values)[:8]
    boundary=dict(complete_order_action_counts={str(w):int((choice==w).sum()) for w in actions.cpu().tolist()},heldout_only_action_counts={str(w):int((alternate==w).sum()) for w in actions.cpu().tolist()},differing_positions=[dict(index=int(i),complete_order_action=int(choice[i]),heldout_only_action=int(alternate[i]),scores=full_scores[i].tolist(),threshold_distance=float(distances[i].min())) for i in different],nearest_threshold_positions=[dict(index=int(i),scores=full_scores[i].tolist(),threshold_distance=float(distances[i].min())) for i in near])
    assert len(different)<=2 and all(float(distances[i].min())<2e-6 for i in different),boundary
    lookup={5:0,6:1,7:2,11:3}
    safe=torch.tensor([bool(truth[i,lookup[int(w)]]) if w else False for i,w in enumerate(choice)])
    print('PB_HIGH_RECALL_POLICY_DIAGNOSTIC '+json.dumps(dict(matmul_precision=torch.get_float32_matmul_precision(),allow_tf32=torch.backends.cuda.matmul.allow_tf32,input_shape=list(hidden.shape))),flush=True)
    result=dict(total=len(choice),accepted=int((choice>0).sum()),tp=int(safe.sum()),fp=int(((choice>0)&~safe).sum()),action_counts={str(w):int((choice==w).sum()) for w in actions.cpu().tolist()})
    assert {k:result[k] for k in ('total','accepted','tp','fp')}==dict(total=7984,accepted=4516,tp=4453,fp=63),result
    expected_hist={'0':3468,'5':216,'6':255,'7':1699,'11':2346}
    if result['action_counts']!=expected_hist:
        assert all(abs(result['action_counts'][k]-v)<=1 for k,v in expected_hist.items()) and float(distances.min())<2e-6,(result,boundary)
    result['floating_point_boundary']=boundary
    result.update(passed=True,checkpoint_sha256=CHECKPOINT_SHA,threshold=threshold,source_checkpoint_sha256=sha(original_path),gpu=torch.cuda.get_device_name(0),runtime_policy_source_sha256=sha(Path(sys.modules[select_action.__module__].__file__)))
    save(out/'high-recall-policy-verified.json',result)
    print('PB_HIGH_RECALL_POLICY_PASS '+json.dumps(result),flush=True)

def fixed_worker(mode):
    import torch
    from sglang.srt.early_exit import batch_evaluate as b
    from sglang.srt.early_exit.smoke import load_runner
    from sglang.srt.early_exit.runtime import get_runtime
    out=Path(os.environ['RUN_DIR']);p=b.protocol()
    args=SimpleNamespace(data=ROOT/'data',examples_per_task=8,replay_examples_per_task=8,model=Path(MODEL),classifier=CLASSIFIER,weight_inventory=p.source_root()/'experiments/perfectblend/references/model-weight-inventory.json')
    p.verify_model(args);selected,development,replay=b.selected_questions(args)
    assert torch.cuda.device_count()==1
    runner,tokenizer=load_runner(args.model,max_batch_size=8,max_total_tokens=131072)
    raw=runner.torch_runner;runtime=get_runtime(raw.model)
    assert (runtime is not None)==(mode!='full')
    if runtime:assert runtime.mode==mode and runtime.checkpoint_sha256==CHECKPOINT_SHA and runtime.routing=='batch_min'
    graph_counts=dict(decode_steps=0,decode_real_row_slots=0)
    original=raw.forward
    def checked(batch,*a,**kw):
        result=original(batch,*a,**kw)
        if batch.forward_mode.is_decode():
            assert result.can_run_graph
            graph_counts['decode_steps']+=1;graph_counts['decode_real_row_slots']+=batch.batch_size
        return result
    raw.forward=checked
    eos=b.eos_tokens(tokenizer);inputs={r['evaluation_index']:p.prompt(tokenizer,r,development) for r in selected}
    trajectory_path=out/'trajectories.json'
    with torch.inference_mode():
        for _ in range(2):b.generate_batch(runner,tokenizer,[inputs[r['evaluation_index']] for r in selected[:8]],limits=[16]*8,eos=eos)
        if mode=='full':
            trajectories={}
            for row in selected:
                i=row['evaluation_index'];result,_=b.generate_batch(runner,tokenizer,[inputs[i]],limits=[128],eos=eos,ignore_eos=True)
                trajectories[str(i)]=dict(forced_ids=result[0]['generated_ids'],prompt_sha256=b.fingerprint(inputs[i]))
            save(trajectory_path,trajectories)
        trajectories=json.loads(trajectory_path.read_text())
        with (out/'timing.jsonl').open('x') as f:
            for repeat in range(3):
                for size in (1,2,4,8):
                    for batch_index,rows in enumerate(b.question_batches(selected,size)):
                        ids=[inputs[r['evaluation_index']] for r in rows]
                        forced=[trajectories[str(r['evaluation_index'])]['forced_ids'] for r in rows]
                        assert len(rows)==size and all(len(v)==128 for v in forced)
                        for r,ids_row in zip(rows,ids):assert trajectories[str(r['evaluation_index'])]['prompt_sha256']==b.fingerprint(ids_row)
                        _,measured=b.generate_batch(runner,tokenizer,ids,limits=[128]*size,eos=eos,forced=forced)
                        measured.update(mode=mode,repeat=repeat,requested_batch_size=size,benchmark=rows[0]['benchmark'],evaluation_indices=[r['evaluation_index'] for r in rows],prompt_sha256=[b.fingerprint(v) for v in ids],forced_trajectory_sha256=b.fingerprint(forced))
                        assert measured['effective_generated_tokens']==128*size and measured['completed_row_waste_tokens']==0
                        f.write(json.dumps(measured)+'\n');f.flush()
                    print('PB_HIGH_RECALL_FIXED_PROGRESS '+json.dumps(dict(mode=mode,repeat=repeat,batch_size=size)),flush=True)
    receipt=dict(mode=mode,routing=runtime.routing if runtime else 'full',passed=True,gpu=torch.cuda.get_device_name(0),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),torch=str(torch.__version__),cuda=torch.version.cuda,trajectory_sha256=sha(trajectory_path),timing_sha256=sha(out/'timing.jsonl'),graph_execution=graph_counts,identity=p.execution_identity(args),threshold=runtime.threshold if runtime else None,classifier_format=runtime.checkpoint_format if runtime else None,server_args={k:getattr(raw.server_args,k) for k in ('dtype','tp_size','pp_size','attention_backend','linear_attn_decode_backend','linear_attn_prefill_backend','cuda_graph_bs','max_total_tokens')})
    save(out/'receipt.json',receipt)
    print('PB_HIGH_RECALL_FIXED_WORKER_PASS '+json.dumps(receipt),flush=True)

def collect(out, requested_modes=MODES):
    receipts={m:json.loads((out/m/'receipt.json').read_text()) for m in requested_modes}
    for key in ('gpu_uuid','gpu','torch','cuda','trajectory_sha256','identity','server_args'):assert all(receipts[m][key]==receipts['full'][key] for m in requested_modes),key
    rows={}
    for mode in requested_modes:
        assert sha(out/mode/'timing.jsonl')==receipts[mode]['timing_sha256']
        rows[mode]=[json.loads(v) for v in (out/mode/'timing.jsonl').read_text().splitlines()]
    result={}
    for size in (1,2,4,8):
        modes={}
        for mode in requested_modes:
            repeats=[]
            for repeat in range(3):
                values=[r for r in rows[mode] if r['requested_batch_size']==size and r['repeat']==repeat]
                assert len(values)==24//size
                total=sum(r['effective_generated_tokens'] for r in values);seconds=sum(r['decode_seconds'] for r in values)
                assert total==24*128
                entry=dict(tokens=total,decode_seconds=seconds,cuda_ms=sum(r['decode_cuda_ms'] for r in values),tokens_per_second=total/seconds)
                if mode=='lazy':
                    for key in ('skipped_layer_tokens','replayed_layer_tokens','pending_layer_tokens','reentry_events','prefill_chunks','prefill_capacity_slots','replayed_segment_tokens'):entry[key]=sum(r['lazy_cache'][key] for r in values)
                if mode=='async':
                    for key in ('common_step_histogram','proposal_histogram'):
                        entry[key]={str(w):sum(r[key][str(w)] for r in values) for w in (0,5,6,7,11)}
                    entry['proposal_coverage']=1-entry['proposal_histogram']['0']/total
                    common_steps=sum(entry['common_step_histogram'].values())
                    entry['common_action_coverage']=1-entry['common_step_histogram']['0']/common_steps
                entry['mean_main_skipped_layers']=sum(sum(int(w)*n for w,n in r['routing_histogram_including_completed_row_waste'].items()) for r in values)/total
                repeats.append(entry)
            modes[mode]=dict(repeats=repeats,tokens_per_second=sum(r['tokens'] for r in repeats)/sum(r['decode_seconds'] for r in repeats))
        for mode in ('async','lazy'):
            if mode not in modes:continue
            modes[mode]['speedup_vs_full']=modes[mode]['tokens_per_second']/modes['full']['tokens_per_second']
        result[str(size)]=modes
    return dict(passed=True,scope='fixed 128-token replay; 8 requests each GSM8K/MATH/MMLU; 3 repeats; quality is a separate 2-example/task batch1 pilot',results=result,receipts=receipts)

def main():
    source=Path(__file__).resolve().parents[2];out=Path(os.environ['RUN_DIR'])
    save(out/'config.yaml',dict(model=MODEL,classifier=str(CLASSIFIER),classifier_sha256=CHECKPOINT_SHA,policy='bce4_linear/longest/cal99',routing='batch-wide minimum, zero if any real row proposes zero',batch_sizes=[1,2,4,8],matched_tokens=128,repeats=3,examples_per_task=8))
    stages=[('policy','experiments.perfectblend.high_recall_run',['--verify']),('smoke','sglang.srt.early_exit.common_smoke',['--model',MODEL,'--classifier',str(CLASSIFIER),'--batch-sizes','1','2','3','4','8','--max-batch-size','8','--steps','16'])]
    for mode in MODES:stages.append((mode,'experiments.perfectblend.high_recall_run',['--fixed-worker',mode]))
    for stage,module,extra in stages:
        sub=out/stage;sub.mkdir();env={k:v for k,v in os.environ.items() if not k.startswith('SGLANG_EARLY_EXIT_')}
        env['RUN_DIR']=str(sub)
        if stage=='async':
            shutil.copyfile(out/'full/trajectories.json',sub/'trajectories.json')
            env.update(SGLANG_EARLY_EXIT_CHECKPOINT=str(CLASSIFIER),SGLANG_EARLY_EXIT_CACHE=stage,SGLANG_EARLY_EXIT_ROUTING='batch_min')
        argv=[sys.executable,'-u',str(source/'experiments/perfectblend/launch_overlay.py'),'--module',module,*extra]
        print('PB_HIGH_RECALL_STAGE '+json.dumps(dict(stage=stage,argv=argv)),flush=True)
        subprocess.run(argv,env=env,check=True)
    result=collect(out);result['policy']=json.loads((out/'policy/high-recall-policy-verified.json').read_text());result['smoke_sha256']=sha(out/'smoke/common-smoke-summary.json')
    result['scope']='fixed 128-token replay; batch-wide minimum; 8 requests each GSM8K/MATH/MMLU; 3 repeats; no free-generation quality claim'
    save(out/'summary.json',result)
    print('PB_HIGH_RECALL_PASS '+json.dumps(dict(passed=True,results=result['results'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--fixed-worker',choices=MODES);args=p.parse_args()
    if args.verify:verify()
    elif args.fixed_worker:fixed_worker(args.fixed_worker)
    else:main()
