"""Fit one five-action shared MLP using frozen, already captured 27B features."""
import argparse
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEED=20261120
SOURCE=32
SKIPS=(5,6,7,11)
ACTIONS=(0,*SKIPS)
BASELINE_SHA='0a9c4ca0c5d8a98a172b97bff1782c10e9f91cae6854398db14ef407a08f5556'


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def emit(tag,**value):
    print('PB_MULTICLASS_'+tag+' '+json.dumps(value,allow_nan=False),flush=True)


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def normalize(hidden):
    hidden=hidden.float()
    return hidden*torch.rsqrt(hidden.square().mean(-1,keepdim=True).clamp_min(1e-12))


class FiveClassGate(nn.Module):
    def __init__(self,width=1024):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(5120,width),nn.ReLU(),nn.Dropout(.1),nn.Linear(width,5))

    def logits(self,hidden):return self.net(normalize(hidden))
    def forward(self,hidden):return self.logits(hidden).softmax(-1)


class BaselineGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(5124,1024),nn.ReLU(),nn.Dropout(.1),nn.Linear(1024,1))
        self.register_buffer('categories',torch.eye(4),persistent=False)

    def forward(self,hidden):
        hidden=normalize(hidden)
        shape=hidden.shape[:-1]
        h=hidden.unsqueeze(-2).expand(*shape,4,5120)
        cat=self.categories.expand(*shape,4,4)
        return self.net(torch.cat((h,cat),-1)).squeeze(-1).sigmoid()


def longest_safe(labels):
    # Safety labels can be nonmonotonic; choose the longest actually safe action.
    ids=torch.arange(1,5,device=labels.device)
    return torch.where(labels,ids,0).max(-1).values


def metrics(chosen,truth):
    chosen=np.asarray(chosen)
    accepted=chosen>0
    safe=truth[np.arange(len(chosen)),np.maximum(chosen-1,0)]
    tp=int((accepted&safe).sum());fp=int((accepted&~safe).sum())
    saved=np.asarray(ACTIONS)[chosen]
    return dict(total=len(chosen),accepted=int(accepted.sum()),tp=tp,fp=fp,
                precision=tp/(tp+fp) if tp+fp else None,coverage=float(accepted.mean()),
                mean_skipped_blocks=float(saved.mean()),expected_block_fraction_saved=float(saved.mean()/64),
                action_counts={str(w):int((saved==w).sum()) for w in ACTIONS})


def chosen_softmax(scores,threshold):
    best=scores.argmax(-1)
    confidence=scores[np.arange(len(scores)),best]
    return np.where(confidence>=threshold,best,0)


def calibrate(scores,truth,goal):
    confidence=scores.max(-1)
    candidates=np.unique(np.r_[np.quantile(confidence,np.linspace(0,1,1001)),1.000001])
    eligible=[]
    for threshold in candidates:
        result=metrics(chosen_softmax(scores,threshold),truth)
        if result['accepted']>=32 and result['precision']>=goal:
            eligible.append((result['mean_skipped_blocks'],result['coverage'],-threshold,float(threshold),result))
    if not eligible:return 1.000001,metrics(np.zeros(len(scores),dtype=np.int64),truth)
    return max(eligible,key=lambda value:value[:3])[-2:]


def baseline_choices(scores,threshold):
    return np.where(scores>=threshold,np.arange(1,5)[None,:],0).max(-1)


def classification(scores,truth):
    prediction=scores.argmax(-1)
    target=longest_safe(torch.from_numpy(truth)).numpy()
    confusion=np.bincount(target*5+prediction,minlength=25).reshape(5,5)
    ce=float(-np.log(np.clip(scores[np.arange(len(scores)),target],1e-30,1.)).mean())
    return dict(accuracy=float((prediction==target).mean()),cross_entropy=ce,
                target_class_counts={str(w):int((target==i).sum()) for i,w in enumerate(ACTIONS)},
                predicted_class_counts={str(w):int((prediction==i).sum()) for i,w in enumerate(ACTIONS)},
                confusion_matrix=confusion.tolist(),confusion_rows='longest_safe_label',confusion_columns='predicted_action',
                unthresholded_policy=metrics(prediction,truth))


@torch.inference_mode()
def scores(model,hidden):
    return torch.cat([model(x.cuda()).cpu() for x in hidden.split(2048)]).numpy()


@torch.inference_mode()
def benchmark(model,hidden):
    results={}
    for batch in (1,8,32,128,1024):
        x=hidden[:batch].cuda()
        for _ in range(100):model(x)
        torch.cuda.synchronize()
        modes={}
        for mode in ('eager','cuda_graph'):
            if mode=='cuda_graph':
                # Models and inputs are static; no CPU action conversion is timed.
                graph=torch.cuda.CUDAGraph()
                stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(5):model(x)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):output=model(x)
                function=graph.replay
            else:function=lambda:model(x)
            gpu=[];wall=[]
            for _ in range(5):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize();tick=time.perf_counter();start.record()
                for _ in range(200):function()
                end.record();torch.cuda.synchronize()
                wall.append((time.perf_counter()-tick)*1e6/200)
                gpu.append(start.elapsed_time(end)*1000/200)
            modes[mode]=dict(cuda_microseconds_median=statistics.median(gpu),cuda_microseconds_groups=gpu,
                            host_microseconds_median=statistics.median(wall),host_microseconds_groups=wall,
                            warmup=100,repetitions_per_group=200,groups=5)
            if mode=='cuda_graph':del graph,output
        results[str(batch)]=modes
    return results


def main(args):
    out=Path(os.environ['RUN_DIR'])
    assert torch.cuda.device_count()==1
    torch.cuda.set_device(0);torch.set_num_threads(4)
    torch.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED)
    baseline_file=args.capture/'classifier.pt'
    assert sha(baseline_file)==BASELINE_SHA
    saved=torch.load(baseline_file,map_location='cpu',weights_only=True)
    assert saved['source']==SOURCE and tuple(saved['skips'])==SKIPS
    config=dict(seed=SEED,source=SOURCE,actions=ACTIONS,width=1024,dropout=.1,optimizer='AdamW',lr=1e-4,
                weight_decay=.03,batch_size=1024,epochs=6,class_weighting='none',
                target='Longest safe action across four native counterfactual labels; else no skip.',
                calibration='Argmax action; reject no-skip or confidence below threshold. Minimum32 accepted. Maximum saved blocks at target safety precision.',
                capture=str(args.capture),baseline_sha256=BASELINE_SHA)
    write(out/'config.yaml',config)
    parts=[];fingerprints=[]
    for rank in range(4):
        path=args.capture/f'capture-{rank}.pt'
        fingerprints.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
        part=torch.load(path,map_location='cpu',weights_only=True)
        assert torch.equal(part['labels'],part['proposals']==part['targets'][:,None])
        parts.append(part)
    hidden=torch.cat([p['hidden'] for p in parts]);truth=torch.cat([p['labels'] for p in parts])
    splits=[];ids=[]
    for part in parts:
        for split,row,count in zip(part['split'],part['row_ids'],part['counts']):
            splits.extend([split]*count);ids.extend([row]*count)
    del parts
    masks={s:torch.tensor([v==s for v in splits]) for s in ('train','calibration','heldout')}
    expected=dict(train=(10000,79865),calibration=(1000,7993),heldout=(1000,7984))
    for split,(wanted,positions) in expected.items():
        assert len({r for s,r in zip(splits,ids) if s==split})==wanted
        assert int(masks[split].sum())==positions
    assert hidden.shape==(95842,5120) and truth.shape==(95842,4)
    emit('DATA',positions={s:int(m.sum()) for s,m in masks.items()},shards=fingerprints)
    gate=FiveClassGate().cuda().train()
    optimizer=torch.optim.AdamW(gate.parameters(),lr=1e-4,weight_decay=.03)
    x=hidden[masks['train']].cuda();target=longest_safe(truth[masks['train']]).cuda()
    curves=[];step=0
    for epoch in range(6):
        total=0.;correct=0
        for index in torch.randperm(len(x),device=x.device).split(1024):
            logits=gate.logits(x[index]);loss=F.cross_entropy(logits,target[index])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            gradient=torch.nn.utils.clip_grad_norm_(gate.parameters(),1.)
            assert torch.isfinite(gradient);optimizer.step();step+=1
            total+=float(loss.detach())*len(index)
            correct+=int((logits.argmax(-1)==target[index]).sum())
        curves.append(dict(epoch=epoch+1,loss=total/len(x),accuracy=correct/len(x),optimizer_steps=step))
        emit('FIT',**curves[-1])
    del optimizer,x,target
    gate.eval()
    baseline=BaselineGate().cuda().eval();baseline.load_state_dict(saved['state_dict'])
    all_scores={s:scores(gate,hidden[m]) for s,m in masks.items()}
    baseline_scores={s:scores(baseline,hidden[masks[s]]) for s in ('calibration','heldout')}
    labels={s:truth[m].numpy() for s,m in masks.items()}
    policies=[];baseline_results=[]
    for goal in (.95,.99):
        threshold,calibration=calibrate(all_scores['calibration'],labels['calibration'],goal)
        policies.append(dict(goal=goal,threshold=threshold,calibration=calibration,
                             heldout=metrics(chosen_softmax(all_scores['heldout'],threshold),labels['heldout'])))
        original=next(p for p in saved['policies'] if p['goal']==goal)
        result={s:metrics(baseline_choices(baseline_scores[s],original['threshold']),labels[s]) for s in ('calibration','heldout')}
        for s in ('calibration','heldout'):
            for key in ('total','accepted','tp','fp'):
                assert result[s][key]==original[s][key],(goal,s,key,result[s],original[s])
        baseline_results.append(dict(goal=goal,threshold=original['threshold'],**result))
    checkpoint=dict(format='five_class_shared_mlp_v1',state_dict=gate.cpu().state_dict(),source=SOURCE,skips=SKIPS,
                    actions=ACTIONS,width=1024,input_hidden_size=5120,normalization='RMS float32 eps clamp1e-12',
                    activation='ReLU',dropout=.1,policies=policies,seed=SEED,label_definition=config['target'])
    torch.save(checkpoint,out/'classifier-multiclass.pt')
    gate.cuda()
    # Exact checkpoint roundtrip on real held-out features before latency timing.
    restored=FiveClassGate().cuda().eval()
    restored.load_state_dict(torch.load(out/'classifier-multiclass.pt',map_location='cpu',weights_only=True)['state_dict'])
    assert np.array_equal(scores(restored,hidden[masks['heldout']]),all_scores['heldout'])
    del restored
    latency={name:benchmark(model,hidden[masks['heldout']]) for name,model in [('baseline_sigmoid',baseline),('five_class_softmax',gate)]}
    parameter_counts={name:sum(p.numel() for p in model.parameters()) for name,model in [('baseline_sigmoid',baseline),('five_class_softmax',gate)]}
    multiply_accumulates=dict(baseline_sigmoid=4*(5124*1024+1024),five_class_softmax=5120*1024+1024*5)
    summary=dict(config=config,curves=curves,positions={s:int(m.sum()) for s,m in masks.items()},
                 classification={s:classification(all_scores[s],labels[s]) for s in masks},
                 policies=policies,baseline_policies=baseline_results,baseline_reproduced_exact_counts=True,
                 parameters=parameter_counts,dense_linear_MAC_per_token=multiply_accumulates,latency=latency,
                 checkpoint_sha256=sha(out/'classifier-multiclass.pt'),input_shards=fingerprints,
                 runtime=dict(torch=str(torch.__version__),gpu=torch.cuda.get_device_name(0),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                              dtype='FP32 weights and normalization; BF16 captured hidden input',tf32_matmul=torch.backends.cuda.matmul.allow_tf32),
                 limits='Labels are teacher-forced full-history one-step counterfactuals; no free-generation or SGLang quality claim.')
    write(out/'summary.json',summary)
    np.savez_compressed(out/'multiclass-scores.npz',**{s+'_scores':v for s,v in all_scores.items()},
                        **{s+'_labels':v for s,v in labels.items()})
    emit('PASS',summary_sha256=sha(out/'summary.json'),checkpoint_sha256=summary['checkpoint_sha256'],policies=policies,
         baseline_policies=baseline_results,parameters=parameter_counts)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True)
    main(p.parse_args())
