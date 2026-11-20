"""New-corpus layer/width selection, native cached labels, and shared gate fitting."""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from engine import (GRID_SKIPS, SOURCES, SKIPS, Route, SharedGate, emit, fork_cache,
                    chat_ids,check_tokenizer,load_model, parity, sha, write)

SEED = 20261120


def records(data, split):
    with (data/f"{split}.jsonl").open() as handle:
        return [json.loads(line) for line in handle]


def window(tokenizer, row, count=8, context=1024):
    messages=[]
    role_map=dict(human="user",gpt="assistant",system="system",user="user",assistant="assistant")
    answer=None
    for item in row["conversations"]:
        role=role_map.get(item["from"])
        if role=="assistant" and any(x["role"]=="user" for x in messages) and item["value"].strip():
            answer=item["value"]
            break
        if role in ("user","system"):
            messages.append(dict(role=role,content=item["value"]))
    if answer is None:
        raise ValueError("sample lacks a user->assistant pair: "+row["id"])
    # This is fixed-record replay. The answer tokens are context, never task labels.
    prompt=chat_ids(tokenizer,messages)
    continuation=tokenizer.encode(answer,add_special_tokens=False)+[tokenizer.convert_tokens_to_ids("<|im_end|>")]
    rng=random.Random(SEED+row["row_index"])
    limit=min(len(continuation),256)
    count=min(count,limit)
    offset=rng.randrange(max(1,limit-count+1))
    before=prompt+continuation[:offset]
    before=before[-(context-count):]
    ids=before+continuation[offset:offset+count]
    return ids,list(range(len(before),len(ids)))


def stats(scores, labels, threshold):
    accepted=scores>=threshold
    tp=int((accepted&labels).sum());fp=int((accepted&~labels).sum())
    positive=int(labels.sum());n=labels.size
    return dict(total=n,accepted=tp+fp,tp=tp,fp=fp,fn=positive-tp,tn=n-positive-fp,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/positive if positive else None,
                coverage=(tp+fp)/n)


def policy(scores, labels, threshold, skips):
    widths=np.asarray(skips)
    choices=np.where(scores>=threshold,widths[None,:],0).argmax(-1)
    accepted=(scores>=threshold).any(-1)
    correct=labels[np.arange(len(labels)),choices]
    tp=int((accepted&correct).sum());fp=int((accepted&~correct).sum())
    saved=np.where(accepted,widths[choices],0)
    return dict(total=len(labels),accepted=int(accepted.sum()),tp=tp,fp=fp,
                precision=tp/(tp+fp) if tp+fp else None,coverage=float(accepted.mean()),
                mean_skipped_blocks=float(saved.mean()),expected_block_fraction_saved=float(saved.mean()/64),
                skipped_width_counts={str(w):int((saved==w).sum()) for w in (0,*skips)})


def threshold_policy(scores, labels, skips, goal=.99):
    # Choose directly for the deployed longest-eligible-width policy, not for
    # an aggregate of interchangeable binary classification decisions.
    candidates=np.unique(np.r_[np.quantile(scores,np.linspace(0,1,1001)),1.000001])
    eligible=[]
    for threshold in candidates:
        result=policy(scores,labels,float(threshold),skips)
        if result["accepted"]>=32 and result["precision"]>=goal:
            eligible.append((result["mean_skipped_blocks"],result["coverage"],-float(threshold),float(threshold),result))
    if not eligible:
        return 1.000001,policy(scores,labels,1.000001,skips)
    selected=max(eligible,key=lambda x:x[:3])
    return selected[-2:]


def fit(hidden, labels, source, skips, epochs=6, width=1024):
    torch.manual_seed(SEED+source)
    gate=SharedGate(source,skips,width).cuda().train()
    optimizer=torch.optim.AdamW(gate.parameters(),lr=1e-4,weight_decay=.03)
    x=hidden.cuda();y=labels.float().cuda()
    positive=y.mean(0)
    # Normalized per-category balanced BCE keeps every sampled conversation.
    wpos=.5/positive.clamp_min(1e-6)
    wneg=.5/(1-positive).clamp_min(1e-6)
    curves=[]
    step=0
    for epoch in range(epochs):
        order=torch.randperm(len(x),device=x.device)
        total=0.
        for index in order.split(1024):
            logits=gate.logits(x[index]);target=y[index]
            weights=torch.where(target.bool(),wpos,wneg)
            loss=(F.binary_cross_entropy_with_logits(logits,target,reduction="none")*weights).sum()/weights.sum()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient=torch.nn.utils.clip_grad_norm_(gate.parameters(),1.)
            assert torch.isfinite(gradient)
            optimizer.step()
            step+=1
            total+=float(loss.detach())*len(index)
        curves.append(dict(epoch=epoch+1,loss=total/len(x),optimizer_steps=step))
        emit("FIT",source=source,width=width,**curves[-1])
    gate.eval()
    return gate,curves


@torch.inference_mode()
def score(gate, hidden):
    return torch.cat([gate.logits(chunk.cuda()).sigmoid().cpu() for chunk in hidden.split(2048)]).numpy()


def execute_workers(args):
    children=[]
    try:
        for rank in range(args.workers):
            children.append(subprocess.Popen([sys.executable,"-u",__file__,*sys.argv[1:],"--rank",str(rank)]))
        for child in children:
            if child.wait():
                raise RuntimeError("a capture worker failed")
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            if child.poll() is None:child.wait(timeout=60)


@torch.inference_mode()
def capture_worker(args,out):
    rank=args.rank
    model,decoder,head,tokenizer,evidence=load_model(args.model,rank,args.weight_inventory)
    device=f"cuda:{rank}"
    smoke=args.stage=="smoke"
    if args.stage in ("pilot","smoke"):
        cohort=records(args.data,"train")[:(8 if smoke else 640)]
        sources=(8,24,48) if smoke else SOURCES
        pairs=[(s,k) for s in sources for k in GRID_SKIPS if s+k<64]
        columns={s:[k for a,k in pairs if a==s] for s in sources}
        if smoke:
            for source in sources:
                checks=parity(model,decoder,head,tokenizer,device,source=source,skips=columns[source])
                emit('PARITY',rank=rank,source=source,checks=checks)
        part=dict(hidden={s:[] for s in sources},labels={s:[] for s in sources},row_ids=[],row_indices=[],counts=[])
        with Route(decoder) as route:
            for done,row in enumerate(cohort[rank::args.workers],1):
                ids,positions=window(tokenizer,row,count=16,context=768)
                inputs=torch.tensor([ids],device=device)
                selected=torch.tensor(positions,device=device)
                final,_,_,_,_=route.run(inputs,None,torch.arange(len(ids),device=device),capture=True,use_cache=False)
                states=dict(route.captures)
                target=head(final[0,selected]).argmax(-1)
                for source in sources:
                    part['hidden'][source].append(states[source][0,selected].cpu())
                    labels=[]
                    for skip in columns[source]:
                        candidate,_,_,_,trace=route.run(states[source],None,torch.arange(len(ids),device=device),
                            source=source,skip=skip,embedded=True,use_cache=False,trace=done==1)
                        prediction=head(candidate[0,selected]).argmax(-1)
                        if done==1:
                            direct,_,_,_,direct_trace=route.run(inputs,None,torch.arange(len(ids),device=device),source=source,skip=skip,use_cache=False,trace=True)
                            assert torch.equal(direct,candidate)
                            assert trace==list(range(source+skip+1,65))
                            assert direct_trace==[d for d in range(1,65) if not source<d<=source+skip]
                        labels.append((prediction==target).cpu())
                    part['labels'][source].append(torch.stack(labels,-1))
                part['row_ids'].append(row['id']);part['row_indices'].append(row['row_index']);part['counts'].append(len(positions))
                if done==1 or done%10==0:emit("PILOT_CAPTURE",rank=rank,done=done,total=len(cohort[rank::args.workers]))
        for source in sources:
            part['hidden'][source]=torch.cat(part['hidden'][source]);part['labels'][source]=torch.cat(part['labels'][source])
        part.update(columns=columns,evidence=evidence)
        torch.save(part,out/f"pilot-{rank}.pt")
        write(out/f"worker-{rank}.json",dict(rank=rank,cohort=len(part['row_ids']),positions=sum(part['counts']),evidence=evidence))
        return
    selection=json.loads((args.pilot/"selection.json").read_text())
    source,skips=selection['source'],selection['skips']
    # Re-run the real cache controls at the chosen source/range in this image.
    checks=parity(model,decoder,head,tokenizer,device,source=source,skips=skips)
    part=dict(hidden=[],labels=[],row_ids=[],split=[],counts=[],targets=[],proposals=[])
    tick=time.monotonic()
    with Route(decoder) as route:
        for split in ("train","calibration","heldout"):
            cohort=records(args.data,split)[rank::args.workers]
            for done,row in enumerate(cohort,1):
                ids,positions=window(tokenizer,row,count=8,context=1024)
                inputs=torch.tensor([ids],device=device)
                _,cache,_,_,_=route.run(inputs[:,:positions[0]],None,torch.arange(positions[0],device=device),source=source)
                hidden,labels,targets,proposals=[],[],[],[]
                for p in positions:
                    prior=fork_cache(cache)
                    final,cache,state,_,_=route.run(inputs[:,p:p+1],cache,torch.tensor([p],device=device),source=source)
                    target=head(final[:,-1]).argmax(-1)
                    hidden.append(state[0,-1].cpu())
                    predictions=[]
                    for skip in skips:
                        branch=fork_cache(prior)
                        candidate,_,_,_,_=route.run(state,branch,torch.tensor([p],device=device),source=source,skip=skip,embedded=True)
                        predictions.append(head(candidate[:,-1]).argmax(-1)[0].cpu())
                    predictions=torch.stack(predictions)
                    labels.append(predictions==target[0].cpu());targets.append(target[0].cpu());proposals.append(predictions)
                part['hidden'].append(torch.stack(hidden));part['labels'].append(torch.stack(labels))
                part['targets'].append(torch.stack(targets));part['proposals'].append(torch.stack(proposals))
                part['row_ids'].append(row['id']);part['split'].append(split);part['counts'].append(len(positions))
                if done==1 or done%25==0:emit("CAPTURE",rank=rank,split=split,done=done,total=len(cohort),seconds=time.monotonic()-tick)
    for field in ('hidden','labels','targets','proposals'):part[field]=torch.cat(part[field])
    assert torch.equal(part['labels'],part['proposals']==part['targets'][:,None])
    torch.save(part,out/f"capture-{rank}.pt")
    write(out/f"worker-{rank}.json",dict(rank=rank,cohort=len(part['row_ids']),positions=sum(part['counts']),evidence=evidence,checks=checks,seconds=time.monotonic()-tick))


def select_pilot(args,out):
    parts=[torch.load(out/f"pilot-{rank}.pt",weights_only=True,map_location='cpu') for rank in range(args.workers)]
    candidates=[]
    fitting_ids={r['id'] for r in records(args.data,'train')[:512]}
    train=[]
    for part in parts:
        for row,count in zip(part['row_ids'],part['counts']):train.extend([row in fitting_ids]*count)
    train=torch.tensor(train,dtype=torch.bool)
    if not (~train).any():train[::4]=False
    for source in parts[0]['columns']:
        skips=parts[0]['columns'][source]
        hidden=torch.cat([p['hidden'][source] for p in parts]);labels=torch.cat([p['labels'][source] for p in parts])
        gate,curves=fit(hidden[train],labels[train],source,skips,epochs=3,width=256)
        scores=score(gate,hidden[~train]);truth=labels[~train].numpy()
        per_pair=[]
        for j,k in enumerate(skips):
            pair_threshold,pair99=threshold_policy(scores[:,j:j+1],truth[:,j:j+1],[k],.99)
            _,pair95=threshold_policy(scores[:,j:j+1],truth[:,j:j+1],[k],.95)
            per_pair.append(dict(source=source,skip=k,agreement=float(truth[:,j].mean()),threshold99=pair_threshold,policy99=pair99,policy95=pair95))
        ranked=sorted(range(len(skips)),key=lambda j:(per_pair[j]['policy99']['mean_skipped_blocks'],per_pair[j]['policy95']['mean_skipped_blocks'],-skips[j]),reverse=True)
        selected=sorted(ranked[:4])
        retained=[skips[j] for j in selected]
        threshold,metrics=threshold_policy(scores[:,selected],truth[:,selected],retained,.99)
        threshold95,metrics95=threshold_policy(scores[:,selected],truth[:,selected],retained,.95)
        candidates.append(dict(source=source,skips=retained,all_scanned_skips=skips,threshold=threshold,metrics=metrics,threshold95=threshold95,metrics95=metrics95,per_pair=per_pair,curves=curves))
        torch.save(dict(state_dict=gate.cpu().state_dict(),source=source,skips=skips,width=256),out/f"pilot-gate-source{source}.pt")
        del gate
    best=max(candidates,key=lambda c:(c['metrics']['mean_skipped_blocks'],c['metrics95']['mean_skipped_blocks'],-c['source']))
    selection=dict(source=best['source'],skips=best['skips'],selection_metric='Maximum mean skipped blocks at 99% pilot-development policy precision; 95% breaks ties.',
                   note=('Smoke execution only, eight conversations with an internal position holdout.' if args.stage=='smoke' else
                         '640 training conversations only: first512 fit, next128 development. This static pilot only selects geometry. Final gate uses native cached labels and separate calibration.'),
                   candidates=candidates)
    write(out/'selection.json',selection)
    emit('SELECTION',source=best['source'],skips=best['skips'],metrics=best['metrics'])


def fit_final(args,out):
    selection=json.loads((args.pilot/'selection.json').read_text())
    source,skips=selection['source'],selection['skips']
    parts=[torch.load(out/f"capture-{rank}.pt",weights_only=True,map_location='cpu') for rank in range(args.workers)]
    hidden=torch.cat([p['hidden'] for p in parts]);labels=torch.cat([p['labels'] for p in parts])
    splits=[];ids=[]
    for part in parts:
        for split,row,count in zip(part['split'],part['row_ids'],part['counts']):splits.extend([split]*count);ids.extend([row]*count)
    for split,wanted in [('train',10000),('calibration',1000),('heldout',1000)]:
        assert len({r for s,r in zip(splits,ids) if s==split})==wanted,(split,wanted)
    masks={s:torch.tensor([x==s for x in splits]) for s in ('train','calibration','heldout')}
    gate,curves=fit(hidden[masks['train']],labels[masks['train']],source,skips)
    score_cal=score(gate,hidden[masks['calibration']]);score_test=score(gate,hidden[masks['heldout']])
    truth_cal=labels[masks['calibration']].numpy();truth_test=labels[masks['heldout']].numpy()
    policies=[]
    for goal in (.95,.99):
        threshold,calibration=threshold_policy(score_cal,truth_cal,skips,goal)
        policies.append(dict(goal=goal,threshold=threshold,calibration=calibration,heldout=policy(score_test,truth_test,threshold,skips)))
    checkpoint=dict(state_dict=gate.cpu().state_dict(),source=source,skips=skips,width=1024,policies=policies)
    torch.save(checkpoint,out/'classifier.pt')
    np.savez_compressed(out/'heldout-scores.npz',calibration_scores=score_cal,heldout_scores=score_test,calibration_labels=truth_cal,heldout_labels=truth_test)
    summary=dict(source=source,skips=skips,train_conversations=10000,calibration_conversations=1000,heldout_conversations=1000,
                 positions={s:int(m.sum()) for s,m in masks.items()},curves=curves,policies=policies,
                 classifier_sha256=sha(out/'classifier.pt'),label_history='Native full-history cached one-step counterfactual; teacher-forced observed answer tokens.')
    write(out/'summary.json',summary)
    emit('TRAINED',**summary)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('smoke','pilot','capture'),required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--weight-inventory',type=Path,required=True)
    p.add_argument('--pilot',type=Path)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--rank',type=int)
    args=p.parse_args()
    out=Path(os.environ['RUN_DIR'])
    if args.rank is not None:
        capture_worker(args,out)
        return
    assert torch.cuda.device_count()==args.workers
    import transformers
    tokenizer=transformers.AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    check_tokenizer(tokenizer)
    for split in ('train','calibration','heldout'):
        for row in records(args.data,split)[:8]:
            ids,positions=window(tokenizer,row)
            assert isinstance(ids,list) and positions and min(positions)>0
    emit('INPUT_PREFLIGHT',conversation_windows=24)
    write(out/'config.yaml',dict(stage=args.stage,seed=SEED,model=str(args.model),data=str(args.data),pilot=str(args.pilot),
                                sources=SOURCES,skips=GRID_SKIPS,workers=args.workers,
                                data_manifest_sha256=sha(args.data/'sample-manifest.json'),
                                weight_inventory_sha256=sha(args.weight_inventory)))
    execute_workers(args)
    if args.stage in ('smoke','pilot'):select_pilot(args,out)
    else:fit_final(args,out)
    emit('PASS',stage=args.stage)


if __name__=='__main__':main()
