"""Full paired GSM8K/MATH/MMLU evaluation and actual native-cache latency."""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

import torch

from engine import Route,SharedGate,chat_ids,check_tokenizer,emit,load_model,parity,sha,write
from math_equivalence import is_equiv


def boxed(text):
    starts=list(re.finditer(r"\\(?:boxed|fbox)\s*\{",text))
    for match in reversed(starts):
        depth=1
        for i in range(match.end(),len(text)):
            if text[i]=='{':depth+=1
            elif text[i]=='}':depth-=1
            if depth==0:return text[match.end():i]
    return None


def grade_math(text,gold):
    prediction=boxed(text.split('</think>')[-1])
    answer=boxed(gold)
    return bool(prediction is not None and answer is not None and is_equiv(prediction,answer)),prediction,answer


def grade_gsm(text,gold):
    final=text.split('</think>')[-1]
    prediction=boxed(final)
    numbers=re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?",prediction if prediction is not None else final)
    prediction=numbers[-1].replace(',','') if numbers else None
    answer=gold.split('####')[-1].strip().replace(',','')
    try:correct=prediction is not None and abs(float(prediction)-float(answer))<1e-8
    except ValueError:correct=False
    return bool(correct),prediction,answer


def mmlu_example(row,answer=False):
    text=row['question']+'\n'
    for label,choice in zip('ABCD',row['choices']):text+=f'{label}. {choice}\n'
    text+='Answer:'
    if answer:text+=' '+ 'ABCD'[row['answer']]+'\n\n'
    return text


def prompt(tokenizer,row,development):
    if row['benchmark']=='mmlu':
        subject=row['subject']
        header=f"The following are multiple choice questions (with answers) about {subject.replace('_',' ')}.\n\n"
        examples=development[subject]
        assert len(examples)==5
        return tokenizer.encode(header+''.join(mmlu_example(x,True) for x in examples)+mmlu_example(row),add_special_tokens=False)
    question=row.get('question',row.get('problem'))
    question+='\nPlease reason step by step, and put your final answer within \\boxed{}.'
    return chat_ids(tokenizer,[dict(role='user',content=question)])


@torch.inference_mode()
def run(decoder,head,tokenizer,ids,device,*,gate=None,threshold=1.,limit=4096,choice_ids=None,forced=None):
    assert len(ids)>=2
    inputs=torch.tensor([ids],device=device)
    torch.cuda.synchronize()
    tick=time.perf_counter()
    result=decoder(input_ids=inputs[:,:-1],use_cache=True,cache_position=torch.arange(len(ids)-1,device=device))
    cache=result.past_key_values
    del result
    torch.cuda.synchronize()
    prefill=time.perf_counter()-tick
    generated=[];skipped=[];choice_scores=None;first_time=None
    eos={tokenizer.convert_tokens_to_ids('<|im_end|>'),tokenizer.convert_tokens_to_ids('<|endoftext|>')}
    current=inputs[:,-1:]
    maximum=len(forced) if forced is not None else (1 if choice_ids is not None else limit)
    torch.cuda.synchronize()
    decode_tick=time.perf_counter()
    route=Route(decoder) if gate is not None else None
    if route is not None:route.__enter__()
    try:
        for offset in range(maximum):
            position=torch.tensor([len(ids)-1+offset],device=device)
            if route is None:
                result=decoder(input_ids=current,past_key_values=cache,cache_position=position,use_cache=True)
                final,cache=result.last_hidden_state,result.past_key_values
                count=0
            else:
                final,cache,_,count,_=route.run(current,cache,position,gate=gate,threshold=threshold)
            logits=head(final[:,-1]).float()
            if choice_ids is not None:
                choice_scores=logits[0,choice_ids].cpu().tolist()
                token=int(logits[0,choice_ids].argmax())
            else:token=int(logits.argmax(-1))
            generated.append(token);skipped.append(count)
            if offset==0:
                torch.cuda.synchronize();first_time=time.perf_counter()-decode_tick
            if choice_ids is not None:break
            if forced is None and token in eos:break
            next_token=forced[offset] if forced is not None else token
            current=torch.tensor([[next_token]],device=device)
    finally:
        if route is not None:route.__exit__(None,None,None)
    torch.cuda.synchronize()
    decode_seconds=time.perf_counter()-decode_tick
    return dict(text=tokenizer.decode(generated,skip_special_tokens=True) if choice_ids is None else 'ABCD'[generated[0]],
                generated_ids=generated,generated_tokens=len(generated),prompt_tokens=len(ids),
                prefill_seconds=prefill,decode_seconds=decode_seconds,total_seconds=prefill+decode_seconds,
                time_to_first_token_seconds=prefill+first_time,
                truncated=forced is None and choice_ids is None and len(generated)==maximum and generated[-1] not in eos,
                skipped_blocks_total=sum(skipped),skip_counts=dict(Counter(skipped)),choice_scores=choice_scores)


def worker(args,out):
    model,decoder,head,tokenizer,evidence=load_model(args.model,args.rank,args.weight_inventory)
    device=f'cuda:{args.rank}'
    checkpoint=torch.load(args.classifier_run/'classifier.pt',weights_only=True,map_location='cpu')
    gate=SharedGate(checkpoint['source'],checkpoint['skips'],checkpoint['width']).to(device).eval()
    gate.load_state_dict(checkpoint['state_dict'])
    selected=next(x for x in checkpoint['policies'] if x['goal']==.99)
    threshold=selected['threshold']
    checks=parity(model,decoder,head,tokenizer,device,source=gate.source,skips=gate.skips)
    with (args.data/'benchmarks.jsonl').open() as handle:
        rows=[json.loads(line) for line in handle]
    development=defaultdict(list)
    for row in rows:
        if row['split']=='dev':development[row['subject']].append(row)
    test=[r for r in rows if r['split']=='test']
    for index,row in enumerate(test):row['evaluation_index']=index
    # Deterministic shuffling spreads math generation and MCQ work across GPUs.
    random.Random(20261120).shuffle(test)
    if args.smoke:
        test=[
            next(r for r in test if r['benchmark']==benchmark) for benchmark in ('gsm8k','math','mmlu')]
        test=test*args.workers
    shard=test[args.rank::args.workers]
    choice_ids=[]
    for answer in 'ABCD':
        encoded=tokenizer.encode(' '+answer,add_special_tokens=False)
        assert len(encoded)==1,(answer,encoded)
        choice_ids.append(encoded[0])
    warm=prompt(tokenizer,shard[0],development)
    for current in (None,gate):run(decoder,head,tokenizer,warm,device,gate=current,threshold=threshold,limit=16)
    if args.smoke:
        # A disabled gate must reproduce the actual native baseline on a real
        # benchmark prompt, including after cache growth across decode steps.
        native=run(decoder,head,tokenizer,warm,device,limit=32)
        disabled=run(decoder,head,tokenizer,warm,device,gate=gate,threshold=1.000001,limit=32)
        assert native['generated_ids']==disabled['generated_ids']
        assert disabled['skipped_blocks_total']==0
        emit('NO_SKIP_CONTROL',rank=args.rank,tokens=native['generated_tokens'],exact_tokens=True)
    path=out/f'predictions-{args.rank}.jsonl'
    assert not path.exists()
    with path.open('w') as handle:
        for done,row in enumerate(shard,1):
            ids=prompt(tokenizer,row,development)
            assert len(ids)<32768, 'unexpected context length; do not silently truncate evaluation prompts'
            options=dict(choice_ids=choice_ids) if row['benchmark']=='mmlu' else dict(limit=2048 if row['benchmark']=='gsm8k' else 4096)
            results={}
            modes=('full','early_exit') if row['evaluation_index']%2==0 else ('early_exit','full')
            for mode in modes:
                result=run(decoder,head,tokenizer,ids,device,gate=gate if mode=='early_exit' else None,threshold=threshold,**options)
                if row['benchmark']=='mmlu':
                    result.update(correct=result['text']=='ABCD'[row['answer']],prediction=result['text'],reference='ABCD'[row['answer']])
                else:
                    grader=grade_gsm if row['benchmark']=='gsm8k' else grade_math
                    correct,prediction,reference=grader(result['text'],row['answer'] if row['benchmark']=='gsm8k' else row['solution'])
                    result.update(correct=correct,prediction=prediction,reference=reference)
                results[mode]=result
            matched={}
            # Same known input tokens for both paths isolate time saved by routing
            # from changes in answer length. This is separate from free generation.
            if not args.smoke and row['evaluation_index']%251==0 and row['benchmark']!='mmlu':
                forced=results['full']['generated_ids'][:128]
                for mode in modes:
                    measured=run(decoder,head,tokenizer,ids,device,gate=gate if mode=='early_exit' else None,threshold=threshold,forced=forced)
                    matched[mode]={k:measured[k] for k in ('generated_tokens','prefill_seconds','decode_seconds','total_seconds','skipped_blocks_total')}
            record=dict(benchmark=row['benchmark'],evaluation_index=row['evaluation_index'],source_file=row['source_file'],index=row['index'],
                        subject=row.get('subject',row.get('type')),level=row.get('level'),results=results,matched_tokens=matched)
            handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');handle.flush()
            if done==1 or done%10==0:
                emit('EVAL',rank=args.rank,done=done,total=len(shard),benchmark=row['benchmark'],
                     full_correct=results['full']['correct'],exit_correct=results['early_exit']['correct'],
                     full_tokens=results['full']['generated_tokens'],exit_tokens=results['early_exit']['generated_tokens'])
    write(out/f'worker-{args.rank}.json',dict(rank=args.rank,count=len(shard),evidence=evidence,checks=checks,
         prediction_sha256=sha(path),classifier_sha256=sha(args.classifier_run/'classifier.pt'),threshold=threshold))


def aggregate(args,out):
    records=[]
    receipts=[]
    for rank in range(args.workers):
        receipt=json.loads((out/f'worker-{rank}.json').read_text())
        receipts.append(receipt)
        path=out/f'predictions-{rank}.jsonl'
        assert receipt['prediction_sha256']==sha(path)
        with path.open() as handle:records.extend(json.loads(line) for line in handle)
    expected=dict(gsm8k=1319,math=5000,mmlu=14042)
    counts=Counter(r['benchmark'] for r in records)
    if not args.smoke:
        assert dict(counts)==expected,counts
        assert len({r['evaluation_index'] for r in records})==sum(expected.values())
    checkpoint=torch.load(args.classifier_run/'classifier.pt',weights_only=True,map_location='cpu')
    deployed=next(p for p in checkpoint['policies'] if p['goal']==.99)
    summary=dict(counts=dict(counts),benchmarks={},classifier_run=str(args.classifier_run),
                 classifier_sha256=sha(args.classifier_run/'classifier.pt'),workers=args.workers,
                 geometry=dict(source=checkpoint['source'],skips=checkpoint['skips']),
                 deployed_policy=deployed,worker_receipts=receipts,
                 settings=dict(dtype='BF16',engine='native Transformers eager SDPA plus FLA GDN',batch_size=1,
                               sampling='greedy',thinking=False,mmlu_shots=5,math_max_new_tokens=4096,gsm8k_max_new_tokens=2048),
                 grader=dict(math='Hendrycks MATH normalized boxed exact match',gsm8k='last numeric / boxed numeric exact match',
                             mmlu='maximum next-token score over single tokens A/B/C/D'))
    for benchmark in expected:
        group=[r for r in records if r['benchmark']==benchmark]
        values={}
        for mode in ('full','early_exit'):
            predictions=[r['results'][mode] for r in group]
            n=len(predictions)
            summed={key:sum(p[key] for p in predictions) for key in ('generated_tokens','prefill_seconds','decode_seconds','total_seconds','skipped_blocks_total')}
            values[mode]=dict(count=n,correct=sum(p['correct'] for p in predictions),accuracy=sum(p['correct'] for p in predictions)/n,
                              truncated=sum(p['truncated'] for p in predictions),**summed,
                              decode_tokens_per_second=summed['generated_tokens']/summed['decode_seconds'],
                              mean_skipped_blocks=summed['skipped_blocks_total']/summed['generated_tokens'])
            values[mode]['percentiles']={key:dict(zip(('p50','p95'),torch.quantile(
                torch.tensor([p[key] for p in predictions],dtype=torch.float64),torch.tensor([.5,.95],dtype=torch.float64)).tolist()))
                for key in ('generated_tokens','prefill_seconds','decode_seconds','total_seconds','time_to_first_token_seconds')}
        values['comparison']=dict(accuracy_delta=values['early_exit']['accuracy']-values['full']['accuracy'],
                                  end_to_end_speedup=values['full']['total_seconds']/values['early_exit']['total_seconds'],
                                  decode_throughput_speedup=values['early_exit']['decode_tokens_per_second']/values['full']['decode_tokens_per_second'],
                                  lost_correct=sum(r['results']['full']['correct'] and not r['results']['early_exit']['correct'] for r in group),
                                  gained_correct=sum(not r['results']['full']['correct'] and r['results']['early_exit']['correct'] for r in group))
        subjects=sorted({r['subject'] for r in group if r['subject'] is not None})
        if subjects:
            values['by_subject']={}
            for subject in subjects:
                selected=[r for r in group if r['subject']==subject]
                values['by_subject'][subject]=dict(count=len(selected),**{
                    mode+'_accuracy':sum(r['results'][mode]['correct'] for r in selected)/len(selected)
                    for mode in ('full','early_exit')})
            if benchmark=='mmlu' and not args.smoke:
                assert len(subjects)==57,subjects
                values['macro_accuracy']={mode:sum(s[mode+'_accuracy'] for s in values['by_subject'].values())/57 for mode in ('full','early_exit')}
        matched=[r['matched_tokens'] for r in group if r['matched_tokens']]
        if matched:
            full=sum(r['full']['decode_seconds'] for r in matched);candidate=sum(r['early_exit']['decode_seconds'] for r in matched)
            values['matched_token_decode']=dict(examples=len(matched),tokens=sum(r['full']['generated_tokens'] for r in matched),speedup=full/candidate,
                                                full_seconds=full,early_exit_seconds=candidate)
        summary['benchmarks'][benchmark]=values
    write(out/'evaluation-summary.json',summary)
    emit('EVAL_PASS',counts=dict(counts),summary=str(out/'evaluation-summary.json'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--weight-inventory',type=Path,required=True)
    p.add_argument('--classifier-run',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--rank',type=int)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();out=Path(os.environ['RUN_DIR'])
    if args.rank is not None:worker(args,out);return
    assert torch.cuda.device_count()==args.workers
    import transformers
    tokenizer=transformers.AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    check_tokenizer(tokenizer)
    write(out/'config.yaml',dict(model=str(args.model),data=str(args.data),classifier_run=str(args.classifier_run),workers=args.workers,smoke=args.smoke,
                                benchmark_sha256=sha(args.data/'benchmarks.jsonl'),classifier_sha256=sha(args.classifier_run/'classifier.pt'),
                                weight_inventory_sha256=sha(args.weight_inventory)))
    children=[]
    try:
        for rank in range(args.workers):children.append(subprocess.Popen([sys.executable,'-u',__file__,*sys.argv[1:],'--rank',str(rank)]))
        for child in children:
            if child.wait():raise RuntimeError('evaluation worker failed')
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            if child.poll() is None:child.wait(timeout=60)
    aggregate(args,out)


if __name__=='__main__':main()
