"""Matched GPU evaluation on deterministic MRQA dev subsets; not paper reproduction."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import string
from typing import Optional
import time
from contextlib import nullcontext
from tqdm.auto import tqdm

import torch
from episetter.data import digest, normalize, write_json
from episetter.model import load_olmo
from episetter.calibration import LayerCalibrator


def official_metrics():
    # Reuse exactly the published pure functions without importing vLLM or changing HF_HOME.
    source = Path(__file__).resolve().parents[1] / 'episetter/mrqa_metrics.py'
    tree = ast.parse(source.read_text())
    names = {'normalize_answer','extract_answer','exact_match_score','acc_score','f1_score','metric_max_over_ground_truths'}
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    env = {'re':re, 'string':string, 'Counter':Counter, 'Optional':Optional}
    exec(compile(tree, str(source), 'exec'), env)
    return env


def generate_prompt_calibrated(runner, calibrator, ids, max_new_tokens):
    """Prefill once with prompt-position calibration, then decode from its modified KV cache."""
    generated=[]
    with torch.no_grad(),runner.hooks(ids.shape[1]-1,calibrator=calibrator):
        output=runner.model(input_ids=ids,use_cache=True)
    token=output.logits[:,-1:].argmax(-1)
    cache=output.past_key_values
    for _ in range(max_new_tokens):
        item=int(token.item());generated.append(item)
        if item==runner.tokenizer.eos_token_id:break
        with torch.no_grad(): output=runner.model(input_ids=token,past_key_values=cache,use_cache=True)
        cache=output.past_key_values
        token=output.logits[:,-1:].argmax(-1)
    return runner.tokenizer.decode(generated,skip_special_tokens=True)

def generate_prompt_calibrated_batch(runner, calibrator, ids, attention_mask, max_new_tokens):
    """Batched calibrated decode with left padding.

    Left padding makes the final prompt token share one position across the
    batch, so the existing prompt-position calibrator remains semantically
    identical to the single-row path. Attention masks are extended during
    cached decoding and EOS is tracked independently per example.
    """
    batch = ids.shape[0]
    generated=[[] for _ in range(batch)]
    finished=torch.zeros(batch,dtype=torch.bool,device=ids.device)
    mask=attention_mask
    with torch.no_grad(),runner.hooks(ids.shape[1]-1,calibrator=calibrator):
        output=runner.model(input_ids=ids,attention_mask=mask,use_cache=True)
    token=output.logits[:,-1:].argmax(-1)
    cache=output.past_key_values
    for _ in range(max_new_tokens):
        for i in range(batch):
            if not bool(finished[i]):
                item=int(token[i,0].item()); generated[i].append(item)
                if item==runner.tokenizer.eos_token_id: finished[i]=True
        if bool(finished.all()): break
        mask=torch.cat([mask,torch.ones((batch,1),device=mask.device,dtype=mask.dtype)],dim=1)
        with torch.no_grad():
            output=runner.model(input_ids=token,past_key_values=cache,
                                attention_mask=mask,use_cache=True)
        cache=output.past_key_values
        token=output.logits[:,-1:].argmax(-1)
    return [runner.tokenizer.decode(x,skip_special_tokens=True) for x in generated]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--method',choices=['base','ours'],required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--bundle',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--limit',type=int,default=100,help='Per dataset; 0 means all eligible samples')
    p.add_argument('--max-new-tokens',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=1,
                   help='Left-padded greedy batches; released HotpotQA uses 2')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--datasets', nargs='*', default=None,
                   help='Optional dataset filenames to run; omitted means all MRQA dev files.')
    args=p.parse_args()
    from episetter.release import check_model
    check_model(args.model)
    if not args.device.startswith('cuda'): p.error('GPU required')
    torch.empty(1,device=args.device)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=args.resume)
    runner=load_olmo(args.model,args.device,'bfloat16')
    cal=None; audit={}
    shift_gates=None
    if args.method=='ours':
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        # Legacy L11--15 calibrators predate the data_sha256 field; their
        # frozen run manifest binds the training bundle separately.
        if state.get('data_sha256') is not None and state['data_sha256'] != digest(args.bundle):
            raise ValueError('Checkpoint/data mismatch')
        cal=LayerCalibrator.restore(state,args.device)
        audit={'status':state['status'], 'parameters':sum(p.numel() for p in cal.parameters())}
    metric=official_metrics()
    bundle=json.loads(Path(args.bundle).read_text())
    exclude_q=set();exclude_c=set()
    for row in bundle['pairs']:
        exclude_q.add(normalize(row['question']));exclude_c.add(normalize(row['context']))
    # Prevent overlaps with all released SHIFT train/valid examples for both methods.
    for split in ('train','valid'):
        for r in json.loads(Path(f'data/shift_official/{split}.json').read_text()):
            exclude_q.add(normalize(r['question']))
            for k in ('context','cf_context'):
                if r[k]: exclude_c.add(normalize(r[k]))
    for r in bundle['protected']: exclude_c.add(normalize(r['prompt']))
    summaries={}; started=time.time()
    paths = sorted(Path('data/mrqa_dev').glob('*.jsonl.gz'))
    if not paths: raise FileNotFoundError('No data/mrqa_dev/*.jsonl.gz files; run from repository root')
    if args.datasets:
        wanted = set(args.datasets)
        paths = [path for path in paths if path.name in wanted or path.stem in wanted]
    for path in paths:
        rows=[]
        with gzip.open(path,'rt') as f:
            for line in f:
                doc=json.loads(line)
                if 'header' in doc:continue
                if normalize(doc['context']) in exclude_c:continue
                for q in doc['qas']:
                    if normalize(q['question']) not in exclude_q:
                        rows.append({'qid':q['qid'],'question':q['question'],'context':doc['context'],'answers':q['answers']})
        rows.sort(key=lambda r:hashlib.sha256(('42'+r['qid']).encode()).hexdigest())
        totals={'EM':0.,'F1':0.,'ACC':0.}; n=0; skipped=0; selected=[]
        prediction_path=root/(path.stem+'.predictions.jsonl')
        completed={}
        if args.resume and prediction_path.exists():
            completed={r['qid']:r for r in map(json.loads,prediction_path.open())}
            for r in completed.values():
                for k,v in r['scores'].items(): totals[k]+=v
            n=len(completed);selected=list(completed)
        # Left-padded batches share the final prompt position. This permits
        # calibrated decoding while keeping the intervention at the exact
        # prompt boundary used by the single-row reference path.
        if args.batch_size > 1 and shift_gates is None:
            pending=[]
            for row in rows:
                if row['qid'] in completed: continue
                system=('You are a helpful assistant that answers questions based on the provided background information.'
                        'The background information may be incorrect, so you should judge whether to believe yourself or to believe the background information.'
                        'You should think about the reasoning process and then provide the answer based on the given context.')
                user=('Background:\n'+row['context']+'\n\nTask Instruction:\n'
                      'Answer the question with the given background information above. '
                      'Provide your final answer in <answer> </answer> tags, '
                      'for example <answer>Petri Alanko</answer>.\n\nQ: '+row['question']+'\n\nA: ')
                messages=[{'role':'system','content':system},{'role':'user','content':user}]
                if getattr(runner.tokenizer, 'chat_template', None):
                    prompt=runner.tokenizer.apply_chat_template(messages, tokenize=False,
                        add_generation_prompt=True, enable_thinking=False)
                else:
                    prompt='System: '+system+'\n\nUser: '+user+'\nAssistant: '
                ids=runner.ids(prompt,False)
                if ids.shape[1]+args.max_new_tokens>2048:
                    skipped+=1; continue
                pending.append((row,prompt,ids.shape[1]))
                if args.limit and n+len(pending)>=args.limit: break
            old_side, old_pad = runner.tokenizer.padding_side, runner.tokenizer.pad_token
            runner.tokenizer.padding_side='left'
            if runner.tokenizer.pad_token_id is None:
                runner.tokenizer.pad_token=runner.tokenizer.eos_token
            with prediction_path.open('a' if args.resume else 'w') as f:
                for start in tqdm(range(0,len(pending),args.batch_size), desc=f"{args.method} {path.stem}", unit="batch", dynamic_ncols=True):
                    batch=pending[start:start+args.batch_size]
                    enc=runner.tokenizer([x[1] for x in batch], add_special_tokens=False,
                        padding=True, return_tensors='pt').to(runner.device)
                    if cal is not None:
                        responses=generate_prompt_calibrated_batch(runner,cal,enc['input_ids'],enc['attention_mask'],args.max_new_tokens)
                    else:
                        with torch.no_grad():
                            out=runner.model.generate(enc['input_ids'], attention_mask=enc['attention_mask'],
                                do_sample=False,max_new_tokens=args.max_new_tokens,use_cache=True,
                                pad_token_id=runner.tokenizer.eos_token_id)
                        responses=[runner.tokenizer.decode(out[i,enc['input_ids'].shape[1]:],skip_special_tokens=True)
                                   for i in range(len(batch))]
                    for i,(row,_,_) in enumerate(batch):
                        response=responses[i]
                        parsed=metric['extract_answer'](response)
                        answer=parsed if parsed is not None else response.strip()
                        scores={name:metric['metric_max_over_ground_truths'](metric[fn],answer,row['answers'])
                                for name,fn in [('EM','exact_match_score'),('F1','f1_score'),('ACC','acc_score')]}
                        for k,v in scores.items(): totals[k]+=v
                        n+=1; selected.append(row['qid'])
                        f.write(json.dumps({'qid':row['qid'],'response':response,'prediction':answer,
                            'format_compliant':parsed is not None,'references':row['answers'],'scores':scores})+'\n')
                    f.flush()
            runner.tokenizer.padding_side, runner.tokenizer.pad_token = old_side, old_pad
            summaries[path.stem]={'n':n,'scores':{k:100*v/n for k,v in totals.items()} if n else {},
                'skipped_long_before_limit':skipped,'selected_qids_sha256':hashlib.sha256('\n'.join(selected).encode()).hexdigest(),
                'source_sha256':digest(path)}
            write_json(root/'summary.json',{'arguments':vars(args),'datasets':summaries,'audit':audit,
                'seconds':time.time()-started,'scope':'MRQA dev subset, greedy, 2048-token budget, 32-token release default; inspect arguments.max_new_tokens for actual budget'})
            continue
        with prediction_path.open('a' if args.resume else 'w') as f:
            for row in tqdm(rows, desc=f"{args.method} {path.stem}", unit="row", dynamic_ncols=True):
                if row['qid'] in completed:continue
                system=('You are a helpful assistant that answers questions based on the provided background information.'
                        'The background information may be incorrect, so you should judge whether to believe yourself or to believe the background information.'
                        'You should think about the reasoning process and then provide the answer based on the given context.')
                user=('Background:\n'+row['context']+'\n\nTask Instruction:\n'
                      'Answer the question with the given background information above. '
                      'Provide your final answer in <answer> </answer> tags, '
                      'for example <answer>Petri Alanko</answer>.\n\nQ: '+row['question']+'\n\nA: ')
                messages=[{'role':'system','content':system},{'role':'user','content':user}]
                if getattr(runner.tokenizer, 'chat_template', None):
                    prompt=runner.tokenizer.apply_chat_template(messages, tokenize=False,
                        add_generation_prompt=True, enable_thinking=False)
                else:
                    prompt='System: '+system+'\n\nUser: '+user+'\nAssistant: '
                ids=runner.ids(prompt,False)
                if ids.shape[1]+args.max_new_tokens>2048:
                    skipped+=1;continue
                if cal is not None:
                    response=generate_prompt_calibrated(runner,cal,ids,args.max_new_tokens)
                else:
                    gate_context=nullcontext()
                    with torch.no_grad(),runner.hooks(ids.shape[1]-1),gate_context:
                        out=runner.model.generate(ids,do_sample=False,max_new_tokens=args.max_new_tokens,
                            use_cache=True,pad_token_id=runner.tokenizer.eos_token_id)
                    response=runner.tokenizer.decode(out[0,ids.shape[1]:],skip_special_tokens=True)
                parsed=metric['extract_answer'](response)
                # Formatting is diagnostic only. Correctness uses the full response when
                # the model does not emit the requested <answer> wrapper.
                answer=parsed if parsed is not None else response.strip()
                scores={name:metric['metric_max_over_ground_truths'](metric[fn],answer,row['answers'])
                        for name,fn in [('EM','exact_match_score'),('F1','f1_score'),('ACC','acc_score')]}
                for k,v in scores.items():totals[k]+=v
                n+=1;selected.append(row['qid'])
                f.write(json.dumps({'qid':row['qid'],'response':response,'prediction':answer,
                                    'format_compliant':parsed is not None,
                                    'references':row['answers'],'scores':scores})+'\n');f.flush()
                if n%20==0:print(args.method,path.stem,n,flush=True)
                if args.limit and n>=args.limit:break
        summaries[path.stem]={'n':n,'scores':{k:100*v/n for k,v in totals.items()} if n else {},
                              'skipped_long_before_limit':skipped,'selected_qids_sha256':hashlib.sha256('\n'.join(selected).encode()).hexdigest(),
                              'source_sha256':digest(path)}
        write_json(root/'summary.json',{'arguments':vars(args),'datasets':summaries,'audit':audit,
                   'seconds':time.time()-started,'scope':'MRQA dev subset, greedy, 2048-token budget, 32-token release default; inspect arguments.max_new_tokens for actual budget'})
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':main()
