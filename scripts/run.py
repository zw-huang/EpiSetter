"""Frozen R5 training and MR/MC transfer; no answer truncation or skipped rows."""
import argparse
import ast
from collections import Counter, defaultdict
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from episetter.data import digest, normalize, write_json, validate_bundle
from episetter.protocol import prompt, outcome, rows

# An output root is chosen before preparation and becomes part of the frozen
# protocol.  Keeping the original CPU-offload attempt immutable lets the
# GPU-resident formal rerun be fully auditable rather than silently changing a
# run in place.
ROOT = Path(os.environ.get('R5_ROOT', PROJECT / 'runs/l11_15'))
MODEL = os.environ.get('EPISETTER_MODEL', '/home/zhenwei/models/olmo')

def prepare():
    """Use the shipped frozen data; never reconstruct splits from legacy runs."""
    import shutil
    ROOT.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((PROJECT / 'data/freeze.json').read_text())
    for name, expected in frozen['files'].items():
        original = PROJECT / 'data' / name
        if digest(original) != expected:
            raise ValueError('Frozen input hash mismatch: ' + name)
        target = ROOT / name
        if not target.exists():
            shutil.copy2(original, target)
        if digest(target) != expected:
            raise ValueError('Run input hash mismatch: ' + name)
    for name in ['freeze.json', 'memory_smoke.json']:
        target = ROOT / name
        if not target.exists():
            shutil.copy2(PROJECT / 'data' / name, target)
    return frozen

def setup():
    import torch
    from episetter.release import check_model
    check_model(MODEL)
    from episetter.model import load_olmo
    runner = load_olmo(MODEL, 'cuda:0', 'bfloat16')
    original = runner.run
    saved_tensors = os.environ.get('R5_SAVED_TENSORS', 'cpu')
    if saved_tensors not in {'cpu', 'gpu'}:
        raise ValueError('R5_SAVED_TENSORS must be cpu or gpu')
    def run(*args, **kwargs):
        # The GPU route is exact serial autograd, validated on the longest
        # frozen answer.  CPU offload remains available for reproducing v1.
        ctx = (torch.autograd.graph.save_on_cpu(pin_memory=False)
               if kwargs.get('grad') and saved_tensors == 'cpu' else nullcontext())
        with ctx:
            return original(*args, **kwargs)
    runner.run = run
    return runner

def smoke():
    import torch
    f = prepare()
    b = json.loads((ROOT/'train_bundle.json').read_text())
    b['model_path'] = str(Path(MODEL).resolve())
    runner = setup()
    row = max((r for r in b['protected'] if r['split']=='train'), key=lambda r:len(runner.ids(r['prompt'])[0])+len(runner.ids(' '+r['answer'], False)[0]))
    # Test the longest protected sequence and both gradient routes before expensive fitting.
    torch.cuda.reset_peak_memory_stats()
    out = runner.run(row['prompt'], row['answer'], grad=True, layer=11, leaf=True)
    for i in [0, len(out['token_logp'][0])-1]:
        g, = torch.autograd.grad(-out['token_logp'][0,i],out['trace']['leaf'],retain_graph=i==0)
        assert torch.isfinite(g).all()
    del out, g
    from episetter.calibration import LayerCalibrator, prefix_kl
    u = torch.zeros(runner.model.config.hidden_size); u[0]=1
    # A single-layer calibration backward is sufficient to gate the graph
    # mechanism. Constructing a five-layer calibration graph here duplicated
    # the full training memory peak and was itself killed with status 137.
    cal = LayerCalibrator([{'layer':11,'direction':u,'scale':1.,'status':'validation_feasible','mode':'protected'}]).to('cuda:0')
    base = runner.run(row['prompt'],row['answer'])
    changed = runner.run(row['prompt'],row['answer'],grad=True,calibrator=cal)
    prefix_kl(base,changed).backward()
    assert all(torch.isfinite(p.grad).all() for p in cal.parameters() if p.grad is not None)
    del changed, base, cal, runner
    import gc
    gc.collect(); torch.cuda.empty_cache()
    write_json(ROOT/'memory_smoke.json', {'status':'passed','longest_protection_id':row['id'], 'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'full_answer_preserved':True})

def train(seed):
    import torch
    from episetter.experiments import fit
    from episetter.calibration import train_calibrator
    f = prepare()
    assert json.loads((ROOT/'memory_smoke.json').read_text())['status']=='passed'
    b = json.loads((ROOT/'train_bundle.json').read_text())
    b['model_path'] = str(Path(MODEL).resolve())
    runner = setup()
    out = ROOT/f'seed{seed}'
    import platform
    import transformers
    from episetter.data import model_manifest
    write_json(out/'run_manifest.json', {'seed':seed,'freeze_sha256':digest(ROOT/'freeze.json'),
        'model_manifest':model_manifest(MODEL),'torch':str(torch.__version__),
        'transformers':transformers.__version__,'python':platform.python_version(),
        'source_hashes':{p:digest(PROJECT/p) for p in ['scripts/run.py','episetter/model.py','episetter/experiments.py','episetter/calibration.py','episetter/core.py']}})
    fit(runner,b,out/'fit',layers=f['layers'],steps=f['direction_steps'],seed=seed,source_limit=None,protection_limit=500,
        protection_basis_cache=ROOT/'basis_cache')
    bank = torch.load(out/'fit/directions.pt',map_location='cpu',weights_only=True)
    if not bank['directions']:
        raise RuntimeError('No validation-feasible protected direction; do not fabricate a model')
    print('VALIDATION_SELECTED_LAYERS', [d['layer'] for d in bank['directions']], flush=True)
    train_calibrator(runner,b,bank,out/'calibration',steps=f['calibration_steps'],seed=seed,validation_every=f['validation_every'])

def evaluate(seed):
    import torch
    from tqdm.auto import tqdm
    from episetter.calibration import LayerCalibrator, calibration_metrics
    prepare()
    test = json.loads((ROOT/'test417.json').read_text())
    runner = setup()
    out = ROOT/f'seed{seed}'
    state = torch.load(out/'calibration/calibrator.pt',map_location='cpu',weights_only=True)
    cal = LayerCalibrator.restore(state,'cuda:0')
    path = out/'predictions.jsonl'
    done = {r['id'] for r in rows(path)} if path.exists() else set()
    with path.open('a') as stream:
        for r in tqdm(test,desc=f'seed{seed} MR/MC test',unit='fact'):
            if r['id'] in done: continue
            rec = {'id':r['id'],'group_id':r['group_id'],'dataset':r['dataset'],'canonical_answer_overlap':r['canonical_answer_overlap'],'methods':{}}
            for method in ['base','trained']:
                vals = {intent:outcome(runner.generate(prompt(r['question'],r['context'],intent,'fewshot'),32,**({'calibrator':cal} if method=='trained' else {})),r,intent) for intent in ['prior','context']}
                vals['PairAcc'] = vals['prior']['parsed']['EM']*vals['context']['parsed']['EM']
                vals['raw_PairAcc'] = vals['prior']['raw']['EM']*vals['context']['raw']['EM']
                rec['methods'][method]=vals
            stream.write(json.dumps(rec,ensure_ascii=False)+'\n'); stream.flush()
    records = rows(path)
    summary = {'seed':seed,'checkpoint_sha256':digest(out/'calibration/calibrator.pt'),'status':state['status'],'active_layers':list(cal.layers),'cohorts':{}}
    for dataset in ['all','MR','MC','strict_conflict']:
        rs = [r for r in records if dataset=='all' or r['dataset']==dataset or (dataset=='strict_conflict' and not r['canonical_answer_overlap'])]
        diff = [r['methods']['trained']['PairAcc']-r['methods']['base']['PairAcc'] for r in rs]
        groups = defaultdict(list)
        for r,d in zip(rs,diff): groups[r['group_id']].append(d)
        rng=random.Random(42); keys=sorted(groups); draws=[]
        for _ in range(2000):
            sampled=[v for k in rng.choices(keys,k=len(keys)) for v in groups[k]]
            draws.append(sum(sampled)/len(sampled))
        draws.sort()
        summary['cohorts'][dataset]={'n':len(rs),'groups':len(keys),'delta':sum(diff)/len(diff),'delta_ci95':[draws[49],draws[1949]],'corrected':sum(d>0 for d in diff),'broken':sum(d<0 for d in diff),'methods':{m:{'PairAcc':sum(r['methods'][m]['PairAcc'] for r in rs)/len(rs),'raw_PairAcc':sum(r['methods'][m]['raw_PairAcc'] for r in rs)/len(rs),**{i+'_EM':sum(r['methods'][m][i]['parsed']['EM'] for r in rs)/len(rs) for i in ['prior','context']}} for m in ['base','trained']}}
    write_json(out/'summary.json',summary)
    b=json.loads((ROOT/'train_bundle.json').read_text())
    write_json(out/'protected_test.json',calibration_metrics(runner,b,cal,'evaluation'))

def report():
    import statistics
    summaries=[json.loads((ROOT/f'seed{s}/summary.json').read_text()) for s in [42,43,44]]
    out={'seeds':summaries,'human_audit':'pending','aggregates':{}}
    for cohort in ['all','MR','MC','strict_conflict']:
        vals=[s['cohorts'][cohort]['methods']['trained']['PairAcc'] for s in summaries]
        out['aggregates'][cohort]={'PairAcc_mean':statistics.mean(vals),'PairAcc_seed_sd':statistics.stdev(vals)}
    write_json(ROOT/'summary.json',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','smoke','train','evaluate','report']);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='smoke':smoke()
    elif a.stage=='train':train(a.seed)
    elif a.stage=='evaluate':evaluate(a.seed)
    else:report()
