"""Preregistered source-only search; no graph or test-label loader is imported."""
import itertools
import json
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
from .artifacts import write_json,sha256
from .training import fit_model,select_features,predict_logits
from .calibration import probabilities
from .metrics import evaluate

DATASETS=('cora','citeseer','pubmed','arxiv','ogbn-products')
SNAPSHOTS=(80,240)


def load_supports(root):
    return {d:torch.load(Path(root)/'banks'/d/'support.pt',map_location='cpu',weights_only=False) for d in DATASETS}


def preregister(root):
    root=Path(root)
    destination=root/'PREREGISTERED.json'
    if destination.exists():
        raise FileExistsError(destination)
    manifests={d:json.loads((root/'banks'/d/'manifest.json').read_text()) for d in DATASETS}
    groups=manifests[DATASETS[0]]['feature_groups']
    for d in DATASETS:
        if manifests[d]['feature_groups']!=groups:
            raise ValueError('feature groups differ across datasets')
    candidates=[]
    for evidence in groups:
        for width,dropout,wd,treatment,view in itertools.product((32,64),(0.,.1),(1e-4,1e-3,1e-2),('proper','sqrt-balanced'),('raw','rank')):
            candidates.append(dict(architecture='mlp',width=width,dropout=dropout,weight_decay=wd,treatment=treatment,view=view,evidence=evidence,lr=.003))
        # Low capacity shared linear scorer is an additional architecture, not a legacy router.
        for wd,treatment,view in itertools.product((1e-4,1e-3,1e-2),('proper','sqrt-balanced'),('raw','rank')):
            candidates.append(dict(architecture='linear',width=32,dropout=0.,weight_decay=wd,treatment=treatment,view=view,evidence=evidence,lr=.01))
    for i,cfg in enumerate(candidates): cfg['id']=f'c{i:04d}'
    payload=dict(seed=42,shots=5,datasets=DATASETS,snapshots=SNAPSHOTS,candidates=candidates,
        feature_groups=groups,selection=['macro_nll_with_0.5pct_tolerance','macro_aurc','worst_neither_auroc','macro_preference_auroc','parameter_count'],
        primary_policy=dict(risk=.10,min_accepted=20,wilson_z=1.6448536269514722),
        secondary_risks=[.05,.10,.15,.20],coverage=[.01,.025,.05,.1,.25,.5,.75,1.],top_k=[20,50,100,200,500,1000],costs=[.1,.2,.3],
        thresholds=[.5,.6,.7,.8,.9,.95,.99],
        support_adaptation=dict(bias_only=True,strength=20.),calibration=dict(strength=2.,min_per_class=5),
        validation_split='historical split-C label-blind hash; see target record',
        deviations=['single seed42; multiseed, complete baselines, ablations and bootstrap deferred by user'],
        input_hashes={str((root/'banks'/d/f).resolve()):sha256(root/'banks'/d/f) for d in DATASETS for f in ('support.pt','validation.pt','query.pt','manifest.json')},
        code_hashes={str(p.resolve()):sha256(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(payload,destination)
    return payload


_WORKER=None

def _init_worker(root,excluded,groups):
    global _WORKER
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _WORKER=(load_supports(root),tuple(excluded),groups)


def _fit_candidate(config):
    supports,excluded,groups=_WORKER
    banks={d:select_features(b,groups[config['evidence']]) for d,b in supports.items()}
    sources=[banks[d] for d in DATASETS if d not in excluded]
    model,transform,snapshots=fit_model(sources,config,snapshots=SNAPSHOTS)
    result=[]
    for epoch,weights in snapshots.items():
        model.load_state_dict(weights)
        held={}
        for d in excluded:
            bank=transform.apply(banks[d]); c,r=predict_logits(model,bank)
            held[d]=evaluate(probabilities(c,r),bank['state'].numpy(),bank['node_ids'].numpy())
        result.append(dict(config={**config,'epochs':epoch},held=held,parameter_count=sum(p.numel() for p in model.parameters())))
    return result


def run_pair(root,pair_index,workers=4):
    root=Path(root); pre=json.loads((root/'PREREGISTERED.json').read_text())
    for path,expected in pre['input_hashes'].items():
        # Every worker loads only support, not validation/query labels.
        if Path(path).name=='support.pt' and sha256(path)!=expected:
            raise ValueError('preregistered support hash changed')
    verify_code(pre)
    pair=list(itertools.combinations(DATASETS,2))[pair_index]
    out=root/'search'/f'pair_{pair_index:02d}'; out.mkdir(parents=True,exist_ok=True)
    fingerprint=sha256(root/'PREREGISTERED.json')
    for cached in out.glob('c*.json'):
        if json.loads(cached.read_text()).get('preregistration_sha256')!=fingerprint:
            raise ValueError('stale source cache')
    todo=[c for c in pre['candidates'] if not (out/(c['id']+'.json')).exists()]
    started=time.time()
    with ProcessPoolExecutor(max_workers=workers,initializer=_init_worker,initargs=(str(root),pair,pre['feature_groups'])) as pool:
        futures={pool.submit(_fit_candidate,c):c for c in todo}
        for i,future in enumerate(as_completed(futures),1):
            config=futures[future]
            rows=future.result()
            write_json(dict(preregistration_sha256=fingerprint,excluded=pair,source_domains=[d for d in DATASETS if d not in pair],rows=rows),out/(config['id']+'.json'))
            if i%10==0 or i==len(todo):
                print(json.dumps(dict(pair=pair,completed=i,total=len(todo),seconds=round(time.time()-started,1))),flush=True)
    write_json(dict(pair=pair,candidates=len(pre['candidates']),complete=True),out/'COMPLETE.json')


def select_target(root,target):
    root=Path(root); pre=json.loads((root/'PREREGISTERED.json').read_text())
    pool={}
    for pi,pair in enumerate(itertools.combinations(DATASETS,2)):
        if target not in pair: continue
        held=next(d for d in pair if d!=target)
        directory=root/'search'/f'pair_{pi:02d}'
        if not (directory/'COMPLETE.json').exists(): raise ValueError('source search incomplete')
        for cfg in pre['candidates']:
            result=json.loads((directory/(cfg['id']+'.json')).read_text())
            if result.get('preregistration_sha256')!=sha256(root/'PREREGISTERED.json'):
                raise ValueError('source result preregistration mismatch')
            if target in result['source_domains'] or held in result['source_domains']:
                raise ValueError('LODO leakage')
            for row in result['rows']:
                key=(cfg['id'],row['config']['epochs'])
                record=pool.setdefault(key,dict(config=row['config'],parameter_count=row['parameter_count'],held={}))
                record['held'][held]=row['held'][held]
    trials=[]
    for row in pool.values():
        held=list(row['held'].values())
        if len(held)!=4: raise ValueError('missing LODO domain')
        def mean(key):
            vals=[m[key] for m in held if m[key] is not None]
            return float(np.mean(vals)) if vals else None
        trust=[m['neither_auroc'] for m in held if m['neither_auroc'] is not None]
        row.update(macro_nll=mean('nll'),macro_aurc=mean('aurc'),
                   worst_neither_auroc=min(trust) if trust else None,macro_preference_auroc=mean('preference_auroc'))
        trials.append(row)
    best_nll=min(t['macro_nll'] for t in trials)
    eligible=[t for t in trials if t['macro_nll']<=best_nll*1.005]
    best=min(eligible,key=lambda t:(t['macro_aurc'],-(t['worst_neither_auroc'] or 0.),
        -(t['macro_preference_auroc'] or 0.),t['parameter_count'],t['config']['epochs'],t['config']['id']))
    return dict(target=target,best=best,trials=trials,best_nll=best_nll,eligible_count=len(eligible),
                selection_population='other four datasets corrected OOF conflict support only')


def verify_code(pre):
    for filename,expected in pre['code_hashes'].items():
        if sha256(filename)!=expected:
            raise ValueError(f'preregistered code changed: {filename}')
