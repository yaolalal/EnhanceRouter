"""Shared train/selection boundaries for augmented Router-v8; Slurm only."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from eargtc.router_augmentation_v8.sources import ROOT,DATASETS,SAFE,GOLD,read,sha,load,require_slurm
RUN=ROOT/'runs/router_aug_search_20260911_v1'
AUG=ROOT/'runs/router_augmentation_v8_20260911_v1'
DIS=ROOT/'runs/router_v8_learned_20260907_v4'
AGR=ROOT/'runs/agreement_increment_20260908_v1'
BASE=ROOT/'runs/router_v8_v6_final'
HEADS=('trust','preference','agreement')

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def guard_bank(bank,meta):
    ids=np.asarray(bank['ids']);parents=np.asarray(bank['node_ids']);n=len(ids)
    if bank.get('split')!='train' or ids.ndim!=1 or parents.shape!=ids.shape or len(np.unique(ids))!=n:raise ValueError('invalid train identity')
    if not set(parents)<=set(meta['gold_ids']) or set(parents)&(set(meta.get('validation_ids',[]))|set(meta.get('query_ids',[]))):raise ValueError('non-gold parent entered fitting')
    if not np.array_equal(np.asarray(bank['weight']),np.ones(n)):raise ValueError('all six views must have equal unit weight')
    for k in ('tab','g','l','gnn_pred','llm_pred'):
        if len(bank[k])!=n:raise ValueError('unaligned '+k)
    return dict(rows=n,parents=len(set(parents)),view_weights='unit',validation_training_rows=0,query_training_rows=0)

def subset(bank,mask):
    n=len(bank['ids']);mask=torch.as_tensor(mask,dtype=torch.bool)
    return {k:(v[mask] if isinstance(v,torch.Tensor) and v.ndim and len(v)==n else v) for k,v in bank.items()}

def parent_folds(bank):
    parents=np.asarray(bank['node_ids']);unique=np.unique(parents);d=bank['dataset']
    keys={int(n):hashlib.sha256(f'{d}|v8-learned-fivefold|{int(n)}'.encode()).digest() for n in unique}
    order=sorted(unique,key=lambda n:(keys[int(n)],int(n)))
    assign={int(n):j%5 for j,n in enumerate(order)}
    return [np.array([assign[int(n)]==k for n in parents],bool) for k in range(5)]

def head_target(bank,head):
    state=np.asarray(bank['target'] if head=='agreement' else bank['state'])
    mask=state!=2 if head=='preference' else np.ones(len(state),bool)
    y=state==0 if head=='preference' else state!=2 if head=='trust' else state.astype(bool)
    return mask,y

def fit_affine_grouped(banks,target,head,fit,predict):
    from eargtc.router_v8_exploration.core import fit_affine
    b=banks[target];out=np.full(len(b['ids']),np.nan);audits=[]
    for i,held in enumerate(parent_folds(b)):
        if not held.any():continue
        training=subset(b,~held);test=subset(b,held)
        assert not set(training['node_ids'].tolist())&set(test['node_ids'].tolist())
        m=fit(dict(banks,**{target:training}));out[held]=predict(m,test)
        audits.append(dict(fold=i,train_parent_ids=sorted(set(training['node_ids'].tolist())),held_parent_ids=sorted(set(test['node_ids'].tolist()))))
    if not np.isfinite(out).all():raise ValueError('incomplete grouped OOF')
    mask,y=head_target(b,head);aff=fit_affine(out[mask],y[mask])
    return dict(affine=aff,oof_logits=out,oof_rows=int(mask.sum()),folds=audits,validation_fit_rows=0,query_fit_rows=0)

def metadata():return {d:read(GOLD/d/'manifest.json') for d in DATASETS}

def load_training(branch):
    require_slurm();banks={};metas=metadata()
    for d in DATASETS:
        p=AUG/'banks'/d;manifest=read(p/'manifest.json');name=branch+'_train.pt'
        if sha(p/name)!=manifest['output_hashes'][name]:raise ValueError('changed augmented bank '+d)
        b=load(p/name);guard_bank(b,metas[d]);banks[d]=b
    return banks

def load_eval(dataset,branch,split):
    require_slurm()
    if split not in ('validation','query'):raise ValueError('invalid inference role')
    from eargtc.router_v8_learned.data import load_bank
    from eargtc.agreement_increment.data import load_bank as load_agreement
    b=load_bank(DIS,dataset,split) if branch=='disagreement' else load_agreement(AGR,dataset,split)
    m=metadata()[dataset];expected=set(m[split+'_ids']);ids=set(b['ids'].tolist())
    if not ids<=expected or ids&set(m['gold_ids']):raise ValueError('evaluation split mismatch')
    if split=='query' and any(k in b for k in ('state','target','labels','y')):raise ValueError('query truth forbidden')
    return b

def historical_fraction(dataset,branch):
    p=load(BASE/'targets'/dataset/(branch+'_pipeline.pt'))
    return int(p['accepted_mask'].sum())/len(p['node_ids'])

def assert_search_open():
    require_slurm()
    if (RUN/'FROZEN.json').exists() or (RUN/'EVALUATION_STARTED.json').exists():raise RuntimeError('search permanently closed')


def engine(name):
    if name=='v6':from . import v6_engine as e
    elif name=='v7':from . import v7_engine as e
    elif name=='expanded':from . import v7_expanded as e
    else:raise ValueError(name)
    return e


def fit(cfg,head,banks,target,engine_name):
    for d,b in banks.items():guard_bank(b,metadata()[d])
    m=engine(engine_name).fit_candidate(cfg,head,banks,target)
    return dict(engine=engine_name,model=m)

def predict(wrapped,bank):return engine(wrapped['engine']).predict_candidate(wrapped['model'],bank)
