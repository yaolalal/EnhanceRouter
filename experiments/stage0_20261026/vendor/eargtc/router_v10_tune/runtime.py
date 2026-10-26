"""Isolated training/input/prediction utilities for the Router-v10 tuning run."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from .policy import assert_open,validate_query
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/router_v10_20260912_v1'
RUN=ROOT/'runs/router_v10_tune_20260912_v1'
DATASETS=('cora','citeseer','pubmed','arxiv','ogbn-products')
STAGES=('regularization','transfer','training')

def require_slurm():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Numerical work requires Slurm')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def load(p):return torch.load(p,weights_only=True,map_location='cpu')
def check_files(seal):
    for p,h in seal['files'].items():
        if sha(p)!=h:raise ValueError('Changed sealed input '+p)
def authenticate(p):
    reg=read(RUN/'REGISTRATION.json');key=str(p)
    if key not in reg['files'] or sha(p)!=reg['files'][key]:raise ValueError('Unregistered/changed file '+key)
def metadata(d):
    p=BASE/'inputs/splits'/d/'gold_manifest.json';authenticate(p);return read(p)
def bank(d,branch,split):
    require_slurm()
    if split not in ('train','validation','query'):raise ValueError(split)
    if split=='query' and not (RUN/'MODELS_FROZEN.json').exists():raise RuntimeError('Query features deferred until model selection freeze')
    p=BASE/'inputs'/split/d/(branch+'.pt');authenticate(p);b=load(p)
    if split=='query':validate_query(b)
    elif split=='train':
        from eargtc.router_aug_search.common import guard_bank
        guard_bank(b,metadata(d))
    else:
        if not set(b['ids'].tolist())<=set(metadata(d)['validation_ids']):raise ValueError('Unexpected validation IDs')
    return b

def predict(w,b):
    if w['engine']=='tune':
        from .engine import predict_candidate
        return predict_candidate(w['model'],b)
    from eargtc.router_aug_search.common import predict as old_predict
    return old_predict(w,b)
def scores(bundle,b):
    if bundle['branch']=='agreement':return dict(score=predict(bundle['models']['agreement'],b))
    z={h:bundle['affine'][h]['slope']*predict(bundle['models'][h],b)+bundle['affine'][h]['bias'] for h in ('trust','preference')}
    p=expit(z['preference']);t=expit(z['trust']);route=(p<bundle['boundary']).astype(int)
    return dict(score=t*np.where(route==0,p,1-p),route=route,q=np.column_stack((t*p,t*(1-p),1-t)))
def correct(bundle,s,b):return np.asarray(b['target'],bool) if bundle['branch']=='agreement' else s['route']==np.asarray(b['state'])
def normalize(model):return dict(slope=1/model['scale'],bias=-model['center']/model['scale'])
