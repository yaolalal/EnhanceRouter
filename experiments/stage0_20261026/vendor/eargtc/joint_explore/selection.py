"""Deterministic selection using validation probabilities only."""
import itertools
import torch
from eargtc.joint_cross_repair.training import _score


def rank(row):
    return (-row['accuracy'],row['nll'],row['id'])


def shortlist(rows):
    out=[]
    for architecture in sorted({r['architecture'] for r in rows}):
        matches=sorted((r for r in rows if r['architecture']==architecture),key=rank)
        out.extend(r['id'] for r in matches[:2])
    return out


def choose_final(families,y):
    if 'legacy' not in families:raise ValueError('legacy fallback required')
    catalog={k:v for k,v in families.items()}
    specs={k:{'components':{k:1.}} for k in families}
    new=sorted(k for k in families if k not in {'legacy','router','expert_mean','gnn','llm'})
    for k in new:
        for w in [.25,.5,.75]:
            name=f'{k}+legacy@{w}'
            catalog[name]=w*families[k]+(1-w)*families['legacy'];specs[name]={'components':{k:w,'legacy':1-w}}
    for a,b in itertools.combinations(new,2):
        for w in [.25,.5,.75]:
            name=f'{a}+{b}@{w}'
            catalog[name]=w*families[a]+(1-w)*families[b];specs[name]={'components':{a:w,b:1-w}}
    rows=[]
    for name,value in catalog.items():
        if value.ndim!=2 or value.shape[0]!=len(y) or not torch.isfinite(value).all():raise ValueError('invalid validation probabilities')
        rows.append({'id':name,**specs[name],**_score(value,y)})
    # Prefer a preserved fallback when probabilities/scores are exactly tied.
    best=min(rows,key=lambda r:(-r['accuracy'],r['nll'],r['id']!='legacy',r['id']))
    return {**best,'trials':rows},catalog
