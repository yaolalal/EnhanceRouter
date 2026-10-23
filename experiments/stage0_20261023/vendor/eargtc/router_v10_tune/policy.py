"""Finite validation selection only; no fitting, labels forbidden in query banks."""
from pathlib import Path
from fractions import Fraction
import numpy as np
from eargtc.agreement_increment.policy import wilson_lower
FRACTIONS=(.0025,.005,.0075,.01,.015,.02,.025,.035,.05,.075,.1,.125,.15,.175,.2,.225,.25,.3,.35,.4,.5,.6,.75,1.)

def assert_open(root):
    if any((Path(root)/name).exists() for name in ('MODELS_FROZEN.json','FROZEN.json','EVALUATION_STARTED.json','COMPLETE.json')):
        raise RuntimeError('Search permanently closed')

def validate_query(bank):
    if set(bank)!={'dataset','ids','tab','g','l','gnn_pred','llm_pred'}:raise ValueError('Unsafe query schema')

def policy_key(p):return (p['precision'],p['wilson_lower'],p['coverage'])
def better(candidate,incumbent):return policy_key(candidate)>policy_key(incumbent)

def choose_policy(score,correct,ids,fractions=FRACTIONS,min_accepted=20):
    s=np.asarray(score,float);ids=np.asarray(ids);y=np.asarray(correct,bool)
    if s.ndim!=1 or s.shape!=ids.shape or y.shape!=s.shape or not np.isfinite(s).all() or len(np.unique(ids))!=len(ids):raise ValueError('Invalid aligned selection vectors')
    order=np.lexsort((ids,-s));good=np.cumsum(y[order]);n=len(s);rows=[]
    for value in fractions:
        f=Fraction(str(value))
        if not 0<f<=1:raise ValueError('Invalid fraction')
        k=(n*f.numerator+f.denominator-1)//f.denominator
        if k<min_accepted:continue
        c=int(good[k-1]);rows.append(dict(fraction=value,accepted=k,correct=c,wrong=k-c,population=n,coverage=k/n,precision=c/k,wilson_lower=wilson_lower(c,k)))
    if not rows:raise ValueError('Insufficient validation support')
    best=max(rows,key=policy_key)
    return dict(best,reject_all=False,min_accepted=min_accepted)
