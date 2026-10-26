"""Predeclared selective policies with deterministic tie handling."""
import math
import numpy as np


def wilson_upper(errors, count, z=1.6448536269514722):
    n=np.asarray(count,dtype=float)
    p=np.asarray(errors,dtype=float)/n
    return (p+z*z/(2*n)+z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def fit_risk_policy(score, correct, *, risk=.10, min_accepted=20):
    score,correct=np.asarray(score,dtype=float),np.asarray(correct,dtype=bool)
    if score.shape != correct.shape or not np.isfinite(score).all():
        raise ValueError('policy inputs must align and be finite')
    order=np.argsort(-score,kind='stable')
    sorted_score=score[order]
    count=np.arange(1,len(score)+1)
    errors=np.cumsum(~correct[order])
    # A threshold must accept the entire tied block, never cherry-pick tied labels.
    ends=np.r_[sorted_score[:-1]!=sorted_score[1:],True] if len(score) else np.array([],bool)
    upper=wilson_upper(errors,count)
    feasible=np.flatnonzero(ends & (count>=min_accepted) & (upper<=risk))
    i=int(feasible[-1]) if len(feasible) else None
    return dict(threshold=float(sorted_score[i]) if i is not None else None,
        target_risk=risk,min_accepted=min_accepted,validation_count=len(score),
        validation_accepted=i+1 if i is not None else 0,
        validation_error_upper_bound=float(upper[i]) if i is not None else None,
        reject_all=i is None)


def apply_policy(q, policy):
    q=np.asarray(q)
    route=np.argmax(q[:,:2],axis=1)
    accept=np.zeros(len(q),bool) if policy['reject_all'] else q[:,:2].max(1)>=policy['threshold']
    return np.where(accept,route,2)


def top_mask(score,node_ids,*,count=None,fraction=None):
    score,node_ids=np.asarray(score),np.asarray(node_ids)
    k=min(len(score),max(0,int(count if count is not None else math.ceil(len(score)*fraction))))
    order=np.lexsort((node_ids,-score))
    mask=np.zeros(len(score),bool); mask[order[:k]]=True
    return mask
