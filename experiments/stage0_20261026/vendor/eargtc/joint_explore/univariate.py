"""Univariate discrimination with explicit missing targets, ties and fixed direction."""
import numpy as np
from scipy.special import ndtr


def rank_info(score):
    x=np.asarray(score,dtype=np.float64)
    order=np.argsort(x,kind='stable')
    s=x[order]; n=len(s)
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    ends=np.r_[starts[1:],n]
    counts=ends-starts
    ranks=np.empty(n,dtype=np.float64)
    ranks[order]=np.repeat((starts+1+ends)/2,counts)
    ties=float(np.sum(counts.astype(np.float64)**3-counts))
    return ranks,ties,len(counts)


def all_auc(score,targets):
    """Return [target] AUC and two-sided asymptotic rank-test p; -1 means excluded.

    Tied observations receive average ranks. One-class targets return NaN;
    constant scores with two classes have AUC .5 and p 1.
    """
    x=np.asarray(score,dtype=np.float64)
    y=np.asarray(targets)
    if x.ndim!=1 or y.ndim!=2 or len(x)!=len(y) or not np.isfinite(x).all():
        raise ValueError('finite scores and aligned target matrix required')
    if not np.isin(y,[-1,0,1]).all(): raise ValueError('binary targets or -1 required')
    auc=np.full(y.shape[1],np.nan); p=auc.copy(); cache={}
    for j in range(y.shape[1]):
        valid=y[:,j]>=0; yy=y[valid,j]; n=len(yy); pos=int(yy.sum()); neg=n-pos
        if not pos or not neg: continue
        key=None if valid.all() else valid.tobytes()
        if key not in cache: cache[key]=rank_info(x[valid])
        rank,ties,_=cache[key]
        auc[j]=(float(rank[yy==1].sum())-pos*(pos+1)/2)/(pos*neg)
        variance=(n+1-ties/(n*(n-1)))/(12*pos*neg)
        p[j]=2*ndtr(-abs(auc[j]-.5)/np.sqrt(variance)) if variance>0 else 1.
    return auc,p


def binary_auc(score,target):
    auc,p=all_auc(score,np.asarray(target)[:,None])
    return (None if np.isnan(auc[0]) else float(auc[0]), None if np.isnan(p[0]) else float(p[0]))


def choose_direction(score,calibration_target):
    auc,_=binary_auc(score,calibration_target)
    return -1 if auc is not None and auc<.5 else 1


def top_order(node_ids,score,direction):
    ids=np.asarray(node_ids); x=np.asarray(score,dtype=np.float64)
    if ids.ndim!=1 or ids.shape!=x.shape or len(np.unique(ids))!=len(ids) or direction not in (-1,1):
        raise ValueError('unique aligned IDs and +/-1 direction required')
    if not np.isfinite(x).all(): raise ValueError('finite score required')
    return np.lexsort((ids,-direction*x))


def bh_adjust(pvalues):
    p=np.asarray(pvalues,dtype=np.float64)
    result=np.full_like(p,np.nan); indices=np.flatnonzero(np.isfinite(p))
    order=indices[np.argsort(p[indices])]; n=len(order)
    if n:
        adjusted=p[order]*n/np.arange(1,n+1)
        result[order]=np.minimum(1,np.minimum.accumulate(adjusted[::-1])[::-1])
    return result
