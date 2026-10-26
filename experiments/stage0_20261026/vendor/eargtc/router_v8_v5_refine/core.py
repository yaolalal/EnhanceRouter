"""Finite validation selection; no optimizer or loss is evaluated here."""
import math


def choose_head(rows,incumbent):
    if not rows:raise ValueError('no candidates')
    old=next(r for r in rows if r['id']==incumbent)
    valid=[r for r in rows if r['full_auc'] is not None and math.isfinite(r['full_auc'])]
    if not valid:return old
    floor=old['full_auc']-.005 if old['full_auc'] is not None else -1.
    valid=[r for r in valid if r['full_auc']>=floor]
    def key(r):
        halves=[x for x in r['half_auc'] if x is not None and math.isfinite(x)]
        worst=min(halves) if halves else r['full_auc']
        return (-(.5*r['full_auc']+.5*worst),-r['full_auc'],r['complexity'],r['id'])
    return min(valid,key=key)


def centered_affine(multiplier,boundary_logit):
    if not math.isfinite(multiplier) or multiplier<=0 or not math.isfinite(boundary_logit):
        raise ValueError('finite positive slope and finite boundary required')
    return dict(slope=float(multiplier),bias=float(-multiplier*boundary_logit))


def precision_summary(score,correct,ids):
    import numpy as np
    score=np.asarray(score);correct=np.asarray(correct,bool);ids=np.asarray(ids)
    order=np.lexsort((ids,-score));counts=[];values=[]
    for f in (.05,.1,.2,.35,.5):
        k=int(np.ceil(len(ids)*f))
        if k>=10:
            counts.append(k);values.append(float(correct[order[:k]].mean()))
    return dict(mean=float(np.mean(values)) if values else float(correct.mean()) if len(correct) else 0.,
        counts=counts,precision=values)


def choose_post(rows):
    return min(rows,key=lambda r:(-min(x['mean'] for x in r['halves']),
        -r['full']['mean'],r['distance_to_identity'],r['id']))
