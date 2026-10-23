"""Select heads using validation calibration AUROC; no loss metrics."""
import math


def choose(rows):
    if not rows:
        raise ValueError('empty candidate list')
    valid=[r for r in rows if r['metrics']['auc'] is not None and math.isfinite(r['metrics']['auc'])]
    if not valid:
        return next((r for r in rows if r['id']=='incumbent'),min(rows,key=lambda r:r['id']))
    return min(valid,key=lambda r:(-r['metrics']['auc'],
        -(r['metrics']['mean_fold_auc'] if r['metrics']['mean_fold_auc'] is not None else -1.),
        r['complexity'],r['id']))


def measure(logits,state,ids,dataset,head):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from eargtc.router_v8_exploration.core import binary_target
    from eargtc.router_v8_learned.experiment import calibration_folds
    mask,y=binary_target(state,head)
    logits=np.asarray(logits,float)
    assert np.isfinite(logits).all()
    pooled=float(roc_auc_score(y,logits[mask])) if len(np.unique(y))==2 else None
    folds=[]
    for held in calibration_folds(ids,dataset):
        hm,hy=binary_target(np.asarray(state)[held],head)
        if len(np.unique(hy))==2:
            folds.append(float(roc_auc_score(hy,logits[held][hm])))
    mean=float(np.mean(folds)) if folds else None
    sem=float(np.std(folds,ddof=1)/np.sqrt(len(folds))) if len(folds)>1 else None
    return dict(auc=pooled,mean_fold_auc=mean,fold_sem=sem,valid_folds=len(folds),
                count=len(y),positive_count=int(y.sum()),negative_count=int(len(y)-y.sum()))
