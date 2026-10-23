"""Conflict-only metrics, with Preference restricted to resolvable conflicts."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score


def binary_metrics(y,p):
    y,p=np.asarray(y),np.asarray(p)
    two=len(np.unique(y))==2
    return {'auroc':float(roc_auc_score(y,p)) if two else None,
            'auprc':float(average_precision_score(y,p)) if two else None}


def risk_curve(q,state,node_ids=None):
    q,state=np.asarray(q),np.asarray(state)
    score=q[:,:2].max(1); route=q[:,:2].argmax(1)
    ids=np.arange(len(q)) if node_ids is None else node_ids
    order=np.lexsort((ids,-score))
    risk=np.cumsum(route[order]!=state[order])/np.arange(1,len(q)+1)
    return risk


def evaluate(q,state,node_ids=None):
    q=np.asarray(q,dtype=float); z=np.asarray(state,dtype=int)
    n=len(z)
    if n==0:
        return {'count':0,'nll':None,'aurc':None,'trust_auroc':None,'preference_auroc':None}
    c=q[:,:2].sum(1); r=np.divide(q[:,0],c,out=np.full(n,.5),where=c>0)
    u=z!=2; route=q[:,:2].argmax(1); correct=route==z
    trust=binary_metrics(u,c); pref=binary_metrics(z[u]==0,r[u])
    curve=risk_curve(q,z,node_ids)
    errors=int((~correct).sum()); ideal=np.r_[np.zeros(n-errors),np.arange(1,errors+1)/np.arange(n-errors+1,n+1)]
    result=dict(count=n,state_counts=np.bincount(z,minlength=3).tolist(),
        nll=float(-np.log(np.maximum(q[np.arange(n),z],1e-15)).mean()),
        brier=float(((q-np.eye(3)[z])**2).sum(1).mean()),
        macro_f1=float(f1_score(z,q.argmax(1),labels=[0,1,2],average='macro',zero_division=0)),
        balanced_accuracy=float(np.mean([(q.argmax(1)[z==k]==k).mean() for k in np.unique(z)])),
        trust_auroc=trust['auroc'],trust_auprc=trust['auprc'],
        neither_auroc=trust['auroc'],neither_auprc=binary_metrics(~u,1-c)['auprc'],
        preference_auroc=pref['auroc'],preference_auprc=pref['auprc'],
        preference_accuracy=float(correct[u].mean()) if u.any() else None,
        gnn_only_recall=float((route[z==0]==0).mean()) if (z==0).any() else None,
        llm_only_recall=float((route[z==1]==1).mean()) if (z==1).any() else None,
        full_conflict_pseudo_accuracy=float(correct.mean()),
        pair_oracle_accuracy=float(u.mean()),gnn_route_fraction=float((route==0).mean()),
        aurc=float(curve.mean()),excess_aurc=float(curve.mean()-ideal.mean()))
    result['preference_gnn_route_fraction']=float((route[u]==0).mean()) if u.any() else None
    result['preference_balanced_accuracy']=float(np.mean([(route[z==k]==k).mean() for k in (0,1) if (z==k).any()])) if u.any() else None
    result['overall_ece']=ece(q.max(1),q.argmax(1)==z)
    result['classwise_ece']={name:ece(q[:,k],z==k) for k,name in enumerate(('G','L','N'))}
    result['risk_at_coverage']={str(f):float(curve[max(0,int(np.ceil(n*f))-1)]) for f in (.1,.25,.5,.75,1.)}
    return result


def selective_metrics(q,state,actions):
    q,z,a=np.asarray(q),np.asarray(state),np.asarray(actions)
    accept=a!=2; correct=a==z
    # Reject never counts as a correct pseudo-label, including state N.
    ac=accept & correct; an=accept & (z==2); aw=accept & (z!=2) & ~correct
    rn=~accept & (z==2); rg=~accept & (z==0); rl=~accept & (z==1)
    counts=[int(v.sum()) for v in (ac,an,aw,rn,rg,rl)]
    n=len(z); k=int(accept.sum())
    return dict(count=n,accepted_count=k,coverage=k/n if n else 0.,
        pseudo_accuracy=float(correct[accept].mean()) if k else None,
        risk=float((~correct[accept]).mean()) if k else None,
        reject_precision=float((z[~accept]==2).mean()) if (~accept).any() else None,
        reject_recall=float((~accept[z==2]).mean()) if (z==2).any() else None,
        error_decomposition=dict(zip(('accepted_correct','accepted_neither','accepted_wrong_expert',
            'rejected_neither','rejected_gnn_only','rejected_llm_only'),counts)))


def ece(probability,correct,bins=15):
    p,y=np.asarray(probability),np.asarray(correct)
    if not len(p): return None
    index=np.minimum((p*bins).astype(int),bins-1)
    total=0.
    for b in range(bins):
        mask=index==b
        if mask.any(): total+=mask.mean()*abs(p[mask].mean()-y[mask].mean())
    return float(total)
