"""Source-fitted interpretable features, convex heads, adapters and honest policies."""
from dataclasses import dataclass
from fractions import Fraction
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from eargtc.router_v8.deployment import validation_halves
from eargtc.router_v8.policy import fit_risk_policy

HEADS=('trust','preference')
FRACTIONS=(.01,.025,.05,.10,.15,.20,.25,.35,.50,.75,1.)


def array(value):
    return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)


def validate_bank(bank,query=False):
    forbidden={'state','labels','label','y','y_true','correctness','target','targets','gnn_correct','llm_correct'}
    if query and forbidden.intersection(bank):
        raise ValueError('query bank must be label-free')
    n=len(bank['node_ids'])
    for key,width in [('candidate_g',27),('candidate_l',27),('symmetric_global',16)]:
        x=array(bank[key])
        if x.shape!=(n,width) or not np.isfinite(x).all():
            raise ValueError(f'invalid finite shape for {key}')
    ids=array(bank['node_ids'])
    if len(np.unique(ids))!=n: raise ValueError('duplicate bank node IDs')
    if len(bank['candidate_names'])!=27 or len(bank['global_names'])!=16:
        raise ValueError('bank feature names must match widths')
    for key in ('gnn_pred','llm_pred'):
        if array(bank[key]).shape!=(n,): raise ValueError(f'invalid {key}')
    if not query:
        z=array(bank['state'])
        if z.shape!=(n,) or not np.isin(z,[0,1,2]).all(): raise ValueError('invalid state')
    return bank


def binary_target(state,head):
    z=array(state)
    if head not in HEADS: raise ValueError('unknown head')
    mask=np.ones(len(z),bool) if head=='trust' else z!=2
    return mask,(z[mask]!=2 if head=='trust' else z[mask]==0).astype(float)


def _raw_features(bank,head,representation):
    nc,ng={'core':(16,9),'context':(19,12),'prototype':(27,16),'stumps':(16,9)}[representation]
    g=array(bank['candidate_g']).astype(float,copy=False)
    l=array(bank['candidate_l']).astype(float,copy=False)
    s=array(bank['symmetric_global']).astype(float,copy=False)
    cand=list(bank['candidate_names']); glob=list(bank['global_names'])
    indices=list(range(10))+list(range(16,nc))
    shared=.5*(g[:,10:16]+l[:,10:16])
    if head=='preference':
        x=np.column_stack((g[:,indices]-l[:,indices],shared,s[:,:ng]))
        names=[f'difference:{cand[i]}' for i in indices]
    else:
        x=np.column_stack((g[:,indices]+l[:,indices],np.abs(g[:,indices]-l[:,indices]),shared,s[:,:ng]))
        names=[f'sum:{cand[i]}' for i in indices]+[f'absdiff:{cand[i]}' for i in indices]
    names += [f'shared:{cand[i]}' for i in range(10,16)]+[f'global:{v}' for v in glob[:ng]]
    return x,names


def _balanced_stats(xs):
    nonempty=[x for x in xs if len(x)]
    if not nonempty: raise ValueError('source transform has no rows')
    mean=np.mean([x.mean(0) for x in nonempty],axis=0)
    var=np.mean([((x-mean)**2).mean(0) for x in nonempty],axis=0)
    scale=np.sqrt(np.maximum(var,0.)); scale[scale<1e-8]=1.
    return mean,scale


@dataclass
class FeatureTransform:
    head: str
    representation: str
    view: str
    base_names: list
    names: list
    mean: np.ndarray
    scale: np.ndarray
    final_mean: np.ndarray
    final_scale: np.ndarray
    rank_references: list
    interactions: list

    @classmethod
    def fit(cls,banks,head,representation,view):
        if view not in ('standard','rank','support_rank'): raise ValueError('unknown view')
        raws=[_raw_features(b,head,representation) for b in banks]
        names=raws[0][1]
        if any(n!=names for _,n in raws): raise ValueError('inconsistent feature schema')
        xs=[x for x,_ in raws]
        if any(not np.isfinite(x).all() for x in xs): raise ValueError('nonfinite source features')
        mean,scale=_balanced_stats(xs)
        refs=[[np.sort(x[:,j]) for j in range(x.shape[1])] for x in xs if len(x)] if view=='rank' else []
        interactions=[]
        if representation in ('context','prototype'):
            # Context is standardized BEFORE multiplying; expert shape is intentionally
            # directly available to Preference, so this is not strict swap equivariance.
            left=[i for i,n in enumerate(names) if n.startswith('difference:' if head=='preference' else 'sum:')][:10]
            left=[left[i] for i in (0,2,6,7,8,9)]
            right=[i for i,n in enumerate(names) if n.startswith('shared:')]
            globals_=[i for i,n in enumerate(names) if n.startswith('global:')][:4]
            if head=='preference': interactions=[(i,j) for i in left for j in right+globals_]
            else: interactions=[(i,j) for i in left for j in globals_]+[(i,i) for i in left+right+globals_]
        fullnames=names+[f'product:({names[i]})*({names[j]})' for i,j in interactions]
        if representation=='stumps':
            cuts=(-1.,0.,1.) if view=='standard' else (-.5,0.,.5)
            fullnames += [f'indicator:({name})>{cut:g}' for name in names for cut in cuts]
        obj=cls(head,representation,view,names,fullnames,mean,scale,np.zeros(len(fullnames)),np.ones(len(fullnames)),refs,interactions)
        # Each source supplies its entire unlabeled support ECDF before final
        # source-balanced scaling; support_rank never pools domain references.
        intermediate=[obj._expand(x,reference=x if view=='support_rank' else None) for x in xs if len(x)]
        obj.final_mean,obj.final_scale=_balanced_stats(intermediate)
        return obj

    def _expand(self,x,reference=None):
        if self.view in ('rank','support_rank'):
            refs_by_domain=self.rank_references
            if self.view=='support_rank':
                if reference is None or not len(reference): raise ValueError('support_rank requires nonempty support reference')
                refs_by_domain=[[np.sort(reference[:,j]) for j in range(reference.shape[1])]]
            z=np.zeros_like(x,dtype=float)
            for refs in refs_by_domain:
                for j,ref in enumerate(refs):
                    z[:,j]+=(np.searchsorted(ref,x[:,j],side='left')+np.searchsorted(ref,x[:,j],side='right'))/(2*len(ref)*len(refs_by_domain))
            z=2*z-1
        else: z=np.clip((x-self.mean)/self.scale,-8.,8.)
        if self.interactions:
            z=np.column_stack([z]+[z[:,i]*z[:,j] for i,j in self.interactions])
        if self.representation=='stumps':
            cuts=(-1.,0.,1.) if self.view=='standard' else (-.5,0.,.5)
            z=np.column_stack([z]+[(z[:,j]>cut).astype(float) for j in range(z.shape[1]) for cut in cuts])
        return z

    def apply(self,bank,reference_bank=None):
        x,names=_raw_features(bank,self.head,self.representation)
        if names!=self.base_names: raise ValueError('feature names changed')
        if not np.isfinite(x).all(): raise ValueError('nonfinite features')
        reference=None
        if self.view=='support_rank':
            if reference_bank is None: raise ValueError('support_rank requires explicit support reference_bank')
            reference,refnames=_raw_features(reference_bank,self.head,self.representation)
            if refnames!=self.base_names: raise ValueError('reference feature names changed')
            if not np.isfinite(reference).all(): raise ValueError('nonfinite reference features')
        return np.ascontiguousarray(np.clip((self._expand(x,reference)-self.final_mean)/self.final_scale,-8.,8.))

    def state_dict(self):
        return dict(self.__dict__)


def binary_objective(theta,xs,ys,l2,pairwise_weight):
    w,b=theta[:-1],theta[-1]
    value=.5*l2*float(w@w)+.5e-8*b*b
    grad=np.r_[l2*w,1e-8*b]
    active=[(x,y) for x,y in zip(xs,ys) if len(y)]
    paired=[(x,y) for x,y in active if np.any(y==1) and np.any(y==0)]
    for x,y in active:
        eta=x@w+b; delta=(expit(eta)-y)/(len(y)*len(active))
        value+=float(np.mean(np.logaddexp(0,eta)-y*eta))/len(active)
        grad[:-1]+=x.T@delta; grad[-1]+=delta.sum()
    if pairwise_weight and paired:
        for x,y in paired:
            xp,xn=x[y==1],x[y==0]
            # Pairwise score matrix, never a [positive,negative,feature] tensor.
            difference=(xp@w)[:,None]-(xn@w)[None,:]
            factor=pairwise_weight/(len(paired)*difference.size)
            value+=factor*float(np.logaddexp(0,-difference).sum())
            delta=-expit(-difference)*factor
            grad[:-1]+=xp.T@delta.sum(1)-xn.T@delta.sum(0)
    return value,grad


def _minimize(fun,start,bounds=None):
    fit=minimize(fun,start,jac=True,method='L-BFGS-B',bounds=bounds,
        options={'maxiter':500,'ftol':1e-12,'gtol':1e-7,'maxls':40})
    if not np.isfinite(fit.x).all() or not np.isfinite(fit.fun):
        raise RuntimeError(f'nonfinite optimization: {fit.message}')
    if not fit.success:
        # Retry the SAME objective, without changing model/hyperparameters.
        fit=minimize(fun,fit.x,jac=True,method='L-BFGS-B',bounds=bounds,
            options={'maxiter':1500,'ftol':1e-11,'gtol':1e-6,'maxls':80})
    if not fit.success: raise RuntimeError(f'optimization failed: {fit.message}')
    return fit


def fit_binary(xs,ys,l2,pairwise_weight):
    xs=[np.asarray(x,dtype=float) for x in xs]; ys=[np.asarray(y,dtype=float) for y in ys]
    width=xs[0].shape[1]
    if any(x.shape!=(len(y),width) or not np.isfinite(x).all() or not np.isin(y,[0.,1.]).all() for x,y in zip(xs,ys)):
        raise ValueError('invalid binary training rows')
    initial=np.zeros(width+1)
    if not any(len(y) for y in ys):
        return dict(coef=initial[:-1],bias=0.,objective=0.,mode='identity_empty')
    fit=_minimize(lambda t:binary_objective(t,xs,ys,l2,pairwise_weight),initial,
        bounds=[(None,None)]*width+[(-30.,30.)])
    return dict(coef=fit.x[:-1],bias=float(fit.x[-1]),objective=float(fit.fun),iterations=int(fit.nit),mode='optimized')


def adapter_basis(shared,local_models):
    deviations=np.stack([np.asarray(m['coef'])-shared['coef'] for m in local_models])
    _,s,vh=np.linalg.svd(deviations,full_matrices=False)
    rank=min(2,int(np.sum(s>max(1e-8,float(s[0])*1e-8))))
    basis=vh[:rank].T.copy()
    # Resolve arbitrary SVD sign for reproducible serialized coordinates.
    for j in range(rank):
        if basis[np.argmax(np.abs(basis[:,j])),j]<0: basis[:,j]*=-1
    return basis


def adapter_objective(theta,projection,offset,y,prior):
    eta=offset+projection@theta[:-1]+theta[-1]
    value=float(np.mean(np.logaddexp(0,eta)-y*eta))+.5*prior*float(theta@theta)
    delta=(expit(eta)-y)/len(y)
    grad=np.r_[projection.T@delta,delta.sum()]+prior*theta
    return value,grad


def fit_adapter_projected(projection,y,offset,rank,prior):
    rank=min(rank,projection.shape[1]); projection=projection[:,:rank]
    if not len(y): return dict(coordinates=[0.]*rank,bias=0.,rank=rank,requested_rank=rank,prior=prior,mode='identity_empty',objective=0.)
    fit=_minimize(lambda t:adapter_objective(t,projection,offset,y,prior),np.zeros(rank+1))
    return dict(coordinates=fit.x[:-1].tolist(),bias=float(fit.x[-1]),rank=rank,
        prior=prior,mode='optimized',objective=float(fit.fun))


def fit_adapter(x,y,shared,basis,rank,prior):
    result=fit_adapter_projected(x@basis,y,x@shared['coef']+shared['bias'],rank,prior)
    result['requested_rank']=rank
    return result


def predict_adapted(x,shared,basis,adaptation):
    rank=adaptation['rank']
    w=shared['coef']+basis[:,:rank]@np.asarray(adaptation['coordinates'])
    return x@w+shared['bias']+adaptation['bias']


def fit_affine(logits,target,strength=2.):
    x,y=np.asarray(logits,float),np.asarray(target,float)
    if x.shape!=y.shape or x.ndim!=1 or not np.isfinite(x).all() or not np.isin(y,[0,1]).all():
        raise ValueError('invalid calibration inputs')
    counts=[int((y==k).sum()) for k in (0,1)]
    if not len(y): return dict(slope=1.,bias=0.,mode='identity_empty',counts=counts,strength=strength)
    bias_only=min(counts)<5
    def objective(theta):
        a,b=(1.,theta[0]) if bias_only else theta
        eta=a*x+b; delta=expit(eta)-y
        loss=float((np.logaddexp(0,eta)-y*eta).sum())+.5*strength*((a-1)**2+b*b)
        grad=np.array([delta.sum()+strength*b]) if bias_only else np.array([delta@x+strength*(a-1),delta.sum()+strength*b])
        return loss,grad
    fit=_minimize(objective,np.array([0.]) if bias_only else np.array([1.,0.]),
        None if bias_only else [(0.05,20.),(None,None)])
    a,b=(1.,float(fit.x[0])) if bias_only else map(float,fit.x)
    return dict(slope=a,bias=b,mode='bias_only' if bias_only else 'positive_affine',counts=counts,strength=strength,objective=float(fit.fun))


def apply_affine(logits,fit):
    return fit['slope']*np.asarray(logits)+fit['bias']


def binary_metrics(logits,y):
    logits,y=np.asarray(logits,float),np.asarray(y,float)
    p=expit(logits)
    return dict(count=len(y),positive_count=int(y.sum()),negative_count=int(len(y)-y.sum()),
        bce=float(np.mean(np.logaddexp(0,logits)-y*logits)) if len(y) else None,
        brier=float(np.mean((p-y)**2)) if len(y) else None,
        auc=float(roc_auc_score(y,logits)) if len(np.unique(y))==2 else None)


def _mean_defined(values):
    values=[x for x in values if x is not None]
    if any(not np.isfinite(x) for x in values): raise ValueError('nonfinite metric cannot enter source selection')
    return float(np.mean(values)) if values else None


def reciprocal_metrics(logits,state,node_ids,dataset,head):
    logits,state,ids=np.asarray(logits),array(state),array(node_ids)
    first,second=validation_halves(ids,dataset)
    rows=[]
    for cal,evaluation in [(first,second),(second,first)]:
        cm,cy=binary_target(state[cal],head); em,ey=binary_target(state[evaluation],head)
        fit=fit_affine(logits[cal][cm],cy)
        metrics=binary_metrics(apply_affine(logits[evaluation][em],fit),ey)
        rows.append(dict(**metrics,calibration=fit,calibration_ids=ids[cal].tolist(),evaluation_ids=ids[evaluation].tolist(),
            calibration_head_count=int(cm.sum()),evaluation_head_count=int(em.sum())))
    return dict(count=sum(r['count'] for r in rows),positive_count=sum(r['positive_count'] for r in rows),
        negative_count=sum(r['negative_count'] for r in rows),
        **{key:(_mean_defined([r[key] for r in rows]) if key=='auc' or all(r[key] is not None for r in rows) else None) for key in ('auc','bce','brier')},
        valid_auc_folds=sum(r['auc'] is not None for r in rows),folds=rows)


def complexity(config):
    return ({'core':0,'context':1,'prototype':2,'stumps':1}[config['representation']],
        config['adapter_rank'],0 if config['view']=='standard' else 1,
        config['pairwise_weight'],-config['l2'],-config['adapter_prior'])


def select_configuration(trials,target):
    aggregates={}
    for row in trials:
        cfg=row['config']; entry=aggregates.setdefault(cfg['id'],dict(config=cfg,held={}))
        for domain,metrics in row['held'].items():
            if domain==target: continue
            if domain in entry['held'] and entry['held'][domain]!=metrics:
                raise ValueError('duplicate inconsistent source metric')
            entry['held'][domain]=metrics
    summaries=[]
    for entry in aggregates.values():
        held=entry['held']; values=list(held.values())
        bces=[m['bce'] for m in values]
        mean_bce=_mean_defined(bces)  # Reject nonfinite even if another domain is empty.
        summaries.append(dict(**entry,macro_bce=mean_bce if all(v is not None for v in bces) else None,
            macro_auc=_mean_defined([m['auc'] for m in values]),macro_brier=_mean_defined([m.get('brier') for m in values]),
            valid_auc_domains=sum(m['auc'] is not None for m in values),complexity=list(complexity(entry['config']))))
    valid=[r for r in summaries if r['macro_bce'] is not None]
    if not valid: raise ValueError('no usable source BCE')
    best_bce=min(r['macro_bce'] for r in valid)
    eligible=[r for r in valid if r['macro_bce']<=best_bce+.05+1e-12]
    best=min(eligible,key=lambda r:(-(r['macro_auc'] if r['macro_auc'] is not None else -1.),r['macro_bce'],tuple(r['complexity']),r['config']['id']))
    return dict(target=target,best=best,best_macro_bce=best_bce,bce_tolerance=.05,trials=sorted(summaries,key=lambda r:r['config']['id']),
        selection='source macro AUC subject to best calibrated BCE + 0.05; BCE, complexity, stable ID ties')


def routing_boundary(preference_logits,state):
    z=array(state); decisive=z!=2; p=expit(np.asarray(preference_logits)[decisive]); y=z[decisive]==0
    if not len(y): return dict(threshold=.5,mode='default_no_decisive',count=0,accuracy=None)
    # >= threshold chooses G. Sweep tied blocks once rather than O(N^2).
    order=np.argsort(p,kind='stable'); ps=p[order]; ys=y[order]
    unique,first=np.unique(ps,return_index=True)
    cumulative=np.r_[0,np.cumsum(np.where(ys,-1,1))]
    candidates=np.r_[0.,unique,np.nextafter(unique,np.inf),np.nextafter(1.,2.),.5]
    position=np.searchsorted(ps,candidates,side='left')
    correct=int(y.sum())+cumulative[position]
    maximum=int(correct.max()); choices=np.flatnonzero(correct==maximum)
    selected=min(choices,key=lambda i:(abs(float(candidates[i])-.5),float(candidates[i])))
    return dict(threshold=float(candidates[selected]),mode='calibration_decisive_accuracy',count=len(y),accuracy=maximum/len(y),tie_rule='nearest 0.5 then smaller threshold')


def score_payload(bank,trust_logits,preference_logits,boundary,indices=None):
    c,r=expit(trust_logits),expit(preference_logits)
    q=np.column_stack((c*r,c*(1-r),1-c)); route=(r<boundary).astype(np.int64)
    idx=slice(None) if indices is None else indices
    result={key:torch.as_tensor(array(bank[key])[idx].copy()) for key in ('node_ids','gnn_pred','llm_pred')}
    result.update(q=torch.from_numpy(q),route=torch.from_numpy(route),selected_score=torch.from_numpy(q[np.arange(len(q)),route]),
        trust_logit=torch.as_tensor(np.asarray(trust_logits)),preference_logit=torch.as_tensor(np.asarray(preference_logits)))
    return result


def top_mask(score,node_ids,fraction):
    score,ids=np.asarray(score),np.asarray(node_ids)
    f=Fraction(str(fraction)); k=min(len(score),max(0,(len(score)*f.numerator+f.denominator-1)//f.denominator))
    order=np.lexsort((ids,-score)); mask=np.zeros(len(score),bool); mask[order[:k]]=True
    return mask


def fit_top_fraction(score,correct,node_ids):
    score,correct,ids=np.asarray(score),np.asarray(correct,bool),np.asarray(node_ids)
    rows=[]
    for f in FRACTIONS:
        mask=top_mask(score,ids,f); n=int(mask.sum())
        rows.append(dict(fraction=f,count=n,accuracy=float(correct[mask].mean()) if n>=10 else None))
    usable=[r for r in rows if r['accuracy'] is not None]
    if not usable: return dict(fraction=None,validation_count=len(score),validation_accepted=0,validation_accuracy=None,reject_all=True,trials=rows)
    best=max(r['accuracy'] for r in usable)
    choice=max((r for r in usable if r['accuracy']>=best-.01-1e-12),key=lambda r:r['fraction'])
    return dict(fraction=choice['fraction'],validation_count=len(score),validation_accepted=choice['count'],
        validation_accuracy=choice['accuracy'],reject_all=False,trials=rows,tolerance=.01,min_accepted=10,tie_rule='descending score then ascending node ID')
