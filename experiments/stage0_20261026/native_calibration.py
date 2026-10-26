"""Fixed-head variants of the historical validation policy; no query input."""
from common import *
import numpy as np
from enhancerouter.router import predict,predict_bundle as scores
from scipy.special import expit,logit
from eargtc.agreement_increment.policy import top_mask,summarize,select_policy
from eargtc.router_v10_tune.policy import FRACTIONS
from eargtc.router_v8_v6.core import precision_profile
from eargtc.router_v8_exploration.core import routing_boundary

def acceptance(score,y,ids,dense=True,tuned=False):
 if tuned and len(ids)>=20:
  from eargtc.router_v10_tune.policy import choose_policy
  return choose_policy(score,y,ids)
 if not dense:return select_policy(score,y,ids,min_accepted=20)
 rows=[dict(fraction=f,**summarize(y,top_mask(score,ids,f))) for f in FRACTIONS]
 valid=[r for r in rows if r['accepted']>=20]
 if not valid:return dict(fraction=0.,reject_all=True,accepted=0,precision=None,reason='insufficient_validation_support',rows=rows)
 return dict(max(valid,key=lambda r:(r['precision'],r['wilson_lower'],r['fraction'])),reject_all=False,rows=rows)
def apply(z,aff,boundary):
 t=expit(aff['trust']['slope']*z['trust']+aff['trust']['bias']);p=expit(aff['preference']['slope']*z['preference']+aff['preference']['bias'])
 route=(p<boundary).astype(int)
 return dict(score=t*np.where(route==0,p,1-p),route=route,q=np.column_stack((t*p,t*(1-p),1-t)))
def compose(a,s,b):return dict(slope=float(a['slope']*s),bias=float(a['bias']*s+b))
def policy_key(p):return (p['precision'] if p['precision'] is not None else -1.,p.get('wilson_lower') or 0.,p.get('coverage',0.))

def calibrate(d,branch,bundle,vb,norms,anchor,reference_fraction):
 ids=np.asarray(vb['ids']);trials=[]
 if not len(ids):
  bundle['policy']=acceptance([],[],[])
  return bundle,dict(method='empty_validation_reject_all',trials=[]),dict(score=np.empty(0))
 if branch=='agreement':
  s=scores(bundle,vb);bundle['policy']=acceptance(s['score'],np.asarray(vb['target'],bool),ids)
  return bundle,dict(method='fixed_agreement_head_dense_K',trials=trials),s
 z={h:predict(w,vb) for h,w in bundle['models'].items()};state=np.asarray(vb['state']);half=np.asarray(vb['cal_mask'])
 # Transport only the historical post-transform constants, never seed42's fitted normalization.
 initial={h:compose(norms[h],anchor['post'][h]['slope'],anchor['post'][h]['bias']) for h in norms}
 boundary=anchor['boundary'];baseline=apply(z,initial,boundary)
 if len(ids)<20:
  bundle.update(affine=initial,boundary=boundary,policy=acceptance(baseline['score'],baseline['route']==state,ids))
  return bundle,dict(method='insufficient_validation_support',trials=[]),baseline
 if d in ('citeseer','arxiv'):
  # Same registered one-factor order/ranges and strict incumbent tie behavior as v10_tune/run.py.
  current=copy.deepcopy(initial);best=acceptance(baseline['score'],baseline['route']==state,ids,tuned=True)
  for h,kind in [(h,k) for h in ('trust','preference') for k in ('slope','bias')]+[('route','boundary')]:
   base=copy.deepcopy(current);base_boundary=boundary;chosen='incumbent'
   values=(.5,.75,1.,1.5,2.) if kind=='slope' else (-2.,-1.,-.5,.5,1.,2.) if kind=='bias' else (.1,.2,.3,.4,.5,.6,.7,.8,.9)
   for v in values:
    a=copy.deepcopy(base);bound=base_boundary
    if kind=='slope':a[h]={k:x*v for k,x in base[h].items()}
    elif kind=='bias':a[h]['bias']+=v
    else:bound=v
    s=apply(z,a,bound);p=acceptance(s['score'],s['route']==state,ids,tuned=True)
    trials.append(dict(head=h,kind=kind,value=v,precision=p['precision'],fraction=p['fraction'],affine=a,boundary=bound))
    if policy_key(p)>policy_key(best):current=a;boundary=bound;best=p;chosen=v
   trials.append(dict(operation_complete=[h,kind],selected=chosen))
  bundle.update(affine=current,boundary=boundary,policy=best)
  return bundle,dict(method='v10_tune_fixed_heads_coordinate_policy',trials=trials),apply(z,current,boundary)
 # Existing expanded-v7 fixed-head routing family, robust profile objective and fallback protection.
 def profile(s):return precision_profile(s['score'],s['route']==state,ids,half,reference_fraction)
 bp=profile(baseline);bestp=bp;bestid='fallback';bestaff=initial;bestbound=boundary;bestscore=baseline
 pz=norms['preference']['slope']*z['preference']+norms['preference']['bias']
 t=routing_boundary(pz,state)['threshold'];center=float(logit(t)) if 0<t<1 else float(pz.min()-1) if t<=0 else float(pz.max()+1)
 centers=[center]+(np.quantile(pz[state!=2],[0,.25,.5,.75,1]).tolist() if (state!=2).any() else [0.])
 for a in (.5,1.,2.):
  for b in (-2.,0.,2.):
   for c in (.5,1.,2.):
    for center in sorted(set(centers)):
     aff={'trust':compose(norms['trust'],a,b),'preference':compose(norms['preference'],c,-c*center)}
     s=apply(z,aff,.5);p=profile(s);cid=f'fixed|fixed|{a}|{b}|{c}|{center}'
     trials.append(dict(id=cid,affine=aff,boundary=.5,profile=p))
     if (-p['robust'],-p['full']['mean'],cid)<(-bestp['robust'],-bestp['full']['mean'],bestid):bestp=p;bestid=cid;bestaff=aff;bestbound=.5;bestscore=s
 original_policy=acceptance(baseline['score'],baseline['route']==state,ids,False)
 candidate_policy=acceptance(bestscore['score'],bestscore['route']==state,ids,False)
 if bestp['robust']<bp['robust']+.002 or (candidate_policy.get('precision') or 0)<(original_policy.get('precision') or 0):
  bestaff=initial;bestbound=boundary;bestscore=baseline;bestid='fallback'
 bundle.update(affine=bestaff,boundary=bestbound,policy=acceptance(bestscore['score'],bestscore['route']==state,ids))
 return bundle,dict(method='expanded_v7_fixed_heads_then_dense_K',selected=bestid,reference_fraction=reference_fraction,trials=trials),bestscore
