"""Calibration/routing labels are excluded from each held validation fold."""
from common import *
from scipy.special import expit
from eargtc.router_v8_exploration.core import fit_affine
from eargtc.agreement_increment.policy import top_mask,summarize
from eargtc.router_v10_tune.policy import FRACTIONS

def folds(ids,dataset):
 ids=np.asarray(ids,dtype=np.int64)
 if len(ids)!=len(np.unique(ids)):raise ValueError('Duplicate validation ID')
 order=sorted(range(len(ids)),key=lambda i:hashlib.sha256(f'{dataset}|v10-tune-val3|{int(ids[i])}'.encode()).digest())
 result=np.empty(len(ids),dtype=np.int64);result[order]=np.arange(len(ids))%3
 return result

def affine(z,y):
 r=fit_affine(np.asarray(z,float),np.asarray(y,float),strength=4.)
 return {k:float(r[k]) for k in ('slope','bias')}

def summary(score,correct,ids,fraction):return summarize(correct,top_mask(score,ids,fraction))

def outfold_metric(score,correct,ids,fold,fraction):
 score=np.asarray(score);correct=np.asarray(correct,bool);ids=np.asarray(ids);fold=np.asarray(fold)
 keep=np.zeros(len(ids),bool);parts=[]
 for k in sorted(set(fold.tolist())):
  m=fold==k;keep[m]=top_mask(score[m],ids[m],fraction);parts.append(summarize(correct[m],keep[m]))
 r=summarize(correct,keep);defined=[p['precision'] for p in parts if p['precision'] is not None]
 r.update(fraction=float(fraction),worst_fold_precision=min(defined) if defined else None,folds=parts)
 return r

def key(r):return (r['wilson_lower'] if r.get('wilson_lower') is not None else -1.,r['precision'] if r.get('precision') is not None else -1.,r.get('worst_fold_precision') or 0.,r.get('coverage',0.))
def improves(a,b):return a.get('precision') is not None and b.get('precision') is not None and a['precision']>=b['precision'] and a['wilson_lower']>=b['wilson_lower']+.002

def select_acceptance(score,correct,ids,fold):
 rows=[outfold_metric(score,correct,ids,fold,f) for f in FRACTIONS]
 valid=[r for r in rows if r['accepted']>=20]
 if not valid:return dict(fraction=0.,reject_all=True,reason='insufficient_validation_support',accepted=0,precision=None,rows=rows)
 return dict(max(valid,key=key),reject_all=False,rows=rows,selection='OOF Wilson lower, precision, worst fold precision, coverage; min20 accepted')

def crossfit_head(z,y,ids,fold,eligible=None):
 z=np.asarray(z);y=np.asarray(y,bool);fold=np.asarray(fold);eligible=np.ones(len(y),bool) if eligible is None else np.asarray(eligible,bool)
 output=np.empty(len(z));audit=[]
 for k in sorted(set(fold.tolist())):
  held=fold==k;fit=(~held)&eligible;a=affine(z[fit],y[fit]);output[held]=expit(a['slope']*z[held]+a['bias'])
  audit.append(dict(fold=int(k),fit_ids=np.asarray(ids)[fit].tolist(),held_ids=np.asarray(ids)[held].tolist(),affine=a))
 return dict(score=output,audit=audit)

def pair_calibration(z,state):
 state=np.asarray(state);dec=state!=2
 return dict(trust=affine(z['trust'],state!=2),preference=affine(z['preference'][dec],state[dec]==0))

def apply_pair(z,cal,recipe):
 a=cal['trust'];b=cal['preference'];t=expit(recipe['trust_scale']*(a['slope']*z['trust']+a['bias'])+recipe['trust_bias']);p=expit(b['slope']*z['preference']+b['bias'])
 route=(p<recipe['boundary']).astype(np.int64)
 return dict(score=t*np.where(route==0,p,1-p),route=route,q=np.column_stack((t*p,t*(1-p),1-t)))

def fit_route(z,state,ids,fraction):
 cal=pair_calibration(z,state);best=None;rows=[]
 for scale in (.5,1.,2.):
  for bias in (-1.,0.,1.):
   for boundary in (.25,.375,.5,.625,.75):
    recipe=dict(trust_scale=scale,trust_bias=bias,boundary=boundary);s=apply_pair(z,cal,recipe);r=summary(s['score'],s['route']==state,ids,fraction)
    complexity=abs(np.log(scale))+abs(bias)+abs(boundary-.5)
    k=(r['wilson_lower'] or -1.,r['precision'] if r['precision'] is not None else -1.,-complexity)
    rows.append(dict(recipe=recipe,metrics=r))
    if best is None or k>best[0]:best=(k,recipe)
 return cal,best[1],rows

def crossfit_pair(z,state,ids,fold,fraction):
 state=np.asarray(state);ids=np.asarray(ids);fold=np.asarray(fold);score=np.empty(len(ids));route=np.empty(len(ids),np.int64);audit=[]
 for k in sorted(set(fold.tolist())):
  held=fold==k;fit=~held;cal,recipe,_=fit_route({h:v[fit] for h,v in z.items()},state[fit],ids[fit],fraction)
  s=apply_pair({h:v[held] for h,v in z.items()},cal,recipe);score[held]=s['score'];route[held]=s['route']
  audit.append(dict(fold=int(k),fit_ids=ids[fit].tolist(),held_ids=ids[held].tolist(),calibration=cal,recipe=recipe))
 return dict(score=score,route=route,audit=audit)
