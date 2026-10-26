"""Controlled refits with B0 fitted input preprocessing used during training."""
from copy import deepcopy
from contextlib import ExitStack
from unittest.mock import patch
import joblib

def input_state(w):
 m=w['model'];f=m.get('fitted',m)
 fields=('transform','mean','scale','indices') if w['engine']=='v6' else ('transform','mode')
 return {k:deepcopy(f[k]) for k in fields if k in f}

def input_hash(w):
 import numpy as np,torch
 def canonical(v):
  if torch.is_tensor(v):return canonical(v.detach().cpu().numpy())
  if isinstance(v,np.ndarray):return np.ascontiguousarray(v)
  if isinstance(v,dict):return {k:canonical(x) for k,x in sorted(v.items())}
  if isinstance(v,(list,tuple)):return [canonical(x) for x in v]
  if hasattr(v,'__dict__'):return dict(type=type(v).__module__+'.'+type(v).__name__,fields=canonical(vars(v)))
  return v
 return joblib.hash(canonical(input_state(w)),hash_name='sha1')

def refit(w,head,banks,target):
 import numpy as np
 from enhancerouter.router import engine
 old=deepcopy(w);m=old['model'];cfg=m['config']
 scope=cfg.get('scope','transfer' if cfg.get('kind')=='neural' else 'target')
 if scope=='source' or (w['engine']=='v6' and m['fitted'].get('kind')=='baseline'):
  return old,dict(refit=False,reason='source-only or fixed rule',input_hash=input_hash(old))
 fixed=input_state(old)
 fitcfg=deepcopy(cfg);fixed_gamma=None
 sk=m.get('estimator',{}).get('estimator')
 if cfg.get('gamma')=='scale' and hasattr(sk,'_gamma'):
  fixed_gamma=float(sk._gamma);fitcfg['gamma']=fixed_gamma
 # Native one-class smoothing is (positives+.5)/(N+1); N=0 gives .5.
 # Do not reuse trained B0 weights when a target-only head loses every row.
 if scope=='target':
  target_bank=banks[target]
  eligible=(np.asarray(target_bank['state'])!=2) if head=='preference' else np.ones(len(target_bank['ids']),bool)
  if not int(eligible.sum()):
   new=deepcopy(old)
   if w['engine'] in ('v7','expanded','tune'):
    new['model']['estimator']=dict(kind='constant',logit=0.)
   elif w['engine']=='v6' and head=='agreement':
    f=new['model']['fitted'];new['model']['fitted']=dict(kind='constant',family=f['family'],config=deepcopy(f['config']),probability=.5,**({'transform':deepcopy(f['transform'])} if 'transform' in f else {}))
   else:raise NotImplementedError('Empty-target fallback needs reviewed estimator schema')
   assert input_hash(new)==input_hash(old)
   return new,dict(refit=True,scope=scope,parents=0,rows=0,no_eligible_training_rows=True,native_empty_prior=.5,raw_logit=0.,input_before=input_hash(old),input_after=input_hash(new),fitted_transform_recomputed=False,output_normalization_frozen=True)
 with ExitStack() as stack:
  import torch
  from threadpoolctl import threadpool_limits
  threads=torch.get_num_threads();torch.set_num_threads(1)
  stack.callback(torch.set_num_threads,threads)
  stack.enter_context(threadpool_limits(limits=1))
  if w['engine'] in ('v7','expanded','tune'):
   transform=m['transform'];cls=type(transform)
   stack.enter_context(patch.object(cls,'fit',new=classmethod(lambda cls,*args,**kw:deepcopy(transform))))
  elif w['engine']=='v6' and cfg.get('kind')=='neural':
   from eargtc.router_v8_gold_training import training
   transform=m['fitted']['transform']
   stack.enter_context(patch.object(training,'_fit_transform',new=lambda *args,**kw:deepcopy(transform)))
   # Source cache keys do not identify a frozen transform. Isolate this fit's cache.
   stack.enter_context(patch.object(engine('v6'),'_SOURCE_CACHE',new=__import__('collections').OrderedDict()))
  elif w['engine']=='v6' and head=='agreement':
   from eargtc.agreement_increment import models
   if 'transform' not in m['fitted']:
    # Native agreement single-class fitting returns before preprocessing.
    assert m['fitted']['kind']=='constant'
   else:
    transform=m['fitted']['transform']
    def frozen_linear(arrays,mode,weight,seed):
     if mode!=transform['feature_mode']:raise ValueError('Frozen feature mode mismatch')
     raw=models._linear_raw(arrays,transform,slice(None)).astype(np.float64)
     x=np.clip((raw-transform['mean'])/transform['scale'],-12.,12.)
     return deepcopy(transform),x
    stack.enter_context(patch.object(models,'_fit_linear_transform',new=frozen_linear))
  else:raise NotImplementedError(('unreviewed frozen-fit schema',w['engine'],head,cfg))
  new=deepcopy(old);new['model']=engine(w['engine']).fit_candidate(deepcopy(fitcfg),head,banks,target)
 f=new['model'].get('fitted',new['model'])
 # A native single-class estimator may omit its unused transform; retain it for audit.
 if 'transform' in fixed and 'transform' not in f:f['transform']=deepcopy(fixed['transform'])
 for k in ('center','scale'):
  if k in m:new['model'][k]=deepcopy(m[k])
 assert input_hash(new)==input_hash(old),'Input transform changed during controlled refit'
 if fixed_gamma is not None:
  fitted_estimator=new['model'].get('estimator',{}).get('estimator')
  if fitted_estimator is not None:assert float(fitted_estimator._gamma)==fixed_gamma
  new['model']['controlled_effective_gamma']=fixed_gamma
 audit=new['model'].get('fit_audit',new['model'].get('training_audit',{}))
 audit['controlled_input_transform']=dict(source='frozen B0',sha1=input_hash(old),fit_calls_replaced=True)
 for k in ('transform_fit_sample_ids','transform_fit_parent_ids','transform_fit_rows','feature_fit_ids','transform_fit_ids'):
  if k in audit:audit[k]=deepcopy(m.get('fit_audit',m.get('training_audit',{})).get(k))
 return new,dict(refit=True,scope=scope,input_before=input_hash(old),input_after=input_hash(new),
  fitted_transform_recomputed=False,output_normalization_frozen=True,frozen_effective_gamma=fixed_gamma)
