"""Finite catalogs fixed before any seed tuning; no access to metrics or data."""
from common import copy,hashlib,json
from tune_engine import _defaults,_model_config

def identify(c):
 c=copy.deepcopy(c);c.pop('id',None);c.pop('target_domain',None);c.pop('stage',None)
 c['id']='t_'+hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:12];return c

def initial(head):
 rows=[]
 for mode in ('prob','context','embed'):
  for scope in ('target','source','pooled'):
   for C in (.03,.3):
    c=_defaults('lr',mode,scope);c['C']=C;rows.append(c)
 for mode in ('prob','full','embed'):
  for scope in ('target','pooled'):
   c=_defaults('trees',mode,scope);c.update(depth=6,leaf=8,n_estimators=160);rows.append(c)
 for mode in ('prob','context','embed'):
  c=_defaults('svc',mode,'pooled');c.update(C=1.,gamma='scale');rows.append(c)
 for mode,scope in (('context','target'),('embed','pooled'),('compact','transfer')):
  c=_defaults('joint_shared' if head!='agreement' and scope=='transfer' else 'neural',mode,scope);c.update(steps=160,adapt_steps=40,seeds=[42,137,2026]);rows.append(c)
 assert len(rows)==30
 return [identify(c) for c in rows]

def refine(w,head):
 c=_model_config(w,head);c.pop('target_domain',None);family=c['family'];rows=[]
 fields=[('C',(.003,.01,.1,1.,3.))] if family in ('lr','svc') else [('depth',(2,4,8,12)),('leaf',(3,5,12,20))] if family=='trees' else [('weight_decay',(.003,.03,.1)),('dropout',(.05,.25,.4)),('steps',(100,280,400)),('adapt_steps',(0,80))]
 fields+=[('scope',('target','source','pooled') if family not in ('neural','joint_shared') else ('source','pooled','transfer')),('target_mass',(.1,.5,.75)),('source_profile',('uniform','small_graphs','large_graphs')),('mode',('prob','context','embed'))]
 for name,values in fields:
  for value in values:
   if c.get(name)==value:continue
   a=copy.deepcopy(c);a[name]=value;rows.append(identify(a))
 seen=set();out=[]
 for x in rows:
  if x['id'] in seen:continue
  seen.add(x['id']);out.append(x)
  if len(out)==20:break
 return out
