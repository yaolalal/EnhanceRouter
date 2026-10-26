"""Original Joint/PURF V1 four-stage adaptive search using own validation only."""
from common import *
import hashlib,itertools,json,torch
from eargtc.joint_explore.joint_purf_v10_v1.training import DEFAULTS,train

def cid(cfg):return hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]

def dedup(configs):return list({cid(c):c for c in configs}.values())

def core():
    return [dict(DEFAULTS,dim=d,cross=c,lr=lr) for d,c,lr in itertools.product((16,32,64),(0.,.25,1.),(.0003,.001,.003))]

def schemes(anchor):
    out=[dict(anchor,gate_lr=g,warmup=w,gate_gold=s) for g,w,s in itertools.product((1.,.1),(0,50),(False,True))]
    out += [dict(anchor,aux=v) for v in (.1,.3)]
    out += [dict(anchor,keep=v) for v in (.1,.5)]
    return dedup(out)

def refinement(winner):
    out=[winner]
    for k,values in [('dim',(16,32,64)),('cross',(0.,.25,1.)),('lr',(.0003,.001,.003))]:
        out += [dict(winner,**{k:v}) for v in values]
    out += [dict(winner,aux=.1),dict(winner,keep=.1),dict(winner,aux=.1,keep=.1)]
    return dedup(out)

def removals(winner):
    out=[winner]
    for k,v in [('gate_lr',1.),('warmup',0),('gate_gold',False),('aux',0.),('keep',0.),('cross',0.)]:
        out.append(dict(winner,**{k:v}))
    # Check both auxiliary regularizers as a pair around the selected structural/optimizer setting.
    out += [dict(winner,aux=a,keep=k) for a,k in itertools.product((0.,.1),(0.,.1))]
    return dedup(out)

def run(f,s,p,h,seed,out):
 root=Path(out)/'joint-search';previous=[];seen={};validation_ids=s['selection_ids'];gold_ids=s['gold_ids']
 assert not set(validation_ids.tolist())&set(gold_ids.tolist())
 def trial(cfg):
  identity=cid(cfg);dest=root/'candidates'/identity
  if identity in seen:return
  if (dest/'result.json').exists():
   row=read(dest/'result.json');assert row['config']==cfg and row['training_seed']==seed
   assert digest(dest/'checkpoint.pt')==row['checkpoint_sha256']
  else:
   assert not (dest/'checkpoint.pt').exists(),'Partial candidate requires explicit recovery audit'
   member=train(f,s,p,h,cfg,seed);save(dest/'checkpoint.pt',member)
   row=dict(id=identity,config=cfg,training_seed=seed,validation=member['checkpoints']['accuracy'],checkpoint_sha256=digest(dest/'checkpoint.pt'),query_truth_used=False,slurm_job=os.environ['SLURM_JOB_ID'])
   write(dest/'result.json',row)
  seen[identity]=row
 def ranked(configs):
  return sorted([seen[cid(c)] for c in dedup(configs)],key=lambda r:(-r['validation']['accuracy'],r['validation']['nll'],r['id']))
 for stage in range(1,5):
  if stage==1:configs=core()
  elif stage==2:configs=schemes(ranked(previous[0])[0]['config'])
  elif stage==3:configs=refinement(ranked(previous[1])[0]['config'])
  else:configs=removals(ranked([c for cs in previous for c in cs])[0]['config'])
  plan=dict(stage=stage,configs=configs,selection='own validation accuracy then NLL then config ID',query_truth_used=False)
  planpath=root/f'stage{stage}.json'
  if planpath.exists():assert read(planpath)==plan
  else:write(planpath,plan)
  for cfg in configs:trial(cfg)
  previous.append(configs)
  print('JOINT_SEARCH',seed,stage,len(seen),flush=True)
 winner=ranked([c for cs in previous for c in cs])[0]
 member=load(root/'candidates'/winner['id']/'checkpoint.pt')
 assert member['config']==winner['config'] and member['training_seed']==seed
 receipt=dict(config=winner['config'],candidates=list(seen.values()),validation=winner['validation'],search='original V1 four stages',query_truth_used=False)
 write(root/'SELECTION.json',receipt)
 return winner['config'],member,receipt
