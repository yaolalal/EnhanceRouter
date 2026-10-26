from common import *
setup()
import torch,joblib
idx=read(RUN/'HISTORICAL_INPUTS_V2.json');rows=[]
for shot,seed,ds in CELLS:
 if seed>44 and shot!=5:continue
 sp=split(shot,seed,ds)
 for branch in ('agreement','disagreement'):
  b=bank(shot,seed,ds,branch,'train');bundle=joblib.load(idx['cells'][key(shot,seed,ds)]['branches'][branch]['bundle'])
  for fraction in (.25,.5,.75,1.):
   parents=torch.tensor(sp['support_ids']);parents=parents[torch.randperm(len(parents),generator=torch.Generator().manual_seed(seed+913))[:max(1,int(len(parents)*fraction+.5))]]
   mask=torch.isin(b['node_ids'],parents)
   for h,w in bundle['models'].items():
    cfg=w['model']['config'];scope=cfg.get('scope','transfer' if cfg.get('kind')=='neural' else 'target')
    eligible=mask & (b['state']!=2) if h=='preference' else mask
    if scope=='target' and not int(eligible.sum()) and cfg.get('config',{}).get('family')!='baseline':
     row=dict(shot=shot,seed=seed,dataset=ds,branch=branch,head=h,fraction=fraction,eligible_rows=0,engine=w['engine'],family=cfg.get('family',cfg.get('kind')));rows.append(row);print('EMPTY_TARGET',row,flush=True)
write(RUN/'EMPTY_TARGET_AUDIT.json',dict(cases=rows,query_labels_read=False,slurm_job=os.environ['SLURM_JOB_ID']))
