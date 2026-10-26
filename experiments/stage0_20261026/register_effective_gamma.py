from common import *
setup()
import joblib,torch,numpy as np
from frozen_fit import refit,input_hash
from enhancerouter.router import predict
cases=[]
for shot,seed,ds in CELLS:
 if seed>44 and shot!=5:continue
 models=joblib.load(folder(shot,seed,ds)/'b0/models.joblib');heads=[]
 for branch,m in models.items():
  for h,w in m.get('models',{}).items():
   cfg=w['model']['config'];sk=w['model'].get('estimator',{}).get('estimator')
   if cfg.get('scope')!='source' and cfg.get('gamma')=='scale' and hasattr(sk,'_gamma'):heads.append(dict(branch=branch,head=h,effective_gamma=float(sk._gamma)))
 if heads:cases.append(dict(key=key(shot,seed,ds),shot=shot,seed=seed,dataset=ds,heads=heads))
write(RUN/'EFFECTIVE_GAMMA_REGISTRATION.json',dict(affected_historical_cells=cases,rule='Freeze fitted B0 SVC gamma, not just gamma=scale rule; existing unaffected historical D1 artifacts remain valid. Affected historical and all new20 D1 use d1_frozen_v2.',query_labels_read=False))
shot,seed,ds,branch,h=3,42,'cora','disagreement','trust';idx=read(RUN/'HISTORICAL_INPUTS_V2.json');w=joblib.load(folder(shot,seed,ds)/'b0/models.joblib')[branch]['models'][h];banks={d:bank(shot,seed,d,branch,'train') for d in DS};vb=bank(shot,seed,ds,branch,'validation')
full,a=refit(w,h,banks,ds);np.testing.assert_allclose(predict(full,vb),predict(w,vb),rtol=0,atol=1e-10)
parents=torch.tensor(split(shot,seed,ds)['support_ids']);parents=parents[torch.randperm(len(parents),generator=torch.Generator().manual_seed(seed+913))[:max(1,int(len(parents)*.25+.5))]]
bs=dict(banks);bs[ds]=subset(bs[ds],torch.isin(bs[ds]['node_ids'],parents));small,b=refit(w,h,bs,ds)
assert small['model']['estimator']['estimator']._gamma==w['model']['estimator']['estimator']._gamma
assert input_hash(small)==input_hash(w)
write(ROOT/'reports/audit-20261026/EFFECTIVE_GAMMA_TEST.json',dict(all_checks_pass=True,full_predictions_exact=True,input_transform_frozen=True,effective_gamma_frozen=True,diagnostics=b,slurm_job=os.environ['SLURM_JOB_ID']))
print('EFFECTIVE_GAMMA_PASS affected_cells',len(cases),flush=True)
