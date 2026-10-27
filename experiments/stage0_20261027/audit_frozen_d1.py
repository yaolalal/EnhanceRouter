"""Authenticate the already completed frozen-setting D1; disclose process differences."""
from d3_recovery import *
setup()
import torch,joblib,copy
from frozen_fit import input_hash
rows=[]
for sh,se,ds in CELLS:
 original=joblib.load(folder(sh,se,ds)/'b0/models.joblib');target_banks={b:bank(sh,se,ds,b,'train') for b in ('agreement','disagreement')}
 for fraction in (.25,.5,.75,1.):
  out=variant_folder(sh,se,ds,f'd1/{fraction}');r=read(out/'COMPLETE.json')
  for rel,h in r['files'].items():assert digest(out/rel)==h,(out,rel)
  fitted=joblib.load(out/'models.joblib');parents=torch.tensor(split(sh,se,ds)['support_ids']);g=torch.Generator().manual_seed(se+913)
  selected=set(parents[torch.randperm(len(parents),generator=g)[:max(1,int(len(parents)*fraction+.5))]].tolist())
  for b,m in original.items():
   for k in set(m)-{'models'}:assert joblib.hash(m[k])==joblib.hash(fitted[b][k]),(sh,se,ds,fraction,b,k)
   for h,w in m.get('models',{}).items():
    new=fitted[b]['models'][h];assert w['engine']==new['engine'] and input_hash(w)==input_hash(new)
    for k in set(w)-{'model'}:assert joblib.hash(w[k])==joblib.hash(new[k]),(b,h,k)
    a=copy.deepcopy(w['model']['config']);c=copy.deepcopy(new['model']['config'])
    if a.get('gamma')=='scale' and c.get('gamma')!='scale':assert c['gamma']==w['model']['estimator']['estimator']._gamma;a['gamma']=c['gamma']
    assert a==c,(sh,se,ds,fraction,b,h)
  rows.append(dict(key=key(sh,se,ds),fraction=fraction,selected_parent_count=len(selected),full_input_transforms_frozen=True,config_calibration_route_identical=True,seal_sha256=digest(out/'COMPLETE.json')))
write(RUN/'FROZEN_D1_SELF_AUDIT_20261027.json',dict(all_checks_pass=True,cells=75,variants=len(rows),rows=rows,no_new_evaluation_outcomes_read=True,definition='Previously registered fully frozen D1',differences_from_full_B0_selection=['No new hyperparameter search','No validation recalibration','B0 fitted input transforms deliberately fixed during reduced-sample fitting','No downstream retraining; D1 originally reports Level1 only'],not_a_claim_of_reselected_pipeline=True,source=metadata(),audit_code_sha256=digest(__file__)))
print('FROZEN_D1_SELF_AUDIT_PASS',len(rows),flush=True)
