from d1_numeric import *
setup()
import joblib
from frozen_fit import input_hash
assert read(RUN/'D1_NUMERIC_PILOT_PASS.json')['all_checks_pass']
seals={}
for sh,se,ds in CELLS:
 base=folder(sh,se,ds)/'b0';original=joblib.load(base/'models.joblib')
 for fraction in (.25,.5,.75,1.):
  out=folder(sh,se,ds)/'d1_numeric'/str(fraction)
  receipt=read(out/'COMPLETE.json')
  assert receipt['models_frozen'] and receipt['query_truth_used'] is False
  for rel,h in receipt['files'].items():assert digest(out/rel)==h
  assert read(out/'controlled-audit.json')['all_checks_pass']
  models=joblib.load(out/'models.joblib')
  for b,m in models.items():
   assert set(m['models'])==set(original[b]['models'])
   assert m.get('tuning_schema')==original[b].get('tuning_schema')
   for head,w in m['models'].items():
    old=original[b]['models'][head]
    assert w['engine']==old['engine'] and locked(w['model']['config'])==locked(old['model']['config'])
    assert input_hash(w)==input_hash(old)
  selection=read(out/'joint-selection.json')
  assert selection['search']=='original V1 four stages'
  assert joint_locked(selection['config'])==joint_locked(read(base/'config.json')['joint'])
  assert all(joint_locked(c['config'])==joint_locked(read(base/'config.json')['joint']) for c in selection['candidates'])
  seals[str(out/'COMPLETE.json')]=digest(out/'COMPLETE.json')
write(RUN/'D1_NUMERIC_ALL75_VERIFIED.json',dict(all_checks_pass=True,cells=75,variants=300,seals=seals,evaluation_truth_read=False,metadata=metadata()))
print('D1_NUMERIC_ALL75_VERIFIED',len(seals),flush=True)
