"""Freeze the user-confirmed D1 constraints before launching numeric tuning."""
from d1_numeric import *
setup()
import joblib
from frozen_fit import input_hash
import fairrouter,enhancerouter.router as runtime
assert fairrouter.engine is runtime.engine and fairrouter.raw is runtime.raw
assert fairrouter.predict_bundle is runtime.predict_bundle
checked=all_sources() | read(HERE/'source-code.json')
for p,e in checked.items():assert digest(p,'md5')==e['md5'],p
rows=[]
for sh,se,ds in CELLS:
 base=folder(sh,se,ds)/'b0'
 for rel,h in read(base/'COMPLETE.json')['files'].items():assert digest(base/rel)==h
 models=joblib.load(base/'models.joblib')
 heads={}
 for branch,m in models.items():
  for head,w in m.get('models',{}).items():
   catalog=native_catalog(w,ds,head)
   assert all(locked(c)==locked(w['model']['config']) for c in catalog)
   heads[branch+'/'+head]=dict(engine=w['engine'],locked=locked(w['model']['config']),
    B0_input_hash=input_hash(w),candidate_count=len(catalog),catalog_sha256=fingerprint(catalog))
 rows.append(dict(key=key(sh,se,ds),heads=heads,B0_seal_sha256=digest(base/'COMPLETE.json')))
record=dict(all_checks_pass=True,user_confirmation='feature mode, training combination and model family fixed to B0; numerical parameters and native calibration may retune',
 fractions=[.25,.5,.75,1.],seeds=list(DEV),rows=rows,
 parent_unit='all six views of each sampled support node',
 fixed=['experts','source banks','prototypes','input preprocessing','model engines/families','feature modes','training scopes','internal RNG/ensemble membership','Joint non-numerical choices and training schedule'],
 head_selection='native OOF head shortlist and branch pairing/adoption, filtered original engine numeric catalog',
 calibration='original per-cell schema, including separate 5-shot non-crossfit flow',
 training='original engine fit_candidate with fitted B0 preprocessing frozen by the existing controlled-refit adapter',
 downstream='same Joint/PURF V1 four-stage adaptive numeric search, trainer, own validation checkpoint and inference; B0 gate_gold and all other non-searched fields fixed',
 Joint_numeric_search_fields=sorted(JOINT_NUMERIC),
 outputs='d1_numeric; old frozen D1 remains separate',
 L1_budget='deferred at user request; no metric choice made here',evaluation_truth_used=False,
 source_md5_checked=len(checked),
 files={p.name:digest(p) for p in Path(__file__).parent.glob('*') if p.suffix in ('.py','.sh')},metadata=metadata())
write(RUN/'D1_NUMERIC_REGISTRATION_20261027.json',record)
print('D1_NUMERIC_REGISTERED',len(rows),'cells',flush=True)
