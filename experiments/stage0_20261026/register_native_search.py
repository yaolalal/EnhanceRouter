from common import *
setup()
import joblib
from native_router_catalog import initial
assert read(ROOT/'reports/audit-20261026/NATIVE_SEARCH_TEST.json')['all_checks_pass']
history=read(RUN/'HISTORICAL_INPUTS_V2.json');rows=[]
for shot,seed,ds in CELLS:
 if seed<=44 or shot==5:continue
 c=history['cells'][f'5/{seed}/{ds}'];refs={}
 for b,v in c['branches'].items():
  m=joblib.load(v['bundle']);refs[b]=dict(path=v['bundle'],sha256=digest(v['bundle']),schema=m.get('tuning_schema'),reference_fraction=m['policy']['fraction'])
 rows.append(dict(shot=shot,seed=seed,dataset=ds,incumbent_recipe=refs,all_incumbent_weights_refit=True))
code={str(p.relative_to(ROOT)):digest(p) for p in HERE.glob('native_*search.py')}
code[str((HERE/'native_router_catalog.py').relative_to(ROOT))]=digest(HERE/'native_router_catalog.py')
write(RUN/'NATIVE_BASELINE_SEARCH_REGISTRATION.json',dict(all_checks_pass=True,new_cells=20,rows=rows,router_initial_catalog={h:initial(h) for h in ('agreement','trust','preference')},router_refinement_limit=20,router_selection='native OOF head shortlist, pair route comparison and Wilson >=.002 improvement without precision decline',joint_selection='original V1 four-stage adaptive own-validation accuracy, NLL, ID',D3='rerun scorer candidate search and complete Joint V1 search with reduced own validation; fixed original expert-training hyperparameters',supersedes_recipe_registration_sha256=digest(RUN/'BASELINE_RECIPE_REGISTRATION.json'),retains_historical55_unchanged=True,query_truth_used=False,code=code,source=metadata()))
print('NATIVE_BASELINE_SEARCH_REGISTERED',len(rows),flush=True)
