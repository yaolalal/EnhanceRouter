"""Audit baseline evidence and shared function identities without new metric choices."""
from d3_recovery import *
setup()
import fairrouter,enhancerouter.router as router,enhancerouter.adapter as adapter
for name in ('engine','raw','predict','predict_bundle','top_mask'):assert getattr(fairrouter,name) is getattr(router,name)
assert fairrouter.prepare_arrays is adapter.prepare_arrays
old=ROOT/'reports/stage0-20261023-historical';receipt=read(old/'COMPLETE.json')
assert receipt['historical_cells']==45 and receipt['all_predictions_exact'] and receipt['all_metrics_match']
for rel,h in receipt['files'].items():assert digest(old/rel)==h,(old,rel)
index=read(ROOT/'configs/inputs.json');history=read(RUN/'HISTORICAL_INPUTS_V2.json')
rows=[]
for shot,seed,ds in CELLS:
 base=folder(shot,seed,ds)/'b0';seal_data=read(base/'COMPLETE.json')
 for rel,h in seal_data['files'].items():assert digest(base/rel)==h,(base,rel)
 sp=split(shot,seed,ds);ev=evaluation_ids(seed,ds)
 assert not set(ev)&set(sp['support_ids']+sp['valid_ids'])
 rows.append(dict(key=key(shot,seed,ds),B0_seal_sha256=digest(base/'COMPLETE.json'),historical=key(shot,seed,ds) in history['cells'],support=len(sp['support_ids']),valid=len(sp['valid_ids']),eval_count=len(ev)))
write(ROOT/'reports/fairrouter-baseline/AUDIT_20261027.json',dict(all_checks_pass=True,historical_reproduction_cells=45,current_B0_cells=75,identical_function_objects=True,rows=rows,evaluation_outcomes_read=False,old_code_freeze_unchanged=True,source=metadata(),audit_code_sha256=digest(__file__)))
print('FAIRROUTER_BASELINE_AUDIT_PASS',len(rows),flush=True)
