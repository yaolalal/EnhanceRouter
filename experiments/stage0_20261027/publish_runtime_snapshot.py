"""Publish post-pilot artifact counts and checks, without reading test outcomes."""
from d3_recovery import *
setup()
import shutil
from datetime import datetime,timezone
assert read(RUN/'D1_NUMERIC_PILOT_PASS.json')['all_checks_pass']
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out=ROOT/'reports/runtime-recovery-20261027'/('snapshot-'+stamp)
out.mkdir()
shutil.copyfile(RUN/'D1_NUMERIC_PILOT_PASS.json',out/'PILOT_PASS.json')
rows=[]
for i in (0,3,13,25,26):
 sh,se,ds=CELLS[i]
 for fraction in (.25,1.):
  p=folder(sh,se,ds)/'d1_numeric'/str(fraction)
  seal_record=read(p/'COMPLETE.json')
  for rel,h in seal_record['files'].items():assert digest(p/rel)==h
  cfg=read(p/'config.json');audit=read(p/'controlled-audit.json')
  assert cfg['new_code_sha256']==digest(Path(__file__).with_name('d1_numeric.py'))
  rows.append(dict(key=key(sh,se,ds),fraction=fraction,seal_sha256=digest(p/'COMPLETE.json'),config_sha256=digest(p/'config.json'),new_code_sha256=cfg['new_code_sha256'],controlled_audit_pass=audit['all_checks_pass'],input_transforms_frozen=audit['input_transforms_frozen'],six_views_per_parent=audit['six_views_per_parent'],source_banks_unchanged=audit['source_banks_unchanged'],original_calibration_schema_retained=audit['native_calibration_schema_retained']))
variants=['b0','d3',*[f'd1_numeric/{f}' for f in (.25,.5,.75,1.)],*[f'd2{a}/{n}' for a in ('a','b') for n in (50,200,1000)],'d4/confidence_router','d4/confidence_preference']
counts={v:sum((folder(sh,se,ds)/v/'COMPLETE.json').is_file() for sh,se,ds in CELLS) for v in variants}
state=dict(timestamp_utc=stamp,pilot_pass=True,pilot_checked_outputs=rows,artifact_counts=counts,count_scope='seal existence; global model checks remain separately gated',L1_budget='deferred',Stage0_decision_complete=False,stage1_started=False,test_outcomes_read=False,source=metadata(),snapshot_code_sha256=digest(__file__),queue=subprocess.check_output(['squeue','-u','wujingyao','-h','-o','%i|%j|%T|%M|%R'],text=True))
write(out/'STATE.json',state)
seal(out,dict(all_checks_pass=True,scope='post-pilot evidence and artifact counts only',test_outcomes_read=False))
print('RUNTIME_SNAPSHOT',out,counts,flush=True)
