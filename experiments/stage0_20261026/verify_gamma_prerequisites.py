"""Strengthen the full original prerequisite set with corrected effective-gamma D1."""
from common import *
setup()
old=RUN/'PREREQUISITES_FROZEN.json';record=read(old)
assert record['all_checks_pass'] and record['historical_B0']==55 and record['historical_D1_cells']==55 and record['native_frontends']==95
assert read(ROOT/'reports/audit-20261026/EFFECTIVE_GAMMA_TEST.json')['all_checks_pass']
for name,h in record['files'].items():assert digest(name)==h,name
for name,e in read(RUN/'HEAD_SOURCE_FILES.json').items():assert digest(name)==e['sha256'],name
registration=RUN/'EFFECTIVE_GAMMA_REGISTRATION.json';affected=read(registration)['affected_historical_cells'];files={}
for c in affected:
 for fraction in (.25,.5,.75,1.):
  out=variant_folder(c['shot'],c['seed'],c['dataset'],f'd1/{fraction}')
  assert 'd1_frozen_v2' in str(out)
  for rel,h in read(out/'COMPLETE.json')['files'].items():
   assert digest(out/rel)==h,(out,rel);files[str(out/rel)]=h
  files[str(out/'COMPLETE.json')]=digest(out/'COMPLETE.json')
write(RUN/'PREREQUISITES_GAMMA_FROZEN.json',dict(all_checks_pass=True,original_prerequisite_sha256=digest(old),original_checked_files=len(record['files']),preserves_all_original_prerequisite_conditions=True,affected_historical_D1_cells=len(affected),corrected_files=files,effective_gamma_registration_sha256=digest(registration),query_evaluation_labels_used=False,source=metadata()))
print('GAMMA_PREREQUISITES_PASS',len(record['files']),len(affected),flush=True)
