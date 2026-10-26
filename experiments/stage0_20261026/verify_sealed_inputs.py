from common import *
setup()
p=RUN/'PREREQUISITES_FROZEN.json';record=read(p)
assert record['all_checks_pass'] and record['historical_B0']==55 and record['historical_D1_cells']==55 and record['native_frontends']==95
for name,h in record['files'].items():assert digest(name)==h,name
for name,e in read(RUN/'HEAD_SOURCE_FILES.json').items():assert digest(name)==e['sha256'],name
write(RUN/f'PREREQUISITE_RECHECKS/{os.environ["SLURM_JOB_ID"]}.json',dict(all_checks_pass=True,prerequisite_sha256=digest(p),checked_files=len(record['files']),preserves_all_original_prerequisite_conditions=True,source=metadata()))
print('SEALED_INPUTS_RECHECK_PASS',len(record['files']),flush=True)
