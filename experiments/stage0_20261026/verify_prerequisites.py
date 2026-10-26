"""Fail closed on missing/mismatched baseline and native-frontend prerequisites."""
from common import *
setup()
import torch
assert read(RUN/'SPLIT_AUDIT.json')['all_checks_pass']
assert read(ROOT/'reports/audit-20261026/NATIVE_VIEW_TESTS.json')['all_checks_pass']
assert read(ROOT/'reports/audit-20261026/FROZEN_FIT_TESTS.json')['all_checks_pass']
assert read(ROOT/'reports/audit-20261026/NATIVE_CALIBRATION_TESTS.json')['all_checks_pass']
assert read(ROOT/'reports/audit-20261026/CONSTANT_AGREEMENT_TEST.json')['all_checks_pass']
assert read(RUN/'EXACT_FIVE_SHOT_REGISTRATION.json')['all_checks_pass']
assert read(ROOT/'reports/audit-20261026/EMPTY_TARGET_TEST.json')['all_checks_pass']
assert read(RUN/'GNN_TRAINING_SEEDS.json')['experiment_split_seeds']==list(DEV)
files={};historical=0;frontends=0
for p,e in read(RUN/'HEAD_SOURCE_FILES.json').items():assert digest(p)==e['sha256'],p
for shot,seed,ds in CELLS:
 sp=split(shot,seed,ds);ev=evaluation_ids(seed,ds);assert set(ev)<=set(sp['unlabeled_ids'])
 if seed<=44 or shot==5:
  out=folder(shot,seed,ds)/'b0';seal_data=read(out/'COMPLETE.json');assert seal_data['historical_native_predictions_equal']
  for rel,h in seal_data['files'].items():
   assert digest(out/rel)==h,(out,rel);files[str(out/rel)]=h
  files[str(out/'COMPLETE.json')]=digest(out/'COMPLETE.json')
  pr=read(out/'predictions-reference.json');assert digest(pr['path'])==pr['sha256']
  for fraction in (.25,.5,.75,1.):
   d1=folder(shot,seed,ds)/f'd1/{fraction}';data=read(d1/'COMPLETE.json')
   for rel,h in data['files'].items():
    assert digest(d1/rel)==h,(d1,rel);files[str(d1/rel)]=h
   files[str(d1/'COMPLETE.json')]=digest(d1/'COMPLETE.json')
  historical+=1
 modes=['small_native'] if seed<=44 or shot==5 else ['standard_native','small_native']
 for mode in modes:
  out=folder(shot,seed,ds)/mode;assert read(out/'HEAD_COMPLETE.json')['all_checks_pass']
  stats=read(out/'head-training.json');assert stats['training_seed']==gnn_training_seed(ds),(out,'wrong expert RNG')
  sup=load(out/'supervision.pt');assert sup['support_ids'].tolist()==sp['support_ids']
  valid=sup['valid_ids'].tolist();assert not set(valid)&(set(ev)|set(sp['support_ids']))
  if mode=='standard_native':assert valid==sp['valid_ids']
  else:assert set(valid)<=set(sp['valid_ids'])
  up=load(out/'upstream.pt');assert set(up)=={'gnn_logits','llm_logits'} and len(up['gnn_logits'])==sp['num_nodes'] and up['gnn_logits'].shape==up['llm_logits'].shape
  assert bool(torch.isfinite(up['gnn_logits']).all()) and bool(torch.isfinite(up['llm_logits']).all())
  for name in ('HEAD_COMPLETE.json','head-training.json','head.pt','upstream.pt','supervision.pt'):
   p=out/name;files[str(p)]=digest(p)
  frontends+=1
assert historical==55 and frontends==95
write(RUN/'PREREQUISITES_FROZEN.json',dict(all_checks_pass=True,historical_B0=historical,historical_D1_cells=historical,native_frontends=frontends,seeds=list(DEV),wrong_RNG_directories_excluded=True,files=files,slurm_job=os.environ['SLURM_JOB_ID'],query_evaluation_labels_used=False))
print('PREREQUISITES_PASS',historical,frontends,flush=True)
