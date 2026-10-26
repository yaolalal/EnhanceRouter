"""Read-only historical ID audit; no labels, resampling, or metric selection."""
from common import *
setup()
import csv
out=ROOT/'reports/audit-20261026/split-audit';out.mkdir(exist_ok=True)
idx=read(ROOT/'configs/inputs.json');splits={};sources={};evaluations={}
for shot in SHOTS:
 for seed in (42,43,44):
  for ds in DS:
   p=Path(idx['cells'][f'{shot}/{seed}/{ds}']['inputs']['split']);splits[(ds,shot,seed)]=read(p);sources[str(p)]=digest(p)
   if shot==10:
    ids=read(p)['evaluation_1000_ids'];assert ids==idx['standard_eval_ids'][f'{seed}/{ds}'];evaluations[(ds,seed)]=ids
for seed in (45,46):
 for ds in DS:
  p=SOURCE/f'runs/router_v10_multiseed_data_20260912_v1/seed{seed}/splits/{ds}_5shot_seed{seed}.json'
  splits[(ds,5,seed)]=read(p);sources[str(p)]=digest(p)
rows=[]
for (ds,eval_seed),ids in evaluations.items():
 assert len(ids)==len(set(ids))==1000
 for (d,shot,split_seed),sp in splits.items():
  if d!=ds:continue
  row=dict(dataset=ds,eval_seed=eval_seed,split_seed=split_seed,shot=shot,eval_count=len(ids),
   eval_train_overlap=len(set(ids)&set(sp['support_ids'])),eval_valid_overlap=len(set(ids)&set(sp['valid_ids'])),
   train_valid_overlap=len(set(sp['support_ids'])&set(sp['valid_ids'])),same_seed=eval_seed==split_seed,
   eval_outside_query=len(set(ids)-set(sp['unlabeled_ids'])))
  rows.append(row)
same=[r for r in rows if r['same_seed']];cross=[r for r in rows if not r['same_seed']]
result=dict(known_10shot_seeds=[42,43,44],known_5shot_seeds=[42,43,44,45,46],missing_10shot_seeds=[45,46],
 evaluation_ids_unchanged=True,same_seed_all_clear=all(r['eval_train_overlap']==r['eval_valid_overlap']==r['eval_outside_query']==r['train_valid_overlap']==0 for r in same),
 cross_seed_all_clear=all(r['eval_train_overlap']==r['eval_valid_overlap']==0 for r in cross),rows=rows,source_sha256=sources,
 slurm_job=os.environ['SLURM_JOB_ID'],no_resampling=True,labels_read=False)
write(out/'RESULTS.json',result)
with (out/'overlap_matrix.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('SAME_SEED',len(same),'clear',result['same_seed_all_clear'],flush=True)
print('CROSS_SEED',len(cross),'clear',result['cross_seed_all_clear'],flush=True)
for r in cross:
 if r['eval_train_overlap'] or r['eval_valid_overlap']:print('OVERLAP',r,flush=True)
print('MISSING historical10shot45/46; never substitute5shot evaluation or resample',flush=True)
