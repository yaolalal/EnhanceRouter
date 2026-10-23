from common import *
setup()
rows=[]
for ds in DS:
 for shot in SHOTS:
  standard=[];small=[];shortage=[];support=[]
  for seed in DEV:
   f=folder(shot,seed,ds)
   a=read(f/'standard/HEAD_COMPLETE.json');b=read(f/'small_stratified/HEAD_COMPLETE.json')
   standard.append(a['validation_count']);small.append(b['validation_count']);shortage.append(len(b['underfilled_classes']));support.append(len(split(shot,seed,ds)['support_ids']))
  rows.append(dict(dataset=ds,shot=shot,completed_seeds=len(DEV),standard_validation_counts=sorted(set(standard)),small_validation_counts=sorted(set(small)),underfilled_classes=sorted(set(shortage)),support_counts=sorted(set(support))))
write(RUN/'PROGRESS-20261023-2100.json',dict(rows=rows,held_out_evaluations=0,note='Training preparation counts only. No new router/Joint test outcomes are available.'))
print(json.dumps(rows,indent=2),flush=True)
