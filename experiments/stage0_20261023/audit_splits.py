from common import *
setup()
import torch
rows=[]
for ds in DS:
 data=graph(ds);truth=data.y.reshape(-1).long();c=canonical(ds)['llm_logits'].shape[1];test=read(RUN/'EVALUATION_IDS.json')[ds]
 for shot,seed,d in CELLS:
  if d!=ds:continue
  s=split(shot,seed,ds);support=torch.tensor(s['support_ids']);valid=torch.tensor(s['valid_ids'])
  sc=torch.bincount(truth[support],minlength=c);vc=torch.bincount(truth[valid],minlength=c)
  assert not(set(s['support_ids'])&set(s['valid_ids'])) and not(set(test)&set(s['support_ids']+s['valid_ids']))
  assert s['evaluation_1000_ids']==test and len(set(test))==1000
  rows.append(dict(dataset=ds,shot=shot,seed=seed,num_classes=c,support_count=len(support),valid_count=len(valid),support_class_counts=sc.tolist(),valid_class_counts=vc.tolist(),D3_available_count=int(vc.clamp_max(shot).sum()),D3_classes_under_k=int((vc<shot).sum())))
 group=[r for r in rows if r['dataset']==ds]
 print('SPLIT_AUDIT',ds,'support range',min(r['support_count'] for r in group),max(r['support_count'] for r in group),'D3 underfilled classes max',max(r['D3_classes_under_k'] for r in group),flush=True)
write(RUN/'SPLIT_AUDIT.json',dict(rows=rows,evaluation_disjoint=True,identical_eval_order_across_all_shots_and_seeds=True,heldout_evaluations=0,
 D3_note='Shrink from original validation; if class has fewer than k available validation labels, retain all available and record count, never silently claim exact k',metadata=metadata()))
