from common import *
setup()
import torch
rows=[]
for ds in DS:
 data=graph(ds);truth=data.y.reshape(-1).long();n=len(truth);c=canonical(ds)['llm_logits'].shape[1];test=set(read(RUN/'EVALUATION_IDS.json')[ds])
 candidates=torch.tensor([i for i in range(n) if not bool(data.train_mask[i]) and i not in test])
 count=torch.bincount(truth[candidates],minlength=c)
 rows.append(dict(dataset=ds,class_counts=count.tolist(),classes_below10=int((count<10).sum()),classes_below3=int((count<3).sum())))
 print(ds,'nontraining non-evaluation class counts',count.tolist(),flush=True)
write(RUN/'SMALL_VALIDATION_POOL_AUDIT.json',dict(rows=rows,labels_used_only_to_define_stratified_validation_pool=True,eval_labels_used=False,metadata=metadata()))
