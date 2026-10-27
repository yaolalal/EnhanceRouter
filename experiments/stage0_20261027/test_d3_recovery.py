from d3_recovery import *
setup()
import torch
from enhancerouter.adapter import validation_halves
rows=[]
for sh,se,ds in CELLS:
 safe=supervision_bank(sh,se,ds,True);up=upstream_bank(sh,se,ds,True);valid=safe['valid_ids'];agreement=up['gnn_logits'].argmax(1)==up['llm_logits'].argmax(1)
 a,b=small_validation_halves(ds,valid,agreement)
 assert not set(a.tolist())&set(b.tolist()) and set(a.tolist()+b.tolist())==set(valid.tolist())
 sizes=[int(agreement[valid].sum()),int((~agreement[valid]).sum())]
 if min(sizes)>=2:
  x,y=validation_halves(ds,valid,agreement);assert torch.equal(a,x) and torch.equal(b,y)
 rows.append(dict(key=key(sh,se,ds),branch_counts=sizes,exceptional=min(sizes)<2))
write(RUN/'D3_SMALL_VALIDATION_EXTENSION_TEST.json',dict(all_checks_pass=True,query_truth_used=False,rows=rows,exceptional=[r for r in rows if r['exceptional']],rule='0-row branch contributes no labels.1-row branch puts its one node in selection; no duplicate/extra/query labels. All >=2 populations retain exact original split.',code_sha256=digest(Path(__file__).with_name('d3_recovery.py')),source=metadata()))
print('D3_SMALL_VALIDATION_EXTENSION_PASS',[r for r in rows if r['exceptional']],flush=True)
