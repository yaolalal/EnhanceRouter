from common import *
setup()
import torch,numpy as np
from frontend import view_rows
from eargtc.models.gnn import MLPHead
rows=[]
for shot in (3,5):
 seed,ds=42,'cora';sp=split(shot,seed,ds);safe=supervision_bank(shot,seed,ds);state=head_state(shot,seed,ds);cl=canonical(ds)
 head=MLPHead(128,state['net.0.weight'].shape[0],state['net.3.weight'].shape[0],dropout=0.);head.load_state_dict(state);head.eval()
 actual,_=view_rows(ds,safe['support_ids'],safe['support_targets'],sp,upstream_bank(shot,seed,ds),head,cl,canonical_edge(ds))
 for branch in ('agreement','disagreement'):
  old=bank(shot,seed,ds,branch,'train');new=actual[branch];errors={}
  assert old.keys()==new.keys(),(shot,branch,'keys')
  for k,v in old.items():
   if torch.is_tensor(v):
    if v.is_floating_point():
     torch.testing.assert_close(new[k],v,rtol=2e-5,atol=2e-5);errors[k]=float((new[k]-v).abs().max()) if v.numel() else 0.
    else:assert torch.equal(new[k],v),(shot,branch,k)
   else:assert new[k]==v,(shot,branch,k)
  row=dict(shot=shot,dataset=ds,branch=branch,exact_ids_and_targets=True,max_errors=errors);rows.append(row);print('NATIVE_VIEWS_PASS',row,flush=True)
write(ROOT/'reports/audit-20261026/NATIVE_VIEW_TESTS.json',dict(all_checks_pass=True,cases=rows,slurm_job=os.environ['SLURM_JOB_ID']))
