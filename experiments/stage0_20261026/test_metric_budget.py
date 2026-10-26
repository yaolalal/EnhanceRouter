from common import *
setup()
import torch
from enhancerouter.router import top_mask
b=dict(ids=torch.arange(570),accepted=torch.zeros(570,dtype=torch.bool));ids=torch.arange(1000)
a=level1_budget(b,ids,.005,'rho_eval_ceil');c=level1_budget(b,ids,.005,'deployed_intersection')
assert a['primary']==3 and c['primary']==0 and a['deployed_intersection']==0
for n in (0,1,570,1000):
 for rho in (0.,.0025,.005,.175,1.):
  bank=dict(ids=torch.arange(n),accepted=torch.zeros(n,dtype=torch.bool))
  assert level1_budget(bank,torch.arange(n),rho,'rho_eval_ceil')['primary']==int(top_mask(torch.zeros(n),torch.arange(n),rho).sum())
write(ROOT/'reports/audit-20261026/METRIC_BUDGET_TEST.json',dict(all_checks_pass=True,Arxiv_example=a,actual_intersection=c,query_labels_read=False,slurm_job=os.environ['SLURM_JOB_ID']))
print('METRIC_BUDGET_PASS',flush=True)
