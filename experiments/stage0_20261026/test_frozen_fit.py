"""Verify exact full-data recovery and frozen transforms on changed training rows."""
from common import *
setup()
import torch,joblib,numpy as np
from enhancerouter.router import predict
from frozen_fit import refit,input_hash
idx=read(ROOT/'configs/inputs.json');rows=[]
examples=[(3,'cora','disagreement','trust'),(3,'arxiv','agreement','agreement'),(3,'citeseer','disagreement','trust'),(10,'pubmed','disagreement','preference'),(5,'ogbn-products','disagreement','preference'),(5,'pubmed','disagreement','preference')]
for shot,ds,branch,h in examples:
 banks={d:load(idx['cells'][f'{shot}/42/{d}']['branches'][branch]['train']) for d in DS}
 cell=idx['cells'][f'{shot}/42/{ds}'];bundle=joblib.load(cell['branches'][branch]['bundle']);w=bundle['models'][h]
 vb=load(cell['branches'][branch]['validation']);ids=vb['ids'][:100];vb={k:(v[:100] if torch.is_tensor(v) and v.ndim and len(v)==len(vb['ids']) else v) for k,v in vb.items()}
 full,a=refit(w,h,banks,ds)
 np.testing.assert_allclose(predict(full,vb),predict(w,vb),rtol=0,atol=1e-10)
 parents=torch.unique(banks[ds]['node_ids']);parents=parents[:max(1,len(parents)//4)]
 changed=dict(banks);changed[ds]=subset(banks[ds],torch.isin(banks[ds]['node_ids'],parents))
 reduced,b=refit(w,h,changed,ds)
 assert input_hash(w)==input_hash(full)==input_hash(reduced)
 row=dict(shot=shot,dataset=ds,head=h,engine=w['engine'],full_predictions_exact=True,input_transform_unchanged=True,reduced_parents=len(parents),input_hash=input_hash(w),full=a,reduced=b);rows.append(row);print('PASS',row,flush=True)
write(ROOT/'reports/audit-20261026/FROZEN_FIT_TESTS.json',dict(all_checks_pass=True,tests=rows,slurm_job=os.environ['SLURM_JOB_ID']))
