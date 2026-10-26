from common import *
setup()
import torch,joblib,numpy as np
from frozen_fit import refit,input_hash
from enhancerouter.router import predict
idx=read(ROOT/'configs/inputs.json');shot,seed,ds=10,42,'pubmed';h='preference';branch='disagreement'
banks={d:bank(shot,seed,d,branch,'train') for d in DS};w=joblib.load(idx['cells'][key(shot,seed,ds)]['branches'][branch]['bundle'])['models'][h]
assert w['model']['config']['scope']=='target';banks[ds]=subset(banks[ds],torch.zeros(len(banks[ds]['ids']),dtype=torch.bool))
new,diag=refit(w,h,banks,ds);vb=bank(shot,seed,ds,branch,'validation');assert np.array_equal(predict(new,vb),np.zeros(len(vb['ids'])))
assert input_hash(new)==input_hash(w)
write(ROOT/'reports/audit-20261026/EMPTY_TARGET_TEST.json',dict(all_checks_pass=True,case=[shot,seed,ds,h],diagnostics=diag,query_labels_used=False,slurm_job=os.environ['SLURM_JOB_ID']))
write(RUN/'NATIVE_EMPTY_RULE.json',dict(preregistered_before_new20_D1=True,rule='For a genuinely empty target-only head, native smoothed constant prior (0+.5)/(0+1)=.5; raw logit0. Preserve B0 input transform, normalization, affine, rho and tie rules. Never reuse B0 learned weights as if trained on fewer rows.',historical55_empty_cases=read(RUN/'EMPTY_TARGET_AUDIT.json')['cases'],test_sha256=digest(ROOT/'reports/audit-20261026/EMPTY_TARGET_TEST.json')))
print('EMPTY_TARGET_PASS',flush=True)
