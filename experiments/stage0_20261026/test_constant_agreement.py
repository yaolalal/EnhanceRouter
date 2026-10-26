from common import *
setup()
import torch,joblib,numpy as np
from frozen_fit import refit,input_hash
from enhancerouter.router import predict
idx=read(ROOT/'configs/inputs.json');shot,seed,ds=3,44,'arxiv';branch='agreement';h='agreement'
banks={d:load(idx['cells'][f'{shot}/{seed}/{d}']['branches'][branch]['train']) for d in DS};c=idx['cells'][f'{shot}/{seed}/{ds}'];w=joblib.load(c['branches'][branch]['bundle'])['models'][h];vb=load(c['branches'][branch]['validation'])
full,a=refit(w,h,banks,ds);np.testing.assert_array_equal(predict(full,vb),predict(w,vb))
parents=torch.tensor(read(c['inputs']['split'])['support_ids']);parents=parents[torch.randperm(len(parents),generator=torch.Generator().manual_seed(seed+913))[:max(1,int(len(parents)*.25+.5))]]
bs=dict(banks);bs[ds]=subset(banks[ds],torch.isin(banks[ds]['node_ids'],parents));reduced,b=refit(w,h,bs,ds)
assert input_hash(w)==input_hash(reduced) and reduced['model']['fitted']['kind']=='constant'
write(ROOT/'reports/audit-20261026/CONSTANT_AGREEMENT_TEST.json',dict(all_checks_pass=True,case=[shot,seed,ds],native_single_class_short_circuit_preserved=True,full_predictions_exact=True,absent_input_transform_stays_absent=True,reduced=b,slurm_job=os.environ['SLURM_JOB_ID']))
print('CONSTANT_AGREEMENT_PASS',flush=True)
