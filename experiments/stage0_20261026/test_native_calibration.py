from common import *
setup()
import joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
from native_calibration import calibrate
from enhancerouter.router import predict_bundle
threadpool_limits(limits=1);torch.set_num_threads(1)
idx=read(ROOT/'configs/inputs.json');anchor=read(SOURCE/'runs/router_v10_multiseed_20260912_v1/ANCHOR.json');rows=[]
for ds in DS:
 for branch in ('agreement','disagreement'):
  source=copy.deepcopy(idx['cells'][f'5/43/{ds}']['branches'][branch]);source['bundle']=str(SOURCE/f'runs/router_v10_multiseed_20260912_v1/seed43/models/{ds}/{branch}/bundle.joblib');old=joblib.load(source['bundle']);vb=load(source['validation'])
  selected=read(Path(source['bundle']).parent/'SELECTION.json');norms=selected['normalization']
  new,receipt,_=calibrate(ds,branch,copy.deepcopy(old),vb,norms,anchor[ds][branch],anchor[ds][branch]['reference_fraction'])
  assert new['policy']['fraction']==old['policy']['fraction'],(ds,branch,'rho changed')
  a=predict_bundle(old,vb);b=predict_bundle(new,vb)
  for k in a:np.testing.assert_allclose(a[k],b[k],rtol=0,atol=1e-10)
  rows.append(dict(dataset=ds,branch=branch,native_policy_recovered=True,fraction=old['policy']['fraction'],method=receipt['method']))
  print('NATIVE5_PASS',rows[-1],flush=True)
write(ROOT/'reports/audit-20261026/NATIVE_CALIBRATION_TESTS.json',dict(all_checks_pass=True,cases=rows,slurm_job=os.environ['SLURM_JOB_ID']))
