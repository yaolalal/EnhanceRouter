from common import *
setup()
import joblib,torch,numpy as np,argparse
from threadpoolctl import threadpool_limits
torch.set_num_threads(1)
threadpool_limits(limits=1)
from compute import route
from enhancerouter.cli import router_recipe
p=argparse.ArgumentParser();p.add_argument('--index',type=lambda s:int(os.environ['SLURM_ARRAY_TASK_ID']) if s.startswith('$') else int(s));a=p.parse_args()
shot,seed,ds=CELLS[a.index];assert seed<=44 or shot==5
c=read(RUN/'HISTORICAL_INPUTS_V2.json')['cells'][key(shot,seed,ds)];out=folder(shot,seed,ds)/'b0'
if (out/'COMPLETE.json').exists():
 for rel,h in read(out/'COMPLETE.json')['files'].items():assert digest(out/rel)==h
 print('B0_ALREADY_VERIFIED',shot,seed,ds,flush=True);raise SystemExit(0)
assert not out.exists()
models={b:joblib.load(c['branches'][b]['bundle']) for b in ('agreement','disagreement')}
routed,scores=route(models,shot,seed,ds)
ref=load(c['inputs']['router'])
for k,v in routed.items():
 if k in ref:torch.testing.assert_close(v,ref[k],rtol=0,atol=1e-10 if v.dtype==torch.float64 else 0)
save(out/'router.pt',routed)
for b,r in scores.items():save(out/f'{b}-scores.pt',r)
with (out/'models.joblib').open('xb') as f:joblib.dump(models,f,compress=3)
# Reuse immutable selected downstream outputs; no model selection is repeated.
write(out/'predictions-reference.json',dict(path=c['predictions'],sha256=digest(c['predictions'])))
write(out/'config.json',dict(dataset=ds,shot=shot,seed=seed,joint=c['config'],router={b:router_recipe(m) for b,m in models.items()},historical_reuse=True,calibration_recomputed=False,validation_policy_changed=False,metadata=metadata()))
seal(out,dict(query_truth_used=False,models_frozen=True,historical_native_predictions_equal=True))
print('B0_REUSED',shot,seed,ds,flush=True)
