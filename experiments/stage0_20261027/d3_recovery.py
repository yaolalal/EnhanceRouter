"""Resume interrupted D3 Joint fitting; preserve all previously produced router files."""
from pathlib import Path
import sys,os
OLD=Path(__file__).resolve().parent.parent/'stage0_20261026'
sys.path.insert(0,str(OLD))
from common import *
from unittest.mock import patch

def small_validation_halves(dataset,valid,agreement):
 import torch
 from enhancerouter.adapter import deterministic_calibration_threshold_split
 selection=[];threshold=[]
 for module in ('conflict','agreement'):
  ids=valid[agreement[valid] if module=='agreement' else ~agreement[valid]]
  if len(ids)<2:
   selection.append(ids);threshold.append(ids[:0])
  else:
   parts=deterministic_calibration_threshold_split(ids,dataset,module)
   selection.append(ids[parts.calibration]);threshold.append(ids[parts.threshold])
 return torch.cat(selection).sort().values,torch.cat(threshold).sort().values

def main(index):
 setup()
 import joblib,torch
 from compute import route,joint,variant_receipt
 import enhancerouter.adapter as adapter
 shot,seed,ds=CELLS[index];out=folder(shot,seed,ds)/'d3'
 if (out/'COMPLETE.json').exists():return
 assert read(out/'router-search/COMPLETE.json')['all_checks_pass']
 models=joblib.load(out/'models.joblib');router,scores=route(models,shot,seed,ds,True)
 prior=load(out/'router.pt')
 for k,v in router.items():assert torch.equal(v,prior[k]),k
 for branch,data in scores.items():
  prior=load(out/f'{branch}-scores.pt')
  for k,v in data.items():
   torch.testing.assert_close(v,prior[k],rtol=0,atol=0,equal_nan=True)
 with patch.object(adapter,'validation_halves',new=small_validation_halves):
  chosen=joint(shot,seed,ds,router,out,True)
 from enhancerouter.cli import router_recipe
 write(out/'config.json',dict(dataset=ds,shot=shot,seed=seed,joint=chosen,router={b:router_recipe(m) for b,m in models.items()},validation_protocol='k labels/class; branch singleton placed in selection, empty threshold for that branch',recovery=dict(original_error='branch validation population has fewer than2 rows',code=str(Path(__file__).resolve()),sha256=digest(__file__),original_router_replayed_exactly=True),metadata=metadata()))
 seal(out,dict(query_truth_used=False,models_frozen=True,recovery_extension=True))
 print('D3_RECOVERED',index,key(shot,seed,ds),flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
