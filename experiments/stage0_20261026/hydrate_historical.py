"""Reuse sealed historical baselines without refitting or recalibration."""
from common import *
setup()
import joblib
assert read(RUN/'PREPARED.json')['all_checks_pass']
idx=read(ROOT/'configs/inputs.json');extension=copy.deepcopy(idx)
def replace(x,seed):
 if isinstance(x,str):return x.replace('seed44',f'seed{seed}')
 if isinstance(x,dict):return {k:replace(v,seed) for k,v in x.items()}
 if isinstance(x,list):return [replace(v,seed) for v in x]
 return x
for seed in (45,46):
 for ds in DS:
  c=replace(idx['cells'][f'5/44/{ds}'],seed);c.pop('expected_sha256',None)
  selection=read(c['selection']);assert selection['training_seed']==seed and not selection['test_truth_loaded']
  c.update(seed=seed,config=selection['config']);c['checkpoint']=str(Path(c['selection']).parent.parent/'candidates'/selection['candidate_id']/'checkpoint.pt')
  assert Path(c['checkpoint']).is_file();extension['cells'][f'5/{seed}/{ds}']=c
if (RUN/'HISTORICAL_INPUTS.json').exists():assert read(RUN/'HISTORICAL_INPUTS.json')==extension
else:write(RUN/'HISTORICAL_INPUTS.json',extension)
for shot,seed,ds in CELLS:
 if seed>44 and shot!=5:continue
 c=extension['cells'][key(shot,seed,ds)];sp=split(shot,seed,ds);old=read(c['inputs']['split'])
 assert all(sp[k]==old[k] for k in ('support_ids','valid_ids','unlabeled_ids'))
 dest=folder(shot,seed,ds)/'standard';dest.mkdir(parents=True,exist_ok=True)
 write(dest/'split.json',sp)
 print('HYDRATE',shot,seed,ds,flush=True)
write(RUN/'HISTORICAL_READY.json',dict(cells=55,preserved_weights=True,preserved_calibration=True,preserved_policy=True,source_index_sha256=digest(RUN/'HISTORICAL_INPUTS.json')))

extra={};known=all_sources()
for seed in (45,46):
 for ds in DS:
  c=extension['cells'][f'5/{seed}/{ds}']
  paths=list(c['inputs'].values())+[c[k] for k in ('checkpoint','predictions','selection','graph')]
  for b in c['branches'].values():paths.extend(b.values())
  for q in paths:
   if q in known or q in extra:continue
   p=Path(q);extra[q]=dict(md5=digest(p,'md5'),sha256=digest(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
  print('HASHED_HISTORY',seed,ds,len(extra),flush=True)
write(RUN/'SOURCE_EXTENSION_FILES.json',extra)
