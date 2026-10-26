from common import *
setup()
import os
out=ROOT/'reports/audit-20261026/discovery.json';found=[];pruned=[];count=0
skip={'chunks','banks','logs','slurm','models','checkpoints','candidates','pipeline','replay','upstream','downstream','text','graph','text-progress-v2','__pycache__','.git','.cache','node_modules'}
for parent,dirs,files in os.walk(SOURCE):
 drop=[d for d in dirs if d in skip];pruned.extend(str(Path(parent)/d) for d in drop);dirs[:]=[d for d in dirs if d not in skip]
 for name in files:
  count+=1;p=Path(parent)/name
  if name.endswith('.json') and ('split' in name.lower() or 'eval' in name.lower() or 'splits' in p.parts):
   if p.stat().st_size>15000000:continue
   if any(s in str(p) for s in ('seed45','seed46','shot10','10shot')):
    try:x=read(p)
    except (ValueError,OSError):continue
    if isinstance(x,dict):
     seed=x.get('split_seed',x.get('seed'));shot=x.get('shots',x.get('shot'))
     if seed in (45,46) or (('seed45' in str(p) or 'seed46' in str(p)) and shot==10):
      r=dict(path=str(p),seed=seed,shot=shot,keys=list(x),sha256=digest(p));found.append(r);print('CANDIDATE',r,flush=True)
write(out,dict(scanned_files=count,candidates=found,pruned_paths=pruned,scope='EALR-GTC, excluding runtime/model/tensor/log/cache directories; all split directories traversed',slurm_job=os.environ['SLURM_JOB_ID']))
print('DISCOVERY_COMPLETE',count,len(found),flush=True)
