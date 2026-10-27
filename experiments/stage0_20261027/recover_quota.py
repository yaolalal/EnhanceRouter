"""Preserve unsealed quota-interrupted outputs, resume only unfinished work."""
from d3_recovery import *
import argparse,time

def quarantine(path):
 if path.exists():
  dest=RUN/'quota-interrupted-20261027'/path.relative_to(RUN)
  dest=dest.with_name(dest.name+'-'+os.environ['SLURM_JOB_ID'])
  dest.parent.mkdir(parents=True,exist_ok=True);path.rename(dest)

def verify_seal(out):
 if not (out/'COMPLETE.json').exists():return False
 try:record=read(out/'COMPLETE.json')
 except (ValueError,KeyError):return False
 for rel,h in record['files'].items():assert digest(out/rel)==h
 return True

def recover(stage,index):
 setup()
 import compute,torch,joblib
 sh,se,ds=CELLS[index]
 if stage=='d3':
  out=folder(sh,se,ds)/'d3'
  if verify_seal(out):return
  # A complete saved router may resume downstream fitting. Otherwise retain
  # every partial file and rerun the original pipeline in an empty directory.
  ready=False
  if (out/'router-search/COMPLETE.json').exists() and (out/'models.joblib').exists():
   try:
    models=joblib.load(out/'models.joblib');router,scores=compute.route(models,sh,se,ds,True)
    prior=load(out/'router.pt')
    for k,v in router.items():assert torch.equal(v,prior[k])
    for b,data in scores.items():
     prior=load(out/f'{b}-scores.pt')
     for k,v in data.items():torch.testing.assert_close(v,prior[k],rtol=0,atol=0,equal_nan=True)
    ready=True
   except Exception as exc:print('QUOTA_PARTIAL_ROUTER',index,repr(exc),flush=True)
  if ready:
   for name in ('checkpoint.pt','predictions.pt','joint-selection.json','config.json','COMPLETE.json'):quarantine(out/name)
   search=out/'joint-search'
   for candidate in (search/'candidates').glob('*'):
    try:
     receipt=read(candidate/'result.json')
     assert digest(candidate/'checkpoint.pt')==receipt['checkpoint_sha256']
    except Exception:quarantine(candidate)
   for stage_file in search.glob('stage*.json'):
    try:read(stage_file)
    except Exception:quarantine(stage_file)
   quarantine(search/'SELECTION.json')
   main(index)
  else:
   quarantine(out)
   import enhancerouter.adapter as adapter
   with patch.object(adapter,'validation_halves',new=small_validation_halves):compute.baseline(index,'d3')
 elif stage=='oracle-banks':
  if all(verify_seal(folder(sh,se,ds)/'oracle'/arm) for arm in ('a','b')):return
  for arm in ('a','b'):
   out=folder(sh,se,ds)/'oracle'/arm
   if not verify_seal(out):quarantine(out)
  compute.oracle_banks(index)
 elif stage=='d2':
  if all(verify_seal(folder(sh,se,ds)/f'd2{arm}/{n}') for arm in ('a','b') for n in (50,200,1000)):return
  for arm in ('a','b'):
   for n in (50,200,1000):
    out=folder(sh,se,ds)/f'd2{arm}/{n}'
    if not verify_seal(out):quarantine(out)
  compute.d2(index)
 print('QUOTA_RECOVERY_DONE',stage,index,flush=True)

def text_recover(index):
 setup();import torch
 ds=DS[index];root=RUN/f'shared/text-progress-v2/{ds}'
 # torch.save can leave a truncated .pt after quota exhaustion. Keep the file
 # in quarantine, recompute only its missing node batch with the frozen LLM.
 for directory in (root,RUN/f'shared/runtime-clean/{ds}'):
  if not directory.exists():continue
  for p in directory.rglob('batch-*.pt'):
   try:load(p)
   except Exception as exc:print('QUOTA_PARTIAL_TEXT',str(p),repr(exc),flush=True);quarantine(p)
 from frontend import text
 dest=RUN/f'shared/text-extra/{ds}.pt'
 if dest.exists():
  try:
   load(dest)
   receipt=read(dest.with_suffix('.json'))
   if receipt['output_md5']==digest(dest,'md5'):return
  except Exception:pass
  quarantine(dest);quarantine(dest.with_suffix('.json'))
 text(ds,extra=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage');parser.add_argument('--index',type=int)
 a=parser.parse_args();index=a.index if a.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
 if a.stage=='text':text_recover(index)
 else:recover(a.stage,index)
