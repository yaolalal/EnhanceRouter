from common import *
setup()
import compute as c
import torch
idx=read(ROOT/'configs/inputs.json');real_load=c.load
c.bank=lambda shot,seed,ds,branch,role,small=False:real_load(idx['cells'][f'5/42/{ds}']['branches'][branch][role])
c.split=lambda shot,seed,ds:read(idx['cells'][f'5/42/{ds}']['inputs']['split'])
def bound_load(p):
 p=Path(p)
 if p.name=='upstream.pt':return real_load(idx['cells'][f'5/42/{p.parents[1].name}']['inputs']['upstream'])
 return real_load(p)
c.load=bound_load
for ds in DS:
 models=c.calibrated(c.fitted_models(5,142,ds),5,142,ds)
 routed,scores=c.route(models,5,142,ds)
 changed,diag=c.changed_models(models,5,142,ds,fraction=.25)
 c.route(changed,5,142,ds)
 for v in ('confidence_router','confidence_preference'):
  r,_=c.route(models,5,142,ds,variant=v,reference=routed)
  for b in (True,False):assert int(r['accepted'][r['agreement']==b].sum())==int(routed['accepted'][routed['agreement']==b].sum())
 print('PREFLIGHT_API_PASS',ds,flush=True)
write(RUN/'PREFLIGHT_API_PASS.json',dict(datasets=DS,scope='API/dimension checks on historical banks; not development experimental results',evaluation_truth_used=False,metadata=metadata()))
