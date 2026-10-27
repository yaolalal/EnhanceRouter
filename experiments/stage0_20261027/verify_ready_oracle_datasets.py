"""Release only datasets whose complete oracle text inputs already exist and match hashes."""
from d3_recovery import *
setup()
import torch
ready=DS[:3];frozen=read(RUN/'PREREQUISITES_FROZEN.json')
assert frozen['all_checks_pass'] and read(RUN/'PREREQUISITES_GAMMA_FROZEN.json')['all_checks_pass']
for p,h in frozen['files'].items():assert digest(p)==h,p
for p,e in all_sources().items():assert digest(p,'md5')==e['md5'],p
files={}
for sh,se,ds in CELLS:
 b=folder(sh,se,ds)/'b0'
 for rel,h in read(b/'COMPLETE.json')['files'].items():assert digest(b/rel)==h,(b,rel)
 if ds not in ready:continue
 selection=read(folder(sh,se,ds)/'oracle/SELECTION.json');ev=set(evaluation_ids(se,ds))
 assert not set(selection['a']+selection['b'])&ev
for ds in ready:
 p=RUN/f'shared/text-extra/{ds}.pt';r=read(p.with_suffix('.json'));assert digest(p,'md5')==r['output_md5']
 cache=load(p);request=read(RUN/f'shared/text-extra-request-{ds}.json')
 assert cache['node_ids'].tolist()==request['node_ids']
 for env in ENVIRONMENTS[1:4]:
  value=cache['environments'][env]
  assert torch.isfinite(value['l']).all() and torch.isfinite(value['llm_prob']).all()
  assert torch.allclose(value['llm_prob'].sum(1),torch.ones(len(cache['node_ids'])),atol=1e-5)
 files[str(p)]=digest(p)
write(RUN/'EARLY_ORACLE_DATASETS_READY.json',dict(all_checks_pass=True,datasets=list(ready),checks='all original2180 frozen files; current75 B0 seals; all original source hashes; all required per-dataset complete oracle text IDs, probabilities and hidden states',files=files,remaining_datasets_wait_for_own_text=True,query_labels_read=False,source=metadata(),verification_code_sha256=digest(__file__)))
print('ORACLE_DATASETS_READY',ready,flush=True)
