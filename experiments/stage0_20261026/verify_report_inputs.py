"""Fresh verification of every required sealed model before evaluation release."""
from common import *
import argparse
args=argparse.ArgumentParser();args.add_argument('scope',choices=('d1','stage0'));scope=args.parse_args().scope
setup()
assert read(RUN/'NATIVE_BASELINE_SEARCH_REGISTRATION.json')['all_checks_pass']
assert read(RUN/'PREREQUISITES_GAMMA_FROZEN.json')['all_checks_pass']
variants=['b0',*[f'd1/{f}' for f in (.25,.5,.75,1.)]]
if scope=='stage0':variants+=['d3',*[f'd2{a}/{n}' for a in ('a','b') for n in (50,200,1000)],'d4/confidence_router','d4/confidence_preference']
seals={}
for shot,seed,ds in CELLS:
 for variant in variants:
  out=variant_folder(shot,seed,ds,variant);receipt=out/'COMPLETE.json';record=read(receipt)
  for rel,h in record['files'].items():assert digest(out/rel)==h,(out,rel)
  seals[str(receipt)]=digest(receipt)
checked=all_sources() | read(HERE/'source-code.json')
for p,e in checked.items():assert digest(p,'md5')==e['md5'],p
out=RUN/f'REPORT_INPUT_CHECKS/{scope}-{os.environ["SLURM_JOB_ID"]}.json'
write(out,dict(all_checks_pass=True,cells=75,variants=variants,seals=seals,source_md5_checked=len(checked),query_evaluation_labels_used=False,scope=scope,source=metadata()))
ready=RUN/f'REPORT_INPUTS_READY_{scope}.json'
if not ready.exists():write(ready,dict(all_checks_pass=True,scope=scope,receipt=str(out),receipt_sha256=digest(out)))
print('REPORT_INPUTS_READY',scope,len(seals),flush=True)
