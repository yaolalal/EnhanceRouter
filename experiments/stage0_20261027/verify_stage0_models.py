from d1_numeric import *
setup()
for name in ('D1_NUMERIC_ALL75_VERIFIED.json','D2_ALL75_VERIFIED.json','D3_ALL75_RECOVERED_VERIFIED.json'):
 assert read(RUN/name)['all_checks_pass']
variants=['b0',*[f'd1/{f}' for f in (.25,.5,.75,1.)],*[f'd1_numeric/{f}' for f in (.25,.5,.75,1.)],
 'd3',*[f'd2{a}/{n}' for a in ('a','b') for n in (50,200,1000)],'d4/confidence_router','d4/confidence_preference']
seals={}
for sh,se,ds in CELLS:
 for variant in variants:
  out=variant_folder(sh,se,ds,variant)
  for rel,h in read(out/'COMPLETE.json')['files'].items():assert digest(out/rel)==h
  seals[str(out/'COMPLETE.json')]=digest(out/'COMPLETE.json')
checked=all_sources() | read(HERE/'source-code.json')
for p,e in checked.items():assert digest(p,'md5')==e['md5'],p
write(RUN/'STAGE0_MODELS_READY_20261027.json',dict(all_models_complete=True,
 stage0_decision_complete=False,L1_budget='deferred at user request',
 cells=75,variants=variants,seals=seals,source_md5_checked=len(checked),query_truth_read=False,metadata=metadata()))
print('STAGE0_MODELS_READY',len(seals),'L1_AND_DECISION_STILL_DEFERRED',flush=True)
