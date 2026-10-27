from d3_recovery import *
setup()
seals={}
for sh,se,ds in CELLS:
 for arm in ('a','b'):
  for n in (50,200,1000):
   out=folder(sh,se,ds)/f'd2{arm}/{n}'
   for rel,h in read(out/'COMPLETE.json')['files'].items():assert digest(out/rel)==h,(out,rel)
   seals[str(out/'COMPLETE.json')]=digest(out/'COMPLETE.json')
write(RUN/'D2_ALL75_VERIFIED.json',dict(all_checks_pass=True,cells=75,variants=450,seals=seals,evaluation_outcomes_read=False,source=metadata()))
print('D2_ALL75_VERIFIED',len(seals),flush=True)
