from d1_numeric import *
setup()
assert read(RUN/'D1_NUMERIC_REGISTRATION_20261027.json')['all_checks_pass']
for index in (0,3,13,25,26):
 sh,se,ds=CELLS[index]
 for fraction in (.25,1.):
  out=folder(sh,se,ds)/'d1_numeric'/str(fraction)
  receipt=read(out/'COMPLETE.json')
  for rel,h in receipt['files'].items():assert digest(out/rel)==h
  assert read(out/'controlled-audit.json')['all_checks_pass']
  assert read(out/'config.json')['new_code_sha256']==digest(Path(__file__).with_name('d1_numeric.py'))
  selection=read(out/'joint-selection.json')
  assert selection['search']=='original V1 four stages'
  assert all(joint_locked(c['config'])==joint_locked(read(folder(sh,se,ds)/'b0/config.json')['joint']) for c in selection['candidates'])
write(RUN/'D1_NUMERIC_PILOT_PASS.json',dict(all_checks_pass=True,cells=[0,3,13,25,26],fractions=[.25,1.],query_truth_used=False,metadata=metadata()))
print('D1_NUMERIC_PILOT_PASS',flush=True)
