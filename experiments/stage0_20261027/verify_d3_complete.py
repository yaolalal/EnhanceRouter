from d3_recovery import *
setup()
files={}
for sh,se,ds in CELLS:
 out=folder(sh,se,ds)/'d3';record=read(out/'COMPLETE.json')
 assert record['models_frozen'] and record['query_truth_used'] is False
 for rel,h in record['files'].items():assert digest(out/rel)==h,(out,rel)
 files[str(out/'COMPLETE.json')]=digest(out/'COMPLETE.json')
write(RUN/'D3_ALL75_RECOVERED_VERIFIED.json',dict(all_checks_pass=True,cells=75,files=files,empty_branch_extension_test_sha256=digest(RUN/'D3_SMALL_VALIDATION_EXTENSION_TEST.json'),query_labels_read=False,source=metadata()))
print('D3_ALL75_VERIFIED',flush=True)
