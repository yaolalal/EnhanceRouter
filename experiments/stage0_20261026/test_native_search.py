from common import *
setup()
import joblib
from compute import fitted_models,calibrated,joint
from native_router_search import select
shot,seed,ds=3,42,'cora'
out=RUN/'native-search-test'
models=calibrated(fitted_models(shot,seed,ds),shot,seed,ds)
models=select(models,shot,seed,ds,False,out)
assert set(models)=={'agreement','disagreement'}
for branch,m in models.items():
 result=read(out/f'router-search/seed{seed}/models/{ds}/{branch}/SELECTION.json')
 assert result['query_fit_rows']==0
from compute import route
router,scores=route(models,shot,seed,ds)
config=joint(shot,seed,ds,router,out)
assert read(out/'joint-selection.json')['search']=='original V1 four stages'
seal(out,dict(all_checks_pass=True,query_truth_used=False))
write(ROOT/'reports/audit-20261026/NATIVE_SEARCH_TEST.json',dict(all_checks_pass=True,query_truth_used=False,cell=key(shot,seed,ds),joint_config=config,source=metadata()))
print('NATIVE_SEARCH_PASS',flush=True)
