"""Contract tests; execute through Slurm only."""
import os, sys, unittest, tempfile
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('ROUTER_SHOTS','3')
class RouterContracts(unittest.TestCase):
 def test_new_shot42_never_historical_split(self):
  import common as c
  self.assertEqual(c.SEEDS,(42,43,44))
  self.assertEqual(c.SHOT,3)
  self.assertIn('/shot3/',str(c.RUN))
  self.assertEqual(c.split_path(42,'cora').name,'cora_3shot_seed42.json')
 def test_forbidden_query_fields_and_gold_parent(self):
  import common as c
  with self.assertRaises(ValueError):c.check_query_keys(c.QUERY_KEYS|{'target'})
  with self.assertRaises(ValueError):c.check_training_identity([3],dict(support_ids=[1],valid_ids=[2],unlabeled_ids=[3]))
 def test_parent_views_stay_together(self):
  import common as c
  from eargtc.router_aug_search.common import parent_folds
  ids=c.torch.tensor([7,9,11]*6);folds=parent_folds(dict(node_ids=ids,dataset='cora'))
  for node in (7,9,11):
   self.assertEqual(sum(bool(f[ids.numpy()==node].any()) for f in folds),1)
 def test_frozen_calibration_cannot_reopen(self):
  import common as c
  old=c.RUN
  try:
   with tempfile.TemporaryDirectory() as d:
    c.RUN=Path(d);(c.RUN/'seed42').mkdir();(c.RUN/'seed42/MODELS_FROZEN.json').write_text('{}')
    with self.assertRaises(RuntimeError):c.assert_open(42)
  finally:c.RUN=old
 def test_no_acceptance_is_null_precision(self):
  from policy import select_acceptance
  import numpy as np
  p=select_acceptance(np.ones(3),np.ones(3,bool),np.arange(3),np.arange(3))
  self.assertEqual(p['fraction'],0);self.assertIsNone(p['precision'])
 def test_available_bundle_with_empty_query_population(self):
  import common as c
  from export import payload
  for branch in c.BRANCHES:
   bundle=dict(branch=branch,models={},tuning_schema='crossfit_v1',policy=dict(fraction=.1))
   b=dict(dataset='cora',ids=c.torch.empty(0,dtype=c.torch.long),gnn_pred=c.torch.empty(0,dtype=c.torch.long),llm_pred=c.torch.empty(0,dtype=c.torch.long))
   result=payload(bundle,b)
   self.assertEqual(len(result['node_ids']),0)
   if branch=='disagreement':self.assertEqual(tuple(result['q'].shape),(0,3))
 def test_real_fit_reload_and_empty_domains(self):
  import common as c
  from tune_engine import _defaults,fit_candidate,predict_candidate
  c.torch.set_num_threads(1);rng=c.np.random.default_rng(1701);banks={}
  for d in c.DATASETS:
   n=0 if d=='pubmed' else 12
   banks[d]=dict(dataset=d,split='train',ids=c.torch.arange(n),node_ids=c.torch.arange(n)//6,
    weight=c.torch.ones(n),tab=c.torch.tensor(rng.normal(size=(n,70))).float(),
    g=c.torch.tensor(rng.normal(size=(n,128))).float(),l=c.torch.tensor(rng.normal(size=(n,4096))).half(),
    state=c.torch.tensor(([0,1,2]*4)[:n]),gnn_pred=c.torch.zeros(n,dtype=c.torch.long),llm_pred=c.torch.ones(n,dtype=c.torch.long))
  cfg=_defaults('lr','prob','source');m=fit_candidate(cfg,'preference',banks,'cora')
  self.assertNotIn('pubmed',m['fit_audit']['fit_parent_ids'])
  self.assertNotIn('cora',m['fit_audit']['fit_parent_ids'])
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'m.joblib';c.joblib.dump(m,path)
   c.np.testing.assert_array_equal(predict_candidate(m,banks['cora']),predict_candidate(c.joblib.load(path),banks['cora']))
 def test_label_free_export_canonical_parity(self):
  import common as c
  from export import payload
  from canonical import canonical_router
  t=c.torch
  ids=t.tensor([2,3]);g=t.tensor([0,0]);l=t.tensor([0,1])
  b=dict(ids=ids,dataset='cora',gnn_pred=g,llm_pred=l)
  bundle=dict(branch='disagreement',unavailable=True,policy=dict(fraction=0))
  empty=payload(bundle,dict(ids=ids[1:],dataset='cora',gnn_pred=g[1:],llm_pred=l[1:]))
  self.assertFalse(bool(empty['accepted_mask'].any()))
  query=dict(node_ids=ids,gnn_pred=g,llm_pred=l,accepted_mask=t.tensor([True,False]),
   selected_score=t.tensor([.9,.2],dtype=t.float64),pseudo_label=t.tensor([0,-1]))
  disagreement=dict(node_ids=ids[1:],route=t.tensor([1]))
  upstream=dict(gnn_logits=t.tensor([[1.,0.]]*4),llm_logits=t.tensor([[1.,0.],[1.,0.],[1.,0.],[0.,1.]]))
  split=dict(num_nodes=4,support_ids=[0],valid_ids=[1],unlabeled_ids=[2,3])
  r=canonical_router(query,disagreement,upstream,split)
  self.assertEqual(r['selected_pred'][3].item(),1)
  self.assertEqual(r['accepted'].tolist(),[False,False,True,False])
if __name__=='__main__':unittest.main()
