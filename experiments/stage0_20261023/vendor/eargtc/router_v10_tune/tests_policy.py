import unittest,tempfile
from pathlib import Path
import numpy as np
from eargtc.router_v10_tune.policy import choose_policy,better,assert_open,validate_query
class PolicyTests(unittest.TestCase):
    def test_free_fraction_max_precision(self):
        s=np.arange(100)[::-1];y=np.r_[np.ones(20),np.zeros(80)].astype(bool)
        p=choose_policy(s,y,np.arange(100),(.1,.2,.5,1.))
        self.assertEqual(p['fraction'],.2);self.assertEqual(p['precision'],1.)
    def test_tie_prefers_supported_coverage(self):
        p=choose_policy(np.ones(100),np.ones(100,bool),np.arange(100),( .2,.5,1.))
        self.assertEqual(p['fraction'],1.);self.assertFalse(better(p,p))
    def test_node_id_tiebreak_and_minimum(self):
        ids=np.arange(30)[::-1];p=choose_policy(np.ones(30),ids<20,ids,(.5,2/3,1.))
        self.assertEqual(p['accepted'],20);self.assertEqual(p['correct'],20)
        with self.assertRaises(ValueError):choose_policy(np.ones(10),np.ones(10),np.arange(10),(.1,1.))
    def test_freeze_permanent(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);assert_open(p);(p/'MODELS_FROZEN.json').write_text('{}')
            with self.assertRaises(RuntimeError):assert_open(p)
    def test_query_no_labels(self):
        safe={k:None for k in ('dataset','ids','tab','g','l','gnn_pred','llm_pred')};validate_query(safe)
        for k in ('y','state','target','labels','correct'):
            with self.assertRaises(ValueError):validate_query(dict(safe,**{k:None}))
if __name__=='__main__':unittest.main()
