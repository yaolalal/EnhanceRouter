import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from eargtc.router_v10_tune import runtime as r
class RuntimeTests(unittest.TestCase):
    def test_query_is_gated_until_model_freeze(self):
        with tempfile.TemporaryDirectory() as t,patch.object(r,'RUN',Path(t)):
            with self.assertRaises(RuntimeError):r.bank('cora','agreement','query')
    def test_hash_modified_input_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/'input.json';p.write_text('{}')
            (root/'REGISTRATION.json').write_text(json.dumps({'files':{str(p):r.sha(p)}}))
            with patch.object(r,'RUN',root):
                r.authenticate(p);p.write_text('{"changed":true}')
                with self.assertRaises(ValueError):r.authenticate(p)
    def test_gold_guard_rejects_validation_parent(self):
        from eargtc.router_aug_search.common import guard_bank
        b=dict(dataset='cora',split='train',ids=torch.tensor([10]),node_ids=torch.tensor([1]),weight=torch.ones(1))
        for k in ('tab','g','l','gnn_pred','llm_pred'):b[k]=torch.zeros(1)
        meta=dict(gold_ids=[0],validation_ids=[1],query_ids=[2])
        with self.assertRaises(ValueError):guard_bank(b,meta)
    def test_neither_never_correct_by_route(self):
        import numpy as np
        b={'state':np.array([0,1,2,2])};s={'route':np.array([0,1,0,1])}
        np.testing.assert_array_equal(r.correct({'branch':'disagreement'},s,b),[True,True,False,False])
if __name__=='__main__':unittest.main()
