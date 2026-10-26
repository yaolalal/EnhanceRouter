"""Slurm contract tests for fresh-shot classifier and report boundaries."""
import importlib.util
import os
from pathlib import Path
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
assert os.environ.get('SLURM_JOB_ID'), 'Numerical tests require Slurm'

class Contracts(unittest.TestCase):
    def module(self, name):
        path = Path(__file__).with_name(name + '.py')
        self.assertTrue(path.exists(), 'Missing new-shot ' + name + ' adapter')
        spec = importlib.util.spec_from_file_location('shots_' + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def split(self):
        return dict(dataset='cora', split_seed=42, shots=3, num_nodes=8,
                    support_ids=[0,1], valid_ids=[2,3], unlabeled_ids=[4,5,6,7],
                    evaluation_full_ids=[4,5,6,7], evaluation_1000_ids=[7,4],
                    evaluation_test_ids=[7,4], test_truth_exported=False)

    def test_split_guards_and_fixed_order(self):
        d = self.module('downstream'); s = self.split()
        d.validate_split(s, 3, 42, 'cora')
        for change in (dict(valid_ids=[1,3]), dict(evaluation_1000_ids=[4,7]),
                       dict(evaluation_test_ids=[7,0]), dict(split_seed=43),
                       dict(test_truth_exported=True), dict(unlabeled_ids=[4,5,6])):
            with self.assertRaises(ValueError): d.validate_split(dict(s, **change), 3,42,'cora')

    def test_exact_three_seeds_and_undefined_precision(self):
        d=self.module('report')
        rows=[dict(dataset='cora', shot=3, population='full_query',seed=s,
                   accuracy=v, macro_f1=v, pseudo_accuracy=None if s==42 else v,
                   pseudo_coverage=v) for s,v in ((42,.2),(43,.4),(44,.6))]
        r=d.aggregate(rows)[0]
        self.assertAlmostEqual(r['accuracy_mean'], .4)
        self.assertAlmostEqual(r['accuracy_std_ddof1'], .2)
        self.assertIsNone(r['pseudo_accuracy_mean'])
        with self.assertRaises(ValueError):d.aggregate(rows[:2])
        with self.assertRaises(ValueError):d.aggregate(rows+[rows[0]])

    def test_freeze_blocks_fitting(self):
        import tempfile
        d=self.module('downstream')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            d.training_open(root)
            (root/'DOWNSTREAM_FROZEN.json').write_text('{}')
            with self.assertRaises(RuntimeError):d.training_open(root)

    def test_fresh_fit_persists_seed_and_reloads_predictions(self):
        import tempfile
        import torch
        d=self.module("downstream")
        torch.manual_seed(1701)
        f={"gnn_embeddings":torch.randn(24,4),"logp_g":torch.randn(24,3).log_softmax(1),
           "logp_l":torch.randn(24,3).log_softmax(1),"selected_pred":torch.arange(24)%3}
        s={"gold_ids":torch.arange(6),"gold_targets":torch.arange(6)%3,
           "selection_ids":torch.arange(6,10),"selection_targets":torch.arange(4)%3}
        pools={"teach_g":torch.arange(10,14),"teach_l":torch.arange(14,18),"agreement":torch.arange(18,22)}
        h=torch.randn(24,5)
        cfg=dict(d.v2train.DEFAULTS,dim=4,max_steps=4,validate_every=2,batch_size=4)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dest=root/"candidate"
            row=d.fit(root,dest,cfg,43,f,s,pools,h,2)
            self.assertEqual(row["training_seed"],43)
            member=torch.load(dest/"checkpoint.pt",weights_only=True)
            model=d.v2train.make_model(f,h,cfg);model.load_state_dict(member["states"]["accuracy"])
            torch.testing.assert_close(d.v2train.predict(model,f,h,s["selection_ids"]),member["validation_probability"],atol=0,rtol=0)
            self.assertEqual(d.fit(root,dest,cfg,43,f,s,pools,h,2),row)
            with self.assertRaises(ValueError):d.result(dest,cfg,42)

if __name__=='__main__':unittest.main()
