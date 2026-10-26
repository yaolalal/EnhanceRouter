"""Six-view coverage, gold-parent and no-query-truth contracts; run only in Slurm."""
import importlib.util
import os
from pathlib import Path
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

class ViewsContract(unittest.TestCase):
    def test_implementation_exists(self):
        self.assertTrue(Path(__file__).with_name('views.py').is_file(), 'six-view implementation missing')

    def test_six_views_gold_parents_and_equal_weights(self):
        if not Path(__file__).with_name('views.py').is_file():self.skipTest('implementation missing')
        from scripts.router_v10_shots_v1.views import assemble_training, ENVIRONMENTS
        import torch
        ids=torch.tensor([1,3]);envs=[]
        for k,name in enumerate(ENVIRONMENTS):
            envs.append(dict(environment=name,node_ids=ids,gnn_prob=torch.tensor([[.8,.2],[.3,.7]]),llm_prob=torch.tensor([[.9,.1],[.9,.1]]),tab70=torch.zeros(2,70),g=torch.zeros(2,128),l=torch.zeros(2,4096)))
        result=assemble_training('fixture',envs,ids,torch.tensor([0,1]),5)
        parents=torch.cat([result[b]['node_ids'] for b in ('agreement','disagreement')])
        self.assertEqual(parents.tolist().count(1),6);self.assertEqual(parents.tolist().count(3),6)
        self.assertEqual(result['agreement']['tab'].shape,(6,43))
        self.assertTrue(torch.equal(result['disagreement']['weight'],torch.ones(6)))
        with self.assertRaisesRegex(ValueError,'six|environment'):
            assemble_training('fixture',envs[:-1],ids,torch.tensor([0,1]),5)
        envs[0]=dict(envs[0],node_ids=torch.tensor([1,4]))
        with self.assertRaisesRegex(ValueError,'parent|gold|support'):
            assemble_training('fixture',envs,ids,torch.tensor([0,1]),5)

    def test_query_schema_and_validation_targets(self):
        if not Path(__file__).with_name('views.py').is_file():self.skipTest('implementation missing')
        from scripts.router_v10_shots_v1.views import evaluation_bank,QUERY_KEYS
        import torch
        ids=torch.tensor([4,2]);g=torch.tensor([0,1]);l=torch.tensor([0,0])
        args=('fixture',ids,torch.zeros(2,70),torch.zeros(2,128),torch.zeros(2,4096),g,l)
        q=evaluation_bank(*args,branch='disagreement')
        self.assertEqual(set(q),QUERY_KEYS);self.assertEqual(q['ids'].tolist(),[2])
        v=evaluation_bank(*args,branch='disagreement',targets=torch.tensor([1,1]))
        self.assertEqual(v['state'].tolist(),[0])

class PublishedBanks(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('VIEWS_TEST_BANK'), 'Set VIEWS_TEST_BANK for published-bank integration')
    def test_real_banks_parent_views_targets_partition_and_fresh_head(self):
        import json
        import torch
        from scripts.router_v10_shots_v1.views import QUERY_KEYS, ENVIRONMENTS, sha
        path=Path(os.environ['VIEWS_TEST_BANK'])
        meta=json.loads((path/'manifest.json').read_text())
        split=json.loads((path.parents[2]/'splits'/f"{meta['dataset']}_{meta['shots']}shot_seed{meta['split_seed']}.json").read_text())
        up=path.parents[1]/'upstream'/meta['dataset']
        load=lambda p: torch.load(p,weights_only=True,map_location='cpu',mmap=True)
        raw=load(up/'pipeline/predictions.pt')
        sup=load(up/'supervision.pt')
        gold=dict(zip(sup['support_ids'].tolist(),sup['support_targets'].tolist()))
        pairs=[]
        for branch in ('agreement','disagreement'):
            b=load(path/(branch+'_train.pt'))
            self.assertEqual(b['tab'].shape[1],43 if branch=='agreement' else 70)
            self.assertTrue(torch.equal(b['weight'],torch.ones(len(b['ids']))))
            self.assertTrue(set(b['node_ids'].tolist())<=set(gold))
            pairs.extend(zip(b['environment_index'].tolist(),b['node_ids'].tolist()))
            clean=b['environment_index']==0
            self.assertTrue(torch.equal(b['gnn_pred'][clean],raw['gnn_logits'][b['node_ids'][clean]].argmax(1)))
            y=torch.tensor([gold[i] for i in b['node_ids'].tolist()])
            if branch=='agreement':self.assertTrue(torch.equal(b['target'],b['gnn_pred']==y))
            else:self.assertTrue(torch.equal(b['state'],torch.where(b['gnn_pred']==y,0,torch.where(b['llm_pred']==y,1,2))))
        self.assertEqual(len(pairs),6*len(gold))
        self.assertEqual(set(pairs),{(e,i) for e in range(6) for i in gold})
        for role,key in [('validation','valid_ids'),('query','unlabeled_ids')]:
            banks=[load(path/f'{b}_{role}.pt') for b in ('agreement','disagreement')]
            all_ids=[i for b in banks for i in b['ids'].tolist()]
            self.assertEqual(len(all_ids),len(set(all_ids)))
            self.assertEqual(set(all_ids),set(split[key]))
            for branch,b in zip(('agreement','disagreement'),banks):
                self.assertTrue(torch.equal(b['gnn_pred'],raw['gnn_logits'][b['ids']].argmax(1)))
                self.assertTrue(torch.equal(b['llm_pred'],raw['llm_logits'][b['ids']].argmax(1)))
                self.assertTrue(bool(((b['gnn_pred']==b['llm_pred'])==(branch=='agreement')).all()))
                owned=set(b['ids'].tolist())
                self.assertEqual(b['ids'].tolist(),[i for i in split[key] if i in owned])
                if role=='query':self.assertEqual(set(b),QUERY_KEYS)
        for name,h in meta['bank_hashes'].items():self.assertEqual(sha(path/name),h)

if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm'
    unittest.main()
