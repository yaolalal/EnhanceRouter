import os
import pickle
import unittest

import numpy as np

from eargtc.router_v10_tune import engine


DATASETS = ('cora', 'citeseer', 'pubmed', 'arxiv', 'ogbn-products')


def banks(split='train'):
    result = {}
    for di, dataset in enumerate(DATASETS):
        n = 24
        rng = np.random.default_rng(100 + di)
        tab = rng.normal(size=(n, 70))
        # Four original examples, each with exactly six equal-weight views.
        parent = np.repeat(np.arange(4) + 1000 * di, 6)
        state = np.tile(np.array([0, 1, 2, 0, 1, 2]), 4)
        target = np.tile(np.array([0, 1]), n // 2)
        result[dataset] = dict(
            dataset=dataset, split=split,
            ids=np.arange(n) + 10000 * di,
            node_ids=parent, weight=np.ones(n),
            tab=tab, g=rng.normal(size=(n, 11)),
            l=rng.normal(size=(n, 13)), state=state, target=target,
        )
    return result


def base(model):
    return {'engine': 'tune', 'model': {'config': model}}


class GridTests(unittest.TestCase):
    def test_each_stage_is_bounded_json_unique_and_keeps_single_factor_moves(self):
        seeds = {
            'lr': dict(family='lr', mode='context', scope='pooled', rank=False,
                       C=.3, seed=42, target_mass=.25, source_profile='uniform'),
            'trees': dict(family='trees', mode='context', scope='pooled', rank=False,
                          depth=8, leaf=12, n_estimators=256, seed=42,
                          target_mass=.25, source_profile='uniform'),
            'svc': dict(family='svc', mode='context', scope='pooled', rank=False,
                        C=1., gamma=.1, seed=42, target_mass=.25,
                        source_profile='uniform'),
            'neural': dict(family='neural', mode='context', scope='transfer', rank=False,
                           width=32, steps=280, adapt_steps=60, weight_decay=.05,
                           dropout=.15, lr=.002, seeds=[42, 137, 2026],
                           target_mass=.25, source_profile='uniform'),
            'joint_shared': dict(family='joint_shared', mode='context', scope='transfer',
                                 rank=False, width=32, steps=280, adapt_steps=60,
                                 weight_decay=.05, dropout=.15, lr=.002,
                                 seeds=[42, 137, 2026], target_mass=.25,
                                 source_profile='uniform', preference_loss_weight=1.,
                                 trust_loss_weight=1.),
        }
        import json
        for family, seed in seeds.items():
            for stage in ('regularization', 'transfer', 'training'):
                rows = engine.candidate_grid('pubmed', 'trust', stage, base(seed))
                self.assertLessEqual(len(rows), 60)
                self.assertEqual(len({r['id'] for r in rows}), len(rows))
                for row in rows:
                    json.dumps(row)
                    self.assertEqual(row['stage'], stage)

    def test_v6_nonrepresentable_and_cora_agreement_get_local_lr_seeds(self):
        wrapped = {'engine': 'v6', 'model': {'kind': 'agreement_baseline'}}
        rows = engine.candidate_grid('cora', 'agreement', 'regularization', wrapped)
        self.assertTrue(rows)
        self.assertTrue(all(r['family'] == 'lr' for r in rows))
        self.assertEqual({'compact', 'context', 'embed'}, {r['mode'] for r in rows})


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.banks = banks()

    def test_rejects_validation_and_unknown_schema(self):
        cfg = dict(family='lr', mode='context', scope='pooled', rank=False,
                   C=.3, seed=42, target_mass=.25, source_profile='uniform')
        bad = banks(split='validation')
        with self.assertRaises(ValueError):
            engine.fit_candidate(cfg, 'trust', bad, 'pubmed')
        bad = banks()
        del bad['cora']['node_ids']
        with self.assertRaises(ValueError):
            engine.fit_candidate(cfg, 'trust', bad, 'pubmed')

    def test_equal_view_weights_preserve_requested_domain_mass_and_parent_audit(self):
        cfg = dict(family='lr', mode='context', scope='pooled', rank=False,
                   C=.3, seed=42, target_mass=.25, source_profile='small_graphs')
        model = engine.fit_candidate(cfg, 'preference', self.banks, 'pubmed')
        audit = model['fit_audit']
        self.assertAlmostEqual(sum(audit['effective_domain_masses'].values()), 1.)
        self.assertAlmostEqual(audit['effective_domain_masses']['pubmed'], .25)
        self.assertGreater(audit['effective_domain_masses']['cora'],
                           audit['effective_domain_masses']['arxiv'])
        self.assertEqual(audit['transform_fit_parent_ids']['cora'],
                         np.asarray(self.banks['cora']['node_ids'])[self.banks['cora']['state'] != 2].tolist())
        self.assertEqual(audit['normalization_fit_parent_ids']['cora'],
                         np.asarray(self.banks['cora']['node_ids'])[self.banks['cora']['state'] != 2].tolist())
        self.assertEqual(audit['view_weights'], 'all unit within each dataset mass')
        self.assertEqual(audit['validation_training_rows'], 0)
        self.assertEqual(audit['query_training_rows'], 0)

    def test_lr_is_deterministic_and_pickle_roundtrip_predicts_identically(self):
        cfg = dict(family='lr', mode='context', scope='pooled', rank=False,
                   C=.3, seed=42, target_mass=.25, source_profile='uniform')
        one = engine.fit_candidate(cfg, 'trust', self.banks, 'pubmed')
        two = engine.fit_candidate(cfg, 'trust', self.banks, 'pubmed')
        z1 = engine.predict_candidate(one, self.banks['cora'])
        z2 = engine.predict_candidate(two, self.banks['cora'])
        np.testing.assert_array_equal(z1, z2)
        restored = pickle.loads(pickle.dumps(one))
        np.testing.assert_array_equal(z1, engine.predict_candidate(restored, self.banks['cora']))

    def test_small_neural_fit_returns_one_logit_per_row(self):
        cfg = dict(family='neural', mode='compact', scope='transfer', rank=False,
                   width=32, steps=1, adapt_steps=1, weight_decay=.01, dropout=0.,
                   lr=.002, seeds=[42, 137, 2026], target_mass=.25,
                   source_profile='uniform')
        model = engine.fit_candidate(cfg, 'trust', self.banks, 'pubmed')
        self.assertEqual(engine.predict_candidate(model, self.banks['cora']).shape, (24,))


    def test_neither_rows_have_zero_preference_logit_gradient(self):
        import torch
        logits = torch.tensor([[.2, .4], [.1, -.3], [-.2, .7]], requires_grad=True)
        state = torch.tensor([0, 1, 2])
        loss = engine.joint_loss(logits, state, preference_loss_weight=1., trust_loss_weight=0.)
        loss.backward()
        self.assertEqual(float(logits.grad[2, 1]), 0.)
        self.assertNotEqual(float(logits.grad[0, 1]), 0.)

    def test_small_joint_fit_is_deterministic_across_separate_heads(self):
        self.banks['cora']['state'][:12] = 2
        cfg = dict(family='joint_shared', mode='compact', scope='transfer', rank=False,
                   width=32, steps=1, adapt_steps=1, weight_decay=.01, dropout=0.,
                   lr=.002, seeds=[42, 137, 2026], target_mass=.25,
                   source_profile='uniform', preference_loss_weight=.5,
                   trust_loss_weight=1.)
        trust = engine.fit_candidate(cfg, 'trust', self.banks, 'pubmed')
        preference = engine.fit_candidate(cfg, 'preference', self.banks, 'pubmed')
        self.assertEqual(trust['shared_pair_id'], preference['shared_pair_id'])
        for a, b in zip(trust['estimator']['states'], preference['estimator']['states']):
            for key in a:
                self.assertTrue(np.array_equal(a[key].numpy(), b[key].numpy()))
        np.testing.assert_array_equal(
            engine.predict_candidate(trust, self.banks['cora']),
            engine.predict_candidate(engine.fit_candidate(cfg, 'trust', self.banks, 'pubmed'),
                                     self.banks['cora']))


if __name__ == '__main__':
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run numerical engine tests through Slurm')
    unittest.main(verbosity=2)
