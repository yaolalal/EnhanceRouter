"""Frozen split identity and forbidden-label tests; execute only under Slurm."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
assert os.environ.get('SLURM_JOB_ID'), 'Numerical tests require Slurm'
try:
    upstream = importlib.import_module('scripts.router_v10_shots_v1.upstream')
except ModuleNotFoundError:
    upstream = None


class FrozenSplitTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(upstream, 'New frozen-split upstream implementation is missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = {'dataset': 'cora', 'shots': 3, 'seed': 42, 'node_count': 8,
                       'train_ids': [0, 2], 'val_ids': [1], 'test_ids': [3, 4, 5, 6, 7],
                       'evaluation_test_ids': [6, 3, 7]}
        self.source['hash'] = hashlib.md5(str([0, 2, 1, 3, 4, 5, 6, 7]).encode()).hexdigest()[:8]
        self.path = self.base / 'cora_3shot_seed42.json'
        self.manifest = self.base / 'manifest.json'
        self.manifest.write_text('{}')

    def normalize(self, source=None, corrupt_sha=False):
        source = source or self.source
        self.path.write_text(json.dumps(source))
        record = {'dataset': 'cora', 'shots': 3, 'seed': 42, 'hash': source['hash'],
                  'train_count': len(source['train_ids']), 'val_count': len(source['val_ids']),
                  'test_pool_count': len(source['test_ids']), 'evaluation_count': len(source['evaluation_test_ids']),
                  'frozen_path': str(self.path), 'frozen_sha256': upstream.sha(self.path)}
        if corrupt_sha:
            record['frozen_sha256'] = '0' * 64
        return upstream.normalize_frozen_split(self.path, record, self.manifest)

    def test_exact_partition_and_evaluation_order_survive(self):
        result = self.normalize()
        self.assertEqual(result['support_ids'], self.source['train_ids'])
        self.assertEqual(result['valid_ids'], self.source['val_ids'])
        self.assertEqual(result['unlabeled_ids'], self.source['test_ids'])
        self.assertEqual(result['evaluation_1000_ids'], [6, 3, 7])
        self.assertEqual(result['evaluation_full_ids'], self.source['test_ids'])
        self.assertEqual(result['reference_sha256'], upstream.sha(self.path))

    def test_rejects_manifest_digest_mismatch(self):
        with self.assertRaises(ValueError):
            self.normalize(corrupt_sha=True)

    def test_rejects_forbidden_partition_overlap(self):
        for key, replacement in [('val_ids', [0]), ('test_ids', [1, 4, 5, 6, 7])]:
            bad = copy.deepcopy(self.source)
            bad[key] = replacement
            with self.assertRaises(ValueError):
                self.normalize(bad)

    def test_rejects_evaluation_outside_query_or_duplicates(self):
        for ids in ([0, 3, 7], [3, 3, 7]):
            bad = copy.deepcopy(self.source)
            bad['evaluation_test_ids'] = ids
            with self.assertRaises(ValueError):
                self.normalize(bad)

    def test_rejects_split_hash_mismatch(self):
        bad = copy.deepcopy(self.source)
        bad['hash'] = '00000000'
        with self.assertRaises(ValueError):
            self.normalize(bad)

    def test_support_counts_allow_sparse_or_absent_classes(self):
        import torch
        self.assertTrue(hasattr(upstream, 'validate_support_counts'), 'Sparse support validator missing')
        counts = upstream.validate_support_counts(torch.tensor([0, 0, 1]), num_classes=3, shot=3)
        self.assertEqual(counts.tolist(), [2, 1, 0])
        with self.assertRaises(ValueError):
            upstream.validate_support_counts(torch.tensor([0, 0, 0, 0]), num_classes=3, shot=3)

    def test_supervision_contains_only_gold_and_validation(self):
        import torch
        labels = torch.tensor([0, 1, 2, 2, 1, 0, 2, 1])
        train, valid, query = (torch.tensor(self.source[k]) for k in ('train_ids', 'val_ids', 'test_ids'))
        poisoned, supervision = upstream.poison_query_labels(labels, train, valid, query)
        self.assertEqual(poisoned.tolist(), [0, 1, 2, -1, -1, -1, -1, -1])
        self.assertEqual(set(supervision), {'support_ids', 'support_targets', 'valid_ids', 'valid_targets'})
        self.assertEqual(supervision['support_targets'].tolist(), [0, 2])
        self.assertEqual(supervision['valid_targets'].tolist(), [1])
        labels[query] = 999
        again, same = upstream.poison_query_labels(labels, train, valid, query)
        self.assertTrue(torch.equal(again, poisoned))
        self.assertTrue(all(torch.equal(same[k], supervision[k]) for k in supervision))


if __name__ == '__main__':
    unittest.main()
