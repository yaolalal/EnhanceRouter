"""Strict, label-isolated interfaces for cached joint repair experiments."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import torch

DATASETS = ('cora', 'citeseer', 'pubmed', 'arxiv', 'ogbn-products')
FEATURE_KEYS = {'node_ids', 'gnn_embeddings', 'logp_g', 'logp_l', 'choose_g', 'agreement', 'selected_pred', 'trust'}
SUPERVISION_KEYS = {'gold_ids', 'gold_targets', 'selection_ids', 'selection_targets', 'threshold_ids', 'test_ids'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(records):
    for path, expected in records.items():
        if sha256(path) != expected:
            raise ValueError(f'input hash mismatch: {path}')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def save_tensor(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    torch.save(value, tmp)
    os.replace(tmp, path)


def id_hash(ids):
    return hashlib.sha256(torch.as_tensor(ids, dtype=torch.long).numpy().tobytes()).hexdigest()


def _ids(ids, n, name):
    if not torch.is_tensor(ids) or ids.dtype != torch.long or ids.ndim != 1:
        raise ValueError(f'{name} must be an int64 vector')
    if ids.unique().numel() != ids.numel() or bool(((ids < 0) | (ids >= n)).any()):
        raise ValueError(f'{name} duplicate/out-of-range IDs')


def validate_features(data):
    if set(data) != FEATURE_KEYS:
        raise ValueError('feature schema forbids labels and unknown fields')
    n = data['node_ids'].numel()
    if n == 0 or not torch.equal(data['node_ids'], torch.arange(n)):
        raise ValueError('features must cover all node IDs in order')
    for key in ['logp_g', 'logp_l', 'gnn_embeddings']:
        value = data[key]
        if value.ndim != 2 or value.shape[0] != n or value.shape[1] < 1 or not bool(torch.isfinite(value).all()):
            raise ValueError(f'invalid {key}')
    if data['logp_g'].shape != data['logp_l'].shape:
        raise ValueError('expert class spaces differ')
    for key in ['choose_g', 'agreement', 'selected_pred', 'trust']:
        if data[key].shape != (n,):
            raise ValueError(f'invalid {key} shape')
    if data['choose_g'].dtype != torch.bool or data['agreement'].dtype != torch.bool:
        raise ValueError('route/agreement must be Boolean')
    if not bool(torch.isfinite(data['trust']).all()) or bool(((data['trust'] < 0) | (data['trust'] > 1)).any()):
        raise ValueError('trust must be a finite probability estimate')
    g, l = data['logp_g'].argmax(1), data['logp_l'].argmax(1)
    if not torch.equal(g == l, data['agreement']):
        raise ValueError('agreement differs from expert predictions')
    if not torch.equal(torch.where(data['choose_g'], g, l), data['selected_pred']):
        raise ValueError('selected prediction differs from frozen route')
    for key in ['logp_g', 'logp_l']:
        if not torch.allclose(data[key].logsumexp(1), torch.zeros(n), atol=2e-5):
            raise ValueError('base log probabilities must be normalized')


def validate_supervision(data, n, c):
    if set(data) != SUPERVISION_KEYS:
        raise ValueError('supervision schema forbids test targets and unknown fields')
    owner = torch.zeros(n, dtype=torch.int8)
    for key in ['gold_ids', 'selection_ids', 'threshold_ids', 'test_ids']:
        ids = data[key]
        _ids(ids, n, key)
        if bool(owner[ids].any()):
            raise ValueError('gold/validation/test split overlap')
        owner[ids] = 1
    if not bool(owner.all()):
        raise ValueError('splits do not partition the node universe')
    for prefix in ['gold', 'selection']:
        y, ids = data[prefix + '_targets'], data[prefix + '_ids']
        if y.dtype != torch.long or y.shape != ids.shape or y.numel() == 0:
            raise ValueError(f'invalid {prefix} supervision')
        if bool(((y < 0) | (y >= c)).any()):
            raise ValueError('supervised class outside class space')


def validate_hidden(data, n, expected_width=4096):
    if set(data) != {'node_ids', 'hidden', 'metadata'}:
        raise ValueError('raw hidden schema forbids extra labels')
    if not torch.equal(data['node_ids'], torch.arange(n)):
        raise ValueError('raw hidden node IDs do not align')
    h = data['hidden']
    if h.shape != (n, expected_width):
        raise ValueError('raw hidden width/count mismatch')
    for start in range(0, n, 4096):
        if not bool(torch.isfinite(h[start:start + 4096]).all()):
            raise ValueError('nonfinite raw hidden')
    from eargtc.router_v7_decision.audit import assert_label_free_cache
    assert_label_free_cache(data)


def validate_pools(pools, features, supervision):
    required = {'teach_g', 'teach_l', 'keep_g', 'keep_l', 'agreement', 'unselected'}
    if set(pools) != required:
        raise ValueError('invalid pool schema')
    n = len(features['node_ids'])
    test = supervision['test_ids']
    for name, ids in pools.items():
        _ids(ids, n, name)
        if not bool(torch.isin(ids, test).all()):
            raise ValueError('pool contains gold/validation nodes')
    same, route = features['agreement'], features['choose_g']
    if not bool((~same[pools['teach_g']] & ~route[pools['teach_g']]).all()):
        raise ValueError('GNN repair must be taught by selected LLM conflicts')
    if not bool((~same[pools['teach_l']] & route[pools['teach_l']]).all()):
        raise ValueError('text repair must be taught by selected GNN conflicts')
    if not bool(same[pools['agreement']].all()):
        raise ValueError('agreement pool contains conflicts')
    for branch, other in [('g','l'),('l','g')]:
        expected = torch.cat([pools['teach_'+other], pools['agreement']]).unique().sort().values
        if not torch.equal(pools['keep_'+branch].sort().values, expected):
            raise ValueError('retention must protect only selected teacher and agreement')
    selected = torch.cat([pools['teach_g'], pools['teach_l'], pools['agreement']])
    expected_unselected = test[~torch.isin(test,selected)].sort().values
    if not torch.equal(pools['unselected'].sort().values,expected_unselected):
        raise ValueError('unselected pool does not complement all selected nodes')


def load_export(path, verify_sources=True):
    path = Path(path)
    complete = read_json(path / 'complete.json')
    if complete.get('schema_version') != 1 or not complete.get('test_labels_exported') is False:
        raise ValueError('invalid safe export completion')
    verify_files({str(path / k): v for k, v in complete['artifacts'].items()})
    meta = read_json(path / 'metadata.json')
    if verify_sources:
        verify_files(meta['source_hashes'])
    features = torch.load(path / 'features.pt', map_location='cpu', weights_only=True, mmap=True)
    supervision = torch.load(path / 'supervision.pt', map_location='cpu', weights_only=True)
    pools = torch.load(path / 'pools.pt', map_location='cpu', weights_only=True)
    hidden = torch.load(meta['llm_hidden_path'], map_location='cpu', weights_only=True, mmap=True)
    validate_features(features)
    validate_supervision(supervision, features['node_ids'].numel(), features['logp_g'].shape[1])
    validate_hidden(hidden, features['node_ids'].numel(), meta['llm_width'])
    expected_pool_keys = {'teach_g', 'teach_l', 'keep_g', 'keep_l', 'agreement', 'unselected'}
    if set(pools) != expected_pool_keys:
        raise ValueError('unexpected pool schema')
    test = supervision['test_ids']
    for name, ids in pools.items():
        _ids(ids, features['node_ids'].numel(), name)
        if not bool(torch.isin(ids, test).all()):
            raise ValueError('pool contains gold/validation nodes')
    validate_pools(pools, features, supervision)
    candidate = torch.tensor(meta['candidate_ids'], dtype=torch.long)
    _ids(candidate, features['node_ids'].numel(), 'candidates')
    if not bool(torch.isin(candidate, test).all()):
        raise ValueError('candidate IDs outside test')
    from .policy import build_pools
    expected = build_pools(test, features['agreement'], features['choose_g'], features['trust'],
                           read_json(path / 'policies.json'), candidate_ids=candidate)
    if any(not torch.equal(pools[k],expected[k]) for k in pools):
        raise ValueError('saved pools do not reproduce frozen top-k policy')
    return features, supervision, pools, hidden['hidden'], meta


def verify_training_link(run_dir, export_dir):
    run_dir, export_dir = Path(run_dir).resolve(), Path(export_dir).resolve()
    complete = read_json(run_dir / 'complete.json')
    if not complete.get('all_checks_pass') or complete.get('test_labels_read') is not False:
        raise ValueError('training did not complete with isolated test labels')
    verify_files({str(run_dir / k): v for k, v in complete['artifacts'].items()})
    verify_files(complete['source_code'])
    config = read_json(run_dir / 'config.json')
    if Path(config['export_dir']).resolve() != export_dir or config['export_complete_sha256'] != sha256(export_dir / 'complete.json'):
        raise ValueError('run is bound to a different export version')
    dataset = read_json(export_dir / 'metadata.json')['dataset']
    if config.get('dataset') != dataset or complete.get('dataset') != dataset:
        raise ValueError('training/export dataset identity differs')
    return config
