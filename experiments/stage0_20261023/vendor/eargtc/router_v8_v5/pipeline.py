"""Label-free Router-v8 v5 delivery. Numerical calls must run through Slurm."""
import hashlib
import json
import os
from pathlib import Path

import torch

DATASETS = ('cora', 'citeseer', 'pubmed', 'arxiv', 'ogbn-products')
COMMON = {'node_ids', 'gnn_pred', 'llm_pred', 'selected_score', 'accepted_mask', 'pseudo_label'}
DISAGREEMENT = COMMON | {'q', 'route', 'selected_action'}
AGREEMENT = COMMON | {'schema', 'dataset'}
MERGED = COMMON | {'schema', 'dataset', 'node_count', 'class_count', 'population', 'label_source'}


def require_slurm():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Router-v8 v5 numerical operations require Slurm')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def _vector(value, dtype, n, name):
    check(isinstance(value, torch.Tensor) and value.device.type == 'cpu', f'{name}: expected CPU tensor')
    check(value.dtype == dtype and value.shape == (n,), f'{name}: invalid dtype/shape')


def _common(payload, node_count, class_count):
    check(type(node_count) is int and node_count > 0, 'invalid node count')
    check(type(class_count) is int and class_count > 0, 'invalid class count')
    ids = payload['node_ids']
    check(isinstance(ids, torch.Tensor) and ids.ndim == 1, 'invalid node IDs')
    n = len(ids)
    for key in ('node_ids', 'gnn_pred', 'llm_pred', 'pseudo_label'):
        _vector(payload[key], torch.long, n, key)
    _vector(payload['accepted_mask'], torch.bool, n, 'accepted_mask')
    # Both frozen deliveries use float64; refusing casts preserves every score bit.
    _vector(payload['selected_score'], torch.float64, n, 'selected_score')
    check(bool(torch.isfinite(payload['selected_score']).all()), 'nonfinite score')
    check(ids.unique().numel() == n, 'duplicate node IDs')
    check(bool(((ids >= 0) & (ids < node_count)).all()), 'node ID out of range')
    for key in ('gnn_pred', 'llm_pred'):
        check(bool(((payload[key] >= 0) & (payload[key] < class_count)).all()), 'prediction out of class range')
    mask, label = payload['accepted_mask'], payload['pseudo_label']
    check(bool((label[~mask] == -1).all()), 'rejected label must be -1')
    check(bool(((label[mask] >= 0) & (label[mask] < class_count)).all()), 'accepted label out of range')
    return n


def merge_branches(dataset, disagreement, agreement, split_ids, node_count, class_count):
    """Join unchanged branch decisions; split_ids uses historical train/validation/query IDs."""
    require_slurm()
    check(dataset in DATASETS, 'unknown dataset')
    check(isinstance(disagreement, dict) and set(disagreement) == DISAGREEMENT, 'unsafe disagreement schema')
    check(isinstance(agreement, dict) and set(agreement) == AGREEMENT, 'unsafe agreement schema')
    check(agreement['schema'] == 'agreement_increment_pipeline_v1' and agreement['dataset'] == dataset,
          'agreement dataset/schema mismatch')
    d, a = disagreement, agreement
    nd, na = _common(d, node_count, class_count), _common(a, node_count, class_count)
    check(bool((d['gnn_pred'] != d['llm_pred']).all()), 'disagreement includes consensus')
    check(bool((a['gnn_pred'] == a['llm_pred']).all()), 'agreement includes conflict')
    for key in ('route', 'selected_action'):
        _vector(d[key], torch.long, nd, key)
    check(bool(((d['route'] == 0) | (d['route'] == 1)).all()), 'invalid disagreement route')
    q = d['q']
    check(isinstance(q, torch.Tensor) and q.device.type == 'cpu' and q.dtype == torch.float64
          and q.shape == (nd, 3), 'invalid disagreement q')
    check(bool(torch.isfinite(q).all()) and bool(((q >= 0) & (q <= 1)).all()), 'invalid q probabilities')
    check(torch.allclose(q.sum(1), torch.ones(nd, dtype=torch.float64), atol=1e-6, rtol=0), 'q does not sum to one')
    check(torch.equal(d['selected_action'], torch.where(d['accepted_mask'], d['route'], 2)), 'invalid selected action')
    routed = torch.where(d['route'] == 0, d['gnn_pred'], d['llm_pred'])
    check(torch.equal(d['pseudo_label'], torch.where(d['accepted_mask'], routed, -1)), 'disagreement label mismatch')
    check(torch.equal(a['pseudo_label'], torch.where(a['accepted_mask'], a['gnn_pred'], -1)), 'agreement label mismatch')
    check(isinstance(split_ids, dict) and all(k in split_ids for k in ('train', 'validation', 'query')), 'missing historical splits')
    split = {}
    for key in ('train', 'validation', 'query'):
        values = split_ids[key]
        check(isinstance(values, list) and all(type(v) is int for v in values), 'invalid split IDs')
        split[key] = torch.tensor(values, dtype=torch.long)
    combined = torch.cat(list(split.values()))
    check(torch.equal(combined.sort().values, torch.arange(node_count)), 'historical splits must be disjoint full-node partition')
    ids = torch.cat([d['node_ids'], a['node_ids']])
    check(ids.unique().numel() == len(ids), 'overlapping branch IDs')
    check(torch.equal(ids.sort().values, split['query'].sort().values), 'union differs from historical query')
    order = ids.argsort()
    out = {key: torch.cat([d[key], a[key]])[order] for key in COMMON}
    out.update(schema='router_v8_v5_pipeline_v1', dataset=dataset, node_count=node_count, class_count=class_count,
               population=torch.cat([torch.zeros(nd, dtype=torch.long), torch.ones(na, dtype=torch.long)])[order],
               label_source=torch.cat([torch.where(d['accepted_mask'], d['route'], -1),
                                       torch.where(a['accepted_mask'], 2, -1)])[order])
    validate_payload(out, dataset)
    # Exact source replay by ID; no fractions, rankings, or scores are recomputed.
    for branch in (d, a):
        pos = torch.searchsorted(out['node_ids'], branch['node_ids'])
        for key in COMMON:
            check(torch.equal(out[key][pos], branch[key]), f'changed branch field: {key}')
    return out


def validate_payload(payload, dataset):
    require_slurm()
    check(isinstance(payload, dict) and set(payload) == MERGED, 'unsafe v5 schema')
    check(dataset in DATASETS and payload['dataset'] == dataset and payload['schema'] == 'router_v8_v5_pipeline_v1',
          'v5 dataset/schema mismatch')
    n = _common(payload, payload['node_count'], payload['class_count'])
    ids = payload['node_ids']
    check(bool((ids[1:] > ids[:-1]).all()), 'v5 node IDs must be sorted')
    for key in ('population', 'label_source'):
        _vector(payload[key], torch.long, n, key)
    pop, source, keep = payload['population'], payload['label_source'], payload['accepted_mask']
    check(bool(((pop == 0) | (pop == 1)).all()), 'unknown population')
    check(torch.equal(pop == 1, payload['gnn_pred'] == payload['llm_pred']), 'population/expert mismatch')
    check(bool((source[~keep] == -1).all()), 'rejected source must be -1')
    check(bool((source[keep & (pop == 1)] == 2).all()), 'accepted agreement source must be consensus')
    ds = source[keep & (pop == 0)]
    check(bool(((ds == 0) | (ds == 1)).all()), 'invalid accepted disagreement source')
    routed = torch.where(source == 1, payload['llm_pred'], payload['gnn_pred'])
    check(torch.equal(payload['pseudo_label'], torch.where(keep, routed, -1)), 'v5 label/source mismatch')
    return payload


def load_pipeline(root, dataset):
    """Read a manifest-authenticated, label-free v5 query delivery."""
    require_slurm()
    check(dataset in DATASETS, 'unknown dataset')
    root = Path(root).resolve()
    complete_path, freeze_path = root / 'COMPLETE.json', root / 'FROZEN.json'
    check(complete_path.is_file() and freeze_path.is_file(), 'v5 delivery is not completed and frozen')
    complete = json.loads(complete_path.read_text())
    check(complete.get('all_checks_pass') is True and sha256(freeze_path) == complete.get('frozen_sha256'),
          'completion/freeze hash mismatch')
    frozen = json.loads(freeze_path.read_text())
    manifest_path = root / 'pipeline/MANIFEST.json'
    check(sha256(manifest_path) == frozen.get('files', {}).get(str(manifest_path)), 'frozen manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    check(manifest.get('schema') == 'router_v8_v5_manifest_v1', 'invalid manifest schema')
    item = manifest['datasets'][dataset]
    check(item['path'] == f'pipeline/{dataset}.pt', 'invalid manifest path')
    path = root / item['path']
    check(sha256(path) == item['sha256'], 'pipeline hash mismatch')
    return validate_payload(torch.load(path, map_location='cpu', weights_only=True), dataset)


def load_joint_pseudo_labels(root, dataset):
    """Accepted query IDs/labels/population only; scores are not implicit loss weights."""
    payload = load_pipeline(root, dataset)
    keep = payload['accepted_mask']
    return {key: payload[key][keep].clone() for key in ('node_ids', 'pseudo_label', 'population')}
