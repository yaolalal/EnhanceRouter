"""Label-isolated selector adapter; historical router features remain unchanged."""
from __future__ import annotations

import os
import time
from fractions import Fraction
from math import ceil
from pathlib import Path

import numpy as np
import torch

from eargtc.joint_five_router import adapter as old
from eargtc.joint_cross_repair.artifacts import (
    DATASETS, id_hash, read_json, save_tensor, sha256, validate_pools, write_json,
)
from eargtc.router_fiveway_topk.policy import rank_exact

ROOT = old.ROOT
ROUTERS = old.ROUTERS
STRATEGIES = ('full', 'random', 'reliability', 'reliability_cluster')
SOURCE_ROOT = ROOT / 'runs/router_fiveway_node_selection_20260907_v2'
BASE_EXPORT = old.DEFAULT_EXPORT
DEFAULT_EXPORT = ROOT / 'runs/joint_selector_export_20260907_v1'
_DECISION_DTYPES = {'node_ids': np.int64, 'selected_pred': np.int64, 'accepted': np.bool_,
                    'score': np.float32, 'module_code': np.int8, 'candidates': np.bool_}
_PSEUDO_DTYPES = {'node_ids': np.int64, 'pseudo_labels': np.int64,
                  'router_score': np.float32, 'module_code': np.int8}
_VERIFIED = {}


def _stamp(path):
    st = Path(path).stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def verify_cached(records):
    """Hash each immutable file once; any identity/size/time change forces rehash."""
    for filename, digest in records.items():
        path = str(Path(filename).resolve())
        before = _stamp(path)
        key = (path, digest)
        cached = _VERIFIED.get(key)
        # Some filesystems coalesce sub-second timestamps. Never memoize a
        # freshly written file until its timestamp window is safely past.
        if cached is not None and cached[0] == before and max(before[3:]) < cached[1] - 1_000_000_000:
            continue
        if sha256(path) != digest or _stamp(path) != before:
            raise ValueError(f'input hash mismatch: {path}')
        _VERIFIED[key] = (before, time.time_ns())


def _arrays(data, schema, name):
    if set(data) != set(schema):
        raise ValueError(f'invalid {name} schema')
    for key, dtype in schema.items():
        value = data[key]
        if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype != dtype:
            raise ValueError(f'invalid {name} {key} dtype or shape')
        if not np.isfinite(value).all():
            raise ValueError(f'nonfinite {name} {key}')


def validate_selection(features, supervision, old_inputs, decision, candidate_mask, policies, pseudo):
    """Validate frozen subset routing and return its exact Joint supervision pools."""
    _arrays(decision, _DECISION_DTYPES, 'decision')
    _arrays(pseudo, _PSEUDO_DTYPES, 'pseudo')
    test = supervision['test_ids']
    ids = test.numpy()
    if not np.array_equal(decision['node_ids'], ids):
        raise ValueError('decision does not match exact original test IDs')
    if any(v.shape != ids.shape for v in decision.values()):
        raise ValueError('misaligned decision shape')
    if candidate_mask.dtype != np.bool_ or not np.array_equal(decision['candidates'], candidate_mask):
        raise ValueError('decision differs from shared candidate mask')
    accepted = decision['accepted']
    if np.any(accepted & ~candidate_mask):
        raise ValueError('accepted nodes outside candidates')
    for name, expected in [('selected_pred', features['selected_pred'][test].numpy()),
                           ('score', features['trust'][test].numpy()),
                           ('module_code', old_inputs['module_code'][test].numpy())]:
        if not np.array_equal(decision[name], expected):
            label = 'prediction' if name == 'selected_pred' else name
            raise ValueError(f'frozen {label} identity mismatch')
    codes = decision['module_code']
    expected_names = {'global'} if np.all(codes == 0) else {'agreement', 'conflict'}
    if set(policies) != expected_names:
        raise ValueError('frozen policy module mismatch')
    replayed = np.zeros(ids.size, dtype=bool)
    for module in expected_names:
        mask = candidate_mask & (codes == old._MODULE_CODES[module])
        fraction = policies[module]['k_fraction']
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
            raise ValueError('invalid frozen k fraction')
        if mask.any():
            replayed[mask] = rank_exact(ids[mask], features['trust'][test].numpy()[mask], float(fraction))
    if not np.array_equal(accepted, replayed):
        raise ValueError('accepted mask does not replay frozen top-k within candidates')
    for key, source in [('node_ids', 'node_ids'), ('pseudo_labels', 'selected_pred'),
                        ('router_score', 'score'), ('module_code', 'module_code')]:
        if not np.array_equal(pseudo[key], decision[source][accepted]):
            raise ValueError(f'pseudo {key} alignment mismatch')
    _, pools = old.build_router_features_and_pools(features, test,
        {key: torch.from_numpy(decision[key]) for key in ('node_ids', 'accepted', 'selected_pred', 'module_code')},
        features['choose_g'], features['trust'])
    validate_pools(pools, features, supervision)
    return pools


def _read_npz(path, schema):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(schema):
            raise ValueError(f'invalid NPZ schema: {path}')
        return {key: data[key] for key in schema}


def _source_records():
    frozen_path = SOURCE_ROOT / 'FROZEN.json'
    digest = (SOURCE_ROOT / 'FROZEN.sha256').read_text().strip()
    verify_cached({str(frozen_path): digest})
    frozen = read_json(frozen_path)
    if frozen.get('status') != 'frozen_label_free_decisions':
        raise ValueError('source selection is not frozen')
    expected = {f'candidates/{d}.npz' for d in DATASETS} | {
        f'decisions/{d}/{s}/{r}.npz' for d in DATASETS for s in STRATEGIES for r in ROUTERS}
    if set(frozen['decision_hashes']) != expected:
        raise ValueError('source selection does not contain all 100 decisions')
    records = {str((SOURCE_ROOT / key).resolve()): value for key, value in frozen['decision_hashes'].items()}
    records.update(frozen['input_hashes'])
    records.update(frozen['code_hashes'])
    records[str(frozen_path.resolve())] = digest
    records[str((SOURCE_ROOT / 'FROZEN.sha256').resolve())] = sha256(SOURCE_ROOT / 'FROZEN.sha256')
    records[str((SOURCE_ROOT / 'CONFIG.json').resolve())] = frozen['config_sha256']
    verify_cached(records)
    config = read_json(SOURCE_ROOT / 'CONFIG.json')
    if (set(config['strategies']) != set(STRATEGIES) or set(config['datasets']) != set(DATASETS)
            or set(config['router_names']) != set(ROUTERS) or config['candidate_fraction'] != .2):
        raise ValueError('source selector configuration drift')
    upstream = Path(config['source_root'])
    if upstream.resolve() != old.SOURCE_ROOT.resolve():
        raise ValueError('source upstream differs from historical router')
    records[str(upstream / 'selection/FROZEN.json')] = config['source_policy_sha256']
    records[str(upstream / 'test/test_inputs.pt')] = config['source_inputs_sha256']
    if config['source_inputs_sha256'] != frozen['source_inputs_sha256']:
        raise ValueError('source input provenance mismatch')
    verify_cached(records)
    # These exports were generated after source FROZEN. Verify exact contents on
    # every preparation/load and additionally bind their bytes into our export.
    for ds in DATASETS:
        for strategy in STRATEGIES:
            for rid in ROUTERS:
                path = SOURCE_ROOT / 'pseudo_labels' / ds / strategy / f'{rid}.npz'
                records[str(path.resolve())] = sha256(path)
    policies = {ds: {rid: {name: {'k_fraction': value['k_fraction']}
                for name, value in config['frozen_k'][ds][rid].items()} for rid in ROUTERS} for ds in DATASETS}
    return records, policies


def _old_cell(router_id, dataset, verify_sources=True):
    path = BASE_EXPORT / router_id / dataset
    meta = read_json(path / 'metadata.json')
    base_path = Path(meta['base_export_dir'])
    base_meta = read_json(base_path / 'metadata.json')
    if verify_sources:
        verify_cached(meta['source_hashes'])
        verify_cached(base_meta['source_hashes'])
    # Reuse all existing strict schema, split, route and pool checks. Upstream
    # hashes have just been verified through the stat-sensitive process cache.
    result = old.load_cell(BASE_EXPORT, router_id, dataset, verify_sources=False)
    inputs = torch.load(path / 'inputs.pt', map_location='cpu', weights_only=True)
    return result, inputs


def _source_cell(strategy, router_id, dataset, features, supervision, old_inputs, policies):
    path = SOURCE_ROOT / 'candidates' / f'{dataset}.npz'
    with np.load(path, allow_pickle=False) as data:
        if not np.array_equal(data['node_ids'], supervision['test_ids'].numpy()):
            raise ValueError('shared candidates do not match exact original test IDs')
        candidate = data[strategy]
    expected_count = len(candidate) if strategy == 'full' else ceil(Fraction('0.2') * len(candidate))
    if candidate.dtype != np.bool_ or candidate.ndim != 1 or int(candidate.sum()) != expected_count:
        raise ValueError('shared candidate budget mismatch')
    decision = _read_npz(SOURCE_ROOT / 'decisions' / dataset / strategy / f'{router_id}.npz', _DECISION_DTYPES)
    pseudo = _read_npz(SOURCE_ROOT / 'pseudo_labels' / dataset / strategy / f'{router_id}.npz', _PSEUDO_DTYPES)
    pools = validate_selection(features, supervision, old_inputs, decision, candidate, policies, pseudo)
    if strategy == 'full' and not np.array_equal(decision['node_ids'][decision['accepted']], old_inputs['accepted_ids'].numpy()):
        raise ValueError('full strategy differs from historical accepted pool')
    return decision, pools


def prepare_all(export_root=DEFAULT_EXPORT):
    """Prepare all 100 cells before publishing the immutable completion barrier."""
    root = Path(export_root).resolve()
    if root.exists():
        raise FileExistsError(f'refusing to overwrite export root: {root}')
    records, policies = _source_records()
    root.mkdir(parents=True)
    metas = {}
    for rid in ROUTERS:
        for ds in DATASETS:
            (f, s, _, h, old_meta), old_inputs = _old_cell(rid, ds)
            for strategy in STRATEGIES:
                decision, pools = _source_cell(strategy, rid, ds, f, s, old_inputs, policies[ds][rid])
                key = f'{strategy}/{rid}/{ds}'
                dest = root / key
                dest.parent.mkdir(parents=True, exist_ok=True)
                stage = dest.with_name(dest.name + f'.staging.{os.getpid()}')
                stage.mkdir()
                inputs = {'node_ids': f['node_ids'], 'test_ids': s['test_ids'],
                          'candidate_ids': s['test_ids'][torch.from_numpy(decision['candidates'])],
                          'accepted_ids': s['test_ids'][torch.from_numpy(decision['accepted'])]}
                meta = {**old_meta, 'strategy': strategy, 'source_hashes': records,
                        'old_cell_dir': str((BASE_EXPORT / rid / ds).resolve()),
                        'old_cell_complete_sha256': sha256(BASE_EXPORT / rid / ds / 'complete.json'),
                        'old_root_complete_sha256': sha256(BASE_EXPORT / 'COMPLETE.json'),
                        'old_source_hashes': old_meta['source_hashes'],
                        'candidate_count': len(inputs['candidate_ids']), 'accepted_count': len(inputs['accepted_ids']),
                        'test_ids_sha256': id_hash(s['test_ids']),
                        'candidate_ids_sha256': id_hash(inputs['candidate_ids']),
                        'pool_counts': {k: len(v) for k, v in pools.items()}, 'policies': policies[ds][rid],
                        'control_note': 'All historical features, route, selected predictions and C trust remain unchanged.'}
                save_tensor(stage / 'inputs.pt', inputs)
                write_json(stage / 'metadata.json', meta)
                write_json(stage / 'complete.json', {'schema_version': 1, 'strategy': strategy,
                    'router_id': rid, 'dataset': ds, 'test_labels_exported': False,
                    'artifacts': {name: sha256(stage / name) for name in ('inputs.pt', 'metadata.json')}})
                os.rename(stage, dest)
                metas[key] = meta
            del f, s, h, old_inputs
    write_json(root / 'COMPLETE.json', {'schema_version': 1, 'cell_count': 100, 'test_labels_exported': False,
        'source_frozen_sha256': sha256(SOURCE_ROOT / 'FROZEN.json'),
        'cells': {key: sha256(root / key / 'complete.json') for key in metas}})
    return metas


def load_cell(export_root, strategy, router_id, dataset, verify_sources=True):
    """Return unchanged historical inputs and the validated selector pseudo pools."""
    if strategy not in STRATEGIES or router_id not in ROUTERS or dataset not in DATASETS:
        raise ValueError('unknown strategy, router or dataset')
    root = Path(export_root).resolve()
    barrier = read_json(root / 'COMPLETE.json')
    expected = {f'{s}/{r}/{d}' for s in STRATEGIES for r in ROUTERS for d in DATASETS}
    if (barrier.get('schema_version') != 1 or barrier.get('cell_count') != 100
            or set(barrier.get('cells', {})) != expected or barrier.get('test_labels_exported') is not False):
        raise ValueError('export does not contain all 100 frozen selector cells')
    key = f'{strategy}/{router_id}/{dataset}'
    path = root / key
    verify_cached({str(path / 'complete.json'): barrier['cells'][key]})
    complete = read_json(path / 'complete.json')
    if (complete.get('schema_version') != 1 or complete.get('test_labels_exported') is not False
            or set(complete.get('artifacts', {})) != {'inputs.pt', 'metadata.json'}
            or any(complete.get(k) != v for k, v in [('strategy', strategy), ('router_id', router_id), ('dataset', dataset)])):
        raise ValueError('invalid selector completion schema or identity')
    verify_cached({str(path / k): v for k, v in complete['artifacts'].items()})
    meta = read_json(path / 'metadata.json')
    if any(meta.get(k) != v for k, v in [('strategy', strategy), ('router_id', router_id), ('dataset', dataset)]):
        raise ValueError('selector metadata identity mismatch')
    if Path(meta['old_cell_dir']).resolve() != (BASE_EXPORT / router_id / dataset).resolve():
        raise ValueError('historical cell identity mismatch')
    verify_cached({str(BASE_EXPORT / 'COMPLETE.json'): meta['old_root_complete_sha256'],
                   str(BASE_EXPORT / router_id / dataset / 'complete.json'): meta['old_cell_complete_sha256']})
    if verify_sources:
        verify_cached(meta['source_hashes'])
        verify_cached(meta['old_source_hashes'])
    (f, s, _, h, old_meta), old_inputs = _old_cell(router_id, dataset, verify_sources)
    for name in ('base_export_dir', 'base_export_complete_sha256', 'base_feature_node_ids_sha256', 'split_counts'):
        if meta[name] != old_meta[name]:
            raise ValueError('historical base provenance mismatch')
    decision, pools = _source_cell(strategy, router_id, dataset, f, s, old_inputs, meta['policies'])
    inputs = torch.load(path / 'inputs.pt', map_location='cpu', weights_only=True)
    expected_inputs = {'node_ids': f['node_ids'], 'test_ids': s['test_ids'],
                       'candidate_ids': s['test_ids'][torch.from_numpy(decision['candidates'])],
                       'accepted_ids': s['test_ids'][torch.from_numpy(decision['accepted'])]}
    if set(inputs) != set(expected_inputs) or any(not torch.equal(inputs[k], v) for k, v in expected_inputs.items()):
        raise ValueError('selector inputs do not reproduce frozen source')
    if (meta['candidate_count'] != len(inputs['candidate_ids']) or meta['accepted_count'] != len(inputs['accepted_ids'])
            or meta['pool_counts'] != {k: len(v) for k, v in pools.items()}
            or meta['test_ids_sha256'] != id_hash(s['test_ids']) or meta['candidate_ids_sha256'] != id_hash(inputs['candidate_ids'])):
        raise ValueError('selector metadata count or ID mismatch')
    return f, s, pools, h, meta
