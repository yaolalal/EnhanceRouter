"""Immutable per-split Fixed V2 training; test truth is opened only after all cells freeze."""
from dataclasses import asdict, replace
from pathlib import Path
import csv
import fcntl
import itertools
import hashlib
import json
import os
import statistics
import time

import torch
from eargtc.joint_cross_repair.artifacts import (
    DATASETS, read_json, write_json, save_tensor, sha256, verify_files,
    validate_features, validate_supervision, validate_pools, validate_hidden,
)
from eargtc.joint_cross_repair.training import TrainConfig, _score, blend, scheme_b_probability
from eargtc.joint_cross_repair_search.bank import RECIPES, SEEDS, Recipe, train_member, member_logits
from eargtc.joint_cross_repair_search.experiment import candidate_bank
from eargtc.joint_cross_repair_search.selection import residual_probability
from eargtc.joint_five_router.experiment import select_candidates, _check_member
from eargtc.joint_five_router.adapter import build_router_features_and_pools
from eargtc.router_bc_split_c.calibration import deterministic_calibration_threshold_split

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / 'runs/multiseed_reproduction_20260909_v1'
ROUTERS = ('r6_legacy_c', 'r6_split_c', 'a20ms_split_c', 'r8_disagreement', 'r8_full')
SPLIT_SEEDS = (42, 43, 44, 45, 46)
REQUIRED = {'predictions.pt', 'validation.pt', 'selection.json', 'metadata.json', 'pools.pt', 'supervision.pt', 'candidate_catalog.json'}


def slurm_only():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('All numerical execution, inspection and tests require Slurm')


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'immutable artifact differs: {path}')
    else:
        write_json(path, value)


def cell_path(root, seed, rid, ds):
    return Path(root) / f'seed{seed}' / 'joint' / rid / ds


def config_path(root):
    return Path(root) / 'joint' / 'CONFIG.json'


def source_records():
    paths = [ROOT / 'scripts/multiseed_reproduction/joint.py', ROOT / 'eargtc/joint_v8/evaluation.py', ROOT / 'eargtc/joint_selector/evaluation.py']
    for folder in ('eargtc/multiseed_joint', 'eargtc/joint_cross_repair',
                   'eargtc/joint_cross_repair_search', 'eargtc/joint_five_router',
                   'eargtc/router_bc_split_c'):
        paths.extend(sorted((ROOT / folder).glob('*.py')))
    return {str(path): sha256(path) for path in paths}


def initialize(root=DEFAULT_RUN, smoke=False):
    slurm_only()
    root = Path(root).resolve()
    cfg = TrainConfig(threads=2, validate_every=5 if smoke else 50,
                      max_steps=10 if smoke else 500, patience=2 if smoke else 5)
    snapshot_path = ROOT / 'artifacts/preserved_versions/joint_v8_fixed_v2_20260908/SNAPSHOT.json'
    snapshot = read_json(snapshot_path)
    verify_files({str(ROOT / name): digest for name, digest in snapshot['files'].items()})
    config = {
        'architecture': 'frozen_V2_gold_anchored_dual_residual64',
        'routers': list(ROUTERS), 'datasets': list(DATASETS), 'split_seeds': list(SPLIT_SEEDS),
        'inner_seeds': [42] if smoke else list(SEEDS),
        'recipes': [asdict(r) for r in (RECIPES[:2] if smoke else RECIPES)],
        'train': asdict(cfg), 'smoke': smoke, 'source_code': source_records(),
        'preserved_snapshot': {str(snapshot_path): sha256(snapshot_path)},
        'torch_version': str(torch.__version__),
        'protocol': 'unchanged_Fixed_V2_conflict_cross_CE_agreement_keep_KL',
        'selection_rule': 'historical_label_blind_dataset_module_node_id_validation_half',
        'outer_seed_reduction': 'metric_mean_and_std_ddof0_no_probability_blending',
        'inner_seed_reduction': 'mean_probability',
    }
    immutable_json(config_path(root), config)
    return config


def input_paths(root, seed, rid, ds):
    base = Path(root) / f'seed{seed}' / 'upstream' / ds
    return {
        'split': Path(root) / 'splits' / f'{ds}_5shot_seed{seed}.json',
        'upstream': base / 'pipeline' / 'predictions.pt',
        'embeddings': base / 'embeddings.pt', 'supervision': base / 'supervision.pt',
        'router': Path(root) / f'seed{seed}' / 'router' / rid / ds / 'predictions.pt',
        'hidden': ROOT / 'runs/router_v7_decision_llm_hidden_20260827_v1' / ds / 'hidden.pt',
    }


def validation_halves(dataset, valid, agreement):
    selection, threshold = [], []
    for module in ('conflict', 'agreement'):
        ids = valid[agreement[valid] if module == 'agreement' else ~agreement[valid]]
        split = deterministic_calibration_threshold_split(ids, dataset, module)
        selection.append(ids[split.calibration])
        threshold.append(ids[split.threshold])
    return torch.cat(selection).sort().values, torch.cat(threshold).sort().values


def prepare_arrays(split, upstream, embeddings, safe, router, dataset):
    """Strict schemas prevent query truth from entering the trainer adapter."""
    if set(upstream) != {'gnn_logits', 'llm_logits'} or set(embeddings) != {'embeddings', 'node_ids'}:
        raise ValueError('unexpected upstream schema')
    if set(safe) != {'support_ids', 'support_targets', 'valid_ids', 'valid_targets'}:
        raise ValueError('unsafe supervision schema')
    core = {'node_ids', 'choose_g', 'trust', 'accepted', 'selected_pred', 'agreement'}
    extras = {'preference_score', 'logp_g', 'logp_l', 'selected_score'}
    if not core.issubset(router) or not set(router).issubset(core | extras):
        raise ValueError('unsafe Router schema')
    n = split['num_nodes']
    if not torch.equal(embeddings['node_ids'], torch.arange(n)):
        raise ValueError('embedding node alignment differs')
    for key in set(router) - core:
        value = router[key]
        expected = upstream['gnn_logits'].shape if key in ('logp_g', 'logp_l') else (n,)
        if value.shape != expected or not bool(torch.isfinite(value).all()):
            raise ValueError('invalid safe Router auxiliary field: ' + key)
        if key in ('logp_g', 'logp_l'):
            source = upstream['gnn_logits' if key == 'logp_g' else 'llm_logits']
            if not torch.allclose(value.exp(), source.float().softmax(1), atol=2e-6):
                raise ValueError('Router auxiliary expert differs from current upstream')
    ids = {key: torch.tensor(split[key], dtype=torch.long) for key in
           ('support_ids', 'valid_ids', 'unlabeled_ids', 'evaluation_full_ids', 'evaluation_1000_ids')}
    if not torch.equal(ids['unlabeled_ids'], ids['evaluation_full_ids']):
        raise ValueError('full evaluation must use complete query')
    if (len(ids['evaluation_1000_ids'].unique()) != len(ids['evaluation_1000_ids'])
            or not bool(torch.isin(ids['evaluation_1000_ids'], ids['unlabeled_ids']).all())):
        raise ValueError('reference evaluation IDs must be unique query subset')
    for key in ('support_ids', 'valid_ids'):
        if not torch.equal(safe[key], ids[key]):
            raise ValueError('safe supervision differs from outer split')
    if safe['support_targets'].dtype != torch.long or safe['support_targets'].shape != ids['support_ids'].shape:
        raise ValueError('invalid support targets')
    if safe['valid_targets'].dtype != torch.long or safe['valid_targets'].shape != ids['valid_ids'].shape:
        raise ValueError('invalid validation targets')
    if not torch.equal(router['node_ids'], torch.arange(n)):
        raise ValueError('Router node alignment differs')
    if router['accepted'].dtype != torch.bool or router['accepted'].shape != (n,):
        raise ValueError('accepted must be a full canonical Boolean vector')
    protected = torch.cat([ids['support_ids'], ids['valid_ids']])
    if bool(router['accepted'][protected].any()):
        raise ValueError('Router accepted protected nodes')
    f = {
        'node_ids': router['node_ids'], 'gnn_embeddings': embeddings['embeddings'].float(),
        'logp_g': upstream['gnn_logits'].float().log_softmax(1),
        'logp_l': upstream['llm_logits'].float().log_softmax(1),
        **{key: router[key] for key in ('choose_g', 'trust', 'selected_pred', 'agreement')},
    }
    validate_features(f)
    selection, threshold = validation_halves(dataset, ids['valid_ids'], f['agreement'])
    lookup = dict(zip(ids['valid_ids'].tolist(), safe['valid_targets'].tolist()))
    s = {'gold_ids': ids['support_ids'], 'gold_targets': safe['support_targets'],
         'selection_ids': selection, 'selection_targets': torch.tensor([lookup[i] for i in selection.tolist()]),
         'threshold_ids': threshold, 'test_ids': ids['unlabeled_ids']}
    validate_supervision(s, n, f['logp_g'].shape[1])
    query = s['test_ids']
    decision = {'node_ids': query, 'accepted': router['accepted'][query],
                'selected_pred': router['selected_pred'][query],
                'module_code': torch.where(f['agreement'][query], 1, 2).to(torch.int8)}
    f, p = build_router_features_and_pools(f, query, decision, f['choose_g'], f['trust'])
    validate_pools(p, f, s)
    accepted = torch.cat([p['teach_g'], p['teach_l'], p['agreement']]).sort().values
    if not torch.equal(accepted, torch.where(router['accepted'])[0]):
        raise ValueError('pool builder changed Router acceptance')
    return f, s, p


def bind_inputs(root, seed, rid, ds, paths):
    """Bind every tensor to producer completion and frozen Router provenance."""
    root = Path(root).resolve(); base = root / f'seed{seed}' / 'upstream' / ds
    completion_path = base / 'COMPLETE.json'; completion = read_json(completion_path)
    if (completion.get('dataset') != ds or completion.get('split_seed') != seed
            or completion.get('all_checks_pass') is not True or completion.get('test_truth_exported') is not False):
        raise ValueError('unsafe or mismatched upstream completion')
    hashes = {str(base / name): digest for name, digest in completion['files'].items()}
    for key in ('upstream', 'embeddings', 'supervision'):
        if str(paths[key]) not in hashes:
            raise ValueError('upstream completion does not bind ' + key)
    hashes[str(completion_path)] = sha256(completion_path)
    audit_path = root / 'SPLIT_AUDIT.json'; audit = read_json(audit_path)
    if (audit.get('all_checks_pass') is not True or audit.get('seeds') != list(SPLIT_SEEDS)
            or {(r['dataset'], r['seed']) for r in audit.get('records', [])} != set(itertools.product(DATASETS, SPLIT_SEEDS))):
        raise ValueError('all-25 split audit required')
    matches = [r for r in audit['records'] if r['dataset'] == ds and r['seed'] == seed]
    if len(matches) != 1 or Path(matches[0]['export']) != paths['split'].relative_to(root):
        raise ValueError('split audit identity mismatch')
    hashes[str(paths['split'])] = matches[0]['export_sha256']
    hashes[str(audit_path)] = sha256(audit_path)
    if rid.startswith('r8_'):
        frozen_path = root / f'seed{seed}' / 'router_work/V8_FROZEN.json'
        frozen = read_json(frozen_path)
        if (frozen.get('split_seed') != seed or frozen.get('test_labels_accessed') is not False
                or frozen.get('datasets') != list(DATASETS) or str(paths['router']) not in frozen.get('files', {})):
            raise ValueError('invalid V8 frozen handoff')
        hashes.update(frozen['files'])
        # The registration binds banks and the safe split/upstream inputs used in training.
        pre_path = root / f'seed{seed}' / 'router_work/r8_disagreement/PREREGISTERED.json'
        pre = read_json(pre_path)
        hashes.update(pre['input_hashes']); hashes.update(pre['code_hashes'])
    else:
        frozen_path = paths['router'].parent / 'FROZEN.json'; frozen = read_json(frozen_path)
        if (frozen.get('router_id') != rid or frozen.get('dataset') != ds or frozen.get('split_seed') != seed
                or frozen.get('query_truth_loaded') is not False or frozen.get('query_truth_used_for_selection') is not False):
            raise ValueError('invalid R6/A20 frozen handoff')
        protocol_path = root / f'seed{seed}' / 'router_training_r6_a20/CONFIG.json'
        protocol = read_json(protocol_path)
        fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        if frozen['input_fingerprint'] != fingerprint or protocol['split_seed'] != seed:
            raise ValueError('R6/A20 frozen protocol mismatch')
        for domain in protocol['inputs'].values():
            for entry in domain.values():
                hashes[entry['path']] = entry['sha256']
        hashes.update({str(ROOT / path): digest for path, digest in protocol['code'].items()})
        hashes[str(protocol_path)] = sha256(protocol_path)
        hashes[str(paths['router'])] = frozen['predictions_sha256']
    hashes[str(frozen_path)] = sha256(frozen_path)
    hashes[str(paths['hidden'])] = sha256(paths['hidden'])
    verify_files(hashes)
    return hashes


def load_inputs(root, seed, rid, ds):
    paths = input_paths(root, seed, rid, ds)
    hashes = bind_inputs(root, seed, rid, ds, paths)
    split = read_json(paths['split'])
    if split['dataset'] != ds or split['split_seed'] != seed or split.get('test_truth_exported') is not False:
        raise ValueError('split identity or truth isolation differs')
    arrays = {key: torch.load(paths[key], map_location='cpu', weights_only=True, mmap=True)
              for key in ('upstream', 'embeddings', 'supervision', 'router')}
    f, s, p = prepare_arrays(split, arrays['upstream'], arrays['embeddings'], arrays['supervision'], arrays['router'], ds)
    hidden = torch.load(paths['hidden'], map_location='cpu', weights_only=True, mmap=True)
    validate_hidden(hidden, split['num_nodes'])
    if hidden['metadata']['dataset'] != ds:
        raise ValueError('hidden dataset identity differs')
    if rid == 'r8_disagreement' and len(p['agreement']):
        raise ValueError('disagreement-only Router accepted agreement nodes')
    return f, s, p, hidden['hidden'], split, hashes


def gold_member(root, seed, ds, inner, recipe, cfg, f, s, p, h, inputs):
    """A lock prevents duplicate writers; identity excludes only unused Router fields."""
    dest = Path(root) / f'seed{seed}' / 'joint_gold' / ds / f'inner{inner}'
    dest.mkdir(parents=True, exist_ok=True)
    path, marker = dest / 'gold_only.pt', dest / 'complete.json'
    identity = {'split_seed': seed, 'dataset': ds, 'inner_seed': inner,
                'inputs': {str(path): inputs[str(path)] for key, path in input_paths(root, seed, '', ds).items() if key != 'router'},
                'train': asdict(cfg), 'recipe': asdict(recipe),
                'config_sha256': sha256(config_path(root)), 'test_labels_read': False}
    with (dest / '.lock').open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        reused = marker.exists()
        if reused:
            old = read_json(marker)
            if old['identity'] != identity:
                raise ValueError('gold reuse inputs/config differ')
            verify_files({str(path): old['sha256']})
            member = torch.load(path, weights_only=True, map_location='cpu')
        else:
            member = train_member(f, s, p, h, recipe, cfg)
            save_tensor(path, member)
            write_json(marker, {'identity': identity, 'sha256': sha256(path)})
        _check_member(member, cfg, recipe, s)
        # Recompute logits: even identical source paths cannot silently change gold inputs.
        for branch in ('g', 'l'):
            for criterion in ('accuracy', 'nll'):
                z = member_logits(member, branch, criterion, f, h, s['selection_ids'], cfg)
                ref = member['validation_logits'][branch][criterion]
                if not torch.allclose(z, ref, atol=2e-6) or not torch.equal(z.argmax(1), ref.argmax(1)):
                    raise ValueError('gold member fails current validation reproduction')
    return member, {'path': str(path), 'sha256': sha256(path), 'already_trained': reused}


def verify_cell(root, seed, rid, ds, cfg_hash):
    dest = cell_path(root, seed, rid, ds)
    item = read_json(dest / 'complete.json')
    if (item.get('split_seed') != seed or item.get('router_id') != rid or item.get('dataset') != ds
            or item.get('config_sha256') != cfg_hash or item.get('test_labels_read') is not False
            or item.get('all_checks_pass') is not True or not REQUIRED.issubset(item.get('artifacts', {}))):
        raise ValueError('invalid Joint cell completion')
    verify_files({str(dest / name): digest for name, digest in item['artifacts'].items()})
    verify_files(read_json(dest / 'metadata.json')['input_hashes'])
    return item


def run_cell(root, seed, rid, ds):
    slurm_only()
    root = Path(root).resolve()
    config = read_json(config_path(root)); cfg_hash = sha256(config_path(root))
    if seed not in config['split_seeds'] or rid not in config['routers'] or ds not in config['datasets']:
        raise ValueError('unknown cell')
    verify_files(config['source_code'])
    dest = cell_path(root, seed, rid, ds); dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        return verify_cell(root, seed, rid, ds, cfg_hash)
    cfg = TrainConfig(**config['train']); torch.set_num_threads(cfg.threads)
    f, s, p, h, split, inputs = load_inputs(root, seed, rid, ds)
    immutable_json(dest / 'INPUTS.json', inputs)
    start = time.monotonic(); members = {}; reused = trained = 0
    for inner, raw in itertools.product(config['inner_seeds'], config['recipes']):
        recipe = Recipe(**raw); member_cfg = replace(cfg, seed=inner)
        folder = dest / 'bank' / f'inner{inner}'; folder.mkdir(parents=True, exist_ok=True)
        path, marker = folder / (recipe.name + '.pt'), folder / (recipe.name + '.json')
        if marker.exists():
            record = read_json(marker)
            if record['config_sha256'] != cfg_hash:
                raise ValueError('member config changed')
            verify_files({str(path): record['sha256']})
            member = torch.load(path, weights_only=True, map_location='cpu')
        else:
            gold = None
            if recipe.name == 'gold_only':
                member, gold = gold_member(root, seed, ds, inner, recipe, member_cfg, f, s, p, h, inputs)
            else:
                member = train_member(f, s, p, h, recipe, member_cfg)
            save_tensor(path, member)
            record = {'config_sha256': cfg_hash, 'sha256': sha256(path), 'gold_cache': gold}
            write_json(marker, record)
        _check_member(member, member_cfg, recipe, s)
        members[(inner, recipe.name)] = member
        if (record.get('gold_cache') or {}).get('already_trained', False):
            reused += 1
        else:
            trained += 1
        print({'split_seed': seed, 'router': rid, 'dataset': ds, 'inner_seed': inner,
               'recipe': recipe.name, 'seconds': round(member['train_seconds'], 2)}, flush=True)
    candidates = candidate_bank(members, f, s, config['inner_seeds'])
    pg, pl = f['logp_g'].exp(), f['logp_l'].exp()
    pb = scheme_b_probability(pg, pl, f['choose_g'], f['agreement'])
    if not torch.equal(pb.argmax(1), f['selected_pred']):
        raise ValueError('Router-B prediction identity differs')
    ids, y = s['selection_ids'], s['selection_targets']
    selection = select_candidates(candidates, pb[ids], y)
    write_json(dest / 'selection.json', selection)
    save_tensor(dest / 'validation.pt', {'node_ids': ids, 'candidates': candidates, 'router': pb[ids]})
    catalog = {c['id']: c for branch in candidates for c in candidates[branch]}
    write_json(dest / 'candidate_catalog.json', {k: {'metadata': c['metadata'], **_score(c['probability'], y)} for k, c in catalog.items()})
    cache = {}
    def full(candidate_id):
        if candidate_id not in cache:
            m = catalog[candidate_id]['metadata']; branch = m['branch']; base = f['logp_' + branch]
            value = (base.exp().unsqueeze(0).repeat(len(config['inner_seeds']), 1, 1) if m['base'] else
                     torch.stack([residual_probability(base, member_logits(members[(inner, m['recipe'])], branch,
                         m['criterion'], f, h, f['node_ids'], replace(cfg, seed=inner)), m['scale'])
                         for inner in config['inner_seeds']]))
            observed, reference = value.mean(0)[ids], catalog[candidate_id]['probability']
            if not torch.allclose(observed, reference, atol=2e-6) or not torch.equal(observed.argmax(1), reference.argmax(1)):
                raise ValueError('full inference differs from validation candidate')
            cache[candidate_id] = value
        return cache[candidate_id]
    families = {}
    for name, pick in selection['families'].items():
        g = full(pick['g_id'] if pick['beta'] * pick['alpha'] > 0 else 'g/base')
        l = full(pick['l_id'] if pick['beta'] * (1 - pick['alpha']) > 0 else 'l/base')
        seeds = blend(g, l, pb.unsqueeze(0), pick['alpha'], pick['beta']); ensemble = seeds.mean(0)
        score = _score(ensemble[ids], y)
        if abs(score['accuracy'] - pick['accuracy']) > 1e-10 or abs(score['nll'] - pick['nll']) > 2e-6:
            raise ValueError('Joint full prediction fails validation reproduction')
        families[name] = {'ensemble': ensemble, 'seeds': seeds}
    b = selection['base_fusion']
    baselines = {'gnn': pg, 'llm': pl, 'router': pb, 'base_fusion': blend(pg, pl, pb, b['alpha'], b['beta'])}
    winner = selection['optimized']['family']
    families['optimized'] = families[winner] if winner in families else {
        'ensemble': baselines[winner], 'seeds': baselines[winner].unsqueeze(0).repeat(len(config['inner_seeds']), 1, 1)}
    save_tensor(dest / 'predictions.pt', {'node_ids': f['node_ids'], 'baselines': baselines, 'families': families})
    save_tensor(dest / 'supervision.pt', s); save_tensor(dest / 'pools.pt', p)
    write_json(dest / 'metadata.json', {'split_seed': seed, 'inner_seeds': config['inner_seeds'], 'router_id': rid,
        'dataset': ds, 'member_count': len(members), 'trained_count': trained, 'reused_count': reused,
        'pool_counts': {k: len(v) for k, v in p.items()}, 'input_hashes': inputs,
        'test_labels_read': False, 'elapsed_seconds': time.monotonic() - start,
        'slurm_job_id': os.environ['SLURM_JOB_ID']})
    verify_files(config['source_code']); verify_files(inputs)
    artifacts = {str(path.relative_to(dest)): sha256(path) for path in sorted(dest.rglob('*'))
                 if path.is_file() and path.name != 'complete.json' and '.tmp.' not in path.name}
    write_json(dest / 'complete.json', {'split_seed': seed, 'router_id': rid, 'dataset': ds,
        'config_sha256': cfg_hash, 'artifacts': artifacts, 'all_checks_pass': True, 'test_labels_read': False})
    return verify_cell(root, seed, rid, ds, cfg_hash)


def freeze(root=DEFAULT_RUN):
    """Validate every requested cell before creating the only test-evaluation gate."""
    slurm_only()
    from eargtc.joint_five_router.evaluation import verify_selection_reconstruction
    from eargtc.joint_v8.evaluation import verify_full_family
    root = Path(root).resolve(); cfg_path = config_path(root); config = read_json(cfg_path)
    verify_files(config['source_code']); verify_files(config['preserved_snapshot'])
    cfg_hash = sha256(cfg_path); files = {str(cfg_path): cfg_hash}; audits = {}
    cells = list(itertools.product(config['split_seeds'], config['routers'], config['datasets']))
    if config['smoke']:
        raise ValueError('smoke runs cannot freeze for test evaluation')
    if (config['split_seeds'] != list(SPLIT_SEEDS) or config['routers'] != list(ROUTERS)
            or config['datasets'] != list(DATASETS) or config['inner_seeds'] != list(SEEDS)):
        raise ValueError('formal population must contain 125 cells with three inner seeds')
    for seed, rid, ds in cells:
        dest = cell_path(root, seed, rid, ds)
        verify_cell(root, seed, rid, ds, cfg_hash)
        f, s, p, hidden, split, inputs = load_inputs(root, seed, rid, ds)
        saved_s = torch.load(dest / 'supervision.pt', weights_only=True, map_location='cpu')
        saved_p = torch.load(dest / 'pools.pt', weights_only=True, map_location='cpu')
        if any(not torch.equal(s[k], saved_s[k]) for k in s) or any(not torch.equal(p[k], saved_p[k]) for k in p):
            raise ValueError('saved supervision/pools do not reproduce safe inputs')
        predictions = torch.load(dest / 'predictions.pt', weights_only=True, map_location='cpu')
        validation = torch.load(dest / 'validation.pt', weights_only=True, map_location='cpu')
        selection = read_json(dest / 'selection.json')
        pb = scheme_b_probability(f['logp_g'].exp(), f['logp_l'].exp(), f['choose_g'], f['agreement'])
        if not torch.equal(predictions['node_ids'], f['node_ids']):
            raise ValueError('prediction node alignment changed')
        for name, value in [('gnn', f['logp_g'].exp()), ('llm', f['logp_l'].exp()), ('router', pb)]:
            if not torch.equal(predictions['baselines'][name], value):
                raise ValueError('saved expert/Router baseline changed')
        if not torch.equal(validation['router'], pb[s['selection_ids']]):
            raise ValueError('validation Router baseline changed')
        if select_candidates(validation['candidates'], validation['router'], s['selection_targets']) != selection:
            raise ValueError('validation-only selection does not reproduce')
        verify_selection_reconstruction(predictions, selection, validation, s['selection_ids'], s['selection_targets'])
        if set(predictions['families']) != {'gold_only', 'weighted', 'scaled', 'joint', 'optimized'}:
            raise ValueError('missing Fixed V2 family')
        for name, value in predictions['families'].items():
            verify_full_family(value, name, f['logp_g'].shape)
        markers = sorted((dest / 'bank').glob('inner*/*.json'))
        if len(markers) != len(SEEDS) * len(RECIPES):
            raise ValueError('formal member bank is incomplete')
        files[str(dest / 'complete.json')] = sha256(dest / 'complete.json')
        audits[f'{seed}/{rid}/{ds}'] = {'member_count': len(markers), 'pool_counts': {k: len(v) for k, v in p.items()},
            'gold_count': len(s['gold_ids']), 'selection_count': len(s['selection_ids']),
            'threshold_count': len(s['threshold_ids']), 'query_count': len(s['test_ids']),
            'validation_reproduced': True}
        print('Pretest validated', seed, rid, ds, flush=True)
        del f, s, p, hidden, predictions, validation, saved_s, saved_p, pb
    lock = {'schema': 'multiseed_joint_fixed_v2_freeze_v1', 'cell_count': len(cells),
            'split_seeds': config['split_seeds'], 'inner_seeds': config['inner_seeds'],
            'routers': config['routers'], 'datasets': config['datasets'],
            'test_labels_read': False, 'files': files, 'cells': audits}
    immutable_json(root / 'joint' / 'FROZEN.json', lock)
    return lock


def verify_frozen(root):
    slurm_only()
    root = Path(root).resolve(); config = read_json(config_path(root))
    lock = read_json(root / 'joint' / 'FROZEN.json')
    cells = list(itertools.product(SPLIT_SEEDS, ROUTERS, DATASETS))
    expected = {str(config_path(root))} | {str(cell_path(root, seed, rid, ds) / 'complete.json') for seed, rid, ds in cells}
    if (config['smoke'] or lock.get('test_labels_read') is not False or lock.get('cell_count') != 125
            or set(lock.get('files', {})) != expected or lock.get('split_seeds') != list(SPLIT_SEEDS)
            or lock.get('inner_seeds') != list(SEEDS) or lock.get('routers') != list(ROUTERS)
            or lock.get('datasets') != list(DATASETS)):
        raise ValueError('all 125 requested cells must freeze before opening test truth')
    verify_files(lock['files']); verify_files(config['source_code']); verify_files(config['preserved_snapshot'])
    for seed, rid, ds in cells:
        verify_cell(root, seed, rid, ds, sha256(config_path(root)))
    return config, lock


def aggregate_rows(rows):
    metrics = ('pseudo_accuracy', 'pseudo_coverage', 'accuracy', 'macro_f1', 'router_accuracy', 'gnn_accuracy', 'llm_accuracy')
    grouped = {}
    for row in rows:
        grouped.setdefault((row['router'], row['dataset'], row['population']), []).append(row)
    out = []
    for (rid, ds, population), group in sorted(grouped.items()):
        if sorted(row['split_seed'] for row in group) != list(SPLIT_SEEDS):
            raise ValueError('outer-seed aggregate requires five distinct split seeds')
        entry = {'router': rid, 'dataset': ds, 'population': population, 'split_count': len(group)}
        for metric in metrics:
            values = [row[metric] for row in group if row[metric] is not None]
            entry[metric + '_defined_splits'] = len(values)
            # A zero-acceptance split has undefined pseudo accuracy. Never disguise
            # a four-split summary as the required five-split mean/std.
            complete = len(values) == len(group)
            entry[metric + '_mean'] = statistics.fmean(values) if complete else None
            entry[metric + '_std_ddof0'] = statistics.pstdev(values) if complete else None
            entry[metric + '_std_ddof1'] = statistics.stdev(values) if complete else None
        out.append(entry)
    return out


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def evaluate(root=DEFAULT_RUN):
    slurm_only()
    from eargtc.joint_five_router.evaluation import classification_report
    root = Path(root).resolve()
    config, lock = verify_frozen(root)  # No graph or query targets are read above this line.
    output = root / 'joint' / 'evaluation'
    if output.exists():
        raise FileExistsError('final evaluation is immutable')
    stage = output.with_name('evaluation.staging')
    if stage.exists():
        raise FileExistsError('evaluation staging exists; inspect before retrying')
    stage.mkdir(parents=True)
    from eargtc.data import _install_pyg_optional_extension_stubs
    _install_pyg_optional_extension_stubs()
    rows, details, label_hashes = [], {}, {}
    for ds in DATASETS:
        split0 = read_json(root / 'splits' / f'{ds}_5shot_seed42.json')
        graph_path = Path(split0['graph_path'])
        verify_files({str(graph_path): split0['graph_sha256']})
        graph = torch.load(graph_path, weights_only=False, map_location='cpu')
        truth = graph.y.reshape(-1).long(); del graph
        label_hashes[str(graph_path)] = split0['graph_sha256']
        if len(truth) != split0['num_nodes']:
            raise ValueError('evaluation graph alignment differs')
        for seed, rid in itertools.product(SPLIT_SEEDS, ROUTERS):
            dest = cell_path(root, seed, rid, ds)
            split = read_json(root / 'splits' / f'{ds}_5shot_seed{seed}.json')
            if split['graph_path'] != str(graph_path) or split['graph_sha256'] != split0['graph_sha256']:
                raise ValueError('outer splits refer to different graph truth')
            pred = torch.load(dest / 'predictions.pt', weights_only=True, map_location='cpu')
            p = torch.load(dest / 'pools.pt', weights_only=True, map_location='cpu')
            router = torch.load(input_paths(root, seed, rid, ds)['router'], weights_only=True, map_location='cpu')
            selection = read_json(dest / 'selection.json')
            accepted = torch.zeros(len(truth), dtype=torch.bool)
            accepted[torch.cat([p['teach_g'], p['teach_l'], p['agreement']])] = True
            for population, key in [('full_query', 'evaluation_full_ids'), ('eval1000', 'evaluation_1000_ids')]:
                ids = torch.tensor(split[key], dtype=torch.long); y = truth[ids]
                metrics = {name: classification_report(value[ids], y) for name, value in pred['baselines'].items()}
                metrics.update({name: classification_report(value['ensemble'][ids], y) for name, value in pred['families'].items()})
                mask = accepted[ids]; accepted_ids = ids[mask]
                pseudo_acc = float((router['selected_pred'][accepted_ids] == truth[accepted_ids]).double().mean()) if len(accepted_ids) else None
                inner = [{ 'inner_seed': inner_seed, **classification_report(value[ids], y)}
                         for inner_seed, value in zip(config['inner_seeds'], pred['families']['optimized']['seeds'])]
                row = {'router': rid, 'dataset': ds, 'split_seed': seed, 'population': population,
                       'query_count': len(ids), 'accepted_count': len(accepted_ids),
                       'pseudo_accuracy': pseudo_acc, 'pseudo_coverage': len(accepted_ids) / len(ids),
                       'accuracy': metrics['optimized']['accuracy'], 'macro_f1': metrics['optimized']['macro_f1'],
                       'nll': metrics['optimized']['nll'], 'router_accuracy': metrics['router']['accuracy'],
                       'gnn_accuracy': metrics['gnn']['accuracy'], 'llm_accuracy': metrics['llm']['accuracy'],
                       'selected_family': selection['optimized']['family']}
                rows.append(row)
                details[f'{seed}/{rid}/{ds}/{population}'] = {
                    **row, 'metrics': metrics, 'inner_seed_metrics': inner,
                    'inner_seed_note': 'Training stochasticity within one outer split; excluded from main split mean/std.'}
            del pred, p, router
        del truth
    summary = aggregate_rows(rows)
    write_csv(stage / 'per_split.csv', rows); write_csv(stage / 'mean_std.csv', summary)
    for population in ('full_query', 'eval1000'):
        write_csv(stage / f'{population}_per_split.csv', [r for r in rows if r['population'] == population])
        write_csv(stage / f'{population}_mean_std.csv', [r for r in summary if r['population'] == population])
    payload = {'config': config, 'cells': details, 'aggregate': summary, 'test_label_hashes': label_hashes,
               'frozen_sha256': sha256(root / 'joint' / 'FROZEN.json')}
    write_json(stage / 'results.json', payload)
    lines = ['# Five-router Joint-Fixed reproduction across five outer splits', '',
        'All 125 cells were trained and selected using safe support/validation targets, audited, and frozen before this evaluation read query truth. '
        'The unchanged Fixed V2 trainer uses inner seeds 42/43/44 and averages their probabilities within each outer split. '
        'Main results average the five split-level metrics from outer split seeds 42–46; probabilities are never averaged across outer splits.', '',
        'Values below are percentages: mean ± population standard deviation (ddof=0), matching the reference common/utils.py convention. '
        'CSV also includes sample standard deviation (ddof=1). Pseudo-label accuracy concerns accepted Router labels; '
        'coverage divides accepted count by the stated evaluation population. Final classification covers every node in that population.', '',
        'Full query and the reference 1000-node sample share each frozen trained model. '
        'The 1000-node membership follows the audited reference algorithm for each split; only seed42 has matching saved historical evaluation artifacts. '
        'These are outer data splits with new gold heads and Router fits; the raw pretrained SSL embeddings and LLM hidden source remain shared frozen dependencies.', '']
    def fmt(row, metric):
        mean, std = row[metric + '_mean'], row[metric + '_std_ddof0']
        return 'undefined' if mean is None else f'{100 * mean:.2f} ± {100 * std:.2f}'
    for population in ('full_query', 'eval1000'):
        lines += [f'## {population}', '', '| Router | Dataset | Pseudo accuracy % | Coverage % | Joint accuracy % | Joint macro-F1 % |',
                  '|---|---|---:|---:|---:|---:|']
        for row in summary:
            if row['population'] == population:
                lines.append(f"| {row['router']} | {row['dataset']} | {fmt(row, 'pseudo_accuracy')} | {fmt(row, 'pseudo_coverage')} | {fmt(row, 'accuracy')} | {fmt(row, 'macro_f1')} |")
        lines.append('')
    by_key = {(r['router'], r['dataset'], r['population']): r for r in summary}
    for population in ('full_query', 'eval1000'):
        for metric in ('pseudo_accuracy', 'accuracy', 'macro_f1', 'pseudo_coverage'):
            matrix = [{'router': rid, **{ds: fmt(by_key[rid, ds, population], metric) for ds in DATASETS}} for rid in ROUTERS]
            write_csv(stage / f'{population}_{metric}_matrix.csv', matrix)
            lines += [f'## {population}: {metric} (mean ± std, %)', '',
                      '| Router | ' + ' | '.join(DATASETS) + ' |',
                      '|---|' + '---:|' * len(DATASETS)]
            lines.extend('| ' + row['router'] + ' | ' + ' | '.join(row[ds] for ds in DATASETS) + ' |' for row in matrix)
            lines.append('')
    (stage / 'REPORT.md').write_text('\n'.join(lines))
    write_json(stage / 'complete.json', {'all_checks_pass': True, 'cell_count': 125, 'metric_rows': len(rows),
        'frozen_sha256': payload['frozen_sha256'], 'test_label_hashes': label_hashes,
        'artifacts': {path.name: sha256(path) for path in stage.iterdir() if path.is_file()}})
    stage.rename(output)
    return output
