"""Controlled selector orchestration using the unchanged frozen V2 trainer/search."""
from dataclasses import asdict, replace
import itertools
from pathlib import Path
import time
import torch
from eargtc.joint_cross_repair.artifacts import DATASETS, read_json, write_json, save_tensor, sha256, verify_files
from eargtc.joint_cross_repair.prepare import DEFAULT_EXPORT as BASE_EXPORT, FROZEN_ROUTER
from eargtc.joint_cross_repair.training import TrainConfig, _score, blend, scheme_b_probability
from eargtc.joint_cross_repair_search.bank import RECIPES, SEEDS, Recipe, train_member, member_logits
from eargtc.joint_cross_repair_search.experiment import candidate_bank
from eargtc.joint_cross_repair_search.selection import residual_probability
from eargtc.joint_five_router.experiment import ROUTER_IDS, FAMILIES, select_candidates, _check_member

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / 'runs/joint_selector_20260907_v1'
OLD_FIVE = ROOT / 'runs/joint_five_router_20260907_v1'
STRATEGIES = ('full', 'random', 'reliability', 'reliability_cluster')
SPEC = ROOT / 'docs/superpowers/specs/2026-09-07-joint-selector-design.md'
REQUIRED_ARTIFACTS = {'selection.json', 'validation.pt', 'predictions.pt', 'fixed_control.pt',
                      'candidate_catalog.json', 'metadata.json'}


def initialize(run_root, export_root=None, smoke=False):
    from .adapter import DEFAULT_EXPORT, SOURCE_ROOT
    from eargtc.joint_five_router.experiment import preflight as old_preflight
    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    export = Path(export_root or DEFAULT_EXPORT).resolve()
    cells = [export / strategy / rid / ds / 'complete.json'
             for strategy, rid, ds in itertools.product(STRATEGIES, ROUTER_IDS, DATASETS)]
    if not all(path.exists() for path in cells):
        raise ValueError('all100 safe selector exports required before search initialization')
    old = old_preflight(OLD_FIVE)
    expected = TrainConfig(threads=2, validate_every=50, max_steps=500, patience=5)
    if (old['train'] != asdict(expected) or old['seeds'] != list(SEEDS)
            or old['recipes'] != [asdict(r) for r in RECIPES]
            or old['architecture'] != 'frozen_V2_gold_anchored_dual_residual64'
            or Path(old['base_export_root']).resolve() != BASE_EXPORT.resolve()):
        raise ValueError('historical training/search/base export provenance differs')
    cfg = replace(expected, validate_every=5, max_steps=10, patience=2) if smoke else expected
    paths = [SPEC]
    for folder in ('eargtc/joint_cross_repair', 'eargtc/joint_cross_repair_search',
                   'eargtc/joint_five_router', 'eargtc/joint_selector',
                   'scripts/joint_cross_repair', 'scripts/joint_cross_repair_search',
                   'scripts/joint_five_router', 'scripts/joint_selector',
                   'eargtc/router_fiveway_topk', 'scripts/router_fiveway_topk'):
        paths.extend(sorted((ROOT / folder).glob('*.py')))
    inputs = cells + [export / 'COMPLETE.json', SOURCE_ROOT / 'FROZEN.json',
                      SOURCE_ROOT / 'FROZEN.sha256', SOURCE_ROOT / 'CONFIG.json', SOURCE_ROOT / 'FINAL_AUDIT.json',
                      OLD_FIVE / 'CONFIG.json', OLD_FIVE / 'FROZEN.json']
    inputs.extend(OLD_FIVE / rid / ds / 'complete.json' for rid, ds in itertools.product(ROUTER_IDS, DATASETS))
    config = {'architecture': old['architecture'], 'strategies': list(STRATEGIES),
              'datasets': list(DATASETS), 'routers': list(ROUTER_IDS),
              'seeds': [42] if smoke else list(SEEDS), 'recipes': [asdict(r) for r in RECIPES],
              'train': asdict(cfg), 'smoke': smoke, 'export_root': str(export),
              'source_root': str(SOURCE_ROOT), 'base_export_root': str(BASE_EXPORT),
              'old_five_root': str(OLD_FIVE), 'old_five_export_root': old['export_root'],
              'label_root': str(FROZEN_ROUTER), 'gold_members': old['gold_members'],
              'source_code': {str(p): sha256(p) for p in paths},
              'input_completions': {str(p): sha256(p) for p in inputs},
              'torch_version': str(torch.__version__),
              'fixed_control_policy': 'historical full optimized candidate IDs, criteria, residual scales, alpha/beta; current member checkpoints'}
    if config['torch_version'] != old['torch_version']:
        raise ValueError('historical torch version differs')
    path = root / 'CONFIG.json'
    if path.exists():
        if read_json(path) != config:
            raise ValueError('immutable search config/source/input mismatch')
    else:
        write_json(path, config)
    return config


def _verify_cell(root, strategy, rid, ds, config_hash):
    dest = Path(root) / strategy / rid / ds
    complete = read_json(dest / 'complete.json')
    if (complete.get('strategy') != strategy or complete.get('router_id') != rid
            or complete.get('dataset') != ds or complete.get('config_sha256') != config_hash
            or complete.get('test_labels_read') is not False or complete.get('all_checks_pass') is not True):
        raise ValueError('invalid cell completion/provenance')
    if not REQUIRED_ARTIFACTS.issubset(complete.get('artifacts', {})):
        raise ValueError('incomplete cell artifacts, including fixed control')
    verify_files({str(dest / k): v for k, v in complete['artifacts'].items()})


def verify_frozen(run_root):
    root = Path(run_root).resolve()
    lock = read_json(root / 'FROZEN.json')
    cells = list(itertools.product(STRATEGIES, ROUTER_IDS, DATASETS))
    expected = {str(root / 'CONFIG.json')} | {str(root / st / r / d / 'complete.json') for st, r, d in cells}
    if (lock.get('strategies') != list(STRATEGIES) or lock.get('routers') != list(ROUTER_IDS)
            or lock.get('datasets') != list(DATASETS) or lock.get('test_labels_read') is not False
            or set(lock['files']) != expected):
        raise ValueError('all100 cells must be safely frozen')
    verify_files(lock['files'])
    config = read_json(root / 'CONFIG.json')
    if any(config[key] != list(values) for key, values in
           [('strategies', STRATEGIES), ('routers', ROUTER_IDS), ('datasets', DATASETS)]):
        raise ValueError('invalid search population')
    verify_files(config['source_code'])
    verify_files(config['input_completions'])
    digest = sha256(root / 'CONFIG.json')
    for strategy, rid, ds in cells:
        _verify_cell(root, strategy, rid, ds, digest)
    return config


def freeze(run_root):
    root = Path(run_root).resolve()
    cells = list(itertools.product(STRATEGIES, ROUTER_IDS, DATASETS))
    paths = [root / st / r / d / 'complete.json' for st, r, d in cells]
    if not all(p.exists() for p in paths):
        return False
    digest = sha256(root / 'CONFIG.json')
    for strategy, rid, ds in cells:
        _verify_cell(root, strategy, rid, ds, digest)
    lock = {'strategies': list(STRATEGIES), 'routers': list(ROUTER_IDS), 'datasets': list(DATASETS),
            'test_labels_read': False, 'files': {str(p): sha256(p) for p in [root / 'CONFIG.json', *paths]}}
    path = root / 'FROZEN.json'
    if path.exists():
        if read_json(path) != lock:
            raise ValueError('frozen run changed')
    else:
        write_json(path, lock)
    verify_frozen(root)
    return True


def preflight(run_root):
    from .adapter import load_cell
    config = verify_frozen(run_root)
    if config['smoke']:
        raise ValueError('smoke cannot evaluate test truth')
    dependencies = {}
    for strategy, rid, ds in itertools.product(STRATEGIES, ROUTER_IDS, DATASETS):
        # Stream large features; deduplicate shared source hashes before verifying.
        f, s, p, h, meta = load_cell(config['export_root'], strategy, rid, ds, verify_sources=False)
        for path, digest in itertools.chain(meta['source_hashes'].items(), meta.get('old_source_hashes', {}).items()):
            if path in dependencies and dependencies[path] != digest:
                raise ValueError('conflicting source provenance')
            dependencies[path] = digest
        del f, s, p, h, meta
    for ds in DATASETS:
        base = Path(config['base_export_root']) / ds
        complete = read_json(base / 'complete.json')
        verify_files({str(base / k): v for k, v in complete['artifacts'].items()})
        dependencies.update(read_json(base / 'metadata.json')['source_hashes'])
    verify_files(dependencies)
    return config


def _assert_same_tensors(left, right, name):
    if set(left) != set(right) or any(not torch.equal(left[k], right[k]) for k in left):
        raise ValueError(f'historical {name} identity differs')


def _reproduce_member(member, cfg, recipe, f, s, h):
    _check_member(member, cfg, recipe, s)
    for branch, criterion in itertools.product(('g', 'l'), ('accuracy', 'nll')):
        observed = member_logits(member, branch, criterion, f, h, s['selection_ids'], cfg)
        reference = member['validation_logits'][branch][criterion]
        if not torch.allclose(observed, reference, atol=2e-6) or not torch.equal(observed.argmax(1), reference.argmax(1)):
            raise ValueError('reused member does not reproduce current validation inputs')


def fixed_control(pick, full, pb, node_ids, seeds):
    """Apply historical hyperparameters to this cell's newly trained candidates."""
    if pick['family'] == 'router':
        probability = pb.unsqueeze(0).repeat(len(seeds), 1, 1)
    else:
        if pick['family'] == 'base_fusion':
            g_id, l_id = 'g/base', 'l/base'
        else:
            g_id, l_id = pick['g_id'], pick['l_id']
        g = full(g_id if pick['alpha'] * pick['beta'] > 0 else 'g/base')
        l = full(l_id if (1 - pick['alpha']) * pick['beta'] > 0 else 'l/base')
        probability = blend(g, l, pb.unsqueeze(0), pick['alpha'], pick['beta'])
    ensemble = pb if pick['family'] == 'router' else (probability[0] if pick['family'] == 'base_fusion' else probability.mean(0))
    return {'node_ids': node_ids, 'ensemble': ensemble, 'seeds': probability,
            'source_selection': pick}


def _verify_full_reproduction(dest, old_dest, selection, predictions):
    if selection != read_json(old_dest / 'selection.json'):
        raise ValueError('full selected configuration differs from historical full')
    old = torch.load(old_dest / 'predictions.pt', weights_only=True, map_location='cpu')
    if not torch.equal(old['node_ids'], predictions['node_ids']):
        raise ValueError('full prediction node IDs differ')
    maximum = 0.
    for family, outputs in predictions['families'].items():
        for key in ('ensemble', 'seeds'):
            observed, reference = outputs[key], old['families'][family][key]
            error = float((observed - reference).abs().max())
            maximum = max(maximum, error)
            if not torch.equal(observed, reference):
                raise ValueError('full probabilities fail historical reproduction')
    for name, observed in predictions['baselines'].items():
        reference = old['baselines'][name]
        maximum = max(maximum, float((observed - reference).abs().max()))
        if not torch.equal(observed, reference):
            raise ValueError('full baseline differs from historical full')
    write_json(dest / 'full_reproduction.json', {'selected_configuration_exact': True,
        'all_family_argmax_equal': True, 'all_probabilities_bitwise_equal': True, 'maximum_probability_error': maximum,
        'old_selection_sha256': sha256(old_dest / 'selection.json'),
        'old_predictions_sha256': sha256(old_dest / 'predictions.pt'), 'test_labels_read': False})


def run_cell(run_root, strategy, router_id, dataset):
    from .adapter import load_cell
    from eargtc.joint_five_router.adapter import load_cell as old_load_cell
    from eargtc.joint_five_router.experiment import _verify_cell as old_verify_cell
    root = Path(run_root).resolve()
    config = read_json(root / 'CONFIG.json')
    start = time.monotonic()
    if strategy not in STRATEGIES or router_id not in ROUTER_IDS or dataset not in DATASETS:
        raise ValueError('unknown selector/Router/dataset cell')
    cfg = TrainConfig(**config['train'])
    torch.set_num_threads(cfg.threads)
    verify_files(config['source_code'])
    verify_files(config['input_completions'])
    dest = root / strategy / router_id / dataset
    dest.mkdir(parents=True, exist_ok=True)
    config_hash = sha256(root / 'CONFIG.json')
    if (dest / 'complete.json').exists():
        _verify_cell(root, strategy, router_id, dataset, config_hash)
        return strategy, router_id, dataset
    f, s, p, h, meta = load_cell(config['export_root'], strategy, router_id, dataset)
    if meta['router_id'] != router_id or meta['dataset'] != dataset or meta['strategy'] != strategy:
        raise ValueError('adapter cell identity differs')
    base = (Path(config['base_export_root']) / dataset).resolve()
    if Path(meta['base_export_dir']).resolve() != base or meta['base_export_complete_sha256'] != sha256(base / 'complete.json'):
        raise ValueError('gold reuse requires identical original safe base export')
    old_root = Path(config['old_five_root'])
    old_dest = old_root / router_id / dataset
    old_verify_cell(old_root, router_id, dataset, sha256(old_root / 'CONFIG.json'))
    old_pick = read_json(old_dest / 'selection.json')['optimized']
    if strategy == 'full':
        old_f, old_s, old_p, old_h, old_meta = old_load_cell(config['old_five_export_root'], router_id, dataset)
        for name, left, right in [('features', f, old_f), ('supervision', s, old_s), ('pools', p, old_p)]:
            _assert_same_tensors(left, right, name)
        if not torch.equal(h, old_h):
            raise ValueError('historical hidden embedding identity differs')
        del old_f, old_s, old_p, old_h, old_meta
    members = {}
    trained = reused = 0
    train_seconds = 0.
    for seed, raw in itertools.product(config['seeds'], config['recipes']):
        recipe = Recipe(**raw)
        member_cfg = replace(cfg, seed=seed)
        folder = dest / 'bank' / f'seed{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        path, marker = folder / (recipe.name + '.pt'), folder / (recipe.name + '.json')
        reuse = None
        if marker.exists():
            record = read_json(marker)
            if record['config_sha256'] != config_hash:
                raise ValueError('bank config changed')
            verify_files({str(path): record['sha256']})
            member = torch.load(path, weights_only=True, map_location='cpu')
            reuse = record.get('reused_from')
            if reuse:
                verify_files({reuse['path']: reuse['sha256']})
        else:
            if not config['smoke'] and strategy == 'full':
                original = old_dest / 'bank' / f'seed{seed}' / (recipe.name + '.pt')
                reuse = {'path': str(original), 'sha256': sha256(original)}
            elif not config['smoke'] and recipe.name == 'gold_only':
                reuse = config['gold_members'][f'{dataset}/{seed}']
            if reuse:
                verify_files({reuse['path']: reuse['sha256']})
                member = torch.load(reuse['path'], weights_only=True, map_location='cpu')
                _reproduce_member(member, member_cfg, recipe, f, s, h)
            else:
                member = train_member(f, s, p, h, recipe, member_cfg)
            save_tensor(path, member)
            write_json(marker, {'sha256': sha256(path), 'config_sha256': config_hash, 'reused_from': reuse})
        _check_member(member, member_cfg, recipe, s)
        members[(seed, recipe.name)] = member
        if reuse:
            reused += 1
        else:
            trained += 1
            train_seconds += member['train_seconds']
        print({'strategy': strategy, 'router': router_id, 'dataset': dataset, 'seed': seed,
               'recipe': recipe.name, 'reused': bool(reuse), 'seconds': round(member['train_seconds'], 2)}, flush=True)
    candidates = candidate_bank(members, f, s, config['seeds'])
    pg, pl = f['logp_g'].exp(), f['logp_l'].exp()
    pb = scheme_b_probability(pg, pl, f['choose_g'], f['agreement'])
    if not torch.equal(pb.argmax(1), f['selected_pred']):
        raise ValueError('Router B identity differs')
    ids, y = s['selection_ids'], s['selection_targets']
    selection = select_candidates(candidates, pb[ids], y)
    write_json(dest / 'selection.json', selection)
    save_tensor(dest / 'validation.pt', {'node_ids': ids, 'candidates': candidates, 'router': pb[ids]})
    catalog = {c['id']: c for branch in candidates for c in candidates[branch]}
    write_json(dest / 'candidate_catalog.json', {k: {'metadata': c['metadata'], **_score(c['probability'], y)} for k, c in catalog.items()})
    full_cache = {}
    def full(candidate_id):
        if candidate_id in full_cache:
            return full_cache[candidate_id]
        m = catalog[candidate_id]['metadata']
        branch, base = m['branch'], f['logp_' + m['branch']]
        if m['base']:
            value = base.exp().unsqueeze(0).repeat(len(config['seeds']), 1, 1)
        else:
            value = torch.stack([residual_probability(base,
                member_logits(members[(seed, m['recipe'])], branch, m['criterion'], f, h,
                              f['node_ids'], replace(cfg, seed=seed)), m['scale']) for seed in config['seeds']])
        observed, reference = value.mean(0)[ids], catalog[candidate_id]['probability']
        if not torch.allclose(observed, reference, atol=2e-6) or not torch.equal(observed.argmax(1), reference.argmax(1)):
            raise ValueError('restored full branch differs from validation candidate')
        full_cache[candidate_id] = value
        return value
    families = {}
    for name, pick in selection['families'].items():
        g = full(pick['g_id'] if pick['beta'] * pick['alpha'] > 0 else 'g/base')
        l = full(pick['l_id'] if pick['beta'] * (1 - pick['alpha']) > 0 else 'l/base')
        seeds = blend(g, l, pb.unsqueeze(0), pick['alpha'], pick['beta'])
        ens = seeds.mean(0)
        score = _score(ens[ids], y)
        if abs(score['accuracy'] - pick['accuracy']) > 1e-10 or abs(score['nll'] - pick['nll']) > 2e-6:
            raise ValueError('Joint full prediction fails validation reproduction')
        families[name] = {'ensemble': ens, 'seeds': seeds}
    b = selection['base_fusion']
    baselines = {'gnn': pg, 'llm': pl, 'router': pb, 'base_fusion': blend(pg, pl, pb, b['alpha'], b['beta'])}
    winner = selection['optimized']['family']
    families['optimized'] = families[winner] if winner in families else {
        'ensemble': baselines[winner], 'seeds': baselines[winner].unsqueeze(0).repeat(len(config['seeds']), 1, 1)}
    predictions = {'node_ids': f['node_ids'], 'baselines': baselines, 'families': families}
    save_tensor(dest / 'predictions.pt', predictions)
    control = fixed_control(old_pick, full, pb, f['node_ids'], config['seeds'])
    control['source_selection_path'] = str(old_dest / 'selection.json')
    control['source_selection_sha256'] = sha256(old_dest / 'selection.json')
    save_tensor(dest / 'fixed_control.pt', control)
    if strategy == 'full' and not config['smoke']:
        _verify_full_reproduction(dest, old_dest, selection, predictions)
    write_json(dest / 'metadata.json', {**meta, 'member_count': len(members), 'trained_count': trained,
        'reused_count': reused, 'train_seconds': train_seconds, 'elapsed_seconds': time.monotonic() - start,
        'architecture': config['architecture'], 'fixed_control_source': str(old_dest / 'selection.json'),
        'test_labels_read': False})
    verify_files(config['source_code'])
    verify_files(config['input_completions'])
    artifacts = {str(q.relative_to(dest)): sha256(q) for q in sorted(dest.rglob('*'))
                 if q.is_file() and q.name != 'complete.json' and '.tmp.' not in q.name}
    write_json(dest / 'complete.json', {'strategy': strategy, 'router_id': router_id, 'dataset': dataset,
        'config_sha256': config_hash, 'artifacts': artifacts, 'all_checks_pass': True, 'test_labels_read': False})
    print({'strategy': strategy, 'router': router_id, 'dataset': dataset, 'complete': True,
           'selected': selection['optimized']}, flush=True)
    return strategy, router_id, dataset
