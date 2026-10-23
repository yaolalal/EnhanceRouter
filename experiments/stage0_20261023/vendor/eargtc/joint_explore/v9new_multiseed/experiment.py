"""Additional 25 v9-new cells; all Fixed V2 numerical routines remain frozen imports."""
from dataclasses import asdict, replace
from pathlib import Path
import csv
import itertools
import json
import os
import time
import torch
from eargtc.multiseed_joint import experiment as historical
from eargtc.multiseed_joint.experiment import (
    DATASETS, SPLIT_SEEDS, REQUIRED, slurm_only, immutable_json, cell_path,
    config_path, prepare_arrays, validation_halves, aggregate_rows, write_csv,
    verify_cell,
)
from eargtc.joint_cross_repair.artifacts import (
    read_json, write_json, save_tensor, sha256, verify_files, validate_hidden,
)
from eargtc.joint_cross_repair.training import TrainConfig, _score, blend, scheme_b_probability
from eargtc.joint_cross_repair_search.bank import RECIPES, SEEDS, Recipe, train_member, member_logits
from eargtc.joint_cross_repair_search.experiment import candidate_bank
from eargtc.joint_cross_repair_search.selection import residual_probability
from eargtc.joint_five_router.experiment import select_candidates, _check_member
ROOT = historical.ROOT
OLD_RUN = historical.DEFAULT_RUN
DEFAULT_RUN = ROOT / 'runs/router_v9_new_joint_multiseed_20260912_v1'
DEPLOYMENT_RUN = (ROOT / 'runs/router_v9_new_final').resolve()
ROUTERS = ('r9_new',)
PREVIOUS_RUN = ROOT / 'runs/router_v9_joint_multiseed_20260911_v1'
PREVIOUS_ROUTERS = (*historical.ROUTERS, 'r8_v6', 'r9')


def validate_formal_config(config):
    old = read_json(config_path(OLD_RUN))
    expected = {'routers': list(ROUTERS), 'datasets': list(DATASETS),
                'split_seeds': list(SPLIT_SEEDS), 'inner_seeds': list(SEEDS),
                'recipes': old['recipes'], 'train': old['train'], 'smoke': False,
                **{k:old[k] for k in ('architecture','protocol','selection_rule',
                   'outer_seed_reduction','inner_seed_reduction','torch_version')}}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError('formal Joint configuration differs: ' + key)


def assert_new_root(root):
    root = Path(root).resolve()
    if root != DEFAULT_RUN.resolve() and not root.is_relative_to(Path('/tmp').resolve()):
        raise ValueError('use the dedicated v9-new run root or an isolated /tmp diagnostic root')
    return root



def assert_training_open(root):
    root = assert_new_root(root)
    for name in ('FROZEN.json','EVALUATION_STARTED.json','EVALUATION_STARTED',
                 'evaluation','evaluation.staging'):
        if (root/'joint'/name).exists():
            raise ValueError('Joint fits are closed after global freeze/evaluation')


def router_parity_records(root):
    """Require all 25 canonical handoffs to reproduce the frozen deployment."""
    root = assert_new_root(root)
    path = root/'DEPLOYMENT_PARITY.json'; gate = read_json(path)
    expected = {'all_checks_pass':True, 'all_exact':True, 'all_decisions_equal':True,
                'query_truth_loaded':False, 'query_truth_used_for_selection':False,
                'router_id':'r9_new', 'split_seeds':list(SPLIT_SEEDS),
                'datasets':list(DATASETS), 'cell_count':25}
    for key,value in expected.items():
        if gate.get(key) != value or (isinstance(value,bool) and gate.get(key) is not value):
            raise ValueError('exact all-25 frozen deployment parity is required: ' + key)
    inputs,code = gate.get('input_hashes'),gate.get('code_hashes')
    if not isinstance(inputs,dict) or not inputs or not isinstance(code,dict) or not code:
        raise ValueError('deployment parity requires complete input/code provenance')
    required = [root/'REGISTRATION.json',root/'ROUTER_RECIPE.json']
    required += [root/f'seed{seed}/INPUT_SEAL.json' for seed in SPLIT_SEEDS]
    required += [path for seed,ds in itertools.product(SPLIT_SEEDS,DATASETS)
                 for path in (DEPLOYMENT_RUN/f'seed{seed}/pipeline/{ds}.pt',
                              root/f'seed{seed}/router/r9_new/{ds}/predictions.pt',
                              root/f'seed{seed}/router/r9_new/{ds}/FROZEN.json')]
    if not {str(p) for p in required}.issubset(inputs):
        raise ValueError('deployment parity must bind all 25 deployed/canonical outputs and handoffs')
    recipe_path = root/'ROUTER_RECIPE.json'
    if gate.get('recipe_sha256') != inputs[str(recipe_path)]:
        raise ValueError('deployment parity recipe identity differs')
    records = {}
    for mapping in (inputs,code,{str(path):sha256(path)}):
        for name,digest in mapping.items():
            if not Path(name).is_absolute() or (name in records and records[name] != digest):
                raise ValueError('conflicting deployment parity provenance')
            records[name] = digest
    verify_files(records)
    return records


def verify_router_parity(root,config):
    records = router_parity_records(root)
    if any(config.get('reference_config',{}).get(name) != digest for name,digest in records.items()):
        raise ValueError('deployment parity differs from frozen Joint configuration')
    return records

def source_records():
    old = read_json(config_path(OLD_RUN))
    verify_files(old['source_code']); verify_files(old['preserved_snapshot'])
    own = list(Path(__file__).parent.glob('*.py')) + [ROOT/'scripts/v9new_multiseed/joint.py']
    return {**old['source_code'], **{str(p): sha256(p) for p in own}}


def initialize(root=DEFAULT_RUN):
    slurm_only(); root = assert_new_root(root)
    parity = router_parity_records(root)
    old = read_json(config_path(OLD_RUN))
    config = {**old, 'routers': list(ROUTERS), 'source_code': source_records(),
              'reference_config': {**parity, **previous_report_records(), **{str(p):sha256(p) for p in (config_path(OLD_RUN),
                  OLD_RUN/'joint/FROZEN.json', OLD_RUN/'joint/evaluation/complete.json')}},
              'historical_run': str(OLD_RUN), 'previous_seven_router_run': str(PREVIOUS_RUN),
              'gold_reuse': False,
              'interpretation': 'fixed Joint recipe; frozen Router-v9-new selected using posthoc query metrics; not untouched-test evaluation',
              'torch_version': str(torch.__version__),
              'gold_parity': 'all_25_cells_all_inner_seeds_four_checkpoint_probabilities'}
    validate_formal_config(config)
    if str(torch.__version__) != old['torch_version']:
        raise ValueError('Joint PyTorch version differs from old benchmark')
    immutable_json(config_path(root), config)
    return config


def input_paths(root, seed, rid, ds):
    root = assert_new_root(root)
    expected = historical.input_paths(OLD_RUN, seed, '', ds)
    # Split/upstream symlinks expose exactly the historical read-only producers.
    for relative, target in [('splits', OLD_RUN/'splits'),
                             (f'seed{seed}/upstream', OLD_RUN/f'seed{seed}/upstream')]:
        if (root/relative).resolve() != target.resolve():
            raise ValueError('safe historical input link differs: ' + relative)
    return {**{k: p.resolve() for k,p in expected.items() if k != 'router'},
            'router': root/f'seed{seed}/router/{rid}/{ds}/predictions.pt'}


def validate_handoff(frozen, seed, rid, ds):
    if (frozen.get('split_seed') != seed or frozen.get('router_id') != rid
            or frozen.get('dataset') != ds or frozen.get('query_truth_loaded') is not False
            or frozen.get('query_truth_used_for_selection') is not False
            or frozen.get('upstream_selection') != 'posthoc query metrics'
            or not frozen.get('predictions_sha256')
            or not frozen.get('input_hashes') or not frozen.get('code_hashes')):
        raise ValueError('unsafe or mismatched v9-new frozen handoff')


def bind_inputs(root, seed, rid, ds, paths):
    base = OLD_RUN/f'seed{seed}/upstream/{ds}'
    completion_path = base/'COMPLETE.json'; completion = read_json(completion_path)
    if (completion.get('dataset') != ds or completion.get('split_seed') != seed
            or completion.get('all_checks_pass') is not True
            or completion.get('test_truth_exported') is not False):
        raise ValueError('unsafe upstream completion')
    hashes = {str(base/name): digest for name,digest in completion['files'].items()}
    for key in ('upstream','embeddings','supervision'):
        if str(paths[key]) not in hashes:
            raise ValueError('completion does not bind ' + key)
    audit_path = OLD_RUN/'SPLIT_AUDIT.json'; audit = read_json(audit_path)
    if (audit.get('all_checks_pass') is not True or audit.get('seeds') != list(SPLIT_SEEDS)
            or {(r['dataset'],r['seed']) for r in audit['records']} != set(itertools.product(DATASETS,SPLIT_SEEDS))):
        raise ValueError('all-25 historical split audit required')
    records = [r for r in audit['records'] if r['dataset']==ds and r['seed']==seed]
    if len(records)!=1 or (OLD_RUN/records[0]['export']).resolve()!=paths['split']:
        raise ValueError('historical split identity differs')
    hashes[str(paths['split'])] = records[0]['export_sha256']
    frozen_path = paths['router'].parent/'FROZEN.json'; frozen = read_json(frozen_path)
    validate_handoff(frozen,seed,rid,ds)
    for records in (frozen['input_hashes'], frozen['code_hashes']):
        for name,digest in records.items():
            if not Path(name).is_absolute() or (name in hashes and hashes[name] != digest):
                raise ValueError('v9-new provenance overrides canonical upstream hash')
            hashes[name] = digest
    hashes[str(paths['router'])] = frozen['predictions_sha256']
    old_cell = historical.cell_path(OLD_RUN,seed,'r8_full',ds)
    old_lock_path = OLD_RUN/'joint/FROZEN.json'
    old_lock = read_json(old_lock_path)
    old_completion_path = old_cell/'complete.json'
    verify_files({str(old_completion_path):old_lock['files'][str(old_completion_path)]})
    old_completion = read_json(old_completion_path)
    old_meta_path = old_cell/'metadata.json'
    verify_files({str(old_meta_path):old_completion['artifacts']['metadata.json']})
    historical_meta = read_json(old_meta_path)
    hidden_digest = historical_meta['input_hashes'][str(paths['hidden'])]
    if str(paths['hidden']) in hashes and hashes[str(paths['hidden'])] != hidden_digest:
        raise ValueError('v9-new provenance overrides historical hidden hash')
    hashes[str(paths['hidden'])] = hidden_digest
    hashes[str(old_lock_path)] = sha256(old_lock_path)
    hashes[str(old_completion_path)] = old_lock['files'][str(old_completion_path)]
    hashes[str(old_meta_path)] = old_completion['artifacts']['metadata.json']
    for path in (completion_path,audit_path,frozen_path):
        digest = sha256(path)
        if str(path) in hashes and hashes[str(path)] != digest:
            raise ValueError('v9-new provenance metadata hash conflict')
        hashes[str(path)] = digest
    verify_files(hashes)
    return hashes


def load_inputs(root, seed, rid, ds):
    paths = input_paths(root,seed,rid,ds); hashes = bind_inputs(root,seed,rid,ds,paths)
    split = read_json(paths['split'])
    if split['dataset']!=ds or split['split_seed']!=seed or split.get('test_truth_exported') is not False:
        raise ValueError('split identity/truth isolation differs')
    arrays = {k:torch.load(paths[k],map_location='cpu',weights_only=True,mmap=True)
              for k in ('upstream','embeddings','supervision','router')}
    f,s,p = prepare_arrays(split,arrays['upstream'],arrays['embeddings'],arrays['supervision'],arrays['router'],ds)
    hidden = torch.load(paths['hidden'],map_location='cpu',weights_only=True,mmap=True)
    validate_hidden(hidden,split['num_nodes'])
    if hidden['metadata']['dataset'] != ds:
        raise ValueError('hidden dataset differs')
    return f,s,p,hidden['hidden'],split,hashes


def verify_gold_parity(root,seed,rid,ds,f,s,h,inputs,config):
    """Fresh fits must reproduce the historical router-independent gold members."""
    records = []
    for inner in SEEDS:
        donor = OLD_RUN/f'seed{seed}/joint_gold/{ds}/inner{inner}'
        marker = read_json(donor/'complete.json'); identity = marker['identity']
        cfg = replace(TrainConfig(**config['train']),seed=inner)
        expected = {'split_seed':seed,'dataset':ds,'inner_seed':inner,
                    'inputs':{str(p):inputs[str(p)] for k,p in input_paths(root,seed,rid,ds).items() if k!='router'},
                    'train':asdict(cfg),'recipe':asdict(RECIPES[0]),
                    'config_sha256':sha256(config_path(OLD_RUN)),'test_labels_read':False}
        if identity != expected:
            raise ValueError('historical gold identity differs')
        path = donor/'gold_only.pt'; verify_files({str(path):marker['sha256']})
        old = torch.load(path,weights_only=True,map_location='cpu')
        fresh = torch.load(cell_path(root,seed,rid,ds)/f'bank/inner{inner}/gold_only.pt',weights_only=True,map_location='cpu')
        _check_member(old,cfg,RECIPES[0],s); _check_member(fresh,cfg,RECIPES[0],s)
        for branch,criterion in itertools.product(('g','l'),('accuracy','nll')):
            a = member_logits(fresh,branch,criterion,f,h,s['selection_ids'],cfg).softmax(1)
            b = member_logits(old,branch,criterion,f,h,s['selection_ids'],cfg).softmax(1)
            if not torch.allclose(a,b,atol=2e-6,rtol=0) or not torch.equal(a.argmax(1),b.argmax(1)):
                raise ValueError('fresh gold checkpoint differs from historical probability')
        records.append({'inner_seed':inner,'donor':str(path),'sha256':marker['sha256'],
                        'four_checkpoints_reproduced':True})
    return records

def run_cell(root, seed, rid, ds):
    slurm_only()
    root = assert_new_root(root)
    config = read_json(config_path(root)); validate_formal_config(config); cfg_hash = sha256(config_path(root))
    if seed not in config['split_seeds'] or rid not in config['routers'] or ds not in config['datasets']:
        raise ValueError('unknown cell')
    verify_router_parity(root,config)
    verify_files(config['source_code']); verify_files(config['reference_config'])
    dest = cell_path(root, seed, rid, ds)
    if (dest / 'complete.json').exists():
        return verify_cell(root, seed, rid, ds, cfg_hash)
    assert_training_open(root)
    dest.mkdir(parents=True, exist_ok=True)
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
    root = assert_new_root(root); cfg_path = config_path(root); config = read_json(cfg_path)
    validate_formal_config(config)
    verify_router_parity(root,config)
    verify_files(config['source_code']); verify_files(config['preserved_snapshot']); verify_files(config['reference_config'])
    cfg_hash = sha256(cfg_path); files = {str(cfg_path): cfg_hash}; audits = {}
    cells = list(itertools.product(config['split_seeds'], config['routers'], config['datasets']))
    if config['smoke']:
        raise ValueError('smoke runs cannot freeze for test evaluation')
    if (config['split_seeds'] != list(SPLIT_SEEDS) or config['routers'] != list(ROUTERS)
            or config['datasets'] != list(DATASETS) or config['inner_seeds'] != list(SEEDS)):
        raise ValueError('formal population must contain 25 cells with three inner seeds')
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
        gold_parity = verify_gold_parity(root,seed,rid,ds,f,s,hidden,inputs,config)
        files[str(dest / 'complete.json')] = sha256(dest / 'complete.json')
        audits[f'{seed}/{rid}/{ds}'] = {'member_count': len(markers), 'pool_counts': {k: len(v) for k, v in p.items()},
            'gold_count': len(s['gold_ids']), 'selection_count': len(s['selection_ids']),
            'threshold_count': len(s['threshold_ids']), 'query_count': len(s['test_ids']),
            'validation_reproduced': True, 'gold_parity': gold_parity}
        print('Pretest validated', seed, rid, ds, flush=True)
        del f, s, p, hidden, predictions, validation, saved_s, saved_p, pb
    lock = {'schema': 'v9new_multiseed_joint_fixed_v2_freeze_v1', 'cell_count': len(cells),
            'split_seeds': config['split_seeds'], 'inner_seeds': config['inner_seeds'],
            'routers': config['routers'], 'datasets': config['datasets'],
            'test_labels_read': False, 'files': files, 'cells': audits}
    immutable_json(root / 'joint' / 'FROZEN.json', lock)
    return lock


def verify_frozen(root):
    slurm_only()
    root = assert_new_root(root); config = read_json(config_path(root))
    validate_formal_config(config)
    lock = read_json(root / 'joint' / 'FROZEN.json')
    cells = list(itertools.product(SPLIT_SEEDS, ROUTERS, DATASETS))
    expected = {str(config_path(root))} | {str(cell_path(root, seed, rid, ds) / 'complete.json') for seed, rid, ds in cells}
    if (config['smoke'] or lock.get('test_labels_read') is not False or lock.get('cell_count') != 25
            or set(lock.get('files', {})) != expected or lock.get('split_seeds') != list(SPLIT_SEEDS)
            or lock.get('inner_seeds') != list(SEEDS) or lock.get('routers') != list(ROUTERS)
            or lock.get('datasets') != list(DATASETS)):
        raise ValueError('all 25 requested cells must freeze before opening test truth')
    verify_files(lock['files']); validate_formal_config(config)
    verify_router_parity(root,config)
    verify_files(config['source_code']); verify_files(config['preserved_snapshot']); verify_files(config['reference_config'])
    if set(lock.get('cells', {})) != {f'{seed}/{rid}/{ds}' for seed,rid,ds in cells}:
        raise ValueError('global freeze has incomplete audit records')
    for seed, rid, ds in cells:
        audit = lock['cells'][f'{seed}/{rid}/{ds}']
        parity = audit.get('gold_parity', [])
        if (audit.get('validation_reproduced') is not True or audit.get('member_count') != 27
                or [r.get('inner_seed') for r in parity] != list(SEEDS)
                or any(r.get('four_checkpoints_reproduced') is not True for r in parity)):
            raise ValueError('global freeze lacks validation/gold checkpoint parity')
        verify_files({r['donor']:r['sha256'] for r in parity})
        verify_cell(root, seed, rid, ds, sha256(config_path(root)))
    return config, lock

def evaluate(root=DEFAULT_RUN):
    slurm_only()
    from eargtc.joint_five_router.evaluation import classification_report
    root = assert_new_root(root)
    config, lock = verify_frozen(root)  # No graph or query targets are read above this line.
    output = root / 'joint' / 'evaluation'
    if output.exists():
        raise FileExistsError('final evaluation is immutable')
    stage = output.with_name('evaluation.staging')
    if stage.exists():
        raise FileExistsError('evaluation staging exists; inspect before retrying')
    stage.mkdir(parents=True)
    immutable_json(root/'joint/EVALUATION_STARTED.json',
                   {'frozen_sha256':sha256(root/'joint/FROZEN.json'),
                    'slurm_job_id':os.environ['SLURM_JOB_ID']})
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
    lines = ['# Router-v9-new Joint-Fixed across five outer splits', '',
        'All 25 cells were trained and selected using safe support/validation targets, audited, and frozen before this evaluation read query truth. '
        'The unchanged Fixed V2 trainer uses inner seeds 42/43/44 and averages their probabilities within each outer split. '
        'Main results average the five split-level metrics from outer split seeds 42–46; probabilities are never averaged across outer splits.', '',
        'Values below are percentages: mean ± population standard deviation (ddof=0), matching the reference common/utils.py convention. '
        'CSV also includes sample standard deviation (ddof=1). Pseudo-label accuracy concerns accepted Router labels; '
        'coverage divides accepted count by the stated evaluation population. Final classification covers every node in that population.', '',
        'Full query and the reference 1000-node sample share each frozen trained model. '
        'The eval1000 membership is identical to the previous benchmark for every outer split. '
        'Each split reuses its historical safe expert-head outputs; Router-v9-new deployment outputs are reproduced exactly and only Joint fits are new. '
        'Raw SSL embeddings and LLM hidden remain frozen external dependencies. '
        'The frozen Router-v9-new was selected using posthoc query metrics; this is fixed-recipe reproduction, '
        'not an untouched-test evaluation.', '']
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
    combined_receipt = combine_reports(stage)
    write_json(stage / 'complete.json', {'all_checks_pass': True, 'cell_count': 25, 'metric_rows': len(rows), 'combined': combined_receipt,
        'frozen_sha256': payload['frozen_sha256'], 'test_label_hashes': label_hashes,
        'artifacts': {path.name: sha256(path) for path in stage.iterdir() if path.is_file()}})
    stage.rename(output)
    return output


def previous_report_records():
    """Bind the sealed seven-Router report using paths relative to its evaluation root."""
    base = PREVIOUS_RUN/'joint/evaluation'
    marker = base/'complete.json'
    complete = read_json(marker)
    combined_path = base/'combined_seven_router/complete.json'
    combined = read_json(combined_path)
    if (complete.get('all_checks_pass') is not True or complete.get('cell_count') != 25
            or complete.get('metric_rows') != 50 or combined.get('cell_count') != 175
            or combined.get('metric_rows') != 350 or combined.get('aggregate_groups') != 70
            or combined.get('historical_rows_preserved_exactly') is not True
            or complete.get('combined') != {**combined, 'completion_sha256':sha256(combined_path)}):
        raise ValueError('previous seven-router evaluation completion differs')
    artifacts = combined.get('artifacts', {})
    if not {'combined_seven_router/per_split.csv', 'combined_seven_router/mean_std.csv'}.issubset(artifacts):
        raise ValueError('previous seven-router report provenance is incomplete')
    records = {str(base/name):digest for name,digest in artifacts.items()}
    records.update({str(base/name):digest for name,digest in complete['artifacts'].items()})
    records.update(combined['historical_completion'])
    lock = PREVIOUS_RUN/'joint/FROZEN.json'
    records[str(lock)] = complete['frozen_sha256']
    lock_payload = read_json(lock)
    old_config = config_path(PREVIOUS_RUN)
    records[str(old_config)] = lock_payload['files'][str(old_config)]
    prior_config = read_json(old_config)
    records.update(prior_config['source_code'])
    records.update(prior_config['preserved_snapshot'])
    for path in (marker, combined_path):
        records[str(path)] = sha256(path)
    verify_files(records)
    return records


def combine_reports(stage):
    """Append v9-new rows in a new directory, retaining historical CSV bytes exactly."""
    stage = Path(stage)
    previous_records = previous_report_records()
    old = PREVIOUS_RUN/'joint/evaluation/combined_seven_router'
    dest = stage/'combined_eight_router'; dest.mkdir()
    def read_rows(path):
        with path.open(newline='') as handle:
            return list(csv.DictReader(handle))
    combined = {}
    for filename,old_count,new_count,keys in [
            ('per_split.csv',350,50,('router','dataset','population','split_seed')),
            ('mean_std.csv',70,10,('router','dataset','population'))]:
        a,b = old/filename,stage/filename
        old_rows,new_rows = read_rows(a),read_rows(b)
        if len(old_rows)!=old_count or len(new_rows)!=new_count:
            raise ValueError('incomplete eight-router table population')
        rows = old_rows+new_rows
        if len({tuple(r[k] for k in keys) for r in rows}) != len(rows):
            raise ValueError('duplicate eight-router table identity')
        if set(r['router'] for r in new_rows)!=set(ROUTERS):
            raise ValueError('new table contains unexpected Router')
        if set(r['router'] for r in old_rows)!=set(PREVIOUS_ROUTERS):
            raise ValueError('historical table Router identities differ')
        for table, routers in ((old_rows, PREVIOUS_ROUTERS), (new_rows, ROUTERS)):
            expected = set(itertools.product(routers, DATASETS, ('full_query','eval1000')))
            if 'split_seed' in keys:
                expected = {(*identity, str(seed)) for identity in expected for seed in SPLIT_SEEDS}
            if {tuple(row[key] for key in keys) for row in table} != expected:
                raise ValueError('eight-router table population identity differs')
        old_bytes,new_bytes = a.read_bytes(),b.read_bytes()
        old_header = old_bytes.splitlines(keepends=True)[0]
        new_header = new_bytes.splitlines(keepends=True)[0]
        if old_header != new_header or not old_bytes.endswith(b'\n'):
            raise ValueError('CSV schema or historical termination differs')
        output = dest/filename
        output.write_bytes(old_bytes+new_bytes[len(new_header):])
        if output.read_bytes()[:len(old_bytes)] != old_bytes or read_rows(output)[:old_count] != old_rows:
            raise ValueError('historical table rows changed')
        combined[filename] = rows
        for population in ('full_query','eval1000'):
            write_csv(dest/f'{population}_{filename}',[r for r in rows if r['population']==population])
    lines = ['# 八种 Router + 相同 Joint-Fixed：五个外层划分', '',
             '旧七种 Router 的 175 个单元、预测与指标保持原样；本表追加 v9-new 的 25 个新单元。', '',
             '每个单元内部平均 inner 42/43/44 概率；表中平均 outer 42–46 的五个指标。'
             '主标准差使用 ddof=0，CSV 同时保留 ddof=1。eval1000 是 full-query 的报告子集，未从伪标签训练中排除。', '',
             'Joint 结构、损失、搜索与回退完全沿用旧实现：agreement 仅进入自身保持 KL；最终输出可能选择原专家融合。'
             'v9-new 逐单元精确复现冻结部署输出，Joint 使用原始固定配方重新拟合。'
             'v9-new 的上游配置由 posthoc query metrics 选择，因此这些结果属于固定配方复现，不能解释为从未观察测试集的独立评估。', '']
    for population in ('full_query','eval1000'):
        lines += [f'## {population}', '', '| Router | Dataset | Pseudo accuracy % | Coverage % | Joint accuracy % | Macro-F1 % |',
                  '|---|---|---:|---:|---:|---:|']
        for row in combined['mean_std.csv']:
            if row['population']!=population:
                continue
            values=[]
            for metric in ('pseudo_accuracy','pseudo_coverage','accuracy','macro_f1'):
                a,b=row[metric+'_mean'],row[metric+'_std_ddof0']
                values.append('undefined' if a=='' else f'{100*float(a):.2f} ± {100*float(b):.2f}')
            lines.append('| '+row['router']+' | '+row['dataset']+' | '+' | '.join(values)+' |')
        lines.append('')
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n')
    receipt = {'cell_count':200,'metric_rows':400,'aggregate_groups':80,
               'historical_rows_preserved_exactly':True,
               'historical_completion':{str(old/'complete.json'):sha256(old/'complete.json')},
               'previous_seven_router_provenance':previous_records,
               'artifacts':{str(p.relative_to(stage)):sha256(p) for p in dest.iterdir() if p.is_file()}}
    write_json(dest/'complete.json',receipt)
    return {**receipt,'completion_sha256':sha256(dest/'complete.json')}
