"""Read-only historical input reuse; all numerical execution requires Slurm."""
import itertools
import os
from pathlib import Path
import shutil
import sys
import torch
from eargtc.joint_explore.joint_purf_v10_v1.protocol import read, write, save, sha, check, slurm
from eargtc.multiseed_joint.experiment import prepare_arrays
from eargtc.joint_cross_repair.artifacts import validate_hidden

REPO = Path(__file__).resolve().parents[3]
RUN = REPO / 'runs/joint_purf_agreece_20260916'
SHOTS_RUN = REPO / 'runs/router_v10_shots_v1_20260914'
FIVE_RUN = REPO / 'runs/joint_purf_v10_v2_20260914'
DATASETS = ('cora', 'citeseer', 'pubmed', 'arxiv', 'ogbn-products')
SHOTS = (3, 5, 10)
SEEDS = (42, 43, 44)
CELLS = tuple(itertools.product(SHOTS, SEEDS, DATASETS))


def key(shot, seed, ds):
    return f'{shot}/{seed}/{ds}'


def cell(shot, seed, ds):
    return RUN / f'shot{shot}/seed{seed}/{ds}'


def training_open():
    slurm()
    if any((RUN / name).exists() for name in ('FROZEN.json', 'EVALUATION_STARTED.json', 'COMPLETE.json')):
        raise RuntimeError('All fitting and selection closed after global freeze')


def verify_code():
    slurm()
    reg = read(RUN / 'REGISTRATION.json')
    check(reg['code_hashes'])
    return reg


def merge(*groups):
    out = {}
    for group in groups:
        for p, h in group.items():
            if p in out and out[p] != h:
                raise ValueError('Conflicting source digest: ' + p)
            out[p] = h
    return out


def validate_split(split, shot, seed, ds):
    if (split['dataset'], split['split_seed']) != (ds, seed) or split.get('shots', split.get('shot', shot)) != shot:
        raise ValueError('Historical split identity differs')
    if split['test_truth_exported'] is not False:
        raise ValueError('Truth was exported in split')
    groups = [set(split[k]) for k in ('support_ids', 'valid_ids', 'unlabeled_ids')]
    if any(len(g) != len(split[k]) for g, k in zip(groups, ('support_ids', 'valid_ids', 'unlabeled_ids'))):
        raise ValueError('Duplicate IDs')
    if any(a & b for a, b in itertools.combinations(groups, 2)) or set.union(*groups) != set(range(split['num_nodes'])):
        raise ValueError('Historical partition invalid')
    if split['evaluation_full_ids'] != split['unlabeled_ids']:
        raise ValueError('Full query order differs')
    fixed = split['evaluation_1000_ids']
    if not fixed or len(fixed) != len(set(fixed)) or not set(fixed) <= groups[2]:
        raise ValueError('Fixed evaluation invalid')
    if 'evaluation_test_ids' in split and fixed != split['evaluation_test_ids']:
        raise ValueError('Fixed evaluation order differs')


def register():
    training_open()
    if (RUN / 'REGISTRATION.json').exists():
        reg = verify_code()
        check(reg['protected'])
        check(reg['inputs_hashes'])
        print('REGISTRATION VERIFIED', flush=True)
        return
    r3path = SHOTS_RUN / 'DOWNSTREAM_REGISTRATION.json'
    r5path = FIVE_RUN / 'REGISTRATION.json'
    f3 = read(SHOTS_RUN / 'DOWNSTREAM_FROZEN.json')
    f5 = read(FIVE_RUN / 'FROZEN.json')
    for obj, count in ((f3, 30), (f5, 25)):
        if obj['all_checks_pass'] is not True or obj['cell_count'] != count:
            raise ValueError('Historical freeze incomplete')
    check({str(r3path): f3['files'][str(r3path)], str(r5path): f5['files'][str(r5path)]})
    r3, r5 = read(r3path), read(r5path)
    check(r3['code_hashes']); check(r5['code_hashes'])
    router_freeze = SHOTS_RUN / 'ROUTER_FROZEN.json'
    check({str(router_freeze): r3['router_frozen_sha256']})
    router_files = read(router_freeze)['files']
    snapshot = REPO / 'artifacts/preserved_versions/joint_v8_fixed_v2_20260908/SNAPSHOT.json'
    snapshot_hashes = {str(REPO / p): h for p, h in read(snapshot)['files'].items()}
    protected = merge(snapshot_hashes, r3['code_hashes'], r5['code_hashes'],
                      {str(p): sha(p) for p in (snapshot, r3path, r5path, SHOTS_RUN/'DOWNSTREAM_FROZEN.json', FIVE_RUN/'FROZEN.json', router_freeze)})
    records = {}
    for shot, seed, ds in CELLS:
        old = (r5 if shot == 5 else r3)['inputs'][f'{seed}/{ds}' if shot == 5 else key(shot, seed, ds)]
        pp = old['paths']
        split = read(pp['split'])
        validate_split(split, shot, seed, ds)
        # Do not serialize a rewritten split or replace any evaluation IDs.
        if shot == 5:
            historical = REPO / f'runs/router_v10_multiseed_data_20260912_v1/seed{seed}/splits/{ds}_5shot_seed{seed}.json'
            if Path(pp['split']).read_bytes() != historical.read_bytes():
                raise ValueError('Five-shot historical split bytes differ')
            final = FIVE_RUN / f'seed{seed}/{ds}/final'
            seal = f5['files']
        else:
            final = SHOTS_RUN / f'shot{shot}/seed{seed}/downstream/{ds}/final'
            seal = f3['files']
            if router_files[pp['router']] != old['hashes'][pp['router']]:
                raise ValueError('Shot Router seal differs')
        control = {name: str(final / name) for name in ('selection.json', 'predictions.pt', 'model.pt')}
        protected = merge(protected, {p: seal[p] for p in control.values()})
        selection = read(control['selection.json'])
        config = selection['config'] if shot == 5 else selection['winner']['config']
        records[key(shot, seed, ds)] = dict(paths=pp, hashes=old['hashes'], baseline=control,
                                          baseline_config=config, historical_split_sha256=old['hashes'][pp['split']])
    inputs_hashes = merge(*(r['hashes'] for r in records.values()))
    check(protected); check(inputs_hashes)
    paths = list((REPO/'eargtc/joint_explore/joint_purf_agreece').glob('*.py'))
    paths += [p for p in (REPO/'scripts/joint_explore/joint_purf_agreece').glob('*') if p.is_file()]
    paths += [REPO/'docs/joint_purf_agreece_plan_20260916.md']
    code = {str(p): sha(p) for p in paths}
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and path.endswith('.py') and Path(path).is_absolute() and Path(path).is_relative_to(REPO):
            code[path] = sha(path)
    for p in code:
        dest = RUN / 'source' / Path(p).relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    write(RUN/'REGISTRATION.json', dict(inputs=records, inputs_hashes=inputs_hashes, protected=protected, code_hashes=code,
          cell_count=45, shots=list(SHOTS), seeds=list(SEEDS), datasets=list(DATASETS), inner_seeds=[],
          router='Router-v10-MS-Original', llm='LLaMA3-8B (historical frozen probabilities and hidden)',
          test_truth_loaded=False, cross_split_selection=False, slurm_job_id=os.environ['SLURM_JOB_ID']))
    print('REGISTERED 45 CELLS', flush=True)


def load(shot, seed, ds):
    reg = verify_code()
    record = reg['inputs'][key(shot, seed, ds)]
    check(record['hashes'])
    pp = record['paths']; split = read(pp['split'])
    validate_split(split, shot, seed, ds)
    arrays = {k: torch.load(pp[k], weights_only=True, map_location='cpu', mmap=True)
              for k in ('upstream', 'embeddings', 'supervision', 'router')}
    f, s, p = prepare_arrays(split, arrays['upstream'], arrays['embeddings'], arrays['supervision'], arrays['router'], ds)
    hidden = torch.load(pp['hidden'], weights_only=True, map_location='cpu', mmap=True)
    validate_hidden(hidden, split['num_nodes'])
    if hidden['metadata']['dataset'] != ds:
        raise ValueError('Hidden dataset mismatch')
    agreement = p['agreement']
    gp, lp = f['logp_g'].argmax(1), f['logp_l'].argmax(1)
    if not torch.equal(gp[agreement], lp[agreement]) or not torch.equal(f['selected_pred'][agreement], gp[agreement]):
        raise ValueError('Agreement hard target differs from expert consensus')
    if any(t.requires_grad for t in (f['gnn_embeddings'], f['logp_g'], f['logp_l'], hidden['hidden'])):
        raise ValueError('Expert inputs must be frozen')
    return f, s, p, hidden['hidden'], split, arrays['router']
