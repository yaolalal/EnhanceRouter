"""Register, train, replay and globally freeze the 25 independent cells."""
import itertools
import os
import time
from pathlib import Path
import torch
from .protocol import *
from .data import load_data
from .search import STAGES, initial_configs, stage_configs, identity, choose, DEFAULT, adapt_previous
from .training import fit, new_model, predict, score


def register():
    slurm()
    if (RUN/'REGISTRATION.json').exists():
        return training_open()
    from scripts.v9new_persistence.persist import verify as verify_persistence, mapped
    receipt = verify_persistence()
    manifest = read(SOURCE/'PERSISTENCE.json')
    inputs = {}
    def bind(path, digest):
        key = str(Path(path).resolve())
        if key in inputs and inputs[key] != digest:
            raise ValueError('Conflicting source digest')
        inputs[key] = digest
    for rel, digest in manifest['copied_files'].items():
        bind(SOURCE/rel, digest)
    for name, digest in manifest['sealed_hash_dependencies'].items():
        bind(mapped(name, manifest), digest)
    for name in ('PERSISTENCE.json','PERSISTENCE_COMPLETE.json'):
        bind(SOURCE/name, sha(SOURCE/name))
    snapshot = ROOT/'artifacts/preserved_versions/joint_v8_fixed_v2_20260908/SNAPSHOT.json'
    preserved = {str(ROOT/name):digest for name,digest in read(snapshot)['files'].items()}
    preserved[str(snapshot)] = sha(snapshot)
    verify(preserved)
    cfg = read(SOURCE/'joint/CONFIG.json')
    assert cfg['split_seeds'] == list(SEEDS) and cfg['datasets'] == list(DATASETS)
    from eargtc.joint_explore.p_fusion.experiment import verify_frozen as verify_previous
    verify_previous()
    previous_hashes={str(PREVIOUS/'FROZEN.json'):sha(PREVIOUS/'FROZEN.json')}
    previous_configs={}
    for seed,ds in itertools.product(SEEDS,DATASETS):
        old_cell=PREVIOUS/f'seed{seed}/{ds}'
        previous_hashes[str(old_cell/'COMPLETE.json')]=sha(old_cell/'COMPLETE.json')
        previous_hashes.update(read(old_cell/'COMPLETE.json')['files'])
        old_selection=read(old_cell/'selection.json')
        previous_configs[f'{seed}/{ds}']=adapt_previous(old_selection['final_config'])
    reg = dict(schema='p_fusion_tune_v1', previous_hashes=previous_hashes,previous_configs=previous_configs, source=str(SOURCE), deployment=str(DEPLOYMENT),
               seeds=list(SEEDS), datasets=list(DATASETS), inner_seeds=[], training_seed_rule='seed_equals_split_seed',
               input_hashes=inputs, code_hashes=code_records(), preserved_hashes=preserved,
               stages=list(STAGES), defaults=DEFAULT, initial_configs={key:initial_configs(int(key.split('/')[0]),cfg) for key,cfg in previous_configs.items()},
               max_search_trials_per_cell=46,max_total_trials_per_cell=48,
               selection_rule='checkpoint_half_accuracy_nll_earliest; search_half_accuracy_nll_id; independent_outer',
               output='softmax(single_fusion_logits)', test_truth_read=False,
               upstream_selection='historical Router-v9-new selected using posthoc query metrics',
               historical_ensemble_inner_seeds=cfg['inner_seeds'], slurm_job_id=os.environ['SLURM_JOB_ID'])
    write(RUN/'SOURCE_VERIFIED.json', receipt)
    write(RUN/'REGISTRATION.json', reg)
    print('REGISTERED', len(inputs), 'input hashes; inner seeds disabled', flush=True)
    return reg


def run_cell(seed, ds):
    reg = training_open()
    dest = cell(seed, ds)
    if (dest/'COMPLETE.json').exists():
        verify(read(dest/'COMPLETE.json')['files'])
        return
    import fcntl
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _train(seed, ds, reg, dest)


def _train(seed, ds, reg, dest):
    started = time.monotonic()
    data, cp, select, f, s, pools, split, metadata = load_data(seed, ds)
    print('INPUTS VERIFIED', seed, ds, metadata['counts'], flush=True)
    write(dest/'INPUTS.json', metadata)
    rows, members, stage_winners = [], {}, {}
    def trial(cfg):
        if cfg['seed'] != seed:
            raise ValueError('Candidate seed mismatch')
        cid = identity(cfg)
        if cid in members:
            return next(r for r in rows if r['id'] == cid)
        path = dest/'trials'/f'{cid}.pt'
        marker = path.with_suffix('.json')
        if marker.exists():
            receipt = read(marker)
            verify({str(path):receipt['sha256']})
            member = load(path)
            if member['config'] != cfg or not torch.equal(member['validation_ids'], select['ids']):
                raise ValueError('Resumed trial identity differs')
        else:
            if len(rows)>=reg['max_total_trials_per_cell']:
                raise ValueError('Registered fit budget exhausted')
            if path.exists():
                raise ValueError('Unsealed trial exists; inspect before resuming')
            member = fit(data, cp, select, cfg)
            save(path, member)
            write(marker, dict(sha256=sha(path), seed=seed, config=cfg, test_truth_read=False))
        row = dict(id=cid, config=cfg, **member['selection'], checkpoint=member['checkpoint'],
                   elapsed_seconds=member['elapsed_seconds'])
        rows.append(row)
        members[cid] = member
        print('TRIAL', seed, ds, cid, 'val', row['accuracy'], 'step', row['checkpoint']['step'], flush=True)
        return row
    incumbent = None
    for stage in STAGES:
        configs = initial_configs(seed,reg['previous_configs'][f'{seed}/{ds}']) if stage == 'training' else stage_configs(stage, incumbent['config'])
        candidates = [trial(cfg) for cfg in configs]
        if incumbent is not None:
            candidates.append(incumbent)
        incumbent = choose(candidates)
        stage_winners[stage] = dict(winner=incumbent['id'], candidate_ids=sorted({r['id'] for r in candidates}),
                                    accuracy=incumbent['accuracy'], nll=incumbent['nll'])
        print('STAGE', seed, ds, stage, incumbent['accuracy'], flush=True)
    if len(rows)>reg['max_search_trials_per_cell']:
        raise ValueError('Registered search budget exceeded before controls')
    final_id = incumbent['id']
    # This paired control is frozen after selection and cannot feed back into the search.
    gold_cfg = {**incumbent['config'], 'pseudo_weight':0., 'kd_weight':0.}
    no_kd_id=trial({**incumbent['config'],'kd_weight':0.})['id']
    gold_id = trial(gold_cfg)['id']
    base_id = identity({**DEFAULT, 'seed':seed})
    wanted = {final_id, gold_id, base_id, no_kd_id} | {r['winner'] for r in stage_winners.values()}
    probabilities = {}
    torch.set_num_threads(1)
    for cid in sorted(wanted):
        member = members[cid]
        model = new_model(data, member['config'])
        model.load_state_dict(member['state'])
        p = predict(model, data, data['node_ids'])
        if not torch.allclose(p[select['ids']], member['validation_probability'], atol=2e-6, rtol=0):
            raise ValueError('Full prediction does not reproduce validation')
        probabilities[cid] = p
    from .baseline import run_fixed
    fixed, fixed_meta = run_fixed(seed, ds, f, s, pools, data['text'])
    prior=load(PREVIOUS/f'seed{seed}/{ds}/predictions.pt')
    if not torch.equal(prior['node_ids'],data['node_ids']) or not torch.allclose(fixed,prior['fixed'],atol=2e-6,rtol=0) or not torch.equal(fixed.argmax(1),prior['fixed'].argmax(1)):
        raise ValueError('Prior Fixed replay differs')
    old_validation=load(PREVIOUS/f'seed{seed}/{ds}/validation.pt')
    for name,part in (('checkpoint',cp),('selection',select)):
        assert all(torch.equal(part[k],old_validation[name][k]) for k in part)
    if len(rows)>reg['max_total_trials_per_cell']:
        raise ValueError('Registered search budget exceeded')
    save(dest/'predictions.pt', dict(node_ids=data['node_ids'], p_fusion=probabilities[final_id],
        fixed=fixed, previous_fusion=prior['p_fusion'], no_kd=probabilities[no_kd_id], gold_only=probabilities[gold_id], initial_fusion=probabilities[base_id],
        stages={name:probabilities[r['winner']] for name,r in stage_winners.items()}))
    save(dest/'validation.pt', dict(checkpoint=cp, selection=select, gold_ids=s['gold_ids'],
                                   query_ids=s['test_ids'], threshold_ids=s['threshold_ids'], accepted_ids=data['accepted_ids']))
    write(dest/'selection.json', dict(seed=seed, inner_seeds=[], final_id=final_id, matched_gold_id=gold_id, no_kd_id=no_kd_id,
        base_id=base_id, stages=stage_winners, trials=rows, final_config=members[final_id]['config'],
        fixed=fixed_meta, test_truth_read=False, elapsed_seconds=sum(r['elapsed_seconds'] for r in rows)))
    verify(metadata['input_hashes'])
    verify(reg['code_hashes'])
    files = {str(p):sha(p) for p in dest.rglob('*') if p.is_file() and p.name not in ('COMPLETE.json','train.lock') and '.tmp' not in p.name}
    write(dest/'COMPLETE.json', dict(seed=seed, dataset=ds, files=files, all_checks_pass=True,
        registration_sha256=sha(RUN/'REGISTRATION.json'), inner_seeds=[], test_truth_read=False))
    print('CELL COMPLETE', seed, ds, flush=True)


def audit_cell(seed, ds):
    torch.set_num_threads(1)
    dest = cell(seed, ds)
    complete = read(dest/'COMPLETE.json')
    assert complete['seed'] == seed and complete['dataset'] == ds and complete['inner_seeds'] == []
    assert complete['registration_sha256'] == sha(RUN/'REGISTRATION.json')
    verify(complete['files'])
    data, cp, select, f, s, pools, split, metadata = load_data(seed, ds)
    pick = read(dest/'selection.json')
    if pick['seed'] != seed or pick['inner_seeds'] != []:
        raise ValueError('Seed contract changed')
    rows = {r['id']:r for r in pick['trials']}
    for cid, row in rows.items():
        member = load(dest/'trials'/f'{cid}.pt')
        assert identity(member['config']) == cid and member['config'] == row['config']
        assert member['seed'] == seed and member['config']['seed'] == seed
        assert torch.equal(member['validation_ids'], select['ids'])
        actual = score(member['validation_probability'], select['targets'])
        assert actual == member['selection'] and all(actual[k] == row[k] for k in actual)
        best = min(member['history'], key=lambda r:(-r['accuracy'],r['nll'],r['step']))
        assert best == member['checkpoint'] == row['checkpoint']
        model = new_model(data,member['config']); model.load_state_dict(member['state'])
        vp = predict(model,data,select['ids'])
        assert torch.allclose(vp,member['validation_probability'],atol=2e-6,rtol=0)
        cp_score = score(predict(model,data,cp['ids']),cp['targets'])
        assert abs(cp_score['accuracy'] - best['accuracy']) < 1e-10
        assert abs(cp_score['nll'] - best['nll']) < 2e-6
    previous = None
    for stage in STAGES:
        record = pick['stages'][stage]
        configs = initial_configs(seed,read(RUN/'REGISTRATION.json')['previous_configs'][f'{seed}/{ds}']) if previous is None else stage_configs(stage,rows[previous]['config'])
        expected = {identity(c) for c in configs}
        if previous is not None:
            expected.add(previous)
        assert expected == set(record['candidate_ids'])
        chosen = choose([rows[cid] for cid in expected])
        if chosen['id'] != record['winner']:
            raise ValueError('Stage selection fails reconstruction')
        previous = chosen['id']
    assert pick['final_id'] == pick['stages']['optimization']['winner']
    saved = load(dest/'predictions.pt')
    if not torch.equal(saved['node_ids'], data['node_ids']):
        raise ValueError('Prediction node order changed')
    validation = load(dest/'validation.pt')
    for name, expected in (('checkpoint',cp),('selection',select)):
        assert all(torch.equal(validation[name][k],v) for k,v in expected.items())
    assert torch.equal(validation['accepted_ids'],data['accepted_ids'])
    for name,expected in (('gold_ids',s['gold_ids']),('query_ids',s['test_ids']),('threshold_ids',s['threshold_ids'])):
        assert torch.equal(validation[name],expected)
    replay_cache={}
    replay_targets=[(pick['final_id'],'p_fusion'),(pick['matched_gold_id'],'gold_only'),(pick['no_kd_id'],'no_kd'),(pick['base_id'],'initial_fusion')]
    replay_targets += [(record['winner'],'stage_'+name) for name,record in pick['stages'].items()]
    for cid, key in replay_targets:
        member = load(dest/'trials'/f'{cid}.pt')
        assert member['config']['seed'] == seed and member['seed'] == seed and member['test_truth_read'] is False
        model = new_model(data, member['config'])
        model.load_state_dict(member['state'])
        if cid not in replay_cache:
            replay_cache[cid]=predict(model,data,data['node_ids'])
        replay=replay_cache[cid]
        expected=saved['stages'][key[6:]] if key.startswith('stage_') else saved[key]
        if not torch.allclose(replay, expected, atol=2e-6, rtol=0) or not torch.equal(replay.argmax(1),expected.argmax(1)):
            raise ValueError('Checkpoint replay differs')
    from eargtc.joint_five_router.experiment import select_candidates
    fv = load(dest/'fixed/validation.pt')
    fp = read(dest/'fixed/selection.json')
    assert select_candidates(fv['candidates'],fv['router'],fv['targets']) == fp
    fixed_score = score(saved['fixed'][s['selection_ids']],s['selection_targets'])
    assert abs(fixed_score['accuracy'] - fp['optimized']['accuracy']) < 1e-10
    from .baseline import run_fixed
    replay_fixed, _ = run_fixed(seed,ds,f,s,pools,data['text'],replay=True)
    assert torch.allclose(replay_fixed,saved['fixed'],atol=2e-6,rtol=0)
    assert torch.equal(replay_fixed.argmax(1),saved['fixed'].argmax(1))
    for origin in read(dest/'fixed/origins.json').values():
        verify({origin['source']:origin['sha256']})
        assert load(origin['source'])['config']['seed'] == seed
    for value in [saved[k] for k in ('p_fusion','fixed','gold_only','initial_fusion','previous_fusion','no_kd')] + list(saved['stages'].values()):
        assert value.shape == (len(data['node_ids']),data['classes'])
        assert bool(torch.isfinite(value).all()) and bool((value >= 0).all())
        assert torch.allclose(value.sum(1),torch.ones(len(value)),atol=2e-6,rtol=0)
    prior=load(PREVIOUS/f'seed{seed}/{ds}/predictions.pt')
    assert torch.equal(saved['previous_fusion'],prior['p_fusion'])
    assert torch.equal(saved['fixed'].argmax(1),prior['fixed'].argmax(1))
    assert pick['trials'] and len(rows)<=48
    search_ids=set().union(*(set(r['candidate_ids']) for r in pick['stages'].values()))
    assert len(search_ids)<=46
    assert set(rows)==search_ids|{pick['matched_gold_id'],pick['no_kd_id']}
    gold=rows[pick['matched_gold_id']]['config'];no_kd=rows[pick['no_kd_id']]['config']
    assert gold['pseudo_weight']==0 and gold['kd_weight']==0 and no_kd['kd_weight']==0
    assert no_kd=={**pick['final_config'],'kd_weight':0.}
    assert gold=={**pick['final_config'],'pseudo_weight':0.,'kd_weight':0.}
    return dict(seed=seed,dataset=ds,checkpoint_replay=True,validation_selection_replayed=True,
                exact_deployment_parity=True,training_seed=seed,inner_seeds=[],test_truth_read=False,
                complete_sha256=sha(dest/'COMPLETE.json'))


def freeze():
    reg = training_open()
    torch.set_num_threads(1)
    cells = list(itertools.product(SEEDS,DATASETS))
    require_all_cells([(s,d) for s,d in cells if (cell(s,d)/'COMPLETE.json').exists()])
    audits = []
    for seed, ds in cells:
        audits.append(audit_cell(seed,ds))
        print('REPLAY VERIFIED',seed,ds,flush=True)
    verify(reg['preserved_hashes']); verify(reg['code_hashes']);verify(reg['previous_hashes'])
    files = {str(cell(s,d)/'COMPLETE.json'):sha(cell(s,d)/'COMPLETE.json') for s,d in cells}
    files[str(RUN/'REGISTRATION.json')] = sha(RUN/'REGISTRATION.json')
    write(RUN/'FROZEN.json',dict(cells=cells,audits=audits,files=files,test_truth_read=False,inner_seeds=[]))


def verify_frozen():
    slurm()
    lock = read(RUN/'FROZEN.json')
    require_all_cells(lock['cells'])
    assert lock['test_truth_read'] is False and lock['inner_seeds'] == [] and len(lock['audits']) == 25
    verify(lock['files'])
    reg = read(RUN/'REGISTRATION.json')
    verify(reg['code_hashes']); verify(reg['preserved_hashes']);verify(reg['previous_hashes'])
    for seed, ds in lock['cells']:
        verify(read(cell(seed,ds)/'COMPLETE.json')['files'])
    return lock
