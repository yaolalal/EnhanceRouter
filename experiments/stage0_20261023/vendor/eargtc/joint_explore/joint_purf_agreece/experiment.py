"""Own-cell training, complete checkpoint selection, and all-45 freeze barrier."""
import fcntl
import os
from pathlib import Path
import tempfile
import torch
from .protocol import RUN, CELLS, cell, key, read, write, save, sha, check, training_open, verify_code, load
from .training import train, make_model, predict, score
from .search import cid, catalog, rank, from_old
from eargtc.joint_explore.joint_purf_v10_v2.protocol import diagnostic


def result(dest, cfg, seed):
    r = read(dest/'result.json')
    check({str(dest/'checkpoint.pt'): r['checkpoint_sha256']})
    if r['config'] != cfg or r['training_seed'] != seed or r['test_truth_loaded'] is not False:
        raise ValueError('Candidate identity differs')
    return r


def fit(base, cfg, seed, f, s, p, h):
    training_open()
    dest = base/cid(cfg)
    if dest.exists():
        return result(dest,cfg,seed)
    member = train(f,s,p,h,cfg,seed)
    base.mkdir(parents=True,exist_ok=True)
    # Publish both files together; a killed write never looks like a finished candidate.
    tmp = Path(tempfile.mkdtemp(prefix='.candidate_',dir=base))
    save(tmp/'checkpoint.pt',member)
    r = dict(id=cid(cfg), config=cfg, training_seed=seed, accuracy=member['checkpoints']['accuracy']['accuracy'],
             nll=member['checkpoints']['accuracy']['nll'], step=member['checkpoints']['accuracy']['step'],
             selected=member['checkpoints']['accuracy'], checkpoint_sha256=sha(tmp/'checkpoint.pt'),
             train_seconds=member['train_seconds'], coverage=member['coverage'],
             selected_coverage=member['checkpoint_coverage']['accuracy'], test_truth_loaded=False,
             slurm_job_id=os.environ['SLURM_JOB_ID'])
    write(tmp/'result.json',r)
    training_open(); tmp.rename(dest)
    print('TRIAL',base.parent, r['id'], r['accuracy'], r['nll'], r['step'], round(r['train_seconds'],2),flush=True)
    return r


def export_model(dest, row, f, s, h, seed, shot):
    member = torch.load(dest/'candidates'/row['id']/'checkpoint.pt',map_location='cpu',weights_only=True)
    model = make_model(f,h,row['config']); model.load_state_dict(member['states']['accuracy'])
    prob = predict(model,f,h,f['node_ids'])
    torch.testing.assert_close(prob[s['selection_ids']],member['validation_probability'],atol=2e-6,rtol=0)
    final = dest/'final'
    save(final/'predictions.pt',dict(node_ids=f['node_ids'],probability=prob,training_seed=seed,split_seed=seed,shots=shot,inner_ensemble=False))
    save(final/'model.pt',dict(state=model.state_dict(),config=row['config'],training_seed=seed,shots=shot))
    return member


def run_cell(index):
    training_open(); reg = verify_code(); shot,seed,ds = CELLS[index]
    dest = cell(shot,seed,ds); dest.mkdir(parents=True,exist_ok=True)
    with (dest/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (dest/'COMPLETE.json').exists():
            check(read(dest/'COMPLETE.json')['files']); return
        f,s,p,h,split,router = load(shot,seed,ds)
        write(dest/'pools.json',dict(counts={k:len(v) for k,v in p.items()},gold_count=len(s['gold_ids']),
              selection_count=len(s['selection_ids']),historical_split_sha256=reg['inputs'][key(shot,seed,ds)]['historical_split_sha256'],test_truth_loaded=False))
        base = dest/'candidates'; rows = {}
        # Same old configuration with only KL removed / agreement CE added, retrained from scratch.
        control_cfg = from_old(reg['inputs'][key(shot,seed,ds)]['baseline_config'])
        control = fit(base,control_cfg,seed,f,s,p,h); rows[control['id']] = control
        diag = diagnostic(f,s,p,torch.load(base/control['id']/'checkpoint.pt',weights_only=True,map_location='cpu')['validation_probability'])
        write(dest/'DIAGNOSTICS.json',diag)
        for stage in range(1,10):
            anchor = rank(list(rows.values()))[0]
            configs = catalog(stage,anchor['config'],diag['class_weight_eligible'])
            spec = dict(stage=stage,configs=configs,anchor_id=anchor['id'],anchor_validation=dict(accuracy=anchor['accuracy'],nll=anchor['nll']),test_truth_loaded=False)
            specpath = dest/f'stage{stage}.json'
            if specpath.exists():
                # On resume reconstruct the original stage-start ranking, not partial current-stage trials.
                if read(specpath) != spec:
                    raise ValueError('Stage plan reconstruction differs')
            else:
                write(specpath,spec)
            for cfg in configs:
                r = fit(base,cfg,seed,f,s,p,h); rows[r['id']] = r
            winner = rank(list(rows.values()))[0]
            write(dest/f'stage{stage}_complete.json',dict(stage=stage,candidate_ids=[cid(c) for c in configs],
                  cumulative_candidates=len(rows),winner_id=winner['id'],accuracy=winner['accuracy'],nll=winner['nll']))
            print('STAGE COMPLETE',shot,seed,ds,stage,len(rows),winner['accuracy'],flush=True)
        ranked = rank(list(rows.values())); winner = ranked[0]
        # Export in a staging directory to preserve recovery after interrupted large prediction writes.
        stage_dir = Path(tempfile.mkdtemp(prefix='.export_',dir=dest))
        # export_model uses candidates in its input root; link read-only candidate directory for lookup.
        (stage_dir/'candidates').symlink_to(base.resolve(),target_is_directory=True)
        member = export_model(stage_dir,winner,f,s,h,seed,shot)
        write(stage_dir/'final/selection.json',dict(winner=winner,ranking=ranked,control_candidate_id=control['id'],
              training_class_counts=member['training_class_counts'],class_weights=member['class_weights'],
              selected_coverage=member['checkpoint_coverage']['accuracy'],test_truth_loaded=False))
        training_open()
        if (dest/'final').exists():
            # A fully published export without the final receipt must replay exactly before reuse.
            for name in ('selection.json',):
                if read(dest/'final'/name) != read(stage_dir/'final'/name):
                    raise ValueError('Partial final differs')
            for name in ('model.pt','predictions.pt'):
                old = torch.load(dest/'final'/name,weights_only=True,map_location='cpu')
                new = torch.load(stage_dir/'final'/name,weights_only=True,map_location='cpu')
                if name=='predictions.pt': torch.testing.assert_close(old['probability'],new['probability'],atol=0,rtol=0)
                else:
                    for k in old['state']: torch.testing.assert_close(old['state'][k],new['state'][k],atol=0,rtol=0)
        else:
            (stage_dir/'final').rename(dest/'final')
        files = {str(x):sha(x) for x in dest.rglob('*') if x.is_file() and not any(part.startswith('.') for part in x.relative_to(dest).parts)}
        write(dest/'COMPLETE.json',dict(files=files,all_checks_pass=True,shot=shot,seed=seed,dataset=ds,
              candidate_count=len(rows),test_truth_loaded=False,slurm_job_id=os.environ['SLURM_JOB_ID']))
        print('CELL COMPLETE',shot,seed,ds,len(rows),flush=True)


def smoke():
    training_open()
    records = []
    for shot,seed,ds in ((3,42,'cora'),(5,43,'citeseer'),(10,44,'ogbn-products')):
        f,s,p,h,split,router = load(shot,seed,ds)
        cfg = dict(catalog(1)[0],max_steps=2,validate_every=1,patience=5,aux_gold=.1,aux_pseudo=.1)
        member = train(f,s,p,h,cfg,seed)
        model = make_model(f,h,cfg); model.load_state_dict(member['states']['accuracy'])
        torch.testing.assert_close(predict(model,f,h,s['selection_ids']),member['validation_probability'],atol=0,rtol=0)
        if len(p['agreement']) and member['coverage']['agreement']['draws'] != 128:
            raise ValueError('Agreement CE sampling inactive')
        records.append(dict(shot=shot,seed=seed,dataset=ds,pools={k:len(v) for k,v in p.items()},hidden_shape=list(h.shape),coverage=member['coverage']))
        print('SMOKE PASS',shot,seed,ds,flush=True)
    write(RUN/'SMOKE.json',dict(all_checks_pass=True,records=records,test_truth_loaded=False,slurm_job_id=os.environ['SLURM_JOB_ID']))


def freeze():
    training_open(); reg = verify_code()
    check(reg['protected']); check(reg['inputs_hashes'])
    files = {str(RUN/'REGISTRATION.json'):sha(RUN/'REGISTRATION.json')}
    for shot,seed,ds in CELLS:
        dest = cell(shot,seed,ds); complete = read(dest/'COMPLETE.json'); check(complete['files'])
        if not complete['all_checks_pass'] or complete['test_truth_loaded'] is not False:
            raise ValueError('Cell incomplete')
        f,s,p,h,split,router = load(shot,seed,ds)
        selection = read(dest/'final/selection.json'); reconstructed = []
        configs = {cid(from_old(reg['inputs'][key(shot,seed,ds)]['baseline_config'])): from_old(reg['inputs'][key(shot,seed,ds)]['baseline_config'])}
        for stage in range(1,10):
            configs.update({cid(c):c for c in read(dest/f'stage{stage}.json')['configs']})
        for candidate_id,cfg in configs.items():
            r = result(dest/'candidates'/candidate_id,cfg,seed)
            m = torch.load(dest/'candidates'/candidate_id/'checkpoint.pt',weights_only=True,map_location='cpu')
            if m['config'] != cfg or m['training_seed'] != seed or m['split_seed'] != seed or m['inner_ensemble']:
                raise ValueError('Checkpoint identity differs')
            torch.testing.assert_close(m['validation_ids'],s['selection_ids'],atol=0,rtol=0)
            observed = score(m['validation_probability'],s['selection_targets'])
            if any(abs(observed[k]-r[k])>1e-12 for k in ('accuracy','nll')) or m['checkpoints']['accuracy'] != r['selected']:
                raise ValueError('Validation ranking does not reproduce')
            reconstructed.append(r)
        if rank(reconstructed) != selection['ranking'] or rank(reconstructed)[0] != selection['winner']:
            raise ValueError('Winner reconstruction differs')
        winner = selection['winner']; final = torch.load(dest/'final/model.pt',weights_only=True,map_location='cpu')
        member = torch.load(dest/'candidates'/winner['id']/'checkpoint.pt',weights_only=True,map_location='cpu')
        if final['config'] != winner['config'] or final['training_seed'] != seed or final['shots'] != shot:
            raise ValueError('Final identity differs')
        for name,t in final['state'].items(): torch.testing.assert_close(t,member['states']['accuracy'][name],atol=0,rtol=0)
        model = make_model(f,h,final['config']); model.load_state_dict(final['state'])
        pred = torch.load(dest/'final/predictions.pt',weights_only=True,map_location='cpu')
        if pred['training_seed'] != seed or pred['shots'] != shot or pred['inner_ensemble']:
            raise ValueError('Prediction identity differs')
        torch.testing.assert_close(pred['node_ids'],f['node_ids'],atol=0,rtol=0)
        torch.testing.assert_close(pred['probability'],predict(model,f,h,f['node_ids']),atol=0,rtol=0)
        files.update(complete['files']); files[str(dest/'COMPLETE.json')]=sha(dest/'COMPLETE.json')
        print('FREEZE VERIFIED',shot,seed,ds,len(reconstructed),flush=True)
    check(reg['protected']); check(reg['inputs_hashes'])
    write(RUN/'FROZEN.json',dict(files=files,all_checks_pass=True,cell_count=45,inner_seeds=[],
          test_truth_loaded=False,cross_split_selection=False,slurm_job_id=os.environ['SLURM_JOB_ID']))
    print('ALL 45 MODELS FROZEN',flush=True)
