"""Audit-only recovery of the immutable v1 search, before any test evaluation.

Slurm diagnosis 99395 found <= 1 ULP exp / 2 ULP mean variation in
Arxiv raw probabilities. No retraining, prediction replacement or reselection.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import torch
from eargtc.joint_v6_unified.experiment import (
    ROOT,DEFAULT_RUN,DATASETS,FAMILIES,SEEDS,require_slurm,read_json,write_json,
    sha256,verify_files,verify_registration,audit_member,score,shortlist,ensemble,
    choose,probability,_raw)

RECOVERY_ROOT=ROOT/'runs/joint_v6_unified_20260910_v2'
RAW_MAX_ABS=6e-8
RAW_MAX_REL=2.5e-7


def compare_raw(saved,replayed):
    require_slurm()
    probability(saved);probability(replayed)
    if saved.shape!=replayed.shape or saved.dtype!=torch.float32 or replayed.dtype!=saved.dtype:
        raise ValueError('Raw replay shape/dtype mismatch')
    difference=(saved.double()-replayed.double()).abs()
    max_abs=float(difference.max())
    relative=difference/replayed.double().abs().clamp_min(torch.finfo(torch.float64).tiny)
    max_relative=float(relative.max())
    changed=int((saved.argmax(1)!=replayed.argmax(1)).sum())
    # Both bounds must hold; an absolute allowance cannot hide a large relative
    # change in a low-probability class. Prediction identity is exact.
    if max_abs>RAW_MAX_ABS or max_relative>RAW_MAX_REL or changed:
        raise ValueError('Original expert changed beyond measured rounding tolerance')
    return dict(exact=torch.equal(saved,replayed),max_abs=max_abs,max_relative=max_relative,
                argmax_changes=changed)


def reader_files(root,original,datasets=DATASETS):
    """Bind the actual reader-visible paths, not only old absolute manifest paths."""
    root=Path(root);original=Path(original);files={}
    for ds in datasets:
        cell=root/ds;source=original/ds
        if not cell.is_symlink() or cell.resolve()!=source.resolve():
            raise ValueError('Recovery cell link changed')
        complete=read_json(cell/'COMPLETE.json')
        files[str(cell/'COMPLETE.json')]=sha256(cell/'COMPLETE.json')
        for path,digest in complete['files'].items():
            relative=Path(path).relative_to(source)
            visible=cell/relative
            if sha256(visible)!=digest:raise ValueError('Reader-visible artifact hash mismatch')
            files[str(visible)]=digest
    return files


def recovery_sources():
    files=[]
    for folder in ('eargtc/joint_v6_unified_audit_v2','scripts/joint_v6_unified_audit_v2'):
        for pattern in ('*.py','*.sbatch'):files.extend(sorted((ROOT/folder).glob(pattern)))
    files.extend([ROOT/'tests/test_joint_v6_unified_audit_v2.py',
                  ROOT/'docs/joint_v6_unified_work/raw_replay_diagnosis.json',
                  DEFAULT_RUN/'recovery_diagnostics/raw_replay_diagnosis.py',
                  DEFAULT_RUN/'slurm/finish_99347.err'])
    return {str(p):sha256(p) for p in files}


def prepare(root=RECOVERY_ROOT):
    require_slurm();root=Path(root).resolve();source=DEFAULT_RUN
    if root==source:raise ValueError('Recovery must use a new directory')
    if any((root/name).exists() for name in ('CONFIG.json','RECOVERY.json','FROZEN.json','evaluation')):
        raise FileExistsError('Recovery directory already initialized')
    if (source/'FROZEN.json').exists() or (source/'evaluation').exists():
        raise ValueError('Recovery requires the original unevaluated search')
    config=verify_registration(source,inputs=True)
    if config['smoke'] or config['datasets']!=list(DATASETS):raise ValueError('Formal five-cell search required')
    bound=recovery_sources()
    diagnosis=read_json(ROOT/'docs/joint_v6_unified_work/raw_replay_diagnosis.json')
    if diagnosis['test_truth_read'] is not False or not diagnosis['registered_source_manifest']['ok']:
        raise ValueError('Invalid diagnostic evidence')
    for ds in DATASETS:
        complete=read_json(source/ds/'COMPLETE.json')
        if (complete['dataset']!=ds or complete['config_sha256']!=sha256(source/'CONFIG.json')
                or complete['test_truth_read'] is not False or complete['member_count']!=18):
            raise ValueError('Invalid source cell')
        verify_files(complete['files'])
        bound[str(source/ds/'COMPLETE.json')]=sha256(source/ds/'COMPLETE.json')
    bound[str(source/'CONFIG.json')]=sha256(source/'CONFIG.json')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'CONFIG.json').open('xb') as f:f.write((source/'CONFIG.json').read_bytes())
    for ds in DATASETS:(root/ds).symlink_to(source/ds,target_is_directory=True)
    bound.update(reader_files(root,source))
    write_json(root/'RECOVERY.json',dict(schema='joint_v6_unified_audit_only_v2',
        original_run=str(source),original_failure_job='99347',diagnostic_job='99395',
        reason='Arxiv raw exp differs by <=1 ULP, mean <=2 ULP; no class changes',
        raw_tolerance=dict(max_abs=RAW_MAX_ABS,max_relative=RAW_MAX_REL,exact_argmax=True),
        checkpoint_tolerance=dict(atol=2e-5,rtol=1e-5,unchanged=True),
        prediction_bytes_unchanged=True,selection_unchanged=True,retrained=False,
        test_truth_read=False,files=bound,slurm_job_id=os.environ['SLURM_JOB_ID']))
    print('Recovery registered; all original predictions and choices preserved.',flush=True)


def audit_cell(root,dataset,device='cpu'):
    require_slurm()
    from eargtc.joint_v6_unified.data import load_data
    from eargtc.joint_v6_unified.training import predict
    config=read_json(root/'CONFIG.json');cell=root/dataset
    complete=read_json(cell/'COMPLETE.json')
    if complete['dataset']!=dataset or complete['config_sha256']!=sha256(root/'CONFIG.json') or complete['test_truth_read'] is not False:
        raise ValueError('Cell is not linked to registered search')
    verify_files(complete['files'])
    train,val,inf,meta=load_data(dataset)
    if meta!=config['metadata'][dataset]:raise ValueError('Data audit changed')
    sel=read_json(cell/'selection.json');by_id={c['id']:c for c in config['recipes']}
    trials=[];members={}
    expected_names=set()
    for cfg in config['recipes']:
        name=f"{cfg['id']}_s42";expected_names.add(name)
        member=torch.load(cell/'members'/(name+'.pt'),weights_only=True,map_location='cpu')
        audit_member(member,cfg,42,train,val)
        trials.append(dict(id=cfg['id'],family=cfg['family'],**score(member['validation_probs'],val['targets'])))
    picked=shortlist(trials)
    if picked!=sel['shortlist'] or trials!=sel['screen_trials']:raise ValueError('Shortlist does not replay')
    validation=torch.load(cell/'validation.pt',weights_only=True,map_location='cpu')
    if not torch.equal(validation['ids'],val['ids']):raise ValueError('Validation node order mismatch')
    if sel['validation_ids_sha256']!=hashlib.sha256(val['ids'].numpy().tobytes()).hexdigest():
        raise ValueError('Validation identity hash mismatch')
    if set(validation['candidates'])!=set(('gnn','llm','average')+FAMILIES):
        raise ValueError('Saved candidate population mismatch')
    # Authenticated saved vectors are the original decision inputs. Recalculation
    # is a separate provenance check, never a replacement decision input.
    candidates={name:validation['candidates'][name] for name in ('gnn','llm','average')}
    drift={'raw_full':{},'raw_validation':{}}
    for name,p in _raw(train,val['ids']).items():
        drift['raw_validation'][name]=compare_raw(candidates[name],p)
    raw=torch.load(cell/'raw.pt',weights_only=True,map_location='cpu')
    if not torch.equal(raw['node_ids'],inf['node_ids']):raise ValueError('Raw node order mismatch')
    for name,p in _raw(train,inf['node_ids']).items():
        drift['raw_full'][name]=compare_raw(raw[name],p)
        if not torch.equal(candidates[name],raw[name][val['ids']]):
            raise ValueError('Saved raw validation/full vectors differ')
    for branch in ('gnn','llm'):
        key='logp_g' if branch=='gnn' else 'logp_l'
        if not torch.equal(raw[branch].argmax(1),train[key][inf['node_ids']].argmax(1)):
            raise ValueError('Original expert class identity changed')
    for family,cid in picked.items():
        payload=torch.load(cell/(family+'.pt'),weights_only=True,map_location='cpu',mmap=True)
        if not torch.equal(payload['node_ids'],inf['node_ids']):raise ValueError('Finalist node order mismatch')
        if payload['seeds'].shape!=(3,len(inf['node_ids']),train['logp_g'].shape[1]):raise ValueError('Seed/node coverage mismatch')
        vv=[]
        for j,seed in enumerate(SEEDS):
            name=f'{cid}_s{seed}';expected_names.add(name)
            member=torch.load(cell/'members'/(name+'.pt'),weights_only=True,map_location='cpu')
            audit_member(member,by_id[cid],seed,train,val)
            replay=predict(member['state'],by_id[cid],train,inf['node_ids'],device)
            if not torch.allclose(replay,payload['seeds'][j],atol=2e-5,rtol=1e-5):raise ValueError('Checkpoint full prediction mismatch')
            if not torch.allclose(replay[val['ids']],member['validation_probs'],atol=2e-5,rtol=1e-5):raise ValueError('Checkpoint validation mismatch')
            vv.append(member['validation_probs']);probability(payload['seeds'][j])
        candidates[family]=ensemble(vv)
    actual={p.stem for p in (cell/'members').glob('*.pt')}
    if actual!=expected_names or len(actual)!=18:raise ValueError('Unexpected member population')
    selection=choose(candidates,val['targets'])
    if any(sel[k]!=selection[k] for k in ('name','accuracy','nll','trials')):raise ValueError('Final selection does not replay')
    validation=torch.load(cell/'validation.pt',weights_only=True,map_location='cpu')
    if not torch.equal(validation['ids'],val['ids']):raise ValueError('Validation node order mismatch')
    for name,p in candidates.items():
        if not torch.equal(validation['candidates'][name],p):raise ValueError('Saved validation prediction mismatch')
    return dict(dataset=dataset,member_count=18,full_checkpoint_replay=True,validation_selection_replayed=True,
                selected=sel['name'],validation_accuracy=sel['accuracy'],test_truth_read=False,raw_replay_drift=drift)


def freeze(root=RECOVERY_ROOT,device='cuda'):
    require_slurm();root=Path(root).resolve()
    if (root/'FROZEN.json').exists() or (root/'evaluation').exists():raise FileExistsError('Already sealed')
    recovery=read_json(root/'RECOVERY.json');verify_files(recovery['files'])
    if recovery['original_run']!=str(DEFAULT_RUN):raise ValueError('Unexpected recovery source')
    if sha256(root/'CONFIG.json')!=sha256(DEFAULT_RUN/'CONFIG.json'):raise ValueError('Original configuration changed')
    config=verify_registration(root,inputs=True)
    visible=reader_files(root,DEFAULT_RUN)
    rows=[]
    for ds in DATASETS:
        row=audit_cell(root,ds,device);rows.append(row)
        print(f"AUDITED {ds}: 18 members, unchanged selection={row['selected']}",flush=True)
    snapshot=Path(config['protected_snapshot'])
    if sha256(snapshot)!=config['protected_snapshot_sha256']:raise ValueError('Preserved snapshot changed')
    protected={str(ROOT/p):h for p,h in read_json(snapshot)['files'].items()}
    verify_files(protected);verify_files(config['input_hashes']);verify_files(config['source_hashes'])
    verify_files(recovery['files']);verify_files(visible)
    if reader_files(root,DEFAULT_RUN)!=visible:raise ValueError('Recovery links changed during audit')
    audit=dict(all_checks_pass=True,cells=rows,protected_files=len(protected),test_truth_read=False,
               audit_version='v2_raw_rounding_only',slurm_job_id=os.environ['SLURM_JOB_ID'])
    write_json(root/'PRETEST_AUDIT.json',audit)
    files={**recovery['files'],**visible}
    for name in ('CONFIG.json','RECOVERY.json','PRETEST_AUDIT.json'):
        files[str(root/name)]=sha256(root/name)
    # Original verify_frozen traverses this mapping directly, including all new
    # executable/diagnostic sources and reader-visible paths through symlinks.
    write_json(root/'FROZEN.json',dict(datasets=list(DATASETS),files=files,
                                     slurm_job_id=os.environ['SLURM_JOB_ID']))
    return audit
