"""Immutable full-coverage residual search; query truth is absent from this module."""
import hashlib
import math
import os
from pathlib import Path
import torch
from eargtc.joint_v6_unified.experiment import (ROOT,DATASETS,SEEDS,require_slurm,read_json,write_json,
    save_tensor,sha256,verify_files,probability,score)
from .selection import build_candidates,select
DEFAULT_RUN=ROOT/'runs/joint_v6_repair_20260911_v1'
INCUMBENT=ROOT/'runs/joint_v6_unified_20260910_v2'
FIXED=ROOT/'runs/router_v8_v6_joint_multiseed_20260910_v1'
LABEL_ROOT=ROOT/'runs/router_bc_improved_structural_oof_fixed_20260729_v1'


def recipes(smoke=False):
    combinations=[(64,p,k) for p in (.02,.1,.5,2.) for k in (0.,.25)]
    combinations += [(128,p,k) for p in (.1,.5) for k in (0.,.25)]
    return [dict(id=f'w{w}_p{p:g}_k{k:g}',width=w,dropout=.1,pseudo_weight=p,keep_weight=k,
        lr=.001,weight_decay=.0001,batch_size=512,max_epochs=2 if smoke else 400,
        validate_every=1 if smoke else 10,patience=2 if smoke else 12) for w,p,k in combinations]


def warm_config(width,smoke=False):
    config=dict(recipes(smoke)[0]);config.update(id=f'warm_w{width}',width=width,pseudo_weight=0.,keep_weight=0.)
    return config


def source_files():
    paths=[]
    for folder in ('eargtc/joint_v6_repair','scripts/joint_v6_repair'):
        for pattern in ('*.py','*.sbatch'):paths+=list((ROOT/folder).glob(pattern))
    paths += list((ROOT/'eargtc/joint_v6_unified').glob('*.py'))
    paths += [ROOT/'docs/joint_v6_repair_work/SPEC.md',ROOT/'eargtc/joint_v6_unified_audit_v2/audit.py']
    return {str(p):sha256(p) for p in sorted(paths)}


def incumbent_bindings(datasets):
    locked=read_json(INCUMBENT/'FROZEN.json');files={str(INCUMBENT/'FROZEN.json'):sha256(INCUMBENT/'FROZEN.json')}
    for ds in datasets:
        cell=INCUMBENT/ds
        name=read_json(cell/'selection.json')['name']
        if name not in ('direct','residual','attention'):raise ValueError('Unsupported incumbent family')
        for fn in ('COMPLETE.json','selection.json','validation.pt',name+'.pt'):
            p=cell/fn;key=str(p)
            if key not in locked['files']:raise ValueError('Incumbent reader path not sealed')
            files[key]=locked['files'][key]
    verify_files(files);return files


def load_incumbent(ds,val=None):
    cell=INCUMBENT/ds;sel=read_json(cell/'selection.json')
    saved=torch.load(cell/'validation.pt',weights_only=True,map_location='cpu')
    if val is not None and not torch.equal(saved['ids'],val['ids']):raise ValueError('Incumbent validation population changed')
    return saved['candidates'][sel['name']],sel



def expected_cell_files(root,ds,config):
    cell=Path(root)/ds
    paths=[cell/name for name in ('predictions.pt','validation.pt','selection.json','raw.pt')]
    for cfg in config['warm_configs']+config['recipes']:
        for seed in SEEDS:
            member=member_path(cell,cfg,seed);paths.extend((member,member.with_suffix('.json')))
    return {str(p) for p in paths}


def prior_result_files():
    """Traverse both previous result seals and their complete linked artifacts."""
    files={}
    def add(records):
        for path,digest in records.items():
            if path in files and files[path]!=digest:raise ValueError('Conflicting historical seals')
            files[path]=digest
    for root,lockpath in ((INCUMBENT,INCUMBENT/'FROZEN.json'),(FIXED,FIXED/'joint/FROZEN.json')):
        lock=read_json(lockpath);add({str(lockpath):sha256(lockpath)});add(lock['files'])
        if root==FIXED:
            for path in lock['files']:
                if Path(path).name=='complete.json':
                    done=read_json(path)
                    if done['all_checks_pass'] is not True or done['test_labels_read'] is not False:raise ValueError('Unsafe prior Fixed cell')
                    add({str(Path(path).parent/name):h for name,h in done['artifacts'].items()})
    path=INCUMBENT/'evaluation/COMPLETE.json';done=read_json(path)
    if done['all_checks_pass'] is not True or done['frozen_sha256']!=sha256(INCUMBENT/'FROZEN.json'):raise ValueError('Invalid Unified evaluation seal')
    add({str(path):sha256(path)});add(done['files'])
    path=FIXED/'joint/evaluation/complete.json';done=read_json(path)
    if done['all_checks_pass'] is not True or done['frozen_sha256']!=sha256(FIXED/'joint/FROZEN.json'):raise ValueError('Invalid Fixed evaluation seal')
    add({str(path):sha256(path)})
    add({str(path.parent/name):h for name,h in done['artifacts'].items()})
    combined=done['combined'];add({str(path.parent/name):h for name,h in combined['artifacts'].items()})
    add({str(path.parent/'combined_six_router/complete.json'):combined['completion_sha256']})
    audit=FIXED/'FINAL_AUDIT.json';record=read_json(audit)
    if record['all_checks_pass'] is not True or record['evaluation_complete_sha256']!=sha256(path):raise ValueError('Invalid Fixed final audit')
    add({str(audit):sha256(audit),str(FIXED/'REPORT.md'):record['report_sha256']})
    return files

def register(root=DEFAULT_RUN,smoke=False):
    require_slurm();root=Path(root).resolve()
    if (root/'CONFIG.json').exists():raise FileExistsError('Existing registered experiment')
    from eargtc.joint_v6_unified.data import load_data
    from eargtc.joint_v6_unified.experiment import verify_frozen as old_verify_frozen
    old_verify_frozen(INCUMBENT)
    datasets=['cora'] if smoke else list(DATASETS);meta={};inputs={}
    for ds in datasets:
        train,val,inf,m=load_data(ds);meta[ds]=m;inputs.update(m['input_hashes']);del train,val,inf
    snapshot=ROOT/'artifacts/preserved_versions/joint_v8_fixed_v2_20260908/SNAPSHOT.json'
    verify_files({str(ROOT/p):h for p,h in read_json(snapshot)['files'].items()})
    files=incumbent_bindings(datasets);prior=prior_result_files();verify_files(prior)
    config=dict(schema='joint_v6_repair_v1',datasets=datasets,seeds=list(SEEDS),smoke=bool(smoke),recipes=recipes(smoke),
        warm_configs=[warm_config(w,smoke) for w in (64,128)],source_hashes=source_files(),input_hashes=inputs,metadata=meta,
        incumbent_files=files,prior_result_files=prior,protected_snapshot=str(snapshot),protected_snapshot_sha256=sha256(snapshot),
        fixed_comparator=str(FIXED/'joint/evaluation/full_query_per_split.csv'),
        fixed_comparator_sha256=sha256(FIXED/'joint/evaluation/full_query_per_split.csv'),
        label_file_hashes={str(LABEL_ROOT/d/'decisions.jsonl'):sha256(LABEL_ROOT/d/'decisions.jsonl') for d in datasets},
        protocol='historical_seed42_full_query_transductive',slurm_job_id=os.environ['SLURM_JOB_ID'])
    write_json(root/'CONFIG.json',config);print('REGISTERED '+str(root),flush=True)
    return config


def verify_registration(root,inputs=False):
    c=read_json(Path(root)/'CONFIG.json')
    if (c['schema']!='joint_v6_repair_v1' or c['recipes']!=recipes(c['smoke']) or c['seeds']!=list(SEEDS)
        or c['warm_configs']!=[warm_config(w,c['smoke']) for w in (64,128)]
        or c['datasets']!=(['cora'] if c['smoke'] else list(DATASETS))):raise ValueError('Registered search changed')
    if set(c['source_hashes'])!=set(source_files()):raise ValueError('Registered source population changed')
    verify_files(c['source_hashes']);verify_files(c['incumbent_files'])
    if inputs:
        verify_files(c['input_hashes']);verify_files(c['prior_result_files'])
    return c


def audit_member(member,cfg,seed,train,val,stage):
    require_slurm()
    if (set(member)!={'stage','config','seed','branches'} or member['stage']!=stage or member['config']!=cfg
        or member['seed']!=seed or set(member['branches'])!={'g','l'}):raise ValueError('Member identity changed')
    for b,record in member['branches'].items():
        history=record['history'];best={};stale=0
        if [r['epoch'] for r in history]!=list(range(len(history))) or not 1<=len(history)-1<=cfg['max_epochs']:
            raise ValueError('Invalid full epoch history')
        for row in history:
            epoch=row['epoch']
            for pool in ('gold','pseudo'):
                count=len(train[pool+'_ids']) if epoch and (pool=='gold' or stage=='refine') else 0
                if (row[pool+'_seen']!=count or row[pool+'_unique']!=count
                    or row[pool+'_batches']!=math.ceil(count/cfg['batch_size'])):raise ValueError('Incomplete full-pool coverage')
            if epoch:
                if any(not math.isfinite(row[k]) for k in ('gold_loss','pseudo_loss','total_loss')):raise ValueError('Nonfinite training objective')
                expected=row['gold_loss']+cfg['pseudo_weight']*row['pseudo_loss']
                if not math.isclose(row['total_loss'],expected,abs_tol=1e-7,rel_tol=1e-7):raise ValueError('Wrong uniform loss weighting')
            scheduled=epoch==0 or epoch%cfg['validate_every']==0
            if not scheduled:
                if row['accuracy'] is not None or row['nll'] is not None:raise ValueError('Unexpected validation event')
                continue
            acc,nll=row['accuracy'],row['nll']
            if not math.isfinite(acc) or not 0<=acc<=1 or not math.isfinite(nll) or nll<0:raise ValueError('Invalid validation metrics')
            if stage=='refine' and epoch==0:continue
            previous=best.get('accuracy');item=dict(epoch=epoch,accuracy=acc,nll=nll)
            for criterion in ('accuracy','nll'):
                def rank(t):return (-t['accuracy'],t['nll'],t['epoch']) if criterion=='accuracy' else (t['nll'],-t['accuracy'],t['epoch'])
                if criterion not in best or rank(item)<rank(best[criterion]):best[criterion]=item
            stale=0 if previous!=best['accuracy'] else stale+1
            if stale>=cfg['patience'] and epoch!=len(history)-1:raise ValueError('Continued beyond patience')
        if len(history)-1<cfg['max_epochs'] and stale<cfg['patience']:raise ValueError('Premature training stop')
        if record['best']!=best:raise ValueError('Checkpoint rank does not replay')
        for criterion in ('accuracy','nll'):
            state=record['states'][criterion];delta=record['validation_delta'][criterion]
            if not state or any(v.device.type!='cpu' or not bool(torch.isfinite(v).all()) for v in state.values()):raise ValueError('Invalid CPU states')
            if delta.shape!=(len(val['ids']),train['logp_'+b].shape[1]) or not bool(torch.isfinite(delta).all()):raise ValueError('Invalid validation delta')
            actual=score((train['logp_'+b][val['ids']]+delta).softmax(1),val['targets'])
            if actual['accuracy']!=best[criterion]['accuracy'] or not math.isclose(actual['nll'],best[criterion]['nll'],abs_tol=2e-6,rel_tol=2e-6):
                raise ValueError('Checkpoint validation score changed')
    return True


def member_path(cell,cfg,seed):return cell/'members'/f"{cfg['id']}_s{seed}.pt"


def get_member(cell,train,val,cfg,seed,device,warm=None):
    from .training import warm_member,refine_member
    path=member_path(cell,cfg,seed);marker=path.with_suffix('.json')
    expected=dict(config_sha256=sha256(cell.parent/'CONFIG.json'),config_id=cfg['id'],seed=seed,
        stage='warm' if warm is None else 'refine',warm_sha256=None if warm is None else sha256(member_path(cell,warm['config'],seed)))
    if marker.exists():
        saved=read_json(marker)
        if saved['identity']!=expected:raise ValueError('Member provenance changed')
        if set(saved['files'])!={str(path)}:raise ValueError('Member checkpoint binding differs')
        verify_files(saved['files']);result=torch.load(path,weights_only=True,map_location='cpu')
    else:
        if path.exists():raise FileExistsError('Partial member exists; preserve for diagnosis')
        result=warm_member(train,val,cfg,seed,device) if warm is None else refine_member(train,val,cfg,seed,device,warm)
        audit_member(result,cfg,seed,train,val,expected['stage']);save_tensor(path,result)
        write_json(marker,dict(identity=expected,files={str(path):sha256(path)}))
    audit_member(result,cfg,seed,train,val,expected['stage'])
    return result


def selected_predictions(selection,candidates,members,train,val,inf,device):
    from .training import predict_delta
    lookup={c['id']:c for v in candidates.values() for c in v}
    outputs={};details={}
    for branch in ('g','l'):
        candidate=lookup[selection[branch+'_id']];metadata=candidate['metadata'];base=train['logp_'+branch]
        if metadata['raw']:
            preds=base.exp().unsqueeze(0).repeat(3,1,1)
        else:
            cfgid=metadata['config_id'];criterion=metadata['criterion'];scale=metadata['scale'];values=[]
            for member in members[cfgid]:
                record=member['branches'][branch]
                delta=predict_delta(record['states'][criterion],member['config'],train,inf['node_ids'],device)
                if not torch.allclose(delta[val['ids']],record['validation_delta'][criterion],atol=2e-5,rtol=1e-5):
                    raise ValueError('Selected checkpoint delta does not replay')
                values.append((base+scale*delta).softmax(1))
            preds=torch.stack(values)
        for p in preds:probability(p)
        if not torch.allclose(preds[:,val['ids']].mean(0),candidate['probability'],atol=2e-5,rtol=1e-5):raise ValueError('Full and saved validation predictions disagree')
        outputs[branch]=preds;details[branch]=metadata
    return selection['alpha']*outputs['g']+(1-selection['alpha'])*outputs['l'],details


def run_cell(root,dataset,device='cuda'):
    require_slurm();root=Path(root).resolve()
    if (root/'FROZEN.json').exists() or (root/'evaluation').exists():raise ValueError('Training is closed')
    config=verify_registration(root)
    if dataset not in config['datasets']:raise ValueError('Unregistered dataset')
    cell=root/dataset
    if cell.is_symlink():raise ValueError('New cells cannot redirect to historical directories')
    cell.mkdir(parents=True,exist_ok=True)
    if (cell/'COMPLETE.json').exists():
        complete=read_json(cell/'COMPLETE.json')
        if (complete['dataset']!=dataset or complete['config_sha256']!=sha256(root/'CONFIG.json')
            or complete['member_count']!=42 or complete['query_truth_read'] is not False):raise ValueError('Unbound completed cell')
        if set(complete['files'])!=expected_cell_files(root,dataset,config):raise ValueError('Canonical cell reader population differs')
        verify_files(complete['files']);return
    from eargtc.joint_v6_unified.data import load_data
    train,val,inf,meta=load_data(dataset)
    if meta!=config['metadata'][dataset]:raise ValueError('Authenticated data changed')
    warms={}
    for cfg in config['warm_configs']:
        for seed in SEEDS:
            warms[cfg['width'],seed]=get_member(cell,train,val,cfg,seed,device)
            print(f"WARM {dataset} {cfg['id']} seed{seed}",flush=True)
    members={}
    for cfg in config['recipes']:
        members[cfg['id']]=[get_member(cell,train,val,cfg,seed,device,warms[cfg['width'],seed]) for seed in SEEDS]
        print(f"REFINED {dataset} {cfg['id']} three seeds",flush=True)
    candidates=build_candidates(members,train,val);incumbent,oldsel=load_incumbent(dataset,val)
    selection=select(candidates,val['targets'],incumbent)
    repair,details=selected_predictions(selection,candidates,members,train,val,inf,device)
    old=torch.load(INCUMBENT/dataset/(oldsel['name']+'.pt'),weights_only=True,map_location='cpu',mmap=True)
    if not torch.equal(old['node_ids'],inf['node_ids']):raise ValueError('Incumbent full node order mismatch')
    chosen=old['seeds'].clone() if selection['winner']=='incumbent' else repair
    selection.update(selected_heads=details,incumbent_family=oldsel['name'])
    save_tensor(cell/'validation.pt',dict(ids=val['ids'],candidates=candidates,incumbent=incumbent))
    save_tensor(cell/'predictions.pt',dict(node_ids=inf['node_ids'],seeds=chosen,repair_seeds=repair))
    write_json(cell/'selection.json',selection)
    pg,pl=train['logp_g'].exp(),train['logp_l'].exp()
    save_tensor(cell/'raw.pt',dict(node_ids=inf['node_ids'],g=pg,l=pl,average=(pg+pl)*.5))
    files={str(p):sha256(p) for p in sorted(cell.rglob('*')) if p.is_file()}
    if set(files)!=expected_cell_files(root,dataset,config):raise ValueError('Canonical cell reader population differs')
    verify_files(config['source_hashes']);verify_files(meta['input_hashes']);verify_files(config['incumbent_files'])
    write_json(cell/'COMPLETE.json',dict(dataset=dataset,config_sha256=sha256(root/'CONFIG.json'),files=files,
        member_count=42,query_truth_read=False,slurm_job_id=os.environ['SLURM_JOB_ID']))
    print(f"COMPLETE {dataset} winner={selection['winner']} validation={selection['accuracy']:.8f}",flush=True)


def same_record(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(same_record(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(same_record(x,y) for x,y in zip(a,b))
    if isinstance(a,float) and isinstance(b,float):return math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-10)
    return type(a)==type(b) and a==b


def audit_cell(root,ds,device):
    from eargtc.joint_v6_unified.data import load_data
    config=read_json(root/'CONFIG.json');cell=root/ds;complete=read_json(cell/'COMPLETE.json')
    if (complete['dataset']!=ds or complete['config_sha256']!=sha256(root/'CONFIG.json')
        or complete['member_count']!=42 or complete['query_truth_read'] is not False):raise ValueError('Invalid completed cell')
    if set(complete['files'])!=expected_cell_files(root,ds,config):raise ValueError('Canonical cell reader population differs')
    verify_files(complete['files'])
    train,val,inf,meta=load_data(ds)
    if meta!=config['metadata'][ds]:raise ValueError('Registered input population changed')
    expected=set();members={}
    for cfg in config['warm_configs']+config['recipes']:
        stage='warm' if cfg['id'].startswith('warm_') else 'refine';group=[]
        for seed in SEEDS:
            path=member_path(cell,cfg,seed);expected.add(path.name)
            mark=read_json(path.with_suffix('.json'))
            if mark['files']!={str(path):complete['files'][str(path)]}:raise ValueError('Member marker reader path changed')
            verify_files(mark['files'])
            parent=None if stage=='warm' else sha256(member_path(cell,warm_config(cfg['width'],config['smoke']),seed))
            identity=dict(config_sha256=sha256(root/'CONFIG.json'),config_id=cfg['id'],seed=seed,stage=stage,warm_sha256=parent)
            if mark['identity']!=identity:raise ValueError('Warm/refinement linkage changed')
            m=torch.load(path,weights_only=True,map_location='cpu');audit_member(m,cfg,seed,train,val,stage);group.append(m)
        if stage=='refine':members[cfg['id']]=group
    if {p.name for p in (cell/'members').glob('*.pt')}!=expected:raise ValueError('Member population changed')
    replay_candidates=build_candidates(members,train,val)
    saved=torch.load(cell/'validation.pt',weights_only=True,map_location='cpu')
    if not torch.equal(saved['ids'],val['ids']):raise ValueError('Validation IDs changed')
    for b,values in replay_candidates.items():
        original=saved['candidates'][b]
        if len(values)!=len(original):raise ValueError('Candidate population changed')
        for a,z in zip(values,original):
            if a['id']!=z['id'] or a['metadata']!=z['metadata']:raise ValueError('Candidate identity changed')
            probability(z['probability'])
            if not torch.allclose(a['probability'],z['probability'],atol=2e-6,rtol=2e-6):raise ValueError('Member candidate probabilities changed')
    incumbent,oldsel=load_incumbent(ds,val)
    if not torch.equal(incumbent,saved['incumbent']):raise ValueError('Saved incumbent changed')
    chosen=read_json(cell/'selection.json')
    replay=select(saved['candidates'],val['targets'],saved['incumbent'])
    if not same_record(replay,{k:chosen[k] for k in replay}):raise ValueError('Saved validation selection does not replay')
    repair,details=selected_predictions(chosen,saved['candidates'],members,train,val,inf,device)
    if details!=chosen['selected_heads'] or oldsel['name']!=chosen['incumbent_family']:raise ValueError('Selected model identity changed')
    predictions=torch.load(cell/'predictions.pt',weights_only=True,map_location='cpu',mmap=True)
    if not torch.equal(predictions['node_ids'],inf['node_ids']):raise ValueError('Prediction node order changed')
    if not torch.allclose(repair,predictions['repair_seeds'],atol=2e-5,rtol=1e-5):raise ValueError('Full repair checkpoint replay failed')
    expected_probs=predictions['repair_seeds']
    if chosen['winner']=='incumbent':
        expected_probs=torch.load(INCUMBENT/ds/(oldsel['name']+'.pt'),weights_only=True,map_location='cpu',mmap=True)['seeds']
    if not torch.equal(predictions['seeds'],expected_probs):raise ValueError('Final selected output changed')
    for p in predictions['seeds']:probability(p)
    from eargtc.joint_v6_unified_audit_v2.audit import compare_raw
    raw=torch.load(cell/'raw.pt',weights_only=True,map_location='cpu')
    if set(raw)!={'node_ids','g','l','average'} or not torch.equal(raw['node_ids'],inf['node_ids']):raise ValueError('Raw control identity changed')
    pg,pl=train['logp_g'].exp(),train['logp_l'].exp()
    drift={k:compare_raw(raw[k],q) for k,q in {'g':pg,'l':pl,'average':(pg+pl)*.5}.items()}
    return dict(dataset=ds,raw_control_replay=drift,member_count=42,all_member_histories_checked=True,both_checkpoint_criteria_checked=True,
        full_selected_checkpoint_replay=True,validation_selection_replayed=True,winner=chosen['winner'],
        validation_accuracy=chosen['accuracy'],query_truth_read=False)


def protected_files(config):
    snapshot=Path(config['protected_snapshot'])
    if sha256(snapshot)!=config['protected_snapshot_sha256']:raise ValueError('Protected snapshot changed')
    return {str(ROOT/p):h for p,h in read_json(snapshot)['files'].items()}


def freeze(root=DEFAULT_RUN,device='cuda'):
    require_slurm();root=Path(root).resolve();config=verify_registration(root,inputs=True)
    if not config['smoke'] and torch.device(device).type!='cuda':raise ValueError('Formal checkpoint audit requires CUDA')
    if (root/'FROZEN.json').exists() or (root/'evaluation').exists():raise FileExistsError('Search already sealed')
    for ds in config['datasets']:
        if not (root/ds/'COMPLETE.json').is_file():raise ValueError('Missing completed dataset: '+ds)
    rows=[]
    for ds in config['datasets']:
        rows.append(audit_cell(root,ds,device));print('AUDITED '+ds,flush=True)
    protected=protected_files(config);verify_files(protected)
    # Recheck every protected artifact of the previous Unified search, including
    # the original training files reached through v2's reader-visible symlinks.
    verify_files(read_json(INCUMBENT/'FROZEN.json')['files'])
    verify_registration(root,inputs=True)
    audit=dict(all_checks_pass=True,cells=rows,protected_files=len(protected),query_truth_read=False,
        replay_device=str(device),slurm_job_id=os.environ['SLURM_JOB_ID'])
    if config['smoke']:
        write_json(root/'SMOKE_AUDIT.json',audit);return audit
    write_json(root/'PRETEST_AUDIT.json',audit)
    files={str(root/n):sha256(root/n) for n in ('CONFIG.json','PRETEST_AUDIT.json')}
    files.update(config['source_hashes']);files.update(config['incumbent_files'])
    for ds in DATASETS:
        path=root/ds/'COMPLETE.json';files[str(path)]=sha256(path);files.update(read_json(path)['files'])
    write_json(root/'FROZEN.json',dict(datasets=list(DATASETS),files=files,query_truth_read=False,slurm_job_id=os.environ['SLURM_JOB_ID']))
    return audit


def verify_frozen(root):
    require_slurm();root=Path(root).resolve();lock=read_json(root/'FROZEN.json')
    if lock['datasets']!=list(DATASETS) or lock['query_truth_read'] is not False:raise ValueError('All-five barrier required')
    verify_files(lock['files']);config=verify_registration(root,inputs=True)
    if config['smoke']:raise ValueError('Smoke cannot evaluate')
    audit=read_json(root/'PRETEST_AUDIT.json')
    if audit['all_checks_pass'] is not True or [r['dataset'] for r in audit['cells']]!=list(DATASETS):raise ValueError('Incomplete audit')
    required={str(root/'CONFIG.json'),str(root/'PRETEST_AUDIT.json')}|set(config['source_hashes'])|set(config['incumbent_files'])
    for ds in DATASETS:
        path=root/ds/'COMPLETE.json';c=read_json(path);required.add(str(path));required.update(c['files'])
        if (c['dataset']!=ds or c['config_sha256']!=sha256(root/'CONFIG.json') or c['member_count']!=42
            or c['query_truth_read'] is not False):raise ValueError('Unbound final cell')
        if set(c['files'])!=expected_cell_files(root,ds,config):raise ValueError('Canonical cell reader population differs')
        verify_files(c['files'])
    if set(lock['files'])!=required:raise ValueError('Final seal omits or adds reader files')
    verify_files(protected_files(config))
    verify_files({config['fixed_comparator']:config['fixed_comparator_sha256']})
    return config
