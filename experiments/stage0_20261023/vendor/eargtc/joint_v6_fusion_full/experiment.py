"""Additive exhaustive fusion over immutable Repair members; no query truth."""
import os
import math
import shutil
from pathlib import Path
import torch
from eargtc.joint_v6_repair import experiment as parent
from eargtc.joint_v6_unified.experiment import (ROOT,DATASETS,SEEDS,require_slurm,read_json,write_json,save_tensor,sha256,verify_files,probability)
from eargtc.joint_v6_unified.data import load_data
from .selection import select_full
DEFAULT_RUN=ROOT/'runs/joint_v6_fusion_full_20260911_v1'
PARENT=parent.DEFAULT_RUN
INCUMBENT=parent.INCUMBENT
LABEL_ROOT=parent.LABEL_ROOT
ALPHAS=[0.,.25,.5,.75,1.]


def source_files():
    paths=[ROOT/'docs/joint_v6_fusion_full_work/SPEC.md']
    for folder in ('eargtc/joint_v6_fusion_full','scripts/joint_v6_fusion_full'):
        for pattern in ('*.py','*.sbatch'):paths+=list((ROOT/folder).glob(pattern))
    return {str(p):sha256(p) for p in sorted(paths)}


def register(root=DEFAULT_RUN):
    require_slurm();root=Path(root).resolve()
    if (root/'CONFIG.json').exists():raise FileExistsError('Full fusion already registered')
    p=parent.verify_registration(PARENT,inputs=True)
    config={k:p[k] for k in ('metadata','label_file_hashes','fixed_comparator','fixed_comparator_sha256',
        'input_hashes','incumbent_files','prior_result_files','protected_snapshot','protected_snapshot_sha256')}
    config.update(schema='joint_v6_fusion_full_v1',parent_root=str(PARENT),parent_config_sha256=sha256(PARENT/'CONFIG.json'),
        source_hashes=source_files(),datasets=list(DATASETS),seeds=list(SEEDS),candidates_per_branch=73,trials_per_cell=26645,
        alphas=ALPHAS,query_truth_read=False,slurm_job_id=os.environ['SLURM_JOB_ID'])
    write_json(root/'CONFIG.json',config);print('REGISTERED '+str(root),flush=True)
    return config


def verify_registration(root,inputs=False):
    config=read_json(Path(root)/'CONFIG.json')
    if (config['schema']!='joint_v6_fusion_full_v1' or config['datasets']!=list(DATASETS) or config['seeds']!=list(SEEDS)
        or config['alphas']!=ALPHAS or config['candidates_per_branch']!=73 or config['trials_per_cell']!=26645
        or config['query_truth_read'] is not False):raise ValueError('Full fusion protocol changed')
    if set(config['source_hashes'])!=set(source_files()):raise ValueError('Source population changed')
    verify_files(config['source_hashes'])
    proot=Path(config['parent_root'])
    if proot!=PARENT or sha256(proot/'CONFIG.json')!=config['parent_config_sha256']:raise ValueError('Parent CONFIG changed')
    p=parent.verify_registration(proot,inputs=inputs)
    for key in ('metadata','label_file_hashes','fixed_comparator','fixed_comparator_sha256','input_hashes','incumbent_files',
                'prior_result_files','protected_snapshot','protected_snapshot_sha256'):
        if config[key]!=p[key]:raise ValueError('Inherited protocol field changed: '+key)
    return config


def parent_cell(config,ds):
    proot=Path(config['parent_root']);path=proot/ds/'COMPLETE.json'
    pconfig=read_json(proot/'CONFIG.json')
    if sha256(proot/'CONFIG.json')!=config['parent_config_sha256']:raise ValueError('Parent CONFIG changed')
    complete=read_json(path)
    if (complete['dataset']!=ds or complete['config_sha256']!=config['parent_config_sha256']
        or complete['member_count']!=42 or complete['query_truth_read'] is not False):raise ValueError('Unbound parent cell')
    if set(complete['files'])!=parent.expected_cell_files(proot,ds,pconfig):raise ValueError('Canonical parent reader population differs')
    verify_files(complete['files'])
    return {str(path):sha256(path),**complete['files']}


def expected_files(cell):return {str(cell/name) for name in ('validation.pt','selection.json','predictions.pt','raw.pt')}


def check_catalog(saved,pconfig,val):
    if set(saved)!={'ids','candidates','incumbent'} or not torch.equal(saved['ids'],val['ids']):raise ValueError('Validation population changed')
    if set(saved['candidates'])!={'g','l'}:raise ValueError('Only two expert branches allowed')
    for branch in ('g','l'):
        expected={branch+'/raw'}|{f"{branch}/{c['id']}/{criterion}/{scale:g}" for c in pconfig['recipes'] for criterion in ('accuracy','nll') for scale in (.25,.5,1.)}
        values=saved['candidates'][branch]
        if len(values)!=73 or {c['id'] for c in values}!=expected:raise ValueError('Full candidate catalog changed')


def predict(config,ds,selection,saved,train,val,inf,device):
    if torch.device(device).type!='cuda':raise ValueError('Full fusion checkpoint replay requires CUDA')
    lookup={c['id']:c for values in saved['candidates'].values() for c in values}
    ids={lookup[selection[b+'_id']]['metadata']['config_id'] for b in ('g','l') if not lookup[selection[b+'_id']]['metadata']['raw']}
    pconfig=read_json(Path(config['parent_root'])/'CONFIG.json');recipes={c['id']:c for c in pconfig['recipes']}
    members={cid:[torch.load(parent.member_path(Path(config['parent_root'])/ds,recipes[cid],seed),weights_only=True,map_location='cpu') for seed in SEEDS] for cid in ids}
    repair,details=parent.selected_predictions(selection,saved['candidates'],members,train,val,inf,device)
    oldsel=read_json(INCUMBENT/ds/'selection.json')
    old=torch.load(INCUMBENT/ds/(oldsel['name']+'.pt'),weights_only=True,map_location='cpu',mmap=True)
    if not torch.equal(old['node_ids'],inf['node_ids']):raise ValueError('Incumbent node identity changed')
    chosen=old['seeds'].clone() if selection['winner']=='incumbent' else repair
    return repair,chosen,details,oldsel


def selection_check(selection,old):
    if len(selection['trials'])!=26645 or any(len(v)!=73 for v in selection['shortlist'].values()):raise ValueError('Incomplete full fusion search')
    if (-selection['accuracy'],selection['nll'])>(-old['accuracy'],old['nll']):raise ValueError('Full fusion regresses versus parent validation choice')


def check_predictions(output,inf,classes):
    if (set(output)!={'node_ids','seeds','repair_seeds'} or not torch.is_tensor(output['node_ids'])
        or output['node_ids'].device.type!='cpu' or output['node_ids'].dtype!=torch.long
        or not torch.equal(output['node_ids'],inf['node_ids'])):raise ValueError('Prediction identity changed')
    for key in ('seeds','repair_seeds'):
        values=output[key]
        if (not torch.is_tensor(values) or values.device.type!='cpu' or values.dtype!=torch.float32
            or values.shape!=(3,len(inf['node_ids']),classes)):raise ValueError('Exact three-seed prediction shape required')
        for value in values:probability(value)


def check_selected_validation(output,selection,val):
    actual=parent.score(output['seeds'][:,val['ids']].mean(0),val['targets'])
    if (actual['accuracy']!=selection['accuracy']
        or not math.isclose(actual['nll'],selection['nll'],abs_tol=2e-6,rel_tol=2e-6)):
        raise ValueError('Selected validation score differs from saved ensemble')


def copy_exclusive(source,target):
    with Path(source).open('rb') as src,Path(target).open('xb') as dst:shutil.copyfileobj(src,dst)


def run_cell(root,ds,device='cuda'):
    require_slurm();root=Path(root).resolve();config=verify_registration(root)
    if ds not in DATASETS:raise ValueError('Unknown dataset')
    if (root/'FROZEN.json').exists() or (root/'evaluation').exists():raise ValueError('Search is closed')
    cell=root/ds
    if cell.is_symlink():raise ValueError('New cell cannot redirect')
    if cell.exists():raise FileExistsError('Preserve existing full fusion cell')
    bound=parent_cell(config,ds);proot=Path(config['parent_root'])
    train,val,inf,meta=load_data(ds)
    if meta!=config['metadata'][ds]:raise ValueError('Input metadata changed')
    saved=torch.load(proot/ds/'validation.pt',weights_only=True,map_location='cpu')
    check_catalog(saved,read_json(proot/'CONFIG.json'),val)
    selection=select_full(saved['candidates'],val['targets'],saved['incumbent'])
    old=read_json(proot/ds/'selection.json');selection_check(selection,old)
    repair,chosen,details,oldsel=predict(config,ds,selection,saved,train,val,inf,device)
    selection.update(selected_heads=details,incumbent_family=oldsel['name'],parent_top3_score={k:old[k] for k in ('accuracy','nll','winner')})
    cell.mkdir()
    copy_exclusive(proot/ds/'validation.pt',cell/'validation.pt');copy_exclusive(proot/ds/'raw.pt',cell/'raw.pt')
    write_json(cell/'selection.json',selection)
    output=dict(node_ids=inf['node_ids'],seeds=chosen,repair_seeds=repair)
    check_predictions(output,inf,train['logp_g'].shape[1]);check_selected_validation(output,selection,val)
    save_tensor(cell/'predictions.pt',output)
    verify_files(bound);verify_registration(root)
    files={p:sha256(p) for p in sorted(expected_files(cell))}
    write_json(cell/'COMPLETE.json',dict(dataset=ds,config_sha256=sha256(root/'CONFIG.json'),files=files,parent_files=bound,
        query_truth_read=False,trials=26645,slurm_job_id=os.environ['SLURM_JOB_ID']))
    print(f"COMPLETE {ds} winner={selection['winner']} validation={selection['accuracy']:.8f} parent={old['accuracy']:.8f}",flush=True)


def verify_cell(root,config,ds):
    cell=root/ds;done=read_json(cell/'COMPLETE.json')
    if (done['dataset']!=ds or done['config_sha256']!=sha256(root/'CONFIG.json') or done['query_truth_read'] is not False
        or done['trials']!=26645 or set(done['files'])!=expected_files(cell)):raise ValueError('Unbound full fusion cell')
    verify_files(done['files']);verify_files(done['parent_files'])
    if done['parent_files']!=parent_cell(config,ds):raise ValueError('Parent reader bindings changed')
    proot=Path(config['parent_root'])
    for name in ('validation.pt','raw.pt'):
        if done['files'][str(cell/name)]!=done['parent_files'][str(proot/ds/name)]:raise ValueError('Copied parent control changed')
    return done


def audit_cell(root,config,ds,device):
    verify_cell(root,config,ds);cell=root/ds;proot=Path(config['parent_root'])
    train,val,inf,meta=load_data(ds)
    if meta!=config['metadata'][ds]:raise ValueError('Input metadata changed')
    saved=torch.load(cell/'validation.pt',weights_only=True,map_location='cpu');check_catalog(saved,read_json(proot/'CONFIG.json'),val)
    selection=select_full(saved['candidates'],val['targets'],saved['incumbent']);actual=read_json(cell/'selection.json')
    old=read_json(proot/ds/'selection.json');selection_check(selection,old)
    if not parent.same_record(selection,{k:actual[k] for k in selection}):raise ValueError('Full selection does not replay')
    repair,chosen,details,oldsel=predict(config,ds,selection,saved,train,val,inf,device)
    if (actual['selected_heads']!=details or actual['incumbent_family']!=oldsel['name']
        or actual['parent_top3_score']!={k:old[k] for k in ('accuracy','nll','winner')}):raise ValueError('Selection provenance changed')
    output=torch.load(cell/'predictions.pt',weights_only=True,map_location='cpu',mmap=True)
    check_predictions(output,inf,train['logp_g'].shape[1]);check_selected_validation(output,selection,val)
    check_predictions(torch.load(proot/ds/'predictions.pt',weights_only=True,map_location='cpu',mmap=True),inf,train['logp_g'].shape[1])
    if not torch.allclose(output['repair_seeds'],repair,atol=2e-5,rtol=1e-5):raise ValueError('Full repair probabilities do not replay')
    expected=chosen if selection['winner']=='incumbent' else output['repair_seeds']
    if not torch.equal(output['seeds'],expected):raise ValueError('Selected probabilities changed')
    for key in ('seeds','repair_seeds'):
        for p in output[key]:probability(p)
    return dict(dataset=ds,trials=26645,candidates_per_branch=73,full_selected_checkpoint_replay=True,validation_selection_replayed=True,
        validation_nonregression=True,query_truth_read=False,winner=selection['winner'],validation_accuracy=selection['accuracy'])


def freeze(root=DEFAULT_RUN,device='cuda'):
    require_slurm();root=Path(root).resolve();config=verify_registration(root,inputs=True)
    proot=Path(config['parent_root']);parent.verify_frozen(proot)
    if torch.device(device).type!='cuda':raise ValueError('Formal full fusion audit requires CUDA')
    if (root/'FROZEN.json').exists() or (root/'evaluation').exists():raise FileExistsError('Already sealed')
    for ds in DATASETS:
        if not (root/ds/'COMPLETE.json').is_file():raise ValueError('All-five full fusion barrier required')
    parent_lock=read_json(proot/'FROZEN.json');rows=[]
    for ds in DATASETS:
        done=verify_cell(root,config,ds)
        if any(parent_lock['files'].get(p)!=h for p,h in done['parent_files'].items()):raise ValueError('Parent cell not bound by its final seal')
        rows.append(audit_cell(root,config,ds,device));print('AUDITED '+ds,flush=True)
    verify_registration(root,inputs=True);verify_files(parent_lock['files'])
    write_json(root/'PRETEST_AUDIT.json',dict(all_checks_pass=True,cells=rows,query_truth_read=False,replay_device=str(device),
        parent_frozen_sha256=sha256(proot/'FROZEN.json'),slurm_job_id=os.environ['SLURM_JOB_ID']))
    files={str(root/n):sha256(root/n) for n in ('CONFIG.json','PRETEST_AUDIT.json')}
    files.update(config['source_hashes']);files.update(parent_lock['files']);files[str(proot/'FROZEN.json')]=sha256(proot/'FROZEN.json')
    for ds in DATASETS:
        path=root/ds/'COMPLETE.json';files[str(path)]=sha256(path);files.update(read_json(path)['files'])
    write_json(root/'FROZEN.json',dict(datasets=list(DATASETS),files=files,query_truth_read=False,slurm_job_id=os.environ['SLURM_JOB_ID']))


def verify_frozen(root):
    require_slurm();root=Path(root).resolve();lock=read_json(root/'FROZEN.json')
    if lock['datasets']!=list(DATASETS) or lock['query_truth_read'] is not False:raise ValueError('All-five seal required')
    verify_files(lock['files']);config=verify_registration(root,inputs=True);proot=Path(config['parent_root'])
    parent.verify_frozen(proot);plock=read_json(proot/'FROZEN.json');audit=read_json(root/'PRETEST_AUDIT.json')
    if (audit['all_checks_pass'] is not True or audit['query_truth_read'] is not False or not audit['replay_device'].startswith('cuda')
        or audit['parent_frozen_sha256']!=sha256(proot/'FROZEN.json') or [r['dataset'] for r in audit['cells']]!=list(DATASETS)):
        raise ValueError('Full fusion audit changed')
    required={str(root/n) for n in ('CONFIG.json','PRETEST_AUDIT.json')}|set(config['source_hashes'])|set(plock['files'])|{str(proot/'FROZEN.json')}
    for ds in DATASETS:
        done=verify_cell(root,config,ds)
        if any(plock['files'].get(p)!=h for p,h in done['parent_files'].items()):raise ValueError('Unsealed parent linkage')
        required.add(str(root/ds/'COMPLETE.json'));required.update(done['files'])
    if set(lock['files'])!=required:raise ValueError('Canonical final reader population changed')
    return config
