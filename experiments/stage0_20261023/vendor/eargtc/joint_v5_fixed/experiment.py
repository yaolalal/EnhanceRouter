"""Five-cell orchestration around the unmodified frozen V2 Joint trainer."""
from dataclasses import asdict,replace
import itertools
import os
import socket
from pathlib import Path
import time
import torch
from eargtc.joint_cross_repair.artifacts import DATASETS,read_json,write_json,save_tensor,sha256,verify_files
from eargtc.joint_cross_repair.prepare import DEFAULT_EXPORT as BASE_EXPORT,FROZEN_ROUTER
from eargtc.joint_cross_repair.training import TrainConfig,_score,blend,scheme_b_probability
from eargtc.joint_cross_repair_search.bank import RECIPES,SEEDS,Recipe,train_member,member_logits
from eargtc.joint_cross_repair_search.experiment import candidate_bank
from eargtc.joint_cross_repair_search.selection import residual_probability
from eargtc.joint_five_router.experiment import select_candidates,_check_member
from eargtc.joint_selector.experiment import _reproduce_member

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_RUN=ROOT/'runs/joint_v5_fixed_20260909_v1'
OLD_FIVE=ROOT/'runs/joint_five_router_20260907_v1'
ROUTERS=('v8_v5',)
SPEC=ROOT/'docs/superpowers/specs/2026-09-09-joint-v5-fixed-design.md'
REQUIRED_ARTIFACTS={'selection.json','validation.pt','predictions.pt','candidate_catalog.json','metadata.json'}


def check_reference_config(old):
    expected=TrainConfig(threads=2,validate_every=50,max_steps=500,patience=5)
    if (old['train']!=asdict(expected) or old['seeds']!=list(SEEDS)
        or old['recipes']!=[asdict(r) for r in RECIPES]
        or old['architecture']!='frozen_V2_gold_anchored_dual_residual64'
        or Path(old['base_export_root']).resolve()!=BASE_EXPORT.resolve()
        or old['torch_version']!=str(torch.__version__)):
        raise ValueError('historical training/search/environment differs')
    return expected


def initialize(run_root,export_root=None,smoke=False):
    from .adapter import DEFAULT_EXPORT
    root=Path(run_root).resolve();root.mkdir(parents=True,exist_ok=True)
    export=Path(export_root or DEFAULT_EXPORT).resolve()
    barrier=read_json(export/'COMPLETE.json')
    expected_keys={f'{r}/{d}' for r,d in itertools.product(ROUTERS,DATASETS)}
    if barrier.get('cell_count')!=5 or set(barrier.get('cells',{}))!=expected_keys or barrier.get('test_labels_exported') is not False:
        raise ValueError('all5 safe exports required')
    verify_files({str(export/k/'complete.json'):v for k,v in barrier['cells'].items()})
    old=read_json(OLD_FIVE/'CONFIG.json');cfg=check_reference_config(old)
    verify_files(old['source_code'])
    # Reuse is bound to the original training inputs, including gold targets.
    verify_files(old['input_completions'])
    from eargtc.joint_cross_repair_search.experiment import preflight as original_preflight
    original_root=Path(old['old_v2_root'])
    original=original_preflight(original_root)
    if Path(original['export_root']).resolve()!=BASE_EXPORT.resolve():
        raise ValueError('original gold training export differs')
    original_bindings={str(original_root/name):sha256(original_root/name) for name in ['SEARCH_CONFIG.json','FROZEN.json']}
    if any(old['input_completions'].get(path)!=digest for path,digest in original_bindings.items()):
        raise ValueError('original gold run not bound to historical config')
    if smoke:cfg=replace(cfg,max_steps=10,validate_every=5,patience=2)
    paths=[SPEC, ROOT/'eargtc/joint_selector/experiment.py', ROOT/'eargtc/joint_selector/evaluation.py']
    paths += [ROOT/name for name in (
        'eargtc/joint_v5_fixed/__init__.py', 'eargtc/joint_v5_fixed/adapter.py',
        'eargtc/joint_v5_fixed/experiment.py', 'eargtc/joint_v5_fixed/evaluation.py',
        'eargtc/joint_v5_fixed/publication.py', 'eargtc/router_v8_v5/__init__.py',
        'eargtc/router_v8_v5/pipeline.py', 'eargtc/joint_v8/adapter.py',
        'scripts/joint_v5_fixed/run.py', 'scripts/joint_v5_fixed/evaluate.py',
        'scripts/joint_v5_fixed/publish.py', 'scripts/joint_v5_fixed/prepare.sbatch',
        'scripts/joint_v5_fixed/initialize.sbatch', 'scripts/joint_v5_fixed/train.sbatch',
        'scripts/joint_v5_fixed/finish.sbatch', 'scripts/joint_v5_fixed/smoke.sbatch')]
    sources={**old['source_code'],**{str(p):sha256(p) for p in paths}}
    history_root=ROOT/'runs/joint_v8_20260908_v1'
    history_files=[history_root/'CONFIG.json',history_root/'FROZEN.json',history_root/'evaluation/performance.csv',
        OLD_FIVE/'evaluation/performance.csv']
    for ds in DATASETS:
        history_files.extend(history_root/'v8_v4'/ds/name for name in ['complete.json','predictions.pt'])
    snapshot=ROOT/'artifacts/preserved_versions/joint_v8_fixed_v2_20260908/SNAPSHOT.json'
    protected={str(ROOT/k):v for k,v in read_json(snapshot)['files'].items()}
    verify_files(protected)
    inputs=[export/'COMPLETE.json',OLD_FIVE/'CONFIG.json',snapshot,*history_files]
    inputs.extend(export/r/d/'complete.json' for r,d in itertools.product(ROUTERS,DATASETS))
    config={'architecture':old['architecture'],'routers':list(ROUTERS),'datasets':list(DATASETS),
        'train':asdict(cfg),'seeds':[42] if smoke else list(SEEDS),'recipes':[asdict(r) for r in RECIPES],
        'history_root':str(history_root),'preserved_snapshot':str(snapshot),'preserved_file_count':len(protected),
        'smoke':smoke,'export_root':str(export),'base_export_root':str(BASE_EXPORT),'label_root':str(FROZEN_ROUTER),
        'old_five_root':str(OLD_FIVE),'gold_members':old['gold_members'],'source_code':sources,
        'input_completions':{**old['input_completions'],**original['input_completions'],**original_bindings,**{str(p):sha256(p) for p in inputs}},'torch_version':str(torch.__version__),
        'protocol':'fixed_V2_v5_disagreement_cross_CE_and_agreement_keep_KL'}
    path=root/'CONFIG.json'
    if path.exists():
        if read_json(path)!=config:raise ValueError('immutable config/input/source mismatch')
    else:write_json(path,config)
    return config


def _verify_cell(root,rid,ds,config_hash):
    dest=Path(root)/rid/ds;complete=read_json(dest/'complete.json')
    if (complete.get('router_id')!=rid or complete.get('dataset')!=ds or complete.get('config_sha256')!=config_hash
        or complete.get('test_labels_read') is not False or complete.get('all_checks_pass') is not True):
        raise ValueError('invalid cell completion/provenance')
    if not REQUIRED_ARTIFACTS.issubset(complete.get('artifacts',{})):raise ValueError('incomplete cell artifacts')
    verify_files({str(dest/k):v for k,v in complete['artifacts'].items()})


def verify_frozen(run_root):
    root=Path(run_root).resolve();lock=read_json(root/'FROZEN.json')
    cells=list(itertools.product(ROUTERS,DATASETS));expected={str(root/'CONFIG.json')}|{str(root/r/d/'complete.json') for r,d in cells}
    if (lock.get('routers')!=list(ROUTERS) or lock.get('datasets')!=list(DATASETS)
        or lock.get('test_labels_read') is not False or set(lock.get('files',{}))!=expected):
        raise ValueError('all5 cells must be safely frozen')
    verify_files(lock['files']);config=read_json(root/'CONFIG.json')
    if config['routers']!=list(ROUTERS) or config['datasets']!=list(DATASETS):raise ValueError('invalid search population')
    verify_files(config['source_code']);verify_files(config['input_completions']);digest=sha256(root/'CONFIG.json')
    for r,d in cells:_verify_cell(root,r,d,digest)
    return config


def freeze(run_root):
    root=Path(run_root).resolve();cells=list(itertools.product(ROUTERS,DATASETS))
    paths=[root/r/d/'complete.json' for r,d in cells]
    if not all(p.exists() for p in paths):return False
    digest=sha256(root/'CONFIG.json')
    for r,d in cells:_verify_cell(root,r,d,digest)
    lock={'routers':list(ROUTERS),'datasets':list(DATASETS),'test_labels_read':False,
          'files':{str(p):sha256(p) for p in [root/'CONFIG.json',*paths]}}
    path=root/'FROZEN.json'
    if path.exists():
        if read_json(path)!=lock:raise ValueError('frozen run changed')
    else:write_json(path,lock)
    verify_frozen(root);return True


def preflight(run_root):
    from .adapter import load_cell
    config=verify_frozen(run_root)
    if config['smoke']:raise ValueError('smoke cannot evaluate test truth')
    dependencies={}
    for rid,ds in itertools.product(ROUTERS,DATASETS):
        f,s,p,h,meta=load_cell(config['export_root'],rid,ds,verify_sources=True)
        for path,digest in itertools.chain(meta['source_hashes'].items(),meta['base_source_hashes'].items()):
            if path in dependencies and dependencies[path]!=digest:raise ValueError('conflicting provenance')
            dependencies[path]=digest
        del f,s,p,h,meta
    verify_files(dependencies)
    return config


def run_cell(run_root,router_id,dataset):
    from .adapter import load_cell
    root=Path(run_root).resolve();config=read_json(root/'CONFIG.json');start=time.monotonic()
    if router_id not in ROUTERS or dataset not in DATASETS:raise ValueError('unknown Router/dataset')
    cfg=TrainConfig(**config['train']);torch.set_num_threads(cfg.threads)
    verify_files(config['source_code']);verify_files(config['input_completions'])
    dest=root/router_id/dataset;dest.mkdir(parents=True,exist_ok=True);config_hash=sha256(root/'CONFIG.json')
    if (dest/'complete.json').exists():
        _verify_cell(root,router_id,dataset,config_hash);return router_id,dataset
    f,s,p,h,meta=load_cell(config['export_root'],router_id,dataset)
    if meta['router_id']!=router_id or meta['dataset']!=dataset:raise ValueError('adapter identity differs')
    base=(Path(config['base_export_root'])/dataset).resolve()
    if Path(meta['base_export_dir']).resolve()!=base or meta['base_export_complete_sha256']!=sha256(base/'complete.json'):
        raise ValueError('gold reuse requires identical original base export')
    members={};trained=reused=0;train_seconds=0.
    for seed,raw in itertools.product(config['seeds'],config['recipes']):
        recipe=Recipe(**raw);member_cfg=replace(cfg,seed=seed)
        folder=dest/'bank'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
        path,marker=folder/(recipe.name+'.pt'),folder/(recipe.name+'.json');reuse=None
        if marker.exists():
            record=read_json(marker)
            if record['config_sha256']!=config_hash:raise ValueError('bank config changed')
            verify_files({str(path):record['sha256']});member=torch.load(path,weights_only=True,map_location='cpu')
            reuse=record.get('reused_from')
            if reuse:verify_files({reuse['path']:reuse['sha256']})
        else:
            if recipe.name=='gold_only' and not config['smoke']:
                reuse=config['gold_members'][f'{dataset}/{seed}'];verify_files({reuse['path']:reuse['sha256']})
                member=torch.load(reuse['path'],weights_only=True,map_location='cpu')
                _reproduce_member(member,member_cfg,recipe,f,s,h)
            else:member=train_member(f,s,p,h,recipe,member_cfg)
            save_tensor(path,member);write_json(marker,{'sha256':sha256(path),'config_sha256':config_hash,'reused_from':reuse})
        _check_member(member,member_cfg,recipe,s);members[(seed,recipe.name)]=member
        if reuse:reused+=1
        else:trained+=1;train_seconds+=member['train_seconds']
        print({'router':router_id,'dataset':dataset,'seed':seed,'recipe':recipe.name,'reused':bool(reuse)},flush=True)
    candidates=candidate_bank(members,f,s,config['seeds'])
    pg,pl=f['logp_g'].exp(),f['logp_l'].exp();pb=scheme_b_probability(pg,pl,f['choose_g'],f['agreement'])
    if not torch.equal(pb.argmax(1),f['selected_pred']):raise ValueError('Router B identity differs')
    ids,y=s['selection_ids'],s['selection_targets'];selection=select_candidates(candidates,pb[ids],y)
    write_json(dest/'selection.json',selection)
    save_tensor(dest/'validation.pt',{'node_ids':ids,'candidates':candidates,'router':pb[ids]})
    catalog={c['id']:c for branch in candidates for c in candidates[branch]}
    write_json(dest/'candidate_catalog.json',{k:{'metadata':c['metadata'],**_score(c['probability'],y)} for k,c in catalog.items()})
    full_cache={}
    def full(candidate_id):
        if candidate_id in full_cache:return full_cache[candidate_id]
        m=catalog[candidate_id]['metadata'];branch,base=m['branch'],f['logp_'+m['branch']]
        if m['base']:value=base.exp().unsqueeze(0).repeat(len(config['seeds']),1,1)
        else:value=torch.stack([residual_probability(base,member_logits(members[(seed,m['recipe'])],branch,m['criterion'],f,h,
                f['node_ids'],replace(cfg,seed=seed)),m['scale']) for seed in config['seeds']])
        observed,reference=value.mean(0)[ids],catalog[candidate_id]['probability']
        if not torch.allclose(observed,reference,atol=2e-6) or not torch.equal(observed.argmax(1),reference.argmax(1)):
            raise ValueError('restored branch differs from validation candidate')
        full_cache[candidate_id]=value;return value
    families={}
    for name,pick in selection['families'].items():
        g=full(pick['g_id'] if pick['beta']*pick['alpha']>0 else 'g/base')
        l=full(pick['l_id'] if pick['beta']*(1-pick['alpha'])>0 else 'l/base')
        seeds=blend(g,l,pb.unsqueeze(0),pick['alpha'],pick['beta']);ensemble=seeds.mean(0)
        score=_score(ensemble[ids],y)
        if abs(score['accuracy']-pick['accuracy'])>1e-10 or abs(score['nll']-pick['nll'])>2e-6:
            raise ValueError('Joint full prediction fails validation reproduction')
        families[name]={'ensemble':ensemble,'seeds':seeds}
    b=selection['base_fusion'];baselines={'gnn':pg,'llm':pl,'router':pb,'base_fusion':blend(pg,pl,pb,b['alpha'],b['beta'])}
    winner=selection['optimized']['family']
    families['optimized']=families[winner] if winner in families else {
        'ensemble':baselines[winner],'seeds':baselines[winner].unsqueeze(0).repeat(len(config['seeds']),1,1)}
    save_tensor(dest/'predictions.pt',{'node_ids':f['node_ids'],'baselines':baselines,'families':families})
    write_json(dest/'metadata.json',{**meta,'member_count':len(members),'trained_count':trained,'reused_count':reused,
        'train_seconds':train_seconds,'elapsed_seconds':time.monotonic()-start,'architecture':config['architecture'],'test_labels_read':False,
        'execution':{'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'array_job_id':os.environ.get('SLURM_ARRAY_JOB_ID'),
                     'array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),'cpus_per_task':os.environ.get('SLURM_CPUS_PER_TASK'),'hostname':socket.gethostname()}})
    verify_files(config['source_code']);verify_files(config['input_completions'])
    artifacts={str(q.relative_to(dest)):sha256(q) for q in sorted(dest.rglob('*'))
               if q.is_file() and q.name!='complete.json' and '.tmp.' not in q.name}
    write_json(dest/'complete.json',{'router_id':router_id,'dataset':dataset,'config_sha256':config_hash,
        'artifacts':artifacts,'all_checks_pass':True,'test_labels_read':False})
    print({'router':router_id,'dataset':dataset,'complete':True,'selected':selection['optimized']},flush=True)
    return router_id,dataset
