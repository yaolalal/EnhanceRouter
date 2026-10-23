"""Immutable trajectory search; only safe exports are opened before freezing."""
from dataclasses import asdict, replace
import itertools
from pathlib import Path
import time
import torch
from eargtc.joint_cross_repair.artifacts import (DATASETS,read_json,write_json,save_tensor,sha256,verify_files,load_export,verify_training_link)
from eargtc.joint_cross_repair.prepare import DEFAULT_EXPORT,FROZEN_ROUTER
from eargtc.joint_cross_repair.training import TrainConfig,scheme_b_probability,_score
from eargtc.joint_cross_repair_search.bank import Recipe
from .policy import make_policies,make_folds,collect_statistics,choose_from_statistics,apply_policy

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_RUN=ROOT/'runs/joint_four_view_20260907_v1'
V1_ROOT=ROOT/'runs/joint_cross_repair_main_20260906_v1'
V2_ROOT=ROOT/'runs/joint_cross_repair_search_20260906_v1'
SPEC=ROOT/'docs/superpowers/specs/2026-09-07-joint-four-view-design.md'
RECIPES=(Recipe('gold_only',0.,0.),Recipe('cross050',.5,.1),Recipe('preserve050',.5,.5))


def initialize(run_root,smoke=False):
    root=Path(run_root).resolve();root.mkdir(parents=True,exist_ok=True)
    paths=[SPEC]
    for folder in ('eargtc/joint_cross_repair','eargtc/joint_cross_repair_search','eargtc/joint_four_view',
                   'scripts/joint_cross_repair','scripts/joint_cross_repair_search','scripts/joint_four_view'):
        paths+=sorted((ROOT/folder).glob('*.py'))
    cfg=TrainConfig(threads=1,max_steps=2 if smoke else 500)
    config={'datasets':list(DATASETS),'seeds':[42] if smoke else [42,43,44],
        'snapshots':[1,2] if smoke else [50,150,300,500], 'recipes':[asdict(x) for x in RECIPES],
        'train':asdict(cfg),'smoke':smoke,'export_root':str(DEFAULT_EXPORT),'router_root':str(FROZEN_ROUTER),
        'v1_root':str(V1_ROOT),'v2_root':str(V2_ROOT),'source_code':{str(p):sha256(p) for p in paths},
        'input_completions':{str(p):sha256(p) for ds in DATASETS for p in
            (DEFAULT_EXPORT/ds/'complete.json',V1_ROOT/ds/'complete.json',V2_ROOT/ds/'complete.json')}}
    path=root/'CONFIG.json'
    if path.exists():
        if read_json(path)!=config:raise ValueError('immutable run configuration differs')
    else:write_json(path,config)
    return config


def _verify_dataset(root,ds,config_hash):
    dest=root/ds;c=read_json(dest/'complete.json')
    if c.get('dataset')!=ds or c.get('config_sha256')!=config_hash or c.get('test_labels_read') is not False or c.get('all_checks_pass') is not True:
        raise ValueError('unsafe or mismatched dataset completion')
    verify_files({str(dest/k):v for k,v in c['artifacts'].items()})


def verify_frozen(run_root):
    root=Path(run_root).resolve();lock=read_json(root/'FROZEN.json')
    expected={str(root/'CONFIG.json')}|{str(root/ds/'complete.json') for ds in DATASETS}
    if lock.get('datasets')!=list(DATASETS) or lock.get('test_labels_read') is not False or set(lock['files'])!=expected:
        raise ValueError('all five safely frozen datasets required')
    verify_files(lock['files']);config=read_json(root/'CONFIG.json')
    if config['datasets']!=list(DATASETS):raise ValueError('dataset configuration changed')
    verify_files(config['source_code']);verify_files(config['input_completions'])
    for ds in DATASETS:_verify_dataset(root,ds,sha256(root/'CONFIG.json'))
    return config


def freeze(run_root):
    root=Path(run_root).resolve()
    if not all((root/d/'complete.json').exists() for d in DATASETS):return False
    paths=[root/'CONFIG.json']+[root/d/'complete.json' for d in DATASETS]
    config_hash=sha256(root/'CONFIG.json')
    for ds in DATASETS:_verify_dataset(root,ds,config_hash)
    lock={'datasets':list(DATASETS),'test_labels_read':False,'files':{str(p):sha256(p) for p in paths}}
    path=root/'FROZEN.json'
    if path.exists():
        if read_json(path)!=lock:raise ValueError('frozen run differs')
    else:write_json(path,lock)
    verify_frozen(root);return True


def preflight(run_root):
    config=verify_frozen(run_root)
    if config['smoke']:raise ValueError('smoke runs cannot access test truth')
    # Source integrity remains transitive to original exports and previous baselines.
    from eargtc.joint_cross_repair_search.experiment import preflight as old_preflight
    old_preflight(config['v2_root'])
    return config


def _verify_baselines(config,ds,export):
    verify_training_link(Path(config['v1_root'])/ds,export)
    root=Path(config['v2_root']);c=read_json(root/ds/'complete.json')
    if c.get('dataset')!=ds or c.get('test_labels_read') is not False or c['config_sha256']!=sha256(root/'SEARCH_CONFIG.json'):
        raise ValueError('V2 provenance mismatch')
    verify_files({str(root/ds/k):v for k,v in c['artifacts'].items()})
    old_config=read_json(root/'SEARCH_CONFIG.json')
    for key in ('export_root','v1_root','router_root'):
        if Path(old_config[key]).resolve()!=Path(config[key]).resolve():raise ValueError('baseline source roots differ')
    verify_files(old_config['source_code'])


def run_dataset(run_root,dataset):
    from .trajectory import train_trajectory,trajectory_logits
    root=Path(run_root).resolve();config=read_json(root/'CONFIG.json');start=time.monotonic()
    cfg=TrainConfig(**config['train']);torch.set_num_threads(cfg.threads)
    if dataset not in DATASETS:raise ValueError('unknown dataset')
    verify_files(config['source_code']);verify_files(config['input_completions'])
    dest=root/dataset;dest.mkdir(exist_ok=True);config_hash=sha256(root/'CONFIG.json')
    if (dest/'complete.json').exists():
        _verify_dataset(root,dataset,config_hash);return dataset
    export=Path(config['export_root'])/dataset
    _verify_baselines(config,dataset,export)
    f,s,p,h,meta=load_export(export)
    if meta['dataset']!=dataset:raise ValueError('export dataset mismatch')
    members={};seeds=config['seeds'];steps=config['snapshots'];recipes=[Recipe(**r) for r in config['recipes']]
    for seed,recipe in itertools.product(seeds,recipes):
        folder=dest/'bank'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
        path=folder/(recipe.name+'.pt');marker=folder/(recipe.name+'.json');member_cfg=replace(cfg,seed=seed)
        if marker.exists():
            record=read_json(marker)
            if record['config_sha256']!=config_hash:raise ValueError('trajectory config differs')
            verify_files({str(path):record['sha256']});member=torch.load(path,weights_only=True,map_location='cpu')
        else:
            member=train_trajectory(f,s['gold_ids'],s['gold_targets'],p,h,recipe,member_cfg,snapshots=tuple(steps))
            save_tensor(path,member);write_json(marker,{'sha256':sha256(path),'config_sha256':config_hash})
        if member['recipe']!=asdict(recipe) or member['config']!=asdict(member_cfg) or member['validation_labels_read'] is not False or member['test_labels_read'] is not False:
            raise ValueError('invalid trajectory provenance')
        for branch in ('g','l'):
            if set(member['states'][branch])!=set(map(str,steps)):raise ValueError('missing fixed snapshots')
        members[(seed,recipe.name)]=member
        print({'dataset':dataset,'seed':seed,'recipe':recipe.name,'train_seconds':round(member['train_seconds'],2)},flush=True)
    ids=s['selection_ids'];branches={'g0':f['logp_g'][ids].exp(),'l0':f['logp_l'][ids].exp()}
    def predict_seeds(branch,recipe_name,step,idx):
        return torch.stack([trajectory_logits(members[(seed,recipe_name)],branch,step,f,h,idx,replace(cfg,seed=seed)).softmax(1) for seed in seeds])
    # Fresh snapshots were fixed before any selection target was read.
    for branch,recipe,step in itertools.product(('g','l'),recipes,steps):
        branches[f'{branch}/{recipe.name}/{step}']=predict_seeds(branch,recipe.name,step,ids).mean(0)
    for branch,step in itertools.product(('g','l'),steps):
        branches[f'{branch}/recipe_mean/{step}']=torch.stack([branches[f'{branch}/{r.name}/{step}'] for r in recipes]).mean(0)
    pairs=[]
    for step in steps:
        pairs += [(f'g/{g.name}/{step}',f'l/{l.name}/{step}') for g,l in itertools.product(recipes,recipes)]
        pairs.append((f'g/recipe_mean/{step}',f'l/recipe_mean/{step}'))
    pg,pl=f['logp_g'].exp(),f['logp_l'].exp();pb=scheme_b_probability(pg,pl,f['choose_g'],f['agreement'])
    if not torch.equal(pb.argmax(1),f['selected_pred']):raise ValueError('frozen Router identity mismatch')
    records=make_policies(pairs);folds=make_folds(ids)
    stats=collect_statistics(records,branches,pb[ids],f['agreement'][ids],s['selection_targets'],folds,
        progress=lambda done,total:print({'dataset':dataset,'policies_scored':done,'total':total},flush=True))
    selection=choose_from_statistics(records,stats['correct'],stats['nll_sum'],stats['counts'])
    write_json(dest/'candidate_catalog.json',records);write_json(dest/'selection.json',selection)
    save_tensor(dest/'validation.pt',{'node_ids':ids,'branches':branches,'router':pb[ids],**stats})
    full_cache={'g0':pg.unsqueeze(0).repeat(len(seeds),1,1),'l0':pl.unsqueeze(0).repeat(len(seeds),1,1)}
    def full(key):
        if key is None:return
        if key in full_cache:return full_cache[key]
        branch,recipe_name,step=key.split('/');step=int(step)
        if recipe_name=='recipe_mean':value=torch.stack([full(f'{branch}/{r.name}/{step}') for r in recipes]).mean(0)
        else:value=predict_seeds(branch,recipe_name,step,f['node_ids'])
        if not torch.allclose(value.mean(0)[ids],branches[key],atol=2e-6) or not torch.equal(value.mean(0)[ids].argmax(1),branches[key].argmax(1)):
            raise ValueError('full branch inference failed validation reproduction')
        full_cache[key]=value;return value
    picks={k:v for k,v in selection['families'].items()}
    picks.update(primary=selection['primary'],direct=selection['direct'])
    outputs={};seed_outputs={}
    for name,pick in picks.items():
        r=pick['candidate'];needed=['g0','l0']+[k for k in (r['g_id'],r['l_id']) if k]
        local={k:full(k).mean(0) for k in needed}
        out=apply_policy(r,local,pb,f['agreement'])
        score=_score(out[ids],s['selection_targets'])
        if abs(score['accuracy']-pick['accuracy'])>1e-10 or abs(score['nll']-pick['nll'])>2e-6:
            raise ValueError('selected policy failed validation reproduction')
        outputs[name]=out
        seed_outputs[name]=torch.stack([apply_policy(r,{k:full(k)[i] for k in needed},pb,f['agreement']) for i in range(len(seeds))])
    v1=torch.load(Path(config['v1_root'])/dataset/'predictions.pt',weights_only=True,map_location='cpu')
    v2=torch.load(Path(config['v2_root'])/dataset/'predictions.pt',weights_only=True,map_location='cpu')
    if not torch.equal(v1['node_ids'],f['node_ids']) or not torch.equal(v2['node_ids'],f['node_ids']):raise ValueError('baseline node IDs differ')
    baselines={'gnn':pg,'llm':pl,'router':pb,'v1':v1['p_joint'],'v2':v2['families']['optimized']['ensemble']}
    save_tensor(dest/'predictions.pt',{'node_ids':f['node_ids'],'baselines':baselines,'methods':outputs,'seeds':seed_outputs})
    write_json(dest/'metadata.json',{'dataset':dataset,'export_complete_sha256':sha256(export/'complete.json'),
        'split_counts':{k:meta[k] for k in ('gold_count','model_selection_count','threshold_count','test_count')},
        'pool_counts':meta['pool_counts'],'pseudo_candidate_count':meta['pseudo_candidate_count'],
        'member_count':len(members),'candidate_count':len(records),'train_seconds':sum(m['train_seconds'] for m in members.values()),
        'total_seconds':time.monotonic()-start,'validation_independent_snapshots':True,
        'gate_supervision':'none: deterministic prediction confidence','test_labels_read':False})
    verify_files(config['source_code']);verify_files(meta['source_hashes'])
    artifacts={str(q.relative_to(dest)):sha256(q) for q in sorted(dest.rglob('*')) if q.is_file() and q.name!='complete.json'}
    write_json(dest/'complete.json',{'dataset':dataset,'config_sha256':config_hash,'artifacts':artifacts,'all_checks_pass':True,'test_labels_read':False})
    print({'dataset':dataset,'complete':True,'primary':selection['primary']['candidate'],'validation_accuracy':selection['primary']['accuracy']},flush=True)
    return dataset
