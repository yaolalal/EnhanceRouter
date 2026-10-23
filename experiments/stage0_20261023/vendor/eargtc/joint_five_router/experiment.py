"""Equal-budget, fixed-architecture Joint search across 25 frozen Router cells."""
from dataclasses import asdict,replace
import itertools
from pathlib import Path
import time
import torch
from eargtc.joint_cross_repair.artifacts import DATASETS,read_json,write_json,save_tensor,sha256,verify_files
from eargtc.joint_cross_repair.prepare import DEFAULT_EXPORT as BASE_EXPORT,FROZEN_ROUTER
from eargtc.joint_cross_repair.training import TrainConfig,_score,blend,scheme_b_probability
from eargtc.joint_cross_repair_search.bank import RECIPES,SEEDS,Recipe,train_member,member_logits
from eargtc.joint_cross_repair_search.experiment import candidate_bank
from eargtc.joint_cross_repair_search.selection import residual_probability,choose_independent,choose_fusion_pair,choose_joint

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_RUN=ROOT/'runs/joint_five_router_20260907_v1'
OLD_V2=ROOT/'runs/joint_cross_repair_search_20260906_v1'
ROUTER_IDS=('r6_legacy_c','r6_split_c','r6_expert_conf_c','a20ms_split_c','a20emb_v2_split_c')
FAMILIES=('gold_only','weighted','scaled','joint')
SPEC=ROOT/'docs/superpowers/specs/2026-09-07-joint-five-router-design.md'


def select_candidates(candidates,pb,y):
    selected={}
    def subset(branch,family):
        def allowed(c):
            m=c['metadata']
            if m['base']:return True
            if family=='gold_only':return m['recipe']=='gold_only' and m['criterion']=='accuracy' and m['scale']==1.
            if family=='weighted':return m['criterion']=='accuracy' and m['scale']==1.
            return True
        return [c for c in candidates[branch] if allowed(c)]
    for family in ('gold_only','weighted','scaled'):
        g=choose_independent(subset('g',family),y);l=choose_independent(subset('l',family),y)
        selected[family]=choose_fusion_pair(g,l,pb,y)
    selected['joint']=choose_joint(candidates['g'],candidates['l'],pb,y)
    base=choose_fusion_pair(candidates['g'][0],candidates['l'][0],pb,y)
    alternatives=[{'family':k,**v} for k,v in selected.items()]
    alternatives.extend([{'family':'base_fusion',**base},{'family':'router','active_repairs':0,**_score(pb,y)}])
    winner=min(alternatives,key=lambda r:(-r['accuracy'],r['nll'],r['active_repairs'],r['family']))
    return {'families':selected,'base_fusion':base,'optimized':winner,'alternatives':alternatives,
        'selection_population':'historical_validation_calibration_half','seed_reduction':'mean_probability'}


def initialize(run_root,export_root=None,smoke=False):
    from .adapter import DEFAULT_EXPORT,SOURCE_ROOT
    root=Path(run_root).resolve();root.mkdir(parents=True,exist_ok=True)
    export=Path(export_root or DEFAULT_EXPORT).resolve()
    if not all((export/r/d/'complete.json').exists() for r in ROUTER_IDS for d in DATASETS):
        raise ValueError('all25 safe Router exports required before search initialization')
    from eargtc.joint_cross_repair_search.experiment import preflight as old_preflight
    old=old_preflight(OLD_V2)
    if Path(old['export_root']).resolve()!=BASE_EXPORT.resolve():raise ValueError('gold reuse base export differs')
    cfg=TrainConfig(threads=2,validate_every=5 if smoke else 50,max_steps=10 if smoke else 500,patience=2 if smoke else 5)
    paths=[SPEC]
    for folder in ('eargtc/joint_cross_repair','eargtc/joint_cross_repair_search','eargtc/joint_five_router',
        'scripts/joint_cross_repair','scripts/joint_cross_repair_search','scripts/joint_five_router','eargtc/router_fiveway_topk'):
        paths+=sorted((ROOT/folder).glob('*.py'))
    inputs=[export/r/d/'complete.json' for r in ROUTER_IDS for d in DATASETS]
    inputs += [SOURCE_ROOT/'manifest.json',OLD_V2/'SEARCH_CONFIG.json',OLD_V2/'FROZEN.json']
    if (export/'COMPLETE.json').exists():inputs.append(export/'COMPLETE.json')
    gold={}
    for ds,seed in itertools.product(DATASETS,SEEDS):
        p=OLD_V2/ds/'bank'/f'seed{seed}'/'gold_only.pt'
        gold[f'{ds}/{seed}']={'path':str(p),'sha256':sha256(p)}
    config={'architecture':'frozen_V2_gold_anchored_dual_residual64','datasets':list(DATASETS),'routers':list(ROUTER_IDS),
        'seeds':[42] if smoke else list(SEEDS),'recipes':[asdict(r) for r in (RECIPES[:2] if smoke else RECIPES)],
        'train':asdict(cfg),'smoke':smoke,'export_root':str(export),'source_root':str(SOURCE_ROOT),
        'base_export_root':str(BASE_EXPORT),'old_v2_root':str(OLD_V2),'label_root':str(FROZEN_ROUTER),
        'source_code':{str(p):sha256(p) for p in paths},'input_completions':{str(p):sha256(p) for p in inputs},
        'gold_members':gold,'torch_version':str(torch.__version__)}
    path=root/'CONFIG.json'
    if path.exists():
        if read_json(path)!=config:raise ValueError('immutable search config/source/input mismatch')
    else:write_json(path,config)
    return config


def _verify_cell(root,rid,ds,config_hash):
    dest=root/rid/ds;c=read_json(dest/'complete.json')
    if c.get('router_id')!=rid or c.get('dataset')!=ds or c.get('config_sha256')!=config_hash or c.get('test_labels_read') is not False or c.get('all_checks_pass') is not True:
        raise ValueError('invalid cell completion/provenance')
    verify_files({str(dest/k):v for k,v in c['artifacts'].items()})


def verify_frozen(run_root):
    root=Path(run_root).resolve();lock=read_json(root/'FROZEN.json')
    expected={str(root/'CONFIG.json')}|{str(root/r/d/'complete.json') for r in ROUTER_IDS for d in DATASETS}
    if lock.get('routers')!=list(ROUTER_IDS) or lock.get('datasets')!=list(DATASETS) or lock.get('test_labels_read') is not False or set(lock['files'])!=expected:
        raise ValueError('all25 cells must be safely frozen')
    verify_files(lock['files']);config=read_json(root/'CONFIG.json')
    if config['routers']!=list(ROUTER_IDS) or config['datasets']!=list(DATASETS):raise ValueError('invalid search population')
    verify_files(config['source_code']);verify_files(config['input_completions'])
    for rid,ds in itertools.product(ROUTER_IDS,DATASETS):_verify_cell(root,rid,ds,sha256(root/'CONFIG.json'))
    return config


def freeze(run_root):
    root=Path(run_root).resolve()
    paths=[root/r/d/'complete.json' for r in ROUTER_IDS for d in DATASETS]
    if not all(p.exists() for p in paths):return False
    for rid,ds in itertools.product(ROUTER_IDS,DATASETS):_verify_cell(root,rid,ds,sha256(root/'CONFIG.json'))
    lock={'routers':list(ROUTER_IDS),'datasets':list(DATASETS),'test_labels_read':False,
        'files':{str(p):sha256(p) for p in [root/'CONFIG.json',*paths]}}
    path=root/'FROZEN.json'
    if path.exists():
        if read_json(path)!=lock:raise ValueError('frozen run changed')
    else:write_json(path,lock)
    verify_frozen(root);return True


def preflight(run_root):
    config=verify_frozen(run_root)
    if config['smoke']:raise ValueError('smoke cannot evaluate test truth')
    dependencies={}
    for rid,ds in itertools.product(ROUTER_IDS,DATASETS):
        export=Path(config['export_root'])/rid/ds
        c=read_json(export/'complete.json')
        verify_files({str(export/k):v for k,v in c['artifacts'].items()})
        meta=read_json(export/'metadata.json')
        if meta['router_id']!=rid or meta['dataset']!=ds:raise ValueError('safe export identity differs')
        dependencies.update(meta['source_hashes'])
    for ds in DATASETS:
        base=Path(config['base_export_root'])/ds
        c=read_json(base/'complete.json')
        verify_files({str(base/k):v for k,v in c['artifacts'].items()})
        dependencies.update(read_json(base/'metadata.json')['source_hashes'])
    verify_files(dependencies)
    return config


def _check_member(member,cfg,recipe,s):
    if member['config']!=asdict(cfg) or member['recipe']!=asdict(recipe) or member.get('test_labels_read') is not False or not torch.equal(member['validation_ids'],s['selection_ids']):
        raise ValueError('member config/recipe/validation mismatch')


def run_cell(run_root,router_id,dataset):
    from .adapter import load_cell
    root=Path(run_root).resolve();config=read_json(root/'CONFIG.json');start=time.monotonic()
    if router_id not in ROUTER_IDS or dataset not in DATASETS:raise ValueError('unknown Router/dataset cell')
    cfg=TrainConfig(**config['train']);torch.set_num_threads(cfg.threads)
    verify_files(config['source_code']);verify_files(config['input_completions'])
    dest=root/router_id/dataset;dest.mkdir(parents=True,exist_ok=True);config_hash=sha256(root/'CONFIG.json')
    if (dest/'complete.json').exists():_verify_cell(root,router_id,dataset,config_hash);return router_id,dataset
    f,s,p,h,meta=load_cell(config['export_root'],router_id,dataset)
    if meta['router_id']!=router_id or meta['dataset']!=dataset:raise ValueError('adapter cell identity differs')
    base_export=(Path(config['base_export_root'])/dataset).resolve()
    if Path(meta['base_export_dir']).resolve()!=base_export or meta['base_export_complete_sha256']!=sha256(base_export/'complete.json'):
        raise ValueError('gold reuse requires identical original safe base export')
    # Only acceptance/route fields differ; the raw bases/embeddings and gold/val splits
    # are verified by the adapter against the identical old safe export.
    members={};trained=0;reused=0
    for seed,raw in itertools.product(config['seeds'],config['recipes']):
        recipe=Recipe(**raw);member_cfg=replace(cfg,seed=seed)
        folder=dest/'bank'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
        path=folder/(recipe.name+'.pt');marker=folder/(recipe.name+'.json')
        reuse=None
        if marker.exists():
            record=read_json(marker)
            if record['config_sha256']!=config_hash:raise ValueError('bank config changed')
            verify_files({str(path):record['sha256']});member=torch.load(path,weights_only=True,map_location='cpu')
            reuse=record.get('reused_from')
        else:
            if recipe.name=='gold_only' and not config['smoke']:
                source=config['gold_members'][f'{dataset}/{seed}'];verify_files({source['path']:source['sha256']})
                member=torch.load(source['path'],weights_only=True,map_location='cpu');reuse=source
                _check_member(member,member_cfg,recipe,s)
                # Safe current supervision is the exact same base export; recompute
                # validation logits from this member to verify raw feature compatibility.
                for branch in ('g','l'):
                    for criterion in ('accuracy','nll'):
                        z=member_logits(member,branch,criterion,f,h,s['selection_ids'],member_cfg)
                        if not torch.allclose(z,member['validation_logits'][branch][criterion],atol=2e-6) or not torch.equal(z.argmax(1),member['validation_logits'][branch][criterion].argmax(1)):
                            raise ValueError('gold-only reused model does not reproduce current inputs')
            else:member=train_member(f,s,p,h,recipe,member_cfg)
            save_tensor(path,member);write_json(marker,{'sha256':sha256(path),'config_sha256':config_hash,'reused_from':reuse})
        _check_member(member,member_cfg,recipe,s)
        members[(seed,recipe.name)]=member
        if reuse:reused+=1
        else:trained+=1
        print({'router':router_id,'dataset':dataset,'seed':seed,'recipe':recipe.name,'reused':bool(reuse),
            'seconds':round(member['train_seconds'],2)},flush=True)
    candidates=candidate_bank(members,f,s,config['seeds'])
    pg,pl=f['logp_g'].exp(),f['logp_l'].exp();pb=scheme_b_probability(pg,pl,f['choose_g'],f['agreement'])
    if not torch.equal(pb.argmax(1),f['selected_pred']):raise ValueError('Router B identity differs')
    ids=s['selection_ids'];y=s['selection_targets'];selection=select_candidates(candidates,pb[ids],y)
    write_json(dest/'selection.json',selection)
    save_tensor(dest/'validation.pt',{'node_ids':ids,'candidates':candidates,'router':pb[ids]})
    catalog={c['id']:c for branch in candidates for c in candidates[branch]}
    write_json(dest/'candidate_catalog.json',{k:{'metadata':c['metadata'],**_score(c['probability'],y)} for k,c in catalog.items()})
    full_cache={}
    def full(candidate_id):
        if candidate_id in full_cache:return full_cache[candidate_id]
        m=catalog[candidate_id]['metadata'];branch=m['branch'];base=f['logp_'+branch]
        if m['base']:value=base.exp().unsqueeze(0).repeat(len(config['seeds']),1,1)
        else:value=torch.stack([residual_probability(base,member_logits(members[(seed,m['recipe'])],branch,m['criterion'],f,h,f['node_ids'],replace(cfg,seed=seed)),m['scale']) for seed in config['seeds']])
        observed=value.mean(0)[ids];reference=catalog[candidate_id]['probability']
        if not torch.allclose(observed,reference,atol=2e-6) or not torch.equal(observed.argmax(1),reference.argmax(1)):
            raise ValueError('restored full branch differs from validation candidate')
        full_cache[candidate_id]=value;return value
    families={}
    for name,pick in selection['families'].items():
        g=full(pick['g_id'] if pick['beta']*pick['alpha']>0 else 'g/base')
        l=full(pick['l_id'] if pick['beta']*(1-pick['alpha'])>0 else 'l/base')
        seeds=blend(g,l,pb.unsqueeze(0),pick['alpha'],pick['beta']);ens=seeds.mean(0)
        score=_score(ens[ids],y)
        if abs(score['accuracy']-pick['accuracy'])>1e-10 or abs(score['nll']-pick['nll'])>2e-6:raise ValueError('Joint full prediction fails validation reproduction')
        families[name]={'ensemble':ens,'seeds':seeds}
    b=selection['base_fusion'];basefusion=blend(pg,pl,pb,b['alpha'],b['beta'])
    baselines={'gnn':pg,'llm':pl,'router':pb,'base_fusion':basefusion}
    winner=selection['optimized']['family']
    families['optimized']=families[winner] if winner in families else {'ensemble':baselines[winner],
        'seeds':baselines[winner].unsqueeze(0).repeat(len(config['seeds']),1,1)}
    save_tensor(dest/'predictions.pt',{'node_ids':f['node_ids'],'baselines':baselines,'families':families})
    new_meta={**meta,'member_count':len(members),'trained_count':trained,'reused_count':reused,
        'train_seconds':sum(m['train_seconds'] for (seed,name),m in members.items() if name!='gold_only' or config['smoke']),
        'elapsed_seconds':time.monotonic()-start,'architecture':config['architecture'],'test_labels_read':False}
    write_json(dest/'metadata.json',new_meta)
    verify_files(config['source_code']);verify_files(config['input_completions'])
    artifacts={str(q.relative_to(dest)):sha256(q) for q in sorted(dest.rglob('*')) if q.is_file() and q.name!='complete.json'}
    write_json(dest/'complete.json',{'router_id':router_id,'dataset':dataset,'config_sha256':config_hash,
        'artifacts':artifacts,'all_checks_pass':True,'test_labels_read':False})
    print({'router':router_id,'dataset':dataset,'complete':True,'selected':selection['optimized']},flush=True)
    return router_id,dataset
