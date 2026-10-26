"""Safe training, ensemble candidate selection and immutable experiment locks."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import torch
from eargtc.joint_cross_repair.artifacts import (DATASETS,read_json,write_json,save_tensor,sha256,
    verify_files,load_export,verify_training_link)
from eargtc.joint_cross_repair.prepare import DEFAULT_EXPORT,FROZEN_ROUTER
from eargtc.joint_cross_repair.training import TrainConfig,_score,blend,scheme_b_probability
from .bank import RECIPES,SEEDS,Recipe,train_member,member_logits
from .selection import residual_probability,choose_independent,choose_fusion_pair,choose_joint

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_RUN=ROOT/'runs/joint_cross_repair_search_20260906_v1'
V1_ROOT=ROOT/'runs/joint_cross_repair_main_20260906_v1'
FAMILIES=('gold_only','weighted','scaled','joint')


def source_manifest():
    paths=[]
    for folder in ['eargtc/joint_cross_repair','scripts/joint_cross_repair',
                   'eargtc/joint_cross_repair_search','scripts/joint_cross_repair_search']:
        paths+=sorted((ROOT/folder).glob('*.py'))
    paths.append(ROOT/'docs/superpowers/specs/2026-09-06-joint-cross-repair-search-design.md')
    return {str(p):sha256(p) for p in paths}


def initialize(run_root,smoke=False):
    run_root=Path(run_root).resolve();run_root.mkdir(parents=True,exist_ok=True)
    train=TrainConfig(threads=2,validate_every=5 if smoke else 50,max_steps=10 if smoke else 500,patience=2 if smoke else 5)
    config={'datasets':list(DATASETS),'seeds':[42] if smoke else list(SEEDS),
            'recipes':[asdict(x) for x in (RECIPES[:2] if smoke else RECIPES)],'train':asdict(train),
            'smoke':smoke,'export_root':str(DEFAULT_EXPORT),'v1_root':str(V1_ROOT),
            'router_root':str(FROZEN_ROUTER),'source_code':source_manifest(),
            'input_completions':{str(p):sha256(p) for ds in DATASETS for p in [DEFAULT_EXPORT/ds/'complete.json',V1_ROOT/ds/'complete.json']}}
    path=run_root/'SEARCH_CONFIG.json'
    if path.exists():
        if read_json(path)!=config:raise ValueError('search config/source/input manifest is immutable')
    else:write_json(path,config)
    return config


def verify_frozen(run_root):
    root=Path(run_root).resolve();locked=read_json(root/'FROZEN.json')
    if locked.get('test_labels_read') is not False or set(locked['datasets'])!=set(DATASETS):
        raise ValueError('all five datasets must be safely frozen')
    expected={str(root/'SEARCH_CONFIG.json')}|{str(root/d/'complete.json') for d in DATASETS}
    if set(locked['files'])!=expected:raise ValueError('incomplete freeze manifest')
    verify_files(locked['files']);config_hash=sha256(root/'SEARCH_CONFIG.json')
    for ds in DATASETS:
        complete=read_json(root/ds/'complete.json')
        if complete.get('dataset')!=ds or not complete.get('all_checks_pass') or complete.get('test_labels_read') is not False:
            raise ValueError('unsafe or mismatched dataset completion')
        if complete['config_sha256']!=config_hash:raise ValueError('dataset bound to different search config')
        verify_files({str(root/ds/k):v for k,v in complete['artifacts'].items()})
    return read_json(root/'SEARCH_CONFIG.json')


def preflight(run_root):
    config=verify_frozen(run_root)
    if config['smoke']:raise ValueError('smoke cannot be a final test experiment')
    verify_files(config['source_code']);verify_files(config['input_completions'])
    for ds in DATASETS:
        export=Path(config['export_root'])/ds
        verify_training_link(Path(config['v1_root'])/ds,export)
        complete=read_json(export/'complete.json')
        verify_files({str(export/k):v for k,v in complete['artifacts'].items()})
        meta=read_json(export/'metadata.json')
        if meta['dataset']!=ds or meta['test_labels_exported'] is not False:raise ValueError('invalid export dataset')
        verify_files(meta['source_hashes'])
    return config


def freeze(run_root):
    root=Path(run_root).resolve()
    if not all((root/d/'complete.json').exists() for d in DATASETS):return False
    paths=[root/'SEARCH_CONFIG.json']+[root/d/'complete.json' for d in DATASETS]
    value={'datasets':list(DATASETS),'test_labels_read':False,'files':{str(p):sha256(p) for p in paths}}
    path=root/'FROZEN.json'
    if path.exists():
        if read_json(path)!=value:raise ValueError('frozen results changed')
    else:write_json(path,value)
    verify_frozen(root)
    return True


def candidate_bank(members,f,s,seeds):
    idx=s['selection_ids'];out={name:[] for name in ['g','l']}
    for branch in ['g','l']:
        base=f['logp_'+branch][idx]
        out[branch].append({'id':branch+'/base','probability':base.exp(),
                            'metadata':{'base':True,'branch':branch}})
        recipes=sorted({name for seed,name in members})
        for name in recipes:
            for criterion in ['accuracy','nll']:
                for scale in [.25,.5,1.]:
                    probs=[residual_probability(base,members[(seed,name)]['validation_logits'][branch][criterion],scale) for seed in seeds]
                    out[branch].append({'id':f'{branch}/{name}/{criterion}/{scale:g}',
                        'probability':torch.stack(probs).mean(0),
                        'metadata':{'base':False,'active_repair':any(members[(seed,name)]['checkpoints'][branch][criterion]['step']>0 for seed in seeds),
                                    'branch':branch,'recipe':name,'criterion':criterion,'scale':scale,
                                    'steps':[members[(seed,name)]['checkpoints'][branch][criterion]['step'] for seed in seeds]}})
    return out


def v1_active_repairs(selection):
    alpha,beta=selection['fusion']['alpha'],selection['fusion']['beta']
    return sum(int(weight>0 and selection[branch]['step']>0)
               for branch,weight in [('g',beta*alpha),('l',beta*(1-alpha))])


def select_families(candidates,pb,y,v1,v1_repairs=2):
    selected={}
    def subset(branch,family):
        def allowed(c):
            m=c['metadata']
            if m['base']:return True
            if family=='gold_only':return m['recipe']=='gold_only' and m['criterion']=='accuracy' and m['scale']==1.
            if family=='weighted':return m['criterion']=='accuracy' and m['scale']==1.
            return True
        return [c for c in candidates[branch] if allowed(c)]
    for family in ['gold_only','weighted','scaled']:
        g=choose_independent(subset('g',family),y);l=choose_independent(subset('l',family),y)
        selected[family]=choose_fusion_pair(g,l,pb,y)
    selected['joint']=choose_joint(candidates['g'],candidates['l'],pb,y)
    base=choose_fusion_pair(candidates['g'][0],candidates['l'][0],pb,y)
    alternatives=[{'family':k,**v} for k,v in selected.items()]
    alternatives += [{'family':'v1','active_repairs':v1_repairs,**_score(v1,y)},
                     {'family':'base_fusion',**base},
                     {'family':'router','active_repairs':0,**_score(pb,y)}]
    winner=min(alternatives,key=lambda r:(-r['accuracy'],r['nll'],r['active_repairs'],r['family']))
    return {'families':selected,'base_fusion':base,'optimized':winner,'alternatives':alternatives,
            'selection_population':'historical_validation_calibration_half','seed_reduction':'mean_probability'}


def run_dataset(run_root,dataset):
    root=Path(run_root).resolve();config=read_json(root/'SEARCH_CONFIG.json')
    torch.set_num_threads(config['train']['threads'])
    verify_files(config['source_code']);verify_files(config['input_completions'])
    config_hash=sha256(root/'SEARCH_CONFIG.json');dest=root/dataset;dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():
        c=read_json(dest/'complete.json')
        if c['config_sha256']!=config_hash or c['dataset']!=dataset or not c['all_checks_pass'] or c['test_labels_read'] is not False:raise ValueError('completed run mismatch')
        verify_files({str(dest/k):v for k,v in c['artifacts'].items()})
        return dataset
    export=Path(config['export_root'])/dataset
    verify_training_link(Path(config['v1_root'])/dataset,export)
    f,s,p,h,meta=load_export(export)
    cfg=TrainConfig(**config['train']);members={}
    for seed in config['seeds']:
        for raw in config['recipes']:
            recipe=Recipe(**raw);part=dest/'bank'/f'seed{seed}';part.mkdir(parents=True,exist_ok=True)
            path=part/(recipe.name+'.pt');marker=part/(recipe.name+'.json')
            member_cfg=replace(cfg,seed=seed)
            if marker.exists():
                record=read_json(marker)
                if record['config_sha256']!=config_hash:raise ValueError('member config mismatch')
                verify_files({str(path):record['sha256']})
                member=torch.load(path,weights_only=True,map_location='cpu')
            else:
                member=train_member(f,s,p,h,recipe,member_cfg)
                save_tensor(path,member)
                write_json(marker,{'sha256':sha256(path),'config_sha256':config_hash})
            if member['recipe']!=raw or member['config']!=asdict(member_cfg) or member['test_labels_read'] is not False or not torch.equal(member['validation_ids'],s['selection_ids']):
                raise ValueError('bank member protocol mismatch')
            members[(seed,recipe.name)]=member
            print(json.dumps({'dataset':dataset,'seed':seed,'recipe':recipe.name,'seconds':member['train_seconds'],
                              'g':member['checkpoints']['g']['accuracy'],'l':member['checkpoints']['l']['accuracy']}),flush=True)
    candidates=candidate_bank(members,f,s,config['seeds'])
    pg,pl=f['logp_g'].exp(),f['logp_l'].exp()
    pb=scheme_b_probability(pg,pl,f['choose_g'],f['agreement'])
    if not torch.equal(pb.argmax(1),f['selected_pred']):raise ValueError('frozen B identity mismatch')
    old=torch.load(Path(config['v1_root'])/dataset/'predictions.pt',weights_only=True,map_location='cpu')
    if not torch.equal(old['node_ids'],f['node_ids']):raise ValueError('V1 node IDs differ')
    idx=s['selection_ids'];y=s['selection_targets']
    old_selection=read_json(Path(config['v1_root'])/dataset/'selection.json')
    selection=select_families(candidates,pb[idx],y,old['p_joint'][idx],v1_active_repairs(old_selection))
    catalog={c['id']:c for name in candidates for c in candidates[name]}
    write_json(dest/'candidate_catalog.json',{k:{'metadata':v['metadata'],**_score(v['probability'],y)} for k,v in catalog.items()})
    write_json(dest/'selection.json',selection)
    full_cache={}
    def full(candidate_id):
        if candidate_id in full_cache:return full_cache[candidate_id]
        m=catalog[candidate_id]['metadata'];branch=m['branch'];base=f['logp_'+branch]
        if m['base']:value=base.exp().unsqueeze(0).repeat(len(config['seeds']),1,1)
        else:
            values=[]
            for seed in config['seeds']:
                z=member_logits(members[(seed,m['recipe'])],branch,m['criterion'],f,h,f['node_ids'],replace(cfg,seed=seed))
                values.append(residual_probability(base,z,m['scale']))
            value=torch.stack(values)
        ref=catalog[candidate_id]['probability'];observed=value.mean(0)[idx]
        if not torch.allclose(observed,ref,atol=2e-6) or not torch.equal(observed.argmax(1),ref.argmax(1)):
            raise ValueError('full prediction does not reproduce selected validation candidate')
        full_cache[candidate_id]=value
        return value
    families={};repaired={}
    for family,pick in selection['families'].items():
        g,l=full(pick['g_id']),full(pick['l_id'])
        per_seed=blend(g,l,pb.unsqueeze(0),pick['alpha'],pick['beta'])
        ensemble=per_seed.mean(0)
        score=_score(ensemble[idx],y)
        if abs(score['accuracy']-pick['accuracy'])>1e-10:raise ValueError('fusion validation accuracy failed to reproduce')
        families[family]={'ensemble':ensemble,'seeds':per_seed}
        repaired[family]={'g':g.mean(0),'l':l.mean(0)}
    basepick=selection['base_fusion'];basefusion=blend(pg,pl,pb,basepick['alpha'],basepick['beta'])
    baselines={'gnn':pg,'llm':pl,'router':pb,'v1':old['p_joint'],'base_fusion':basefusion}
    winner=selection['optimized']['family']
    optimized=families[winner] if winner in families else {'ensemble':baselines[winner],
                    'seeds':baselines[winner].unsqueeze(0).repeat(len(config['seeds']),1,1)}
    families['optimized']=optimized
    save_tensor(dest/'predictions.pt',{'node_ids':f['node_ids'],'baselines':baselines,'families':families,'repaired':repaired})
    write_json(dest/'dataset_metadata.json',{'dataset':dataset,'export_dir':str(export),'export_complete_sha256':sha256(export/'complete.json'),
               'split_counts':{k:meta[k] for k in ['gold_count','model_selection_count','threshold_count','test_count']},
               'pool_counts':meta['pool_counts'],'candidate_count':meta['pseudo_candidate_count'],'protocol':meta['protocol'],
               'member_count':len(members),'train_seconds':sum(x['train_seconds'] for x in members.values())})
    verify_files(config['source_code']);verify_files(meta['source_hashes'])
    artifacts={str(path.relative_to(dest)):sha256(path) for path in sorted(dest.rglob('*')) if path.is_file() and path.name!='complete.json'}
    write_json(dest/'complete.json',{'dataset':dataset,'config_sha256':config_hash,'all_checks_pass':True,
                                   'test_labels_read':False,'artifacts':artifacts})
    print(json.dumps({'dataset':dataset,'status':'complete','validation_winner':selection['optimized']}),flush=True)
    return dataset
