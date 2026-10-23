"""Bounded, resumable Joint exploration with a ten-cell test barrier."""
from dataclasses import asdict
import itertools
import os
from pathlib import Path
import time
import torch
from eargtc.joint_cross_repair.artifacts import DATASETS,read_json,write_json,sha256,verify_files,save_tensor
from eargtc.joint_cross_repair.training import scheme_b_probability,_score
from eargtc.joint_v8.adapter import load_cell,DEFAULT_EXPORT,ROUTERS
from .selection import shortlist,choose_final,rank

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_RUN=ROOT/'runs/joint_explore_20260908_v1'
LEGACY=ROOT/'runs/joint_v8_20260908_v1'
PRESERVED=ROOT/'artifacts/preserved_versions/joint_v8_fixed_v2_20260908'
SPEC=ROOT/'docs/superpowers/specs/2026-09-08-joint-explore-design.md'
REQUIRED={'selection.json','shortlist.json','validation.pt','predictions.pt','metadata.json','retrieval.pt'}


def close(a,b,message):
    if a.shape!=b.shape or not torch.allclose(a,b,atol=3e-5,rtol=0) or not torch.equal(a.argmax(1),b.argmax(1)):
        raise ValueError(message)


def initialize(root,smoke=False):
    from eargtc.joint_v8.experiment import preflight
    from .training import ARCHITECTURES,PRESETS,make_settings
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    preserved=read_json(PRESERVED/'COMPLETE.json')
    if preserved.get('all_checks_pass') is not True or preserved['snapshot_sha256']!=sha256(PRESERVED/'SNAPSHOT.json'):
        raise ValueError('verified previous-version archive required')
    old=preflight(LEGACY)
    sources=dict(old['source_code'])
    for folder in ['eargtc/joint_explore','scripts/joint_explore']:
        for path in sorted((ROOT/folder).iterdir()):
            if path.suffix in {'.py','.sbatch'}:sources[str(path)]=sha256(path)
    sources.update({str(p):sha256(p) for p in [SPEC,ROOT/'AGENTS.md']})
    inputs={**old['input_completions'],**read_json(LEGACY/'FROZEN.json')['files']}
    inputs.update({str(p):sha256(p) for p in [LEGACY/'FROZEN.json',PRESERVED/'COMPLETE.json',PRESERVED/'SNAPSHOT.json']})
    for rid,ds in itertools.product(ROUTERS,DATASETS):
        cell=LEGACY/rid/ds
        inputs.update({str(cell/name):expected for name,expected in read_json(cell/'complete.json')['artifacts'].items()})
    snapshot=read_json(PRESERVED/'SNAPSHOT.json')['files']
    for name in ['performance.csv','complete.json']:
        path=LEGACY/'evaluation'/name
        inputs[str(path)]=snapshot[str(path.relative_to(ROOT))]
    verify_files(inputs)
    settings={f'{a}/{p}':asdict(make_settings(a,p,device='cpu',threads=4,**({'max_steps':4,'validate_every':2,'patience':2} if smoke else {})))
              for a,p in itertools.product(ARCHITECTURES,PRESETS)}
    config={'routers':list(ROUTERS),'datasets':list(DATASETS),'smoke':smoke,'settings':settings,'seeds':[42,43,44],
        'legacy_root':str(LEGACY),'export_root':str(DEFAULT_EXPORT),'label_root':old['label_root'],
        'source_code':sources,'input_bindings':inputs,'archive_complete_sha256':sha256(PRESERVED/'COMPLETE.json'),
        'screening':'24 seed42; top2 per architecture; add43/44; final checkpoint criterion from3seed mean',
        'expected_neural_members_per_cell':40,'test_labels_read':False,'torch_version':str(torch.__version__)}
    path=root/'CONFIG.json'
    if path.exists():
        if read_json(path)!=config:raise ValueError('immutable configuration changed')
    else:write_json(path,config)
    return config


def verify_config(root):
    config=read_json(Path(root)/'CONFIG.json');verify_files(config['source_code']);verify_files(config['input_bindings'])
    if config['torch_version']!=str(torch.__version__):raise ValueError('torch version changed')
    return config


def check_complete(root,rid,ds):
    root=Path(root);cell=root/rid/ds;c=read_json(cell/'complete.json')
    if c.get('router')!=rid or c.get('dataset')!=ds or c.get('config_sha256')!=sha256(root/'CONFIG.json') or c.get('test_labels_read') is not False:
        raise ValueError('bad cell provenance')
    if not REQUIRED.issubset(c.get('artifacts',{})):raise ValueError('incomplete artifacts')
    verify_files({str(cell/k):v for k,v in c['artifacts'].items()})


def freeze(root):
    root=Path(root);cells=list(itertools.product(ROUTERS,DATASETS))
    if not all((root/r/d/'complete.json').exists() for r,d in cells):return False
    verify_config(root)
    for r,d in cells:check_complete(root,r,d)
    paths=[root/'CONFIG.json',*[root/r/d/'complete.json' for r,d in cells]]
    value={'cell_count':10,'test_labels_read':False,'files':{str(p.resolve()):sha256(p) for p in paths}}
    lock=root/'FROZEN.json'
    if lock.exists():
        if read_json(lock)!=value:raise ValueError('frozen experiment changed')
    else:write_json(lock,value)
    return True


def verify_frozen(root):
    root=Path(root).resolve();lock=read_json(root/'FROZEN.json');config=verify_config(root)
    expected={str(root/'CONFIG.json')}|{str(root/r/d/'complete.json') for r,d in itertools.product(ROUTERS,DATASETS)}
    if lock.get('cell_count')!=10 or lock.get('test_labels_read') is not False or set(lock['files'])!=expected:raise ValueError('all10 freeze required')
    verify_files(lock['files'])
    for r,d in itertools.product(ROUTERS,DATASETS):check_complete(root,r,d)
    return config


def legacy_inputs(config,rid,ds,f,s):
    path=Path(config['legacy_root'])/rid/ds
    pack=torch.load(path/'predictions.pt',map_location='cpu',weights_only=True)
    if not torch.equal(pack['node_ids'],f['node_ids']):raise ValueError('legacy full node IDs differ')
    pg,pl=f['logp_g'].exp(),f['logp_l'].exp();pb=scheme_b_probability(pg,pl,f['choose_g'],f['agreement'])
    for name,value in [('gnn',pg),('llm',pl),('router',pb)]:close(pack['baselines'][name],value,'old expert/Router changed')
    old=pack['families']['optimized'];pick=read_json(path/'selection.json')['optimized']
    score=_score(old['ensemble'][s['selection_ids']],s['selection_targets'])
    if abs(score['accuracy']-pick['accuracy'])>1e-10 or abs(score['nll']-pick['nll'])>3e-5:raise ValueError('legacy validation differs')
    return old,{'router':pb,'gnn':pg,'llm':pl,'expert_mean':(pg+pl)/2}


def run_cell(root,rid,ds):
    from .training import TrainSettings,train_candidate,predict_candidate
    from .retrieval import fit_retrieval,predict_retrieval
    root=Path(root).resolve();config=verify_config(root);dest=root/rid/ds;dest.mkdir(parents=True,exist_ok=True)
    if rid not in ROUTERS or ds not in DATASETS:raise ValueError('unknown cell')
    if (dest/'complete.json').exists():check_complete(root,rid,ds);return
    start=time.monotonic();digest=sha256(root/'CONFIG.json')
    device=next(iter(config['settings'].values()))['device']
    torch.set_num_threads(next(iter(config['settings'].values()))['threads'])
    f,s,p,h,meta=load_cell(config['export_root'],rid,ds)
    legacy,baselines=legacy_inputs(config,rid,ds,f,s)
    bank={}
    def member(key,seed):
        if (key,seed) in bank:return bank[key,seed]
        raw={**config['settings'][key],'seed':seed};settings=TrainSettings(**raw)
        folder=dest/'bank'/key;folder.mkdir(parents=True,exist_ok=True);path=folder/f'seed{seed}.pt';marker=path.with_suffix('.json')
        if marker.exists():
            record=read_json(marker)
            if record['config_sha256']!=digest:raise ValueError('member config differs')
            verify_files({str(path):record['sha256']});value=torch.load(path,map_location='cpu',weights_only=True)
        else:
            value=train_candidate(f,s,p,h,settings)
            if value['settings']!=raw:raise ValueError('trainer changed settings')
            save_tensor(path,value);write_json(marker,{'sha256':sha256(path),'config_sha256':digest})
        if value['settings']!=raw or value.get('test_labels_read') is not False or not torch.equal(value['validation_ids'],s['selection_ids']):raise ValueError('invalid member')
        bank[key,seed]=value
        print({'router':rid,'dataset':ds,'candidate':key,'seed':seed,'trained':True},flush=True)
        return value
    rows=[]
    for key in config['settings']:
        value=member(key,42);choices=[]
        for criterion,prob in value['validation_probability'].items():choices.append({'id':key,'architecture':key.split('/')[0],'criterion':criterion,**_score(prob,s['selection_targets'])})
        rows.append(min(choices,key=rank))
    retained=shortlist(rows)
    shortlist_value={'rows':rows,'retained':retained,'test_labels_read':False}
    shortlist_path=dest/'shortlist.json'
    if shortlist_path.exists() and read_json(shortlist_path)!=shortlist_value:raise ValueError('screening changed on resume')
    write_json(shortlist_path,shortlist_value)
    for key,seed in itertools.product(retained,[43,44]):member(key,seed)
    neural={};nn_val={}
    for key in retained:
        for criterion in ['accuracy','nll']:
            prob=torch.stack([bank[key,seed]['validation_probability'][criterion] for seed in config['seeds']]).mean(0)
            row={'id':key+'/'+criterion,'key':key,'criterion':criterion,**_score(prob,s['selection_targets'])};architecture=key.split('/')[0]
            if architecture not in neural or rank(row)<rank(neural[architecture]):neural[architecture]=row;nn_val[architecture]=prob
    retrieval_path=dest/'retrieval.pt'
    if (dest/'retrieval.json').exists():
        record=read_json(dest/'retrieval.json')
        if record['config_sha256']!=digest:raise ValueError('retrieval config differs')
        verify_files({str(retrieval_path):record['sha256']});retrieval=torch.load(retrieval_path,weights_only=True)
    else:
        best,history=fit_retrieval(f,s,p,h,device=device);retrieval={'families':best,'history':history,'test_labels_read':False}
        save_tensor(retrieval_path,retrieval);write_json(dest/'retrieval.json',{'config_sha256':digest,'sha256':sha256(retrieval_path)})
    ids,y=s['selection_ids'],s['selection_targets']
    family_val={**{k:v[ids] for k,v in baselines.items()},'legacy':legacy['ensemble'][ids],**nn_val,
                **{k:v['validation_probability'] for k,v in retrieval['families'].items()}}
    selected,catalog=choose_final(family_val,y)
    selection={'optimized':selected,'neural':neural,'retrieval':{k:{z:v[z] for z in ['family','config','accuracy','nll']} for k,v in retrieval['families'].items()},'validation_population':'original_model_selection_half'}
    write_json(dest/'selection.json',selection)
    save_tensor(dest/'validation.pt',{'node_ids':ids,'families':family_val,'catalog':catalog})
    full={'legacy':legacy}
    for name,value in baselines.items():full[name]={'ensemble':value,'seeds':value.unsqueeze(0).repeat(3,1,1)}
    for architecture,pick in neural.items():
        values=torch.stack([predict_candidate(bank[pick['key'],seed],f,h,f['node_ids'],pick['criterion'],device=device,batch_size=config['settings'][pick['key']]['inference_batch']) for seed in config['seeds']])
        full[architecture]={'ensemble':values.mean(0),'seeds':values}
    for name,value in retrieval['families'].items():
        prob=predict_retrieval(value,f,h,f['node_ids'],device=device);full[name]={'ensemble':prob,'seeds':prob.unsqueeze(0).repeat(3,1,1)}
    for name,value in full.items():close(value['ensemble'][ids],family_val[name],name+' validation replay differs')
    components=selected['components'];seeds=sum(w*full[k]['seeds'] for k,w in components.items())
    full['optimized']={'ensemble':seeds.mean(0),'seeds':seeds}
    close(full['optimized']['ensemble'][ids],catalog[selected['id']],'final selection replay differs')
    save_tensor(dest/'predictions.pt',{'node_ids':f['node_ids'],'families':full})
    if len(bank)!=40:raise ValueError('expected24screen+16confirmation members')
    write_json(dest/'metadata.json',{**meta,'neural_members':len(bank),'screened':24,'confirmed_configurations':8,
        'elapsed_seconds':time.monotonic()-start,'test_labels_read':False,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
        'selected':selected['id'],'device':device,'gpu':torch.cuda.get_device_name(0) if device.startswith('cuda') else None})
    verify_files(config['source_code']);verify_files(config['input_bindings'])
    artifacts={str(path.relative_to(dest)):sha256(path) for path in sorted(dest.rglob('*')) if path.is_file() and path.name!='complete.json' and '.tmp.' not in path.name}
    write_json(dest/'complete.json',{'router':rid,'dataset':ds,'config_sha256':digest,'test_labels_read':False,'artifacts':artifacts})
    print({'router':rid,'dataset':ds,'complete':True,'selected':selected['id']},flush=True)
