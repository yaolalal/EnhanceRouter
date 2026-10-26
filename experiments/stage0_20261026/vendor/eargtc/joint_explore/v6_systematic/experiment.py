"""Immutable staged controller; no evaluation module or raw truth accessor import.

The CLI bootstraps the code seal before importing this module. Functions also
verify it at entry, after legacy data loading, before publishing, and on reuse.
"""
from __future__ import annotations
import gc
import itertools
import os
from pathlib import Path
import time

from .manifest import (REPO,DEFAULT_ROOT,OLD_ROOT,slurm_only,sha256,digest,read_json,
    immutable_json,merge_hashes,verify_files,root_path,training_open,loaded_sources,
    bootstrap,write_bundle,verify_bundle,bundle_records,tensor_load)
from .plans import (DATASETS,OUTERS,INNERS,PHASES,COARSE,GRIDS,COORDINATES,
    anchor,config_digest,trial,stage_a,stage_b,stage_c,with_budget,coarse_trials,
    adjacent_trials,neighborhood,make_plan,expected_grid,validate_request,rank_rows,cell_key,cell_trials,independent_trials)
from .scoring import (score,fixed_score,branch_probability,fused_probability,calibrate,
    pb_calibrate,scheme_b,mean_probability)


def _paths(root,stage):
    return Path(root)/'stages'/stage


def _binding(root):
    return {str(Path(root)/name):sha256(Path(root)/name) for name in ('CONFIG.json','CODE_SEAL.json','INPUT_SEAL.json')}


def _guard(root,fit=False):
    slurm_only(); root=root_path(root)
    if fit: training_open(root)
    bootstrap(root)
    cfg=read_json(root/'CONFIG.json')
    if cfg.get('datasets')!=list(DATASETS) or cfg.get('outers')!=list(OUTERS) or cfg.get('inners')!=list(INNERS) or cfg.get('phases')!=list(PHASES):
        raise ValueError('formal registration grid differs')
    registered=verify_bundle(root/'registration')
    verify_files(registered['refs'])
    if registered['refs']!=_binding(root):
        raise ValueError('registration identity differs')
    return root,cfg


def _declared_file(candidate):
    candidate=Path(candidate)
    if candidate.is_dir():
        marker=candidate/'complete.json'
        if not marker.is_file(): raise ValueError('unknown historical directory hash convention: '+str(candidate))
        return marker
    return candidate


def _json_file_hashes(value,relative_base=REPO):
    """Extract explicit existing file/digest records, never score historical data."""
    out={}
    if isinstance(value,dict):
        for k,v in value.items():
            if isinstance(k,str) and isinstance(v,str) and len(v)==64:
                candidate=_declared_file(Path(k) if Path(k).is_absolute() else Path(relative_base)/k)
                if candidate.is_file() or '/' in k or Path(k).suffix in ('.json','.pt','.py','.csv','.md','.tar','.gz'):
                    out=merge_hashes(out,{str(candidate.resolve()):v})
            if k in ('sha256','digest') and isinstance(v,str) and len(v)==64:
                p=value.get('path')
                if isinstance(p,str):
                    candidate=_declared_file(Path(p) if Path(p).is_absolute() else Path(relative_base)/p)
                    out=merge_hashes(out,{str(candidate.resolve()):v})
            if isinstance(v,(dict,list)): out=merge_hashes(out,_json_file_hashes(v,relative_base=relative_base))
    elif isinstance(value,list):
        for v in value: out=merge_hashes(out,_json_file_hashes(v,relative_base=relative_base))
    return out


def _history_records():
    old_config=OLD_ROOT/'joint/CONFIG.json'
    if not old_config.exists():
        old_config=OLD_ROOT/'joint/config.json'
    frozen=OLD_ROOT/'joint/FROZEN.json'
    old=read_json(frozen)
    if _json_file_hashes(old).get(str(old_config))!=sha256(old_config):
        raise ValueError('historical v6 CONFIG is not bound by original FROZEN')
    initial=[old_config,frozen,REPO/'artifacts/preserved_versions/joint_v8_fixed_v2_20260908/SNAPSHOT.json']
    initial.extend(p for p in initial[-1].parent.iterdir() if p.is_file() and p.name!='SNAPSHOT.json')
    for pattern in ('joint_v6_repair_20260911_v1','joint_v6_unified_20260910_v2','joint_v6_fusion_full*','joint_v6_fusionfull*'):
        for run in (REPO/'runs').glob(pattern):
            initial.extend(run.glob('FROZEN.json'))
    # Descend completion manifests only; result tables are authenticated bytes.
    records={}; pending=list(initial); visited=set()
    while pending:
        p=Path(pending.pop()).resolve()
        if p in visited: continue
        visited.add(p); records=merge_hashes(records,{str(p):sha256(p)})
        if p.suffix=='.json':
            base=REPO if p.name in ('SNAPSHOT.json','CONFIG.json','config.json') else p.parent
            nested=_json_file_hashes(read_json(p),relative_base=base); records=merge_hashes(records,nested)
            pending.extend(Path(q) for q in nested if Path(q).name.lower() in ('complete.json','frozen.json','snapshot.json'))
    # Preserve every existing file in prior three new experiment roots, including
    # published reports/checkpoints that are descendants of their freeze markers.
    for p in initial:
        if p.name=='FROZEN.json' and p.parent!=OLD_ROOT/'joint':
            for file in p.parent.rglob('*'):
                if file.is_file(): records=merge_hashes(records,{str(file.resolve()):sha256(file)})
    verify_files(records)
    return records


def register(root=DEFAULT_ROOT):
    slurm_only(); root=root_path(root); training_open(root)
    if root.exists() and any(p.name!='preflight' for p in root.iterdir()):
        raise FileExistsError('registration permits only an existing preflight directory')
    audit_path=root/'preflight/INPUT_AUDIT.json'; audit=read_json(audit_path)
    if audit.get('all_checks_pass') is not True or audit.get('query_truth_read') is not False:
        raise ValueError('successful independent safe-input preflight required')
    independent={(r['dataset'],r['outer']):r for r in audit['records']}
    if set(independent)!=set(itertools.product(DATASETS,OUTERS)) or len(audit['records'])!=len(independent):
        raise ValueError('preflight must cover exactly all 25 cells')
    from .data import load_safe_data,_id_hash
    from .training import TrainConfig
    import torch
    sources={}; inputs=_history_records(); cells={}
    for ds,outer in itertools.product(DATASETS,OUTERS):
        train,fit,select,inference,meta=load_safe_data(outer,ds)
        accepted=torch.cat([train[k] for k in ('disagreement_choose_g_ids','disagreement_choose_l_ids','agreement_ids')]).sort().values
        pre=independent[ds,outer]
        if _id_hash(accepted)!=pre['pseudo_ids_sha256'] or _id_hash(train['selected_pred'][accepted])!=pre['pseudo_targets_sha256']:
            raise ValueError('independent preflight pseudo ID/target identity differs')
        inputs=merge_hashes(inputs,meta['input_hashes']); sources=merge_hashes(sources,meta['source_hashes'])
        split_path=OLD_ROOT/'splits'/f'{ds}_5shot_seed{outer}.json'
        split=read_json(split_path)
        inputs=merge_hashes(inputs,{str(split_path):sha256(split_path)})
        if not torch.equal(inference['query_ids'],torch.tensor(split['evaluation_full_ids'],dtype=torch.long)):
            raise ValueError('full-query ID order differs from sealed split')
        cache=root/'inputs'/ds/str(outer)
        write_bundle(cache,{'validation.pt':{'fit':fit,'select':select,'inference':inference},'metadata.json':meta,'split.json':split},{'dataset':ds,'outer':outer,'test_labels_read':False})
        cells[f'{ds}/{outer}']=bundle_records(cache)
        del train,fit,select,inference; gc.collect()
    source_paths=list(Path(__file__).parent.glob('*.py'))+list((REPO/'scripts/joint_explore/v6_systematic').glob('*'))+[REPO/'docs/superpowers/specs/2026-09-11-joint-v6-systematic.md']
    sources=merge_hashes(sources,loaded_sources(),{str(p.resolve()):sha256(p) for p in source_paths if p.is_file()})
    # Validate all static configurations while numerical imports are available.
    for t in stage_a()+stage_b()+stage_c('raw_concat')+coarse_trials(anchor()): TrainConfig.from_mapping(t['config'])
    inputs=merge_hashes(inputs,{str(audit_path):sha256(audit_path)},audit['router_parity_hashes'])
    verify_files(inputs); verify_files(sources)
    immutable_json(root/'CODE_SEAL.json',{'files':sources,'note':'audited source closure, not a general process sandbox'})
    immutable_json(root/'INPUT_SEAL.json',{'files':inputs,'cells':cells})
    config={'datasets':list(DATASETS),'outers':list(OUTERS),'selection_unit':'dataset_outer_independent','inners':list(INNERS),'phases':list(PHASES),'stage_a':stage_a(),'stage_b':stage_b(),'stage_c':stage_c('LOCKED_ARCHITECTURE'),'coarse12':[list(x) for x in COARSE],'coordinate_order':list(COORDINATES),'coordinate_grids':GRIDS,'budgets':[120,240,480],'calibration':{'hpo_diagnostic_scales':[.25,.5,1.],'final_scales':[0.,.25,.5,1.],'alpha':[0.,.25,.5,.75,1.],'beta':[0.,.5,1.],'pb_convention':'(1-beta)*calibrated + beta*pB'},'ranking':'own outer select accuracy, NLL, registered ID order; fixed alpha=.5 scales=1; no cross-outer selection','test_labels_read':False,'torch_version':str(torch.__version__),'old_root':str(OLD_ROOT)}
    immutable_json(root/'CONFIG.json',config)
    write_bundle(root/'registration',{'receipt.json':{'cell_count':25,'test_labels_read':False}}, {'phase':'registration'},refs=_binding(root))
    bootstrap(root)
    _create_plan(root,make_plan('A',independent_trials(lambda ds,outer:stage_a()),_binding(root)))
    return status(root)


def _create_plan(root,plan):
    base=_paths(root,plan['stage'])
    if (base/'plan.json').exists():
        if read_json(base/'plan.json')!=plan: raise ValueError('existing phase plan differs')
        return
    immutable_json(base/'plan.json',plan)


def _verify_plan(root,stage):
    if stage not in PHASES: raise ValueError('unknown stage')
    plan=read_json(_paths(root,stage)/'plan.json')
    if plan['stage']!=stage: raise ValueError('phase identity differs')
    verify_files(plan['parents'])
    idx=PHASES.index(stage)
    expected_parents=_binding(root)
    if idx:
        previous=_paths(root,PHASES[idx-1])/'lock'
        for ancestor in PHASES[:idx]: _verify_lock(root,ancestor)
        expected_parents[str(previous/'complete.json')]=sha256(previous/'complete.json')
    if plan['parents']!=expected_parents:
        raise ValueError('phase ancestors differ')
    # Regenerate all dynamic grids from the immutable previous lock.
    expected=_next_plan(root,PHASES[idx-1]) if idx else make_plan('A',independent_trials(lambda ds,outer:stage_a()),_binding(root))
    if plan!=expected: raise ValueError('phase expected grid differs from registered rules')
    return plan


def _verify_lock(root,stage):
    path=_paths(root,stage)
    lock=verify_bundle(path/'lock',{'stage':stage,'plan_sha256':sha256(path/'plan.json')})
    verify_files(read_json(path/'plan.json')['parents'])
    return read_json(path/'lock/result.json')


def _cached(root,ds,outer):
    path=Path(root)/'inputs'/ds/str(outer)
    seal=read_json(Path(root)/'INPUT_SEAL.json')
    if seal['cells'].get(f'{ds}/{outer}')!=bundle_records(path):
        raise ValueError('input cache differs from registration')
    return tensor_load(path/'validation.pt'),read_json(path/'metadata.json')


def _member_path(root,ds,outer,cfg,seed):
    return Path(root)/'members'/ds/str(outer)/config_digest(cfg)/str(seed)


def _member_identity(root,ds,outer,cfg,seed):
    return {'dataset':ds,'outer':outer,'config':cfg,'config_digest':config_digest(cfg),'inner_seed':seed,'binding':_binding(root),'input_marker_sha256':sha256(Path(root)/'inputs'/ds/str(outer)/'complete.json'),'test_labels_read':False}


def _member(root,ds,outer,cfg,seed):
    path=_member_path(root,ds,outer,cfg,seed)
    verify_bundle(path,_member_identity(root,ds,outer,cfg,seed))
    value=tensor_load(path/'member.pt')
    if value['training']['config']!=cfg or value['training']['seed']!=seed or value['training']['test_labels_read'] is not False:
        raise ValueError('member numerical identity differs')
    cache,_=_cached(root,ds,outer)
    import torch
    for split in ('fit','select'):
        if not torch.equal(value[split]['ids'],cache[split]['ids']): raise ValueError('member validation ID alignment differs')
    return value


def _ensure_member(root,ds,outer,cfg,seed,data,device):
    path=_member_path(root,ds,outer,cfg,seed)
    if path.exists():
        _member(root,ds,outer,cfg,seed); return path
    training_open(root); bootstrap(root)
    from .training import TrainConfig,train_member,predict
    train,fit,select,_,_=data
    t=time.monotonic()
    result=train_member(train,fit,TrainConfig.from_mapping(cfg),seed,device)
    states={b:result['checkpoints'][b]['state'] for b in ('g','l')}
    value={'training':result}
    for name,part in (('fit',fit),('select',select)):
        value[name]={'ids':part['ids'],'branches':predict(states,TrainConfig.from_mapping(cfg),train,part['ids'],device)}
    bootstrap(root); training_open(root)
    write_bundle(path,{'member.pt':value,'runtime.json':{'seconds':time.monotonic()-t,'device':str(device),'job_id':os.environ['SLURM_JOB_ID']}},_member_identity(root,ds,outer,cfg,seed))
    return path


def _cell_path(root,stage,ds,outer):
    return _paths(root,stage)/'cells'/ds/str(outer)


def _trial_pair(t):
    return t['pair'] if 'pair' in t else {'g':t['config'],'l':t['config']}


def _verify_cell(root,stage,ds,outer,plan):
    path=_cell_path(root,stage,ds,outer)
    pairs={t['id']:_trial_pair(t) for t in plan['trials'][ds] if outer in t.get('outers',plan['outers'])}
    expected={'stage':stage,'dataset':ds,'outer':outer,'plan_sha256':sha256(_paths(root,stage)/'plan.json'),'binding':_binding(root)}
    marker=verify_bundle(path,expected)
    aliases=read_json(path/'aliases.json')
    wanted={f'{name}/{b}/{seed}':str(_member_path(root,ds,outer,cfg,seed)) for name,pair in pairs.items() for b,cfg in pair.items() for seed in plan['inners']}
    if aliases!=wanted: raise ValueError('cell expected member grid differs')
    refs={}
    for name,pair in pairs.items():
        for cfg in {config_digest(c):c for c in pair.values()}.values():
            for seed in plan['inners']:
                p=_member_path(root,ds,outer,cfg,seed)
                verify_bundle(p,_member_identity(root,ds,outer,cfg,seed))
                refs=merge_hashes(refs,bundle_records(p))
    if stage=='FINAL':
        aliases=read_json(path/'pair_aliases.json')
        canonical={}; wanted_aliases={}
        for name,pair in pairs.items():
            h=digest(pair)
            canonical.setdefault(h,name); wanted_aliases[name]=canonical[h]
        if aliases!=wanted_aliases: raise ValueError('final dedup aliases differ')
        choices=read_json(path/'calibration.json')
        if set(choices)!=set(aliases.values()): raise ValueError('final calibration coverage differs')
        _,meta=_cached(root,ds,outer)
        for name,choice in choices.items():
            for b,cfg in pairs[name].items():
                scale=choice['noPB']['scale_'+b]
                if choice['noPB'].get('residual_'+b)!=cfg['residual']: raise ValueError('calibration residual mode differs from branch configuration')
                if scale not in ((0.,.25,.5,1.) if cfg['residual'] else (1.,)):
                    raise ValueError('final residual scale is outside registration')
                for seed in INNERS:
                    q=root/'query_members'/ds/str(outer)/digest({'config':cfg,'branch':b,'seed':seed,'scale':scale})
                    identity={'dataset':ds,'outer':outer,'branch':b,'seed':seed,'config_digest':config_digest(cfg),'scale':scale,'binding':_binding(root),'query_ids_sha256':meta['identity_hashes']['query_ids'],'tuned_sha256':sha256(root/'TUNED_LOCK/complete.json')}
                    verify_bundle(q,identity)
                    refs=merge_hashes(refs,bundle_records(q))
        if marker['refs']!=refs: raise ValueError('final exact member/query reference closure differs')
    elif marker['refs']!=refs: raise ValueError('cell reference closure differs')
    return marker


def run_stage_cell(root,stage,dataset,outer,device='cuda'):
    root,_=_guard(root,fit=True); plan=_verify_plan(root,stage); validate_request(plan,dataset,outer)
    if stage=='FINAL': return final_cell(root,dataset,outer,device)
    dest=_cell_path(root,stage,dataset,outer)
    if dest.exists(): _verify_cell(root,stage,dataset,outer,plan); return {'status':'verified_reuse','stage':stage,'dataset':dataset,'outer':outer}
    from .data import load_safe_data
    data=load_safe_data(outer,dataset); bootstrap(root)
    _,meta=_cached(root,dataset,outer)
    if any(data[4][k]!=meta[k] for k in ('dataset','outer','input_hashes','identity_hashes','counts')):
        raise ValueError('safe inputs differ from registration')
    aliases={}; refs={}
    for t in cell_trials(plan,dataset,outer):
        for seed in plan['inners']:
            p=_ensure_member(root,dataset,outer,t['config'],seed,data,device)
            refs=merge_hashes(refs,bundle_records(p))
            for b in ('g','l'): aliases[f'{t["id"]}/{b}/{seed}']=str(p)
    bootstrap(root); training_open(root); verify_files(data[4]['input_hashes'])
    write_bundle(dest,{'aliases.json':aliases},{'stage':stage,'dataset':dataset,'outer':outer,'plan_sha256':sha256(_paths(root,stage)/'plan.json'),'binding':_binding(root)},refs)
    return {'status':'complete','stage':stage,'dataset':dataset,'outer':outer,'members':len(set(aliases.values()))}


def _trial_readout(root,ds,outer,t,inners,final_calibration=False):
    cache,_=_cached(root,ds,outer); pair=_trial_pair(t)
    members={b:[_member(root,ds,outer,pair[b],seed) for seed in inners] for b in ('g','l')}
    out={}
    fit={b:[m['fit']['branches'][b] for m in members[b]] for b in ('g','l')}
    choice=calibrate(fit['g'],fit['l'],cache['fit']['targets'],pair['g']['residual'],pair['l']['residual'],include_zero=final_calibration)
    for name in ('fit','select'):
        branches={b:[m[name]['branches'][b] for m in members[b]] for b in ('g','l')}
        out[name]=fixed_score([x['probabilities'] for x in branches['g']],[x['probabilities'] for x in branches['l']],cache[name]['targets'])
        out[name+'_calibrated']=score(fused_probability(branches['g'],branches['l'],choice),cache[name]['targets'])
    out['calibration']=choice
    out['parameter_counts']={b:members[b][0]['training']['parameter_counts'] for b in ('g','l')}
    out['checkpoint_updates']={b:[m['training']['checkpoints'][b]['update'] for m in members[b]] for b in ('g','l')}
    out['members']={}
    for b in ('g','l'):
        summaries=[]
        for m in members[b]:
            logs=m['training']['logs']; cfg=pair[b]
            summaries.append({'inner_seed':m['training']['seed'],'config':cfg,
                'fit_checkpoint':{k:v for k,v in m['training']['checkpoints'][b].items() if k!='state'},
                'select':score(m['select']['branches'][b]['probabilities'],cache['select']['targets']),
                'pool_exposure':{pool:{key:sum(r['branches'][b]['pools'][pool][key] for r in logs) for key in ('seen','unique')} for pool in ('gold','ce','kl')},
                'loss_mean':{key:sum(r['branches'][b]['losses'][key] for r in logs)/len(logs) for key in ('gold_ce','pseudo_ce','teacher_kl','total')},
                'loss_last':logs[-1]['branches'][b]['losses']})
        out['members'][b]=summaries
    return out


def _aggregate_trials(root,plan):
    rows=[]; aggregates={}
    for ds in plan['trials']:
        for outer in plan['outers']:
            ts=cell_trials(plan,ds,outer); aggregate=[]
            for t in ts:
                r=_trial_readout(root,ds,outer,t,plan['inners'])
                rows.append({'dataset':ds,'outer':outer,'id':t['id'],'eligible':t['eligible'],**r})
                aggregate.append({'id':t['id'],'config':t['config'],'eligible':t['eligible'],**r['select']})
            aggregates[cell_key(ds,outer)]=rank_rows(aggregate,[t['id'] for t in ts])
    return rows,aggregates


def _ledger_update(previous,aggregates,stage):
    ledger={}
    for ds,rs in aggregates.items():
        entries=list(previous.get(ds,[])); seen={config_digest(x['config']) for x in entries}
        for r in rs:
            h=config_digest(r['config'])
            if r['eligible'] and h not in seen:
                entries.append({**r,'id':f'{stage}/{r["id"]}'}); seen.add(h)
        ledger[ds]=entries
    return ledger


def _pair_selection(root,plan):
    choices={}; rows=[]
    for ds in plan['trials']:
        for outer in plan['outers']:
            ts=cell_trials(plan,ds,outer); candidates=[]
            for g,l in itertools.product(ts,repeat=2):
                candidate={'id':g['id']+'__'+l['id'],'pair':{'g':g['config'],'l':l['config']},'sources':{'g':g['id'],'l':l['id']},'shared':g['id']==l['id']}
                r=_trial_readout(root,ds,outer,candidate,INNERS)
                rows.append({'dataset':ds,'outer':outer,'id':candidate['id'],**r})
                candidates.append(dict(candidate,**r['select']))
            ranked=rank_rows(candidates,[r['id'] for r in candidates])
            choices[cell_key(ds,outer)]={'winner':ranked[0],'shared_best':next(r for r in ranked if r['shared']),'all_pairs':ranked}
    return choices,rows


def close_stage(root,stage):
    root,_=_guard(root,fit=True); plan=_verify_plan(root,stage)
    if stage=='FINAL': return _close_final(root,plan)
    dest=_paths(root,stage)/'lock'
    if dest.exists():
        result=_verify_lock(root,stage); _choice_lock(root,stage,result); _create_plan(root,_next_plan(root,stage)); return status(root)
    refs={str(_paths(root,stage)/'plan.json'):sha256(_paths(root,stage)/'plan.json')}
    for ds,outer in itertools.product(DATASETS,plan['outers']):
        _verify_cell(root,stage,ds,outer,plan)
        refs=merge_hashes(refs,bundle_records(_cell_path(root,stage,ds,outer)))
    rows,aggregates=_aggregate_trials(root,plan)
    choices={ds:next(r for r in rs if r['eligible']) for ds,rs in aggregates.items()}
    result={'stage':stage,'choices':choices,'rows':rows,'aggregates':aggregates,'test_labels_read':False}
    if stage.startswith('H'):
        previous={} if stage=='H00_coarse' else _verify_lock(root,PHASES[PHASES.index(stage)-1])['ledger']
        ledger=_ledger_update(previous,aggregates,stage)
        result['ledger']=ledger
        if stage=='H11_replicate':
            result['tuned'],result['pair_rows']=_pair_selection(root,plan)
    bootstrap(root); training_open(root); verify_files(read_json(root/'INPUT_SEAL.json')['files'])
    write_bundle(dest,{'result.json':result,'validation.csv':_csv_validation(rows)},{'stage':stage,'plan_sha256':sha256(_paths(root,stage)/'plan.json')},refs)
    _choice_lock(root,stage,result)
    _create_plan(root,_next_plan(root,stage))
    return status(root)


def _choice_lock(root,stage,result):
    if stage not in ('C','H11_replicate'): return
    name='FRAMEWORK_LOCK' if stage=='C' else 'TUNED_LOCK'
    path=root/name; choices=result.get('tuned',result['choices']); refs=bundle_records(_paths(root,stage)/'lock')
    if path.exists():
        marker=verify_bundle(path,{'stage':stage})
        if marker['refs']!=refs or read_json(path/'choices.json')!=choices: raise ValueError('choice lock differs')
    else:
        write_bundle(path,{'choices.json':choices},{'stage':stage},refs)


def _csv_validation(rows):
    import csv,io
    f=io.StringIO(); fields=['dataset','outer','id','eligible','fit_accuracy','fit_nll','select_accuracy','select_nll','calibrated_select_accuracy','calibrated_select_nll']
    writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
    for r in rows:
        writer.writerow({'dataset':r['dataset'],'outer':r['outer'],'id':r['id'],'eligible':r.get('eligible',True),'fit_accuracy':r['fit']['accuracy'],'fit_nll':r['fit']['nll'],'select_accuracy':r['select']['accuracy'],'select_nll':r['select']['nll'],'calibrated_select_accuracy':r['select_calibrated']['accuracy'],'calibrated_select_nll':r['select_calibrated']['nll']})
    return f.getvalue()


def _next_plan(root,previous):
    idx=PHASES.index(previous)+1
    if idx>=len(PHASES): raise ValueError('no phase follows FINAL')
    name=PHASES[idx]; result=_verify_lock(root,previous)
    parents=_binding(root); parents[str(_paths(root,previous)/'lock/complete.json')]=sha256(_paths(root,previous)/'lock/complete.json')
    def build(ds,outer):
        key=cell_key(ds,outer)
        if name=='B': return stage_b()
        if name=='C': return stage_c(result['choices'][key]['config']['architecture'])
        if name=='H00_coarse': return coarse_trials(result['choices'][key]['config'])
        if name=='H11_replicate':
            rs=result['ledger'][key]; best=rank_rows(rs,[r['id'] for r in rs])[:3]
            if len(best)!=3: raise ValueError('HPO requires three unique finalists per outer')
            return [trial(f'finalist{i}',r['config'],source=r['id']) for i,r in enumerate(best)]
        if name=='FINAL':
            verify_bundle(root/'TUNED_LOCK')
            pair=result['tuned'][key]['winner']['pair']; controls=neighborhood(pair)
            for cfg in pair.values():
                if not cfg['disagreement_all'] or not cfg['agreement_ce'] or cfg['pseudo_sampling']!='full_coverage' or cfg['gold_sampling']!='full_coverage': raise ValueError('final hard constraints violated')
            return [{'id':control,'pair':value,'eligible':False,
                'branch_residual_states':{b:{'selected':pair[b]['residual'],'control':value[b]['residual']} for b in ('g','l')},
                'architecture_control_note':('capacity' if any(c['architecture']=='projected_concat_wide' for c in pair.values()) else 'fusion') if control=='projected_concat_fusion' else 'architecture_block' if control=='raw_concat_block' else None,
                'coupled_warm_teacher_removal':control=='warm_off' and any(c['teacher']=='warm' for c in pair.values())} for control,value in controls.items()]
        incumbent=result['choices'][key]['config']
        if name=='H10_anchor':
            framework=_verify_lock(root,'C')['choices'][key]['config']
            return [trial('incumbent',incumbent),trial('same_budget_anchor',with_budget(framework,incumbent['updates']))]
        if name=='H09_budget': return [trial(f'budget{n}',with_budget(incumbent,n)) for n in (120,240,480)]
        coordinate=name.split('_',1)[1]
        return [trial('incumbent',incumbent)]+adjacent_trials(incumbent,coordinate)
    return make_plan(name,independent_trials(build),parents)


def status(root=DEFAULT_ROOT):
    root,_=_guard(root)
    if (root/'FROZEN.json').exists(): return {'status':'frozen','next':'evaluate' if not (root/'evaluation').exists() else 'receipt'}
    for stage in PHASES:
        p=_paths(root,stage)
        if (p/'plan.json').exists() and not (p/'lock/complete.json').exists():
            plan=_verify_plan(root,stage)
            missing=[{'dataset':ds,'outer':o} for ds,o in itertools.product(DATASETS,plan['outers']) if not (_cell_path(root,stage,ds,o)/'complete.json').exists()]
            return {'status':'stage_open','stage':stage,'next':'run-stage-cell' if missing else 'close-stage','missing_cells':missing,'expected_member_aliases':len(expected_grid(plan))}
    return {'status':'final_closed','next':'freeze'}


def final_cell(root,dataset,outer,device='cuda'):
    root,_=_guard(root,fit=True); plan=_verify_plan(root,'FINAL'); validate_request(plan,dataset,outer)
    tuned=verify_bundle(root/'TUNED_LOCK')
    if tuned['stage']!='H11_replicate': raise ValueError('final members require TUNED lock')
    dest=_cell_path(root,'FINAL',dataset,outer)
    if dest.exists():
        _verify_cell(root,'FINAL',dataset,outer,plan); return {'status':'verified_reuse','dataset':dataset,'outer':outer}
    from .data import load_safe_data
    from .training import TrainConfig,predict_branch
    import torch
    data=load_safe_data(outer,dataset); bootstrap(root)
    train,fit,select,inference,meta=data
    cached,registered_meta=_cached(root,dataset,outer)
    if any(meta[k]!=registered_meta[k] for k in ('dataset','outer','input_hashes','identity_hashes','counts')): raise ValueError('final safe input identity differs')
    aliases={}; refs={}; predictions={}; validation=[]; calibration_choices={}; pair_aliases={}; seen_pairs={}
    for t in plan['trials'][dataset]:
        if outer not in t['outers']: continue
        pair=t['pair']; pair_key=digest(pair)
        for branch,cfg in pair.items():
            for seed in INNERS:
                p=_ensure_member(root,dataset,outer,cfg,seed,data,device)
                refs=merge_hashes(refs,bundle_records(p)); aliases[f'{t["id"]}/{branch}/{seed}']=str(p)
        if pair_key in seen_pairs:
            canonical=seen_pairs[pair_key]; pair_aliases[t['id']]=canonical
            continue
        seen_pairs[pair_key]=t['id']; pair_aliases[t['id']]=t['id']
        members={b:[_member(root,dataset,outer,pair[b],seed) for seed in INNERS] for b in ('g','l')}
        fit_parts={b:[m['fit']['branches'][b] for m in members[b]] for b in ('g','l')}
        calibration=calibrate(fit_parts['g'],fit_parts['l'],fit['targets'],pair['g']['residual'],pair['l']['residual'],include_zero=True)
        fit_cal=fused_probability(fit_parts['g'],fit_parts['l'],calibration)
        pb_choice=pb_calibrate(fit_cal,scheme_b(train,fit['ids']),fit['targets'])
        calibration_choices[t['id']]={'noPB':calibration,'pB_diagnostic':pb_choice,'fit_ids_sha256':registered_meta['identity_hashes']['checkpoint_ids'],'selection_used':False,'zero_scale_branches':[b for b in ('g','l') if calibration['scale_'+b]==0.],'zero_scale_note':'raw expert fallback; disabled correction is not a training gain'}
        readout=_trial_readout(root,dataset,outer,t,INNERS,final_calibration=True)
        validation.append({'dataset':dataset,'outer':outer,'id':t['id'],**readout})
        # Query members are inferred only after choices lock. Save their branch
        # probabilities at both required scales; no targets enter this routine.
        query_parts={b:[] for b in ('g','l')}; fixed_parts={b:[] for b in ('g','l')}
        for b,cfg in pair.items():
            scale=calibration['scale_'+b]
            for seed,m in zip(INNERS,members[b]):
                inference_id=digest({'config':cfg,'branch':b,'seed':seed,'scale':scale})
                qp=root/'query_members'/dataset/str(outer)/inference_id
                identity={'dataset':dataset,'outer':outer,'branch':b,'seed':seed,'config_digest':config_digest(cfg),'scale':scale,'binding':_binding(root),'query_ids_sha256':registered_meta['identity_hashes']['query_ids'],'tuned_sha256':sha256(root/'TUNED_LOCK/complete.json')}
                if qp.exists():
                    verify_bundle(qp,identity); pred=tensor_load(qp/'probabilities.pt')
                else:
                    logits=predict_branch(m['training']['checkpoints'][b]['state'],TrainConfig.from_mapping(cfg),train,inference['query_ids'],b,device)
                    pred={'ids':inference['query_ids'],'fixed':logits['probabilities'],'calibrated':branch_probability([logits],scale,residual=cfg['residual'])}
                    bootstrap(root); training_open(root)
                    write_bundle(qp,{'probabilities.pt':pred},identity,bundle_records(_member_path(root,dataset,outer,cfg,seed)))
                if not torch.equal(pred['ids'],inference['query_ids']): raise ValueError('query member ID alignment differs')
                refs=merge_hashes(refs,bundle_records(qp))
                query_parts[b].append(pred['calibrated']); fixed_parts[b].append(pred['fixed'])
        fixed=.5*(mean_probability(fixed_parts['g'])+mean_probability(fixed_parts['l']))
        cal=calibration['alpha']*mean_probability(query_parts['g'])+(1-calibration['alpha'])*mean_probability(query_parts['l'])
        beta=pb_choice['beta']; pb=(1-beta)*cal+beta*scheme_b(train,inference['query_ids'])
        predictions[t['id']]={'fixed':fixed,'calibrated_noPB':cal,'pB_diagnostic':pb}
        del members,query_parts,fixed_parts; gc.collect()
    bootstrap(root); training_open(root); verify_files(meta['input_hashes'])
    # _verify_cell validates member refs plus the separately sealed query refs.
    write_bundle(dest,{'aliases.json':aliases,'pair_aliases.json':pair_aliases,'predictions.pt':{'query_ids':inference['query_ids'],'pB':scheme_b(train,inference['query_ids']),'candidates':predictions},'validation.json':validation,'baseline_validation.pt':{'fit':scheme_b(train,fit['ids']),'select':scheme_b(train,select['ids'])},'calibration.json':calibration_choices}, {'stage':'FINAL','dataset':dataset,'outer':outer,'plan_sha256':sha256(_paths(root,'FINAL')/'plan.json'),'binding':_binding(root)},refs)
    _verify_cell(root,'FINAL',dataset,outer,plan)
    return {'status':'complete','stage':'FINAL','dataset':dataset,'outer':outer,'unique_pairs':len(predictions)}


def _close_final(root,plan):
    dest=_paths(root,'FINAL')/'lock'
    if dest.exists(): _verify_lock(root,'FINAL'); return status(root)
    refs={str(_paths(root,'FINAL')/'plan.json'):sha256(_paths(root,'FINAL')/'plan.json')}; rows=[]
    for ds,outer in itertools.product(DATASETS,OUTERS):
        _verify_cell(root,'FINAL',ds,outer,plan)
        path=_cell_path(root,'FINAL',ds,outer)
        refs=merge_hashes(refs,bundle_records(path))
        for r in read_json(path/'validation.json'):
            rows.append(r)
    effects=[]
    for ds in DATASETS:
        for outer in OUTERS:
            p=_cell_path(root,'FINAL',ds,outer); aliases=read_json(p/'pair_aliases.json')
            byid={r['id']:r for r in rows if r['dataset']==ds and r['outer']==outer}
            selected=byid[aliases['selected']]
            for name,canonical in aliases.items():
                control=byid[canonical]
                effects.append({'dataset':ds,'outer':outer,'control':name,'canonical':canonical,'fixed_accuracy_selected_minus_control':selected['select']['accuracy']-control['select']['accuracy'],'calibrated_accuracy_selected_minus_control':selected['select_calibrated']['accuracy']-control['select_calibrated']['accuracy'],'note':'matched own-outer tuned model; conditional effect, no retuning'})
    bootstrap(root); training_open(root)
    write_bundle(dest,{'result.json':{'stage':'FINAL','rows':rows,'paired_outer_effects':effects,'test_labels_read':False},'validation.csv':_csv_validation(rows)}, {'stage':'FINAL','plan_sha256':sha256(_paths(root,'FINAL')/'plan.json')},refs)
    return status(root)


def freeze(root=DEFAULT_ROOT):
    root,cfg=_guard(root,fit=True)
    plan=_verify_plan(root,'FINAL'); _verify_lock(root,'FINAL')
    refs=_binding(root)
    for stage in PHASES:
        _verify_plan(root,stage); _verify_lock(root,stage)
        refs=merge_hashes(refs,bundle_records(_paths(root,stage)/'lock'))
    for name in ('registration','FRAMEWORK_LOCK','TUNED_LOCK'):
        refs=merge_hashes(refs,bundle_records(root/name))
    input_seal=read_json(root/'INPUT_SEAL.json'); verify_files(input_seal['files'])
    for records in input_seal['cells'].values(): verify_files(records); refs=merge_hashes(refs,records)
    checks=[]
    import torch
    for ds,outer in itertools.product(DATASETS,OUTERS):
        cache,_=_cached(root,ds,outer)
        for a,b in itertools.combinations(('fit','select','threshold','query'),2):
            ids=lambda k: cache[k]['ids'] if k in ('fit','select') else cache['inference'][k+'_ids']
            if torch.isin(ids(a),ids(b)).any(): raise ValueError('validation/query partitions overlap')
        _verify_cell(root,'FINAL',ds,outer,plan)
        p=_cell_path(root,'FINAL',ds,outer); pred=tensor_load(p/'predictions.pt')
        if not torch.equal(pred['query_ids'],cache['inference']['query_ids']): raise ValueError('frozen query IDs differ')
        aliases=read_json(p/'pair_aliases.json')
        expected={t['id'] for t in plan['trials'][ds] if outer in t['outers']}
        if set(aliases)!=expected or set(pred['candidates'])!=set(aliases.values()): raise ValueError('final candidate coverage differs')
        for probabilities in pred['candidates'].values():
            if set(probabilities)!={'fixed','calibrated_noPB','pB_diagnostic'}: raise ValueError('posthoc control coverage differs')
            for value in probabilities.values():
                if value.shape[0]!=len(pred['query_ids']) or not torch.isfinite(value).all() or (value<0).any() or not torch.allclose(value.sum(1),torch.ones(len(value)),atol=2e-5): raise ValueError('invalid frozen query probabilities')
        verify_final_probabilities(root,ds,outer,plan)
        # Reproduce saved validation selection/calibration from immutable members.
        for saved in read_json(p/'validation.json'):
            t=next(t for t in cell_trials(plan,ds,outer) if t['id']==saved['id'])
            if _trial_readout(root,ds,outer,t,INNERS,final_calibration=True)!={k:v for k,v in saved.items() if k not in ('dataset','outer','id')}: raise ValueError('saved validation readout does not reproduce')
        refs=merge_hashes(refs,bundle_records(p))
        checks.append({'dataset':ds,'outer':outer,'candidate_count':len(expected),'validation_reproduced':True,'query_ids_verified':True})
    bootstrap(root)
    lock={'datasets':list(DATASETS),'outers':list(OUTERS),'inners':list(INNERS),'cell_count':25,'test_labels_read':False,'files':refs,'input_seal_sha256':sha256(root/'INPUT_SEAL.json'),'checks':checks,'slurm_job_id':os.environ['SLURM_JOB_ID']}
    immutable_json(root/'FROZEN.json',lock)
    return {'status':'frozen','next':'evaluate','cell_count':25}


def verify_globalfreeze(root):
    root,cfg=_guard(root)
    lock=read_json(root/'FROZEN.json')
    if lock.get('datasets')!=list(DATASETS) or lock.get('outers')!=list(OUTERS) or lock.get('inners')!=list(INNERS) or lock.get('cell_count')!=25 or lock.get('test_labels_read') is not False:
        raise ValueError('complete all-25-cell freeze is required')
    if {(r['dataset'],r['outer']) for r in lock['checks']}!=set(itertools.product(DATASETS,OUTERS)) or not all(r['validation_reproduced'] and r['query_ids_verified'] for r in lock['checks']): raise ValueError('freeze validation audit incomplete')
    verify_files(lock['files']); verify_files(read_json(root/'INPUT_SEAL.json')['files'])
    for stage in PHASES: _verify_plan(root,stage); _verify_lock(root,stage)
    plan=_verify_plan(root,'FINAL')
    for ds,outer in itertools.product(DATASETS,OUTERS): _verify_cell(root,'FINAL',ds,outer,plan)
    bootstrap(root)
    return root,cfg,lock


def verify_final_probabilities(root,ds,outer,plan):
    """Reproduce final saved probabilities using sealed query member caches."""
    import torch
    path=_cell_path(root,'FINAL',ds,outer); saved=tensor_load(path/'predictions.pt')
    cache,_=_cached(root,ds,outer)
    if not torch.equal(saved['query_ids'],cache['inference']['query_ids']): raise ValueError('query IDs differ')
    calibration=read_json(path/'calibration.json'); baseline=tensor_load(path/'baseline_validation.pt')
    validation={r['id']:r for r in read_json(path/'validation.json')}
    for name,choice in calibration.items():
        pair=next(t['pair'] for t in cell_trials(plan,ds,outer) if t['id']==name)
        if choice['noPB']!=validation[name]['calibration']: raise ValueError('saved calibration choice differs from reproduced validation')
        fit_members={b:[_member(root,ds,outer,pair[b],seed)['fit']['branches'][b] for seed in INNERS] for b in ('g','l')}
        fit_cal=fused_probability(fit_members['g'],fit_members['l'],choice['noPB'])
        if pb_calibrate(fit_cal,baseline['fit'],cache['fit']['targets'])!=choice['pB_diagnostic']:
            raise ValueError('pB diagnostic FIT choice fails reproduction')
        fixed_parts={b:[] for b in ('g','l')}; parts={b:[] for b in ('g','l')}
        for b,cfg in pair.items():
            scale=choice['noPB']['scale_'+b]
            for seed in INNERS:
                q=root/'query_members'/ds/str(outer)/digest({'config':cfg,'branch':b,'seed':seed,'scale':scale})
                pred=tensor_load(q/'probabilities.pt')
                if not torch.equal(pred['ids'],saved['query_ids']): raise ValueError('query member IDs differ')
                fixed_parts[b].append(pred['fixed']); parts[b].append(pred['calibrated'])
        fixed=.5*(mean_probability(fixed_parts['g'])+mean_probability(fixed_parts['l']))
        alpha=choice['noPB']['alpha']; beta=choice['pB_diagnostic']['beta']
        cal=alpha*mean_probability(parts['g'])+(1-alpha)*mean_probability(parts['l'])
        expected={'fixed':fixed,'calibrated_noPB':cal,'pB_diagnostic':(1-beta)*cal+beta*saved['pB']}
        if any(not torch.equal(value,saved['candidates'][name][key]) for key,value in expected.items()):
            raise ValueError('saved final fusion probabilities fail reproduction')
    return True
