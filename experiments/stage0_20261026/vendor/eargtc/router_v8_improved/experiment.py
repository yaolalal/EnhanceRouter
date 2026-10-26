"""Preregistered source deployment simulation and label-free target inference.

Search deserializes support and validation only. Query is first deserialized by
run_target, after independent source-only configuration selection is complete.
"""
import hashlib
import itertools
import json
import os
import tempfile
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from .core import (HEADS,FeatureTransform,array,validate_bank,binary_target,fit_binary,
    adapter_basis,fit_adapter,fit_adapter_projected,predict_adapted,reciprocal_metrics,
    select_configuration,validation_halves,fit_affine,apply_affine,routing_boundary,
    score_payload,fit_top_fraction,fit_risk_policy,binary_metrics)

DATASETS=('cora','citeseer','pubmed','arxiv','ogbn-products')
_WORKER=None


def candidate_grid():
    result=[]
    for base,(rep,view,l2,pair) in enumerate(itertools.product(
            ('core','context','prototype'),('standard','rank'),(.01,.1,1.,10.),(0.,.25,1.,4.))):
        for rank,prior in itertools.product((0,1,2),(.01,.1,1.,10.)):
            result.append(dict(id=f'c{len(result):04d}',base_id=f'b{base:03d}',representation=rep,
                view=view,l2=l2,pairwise_weight=pair,adapter_rank=rank,adapter_prior=prior))
    return result


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def fingerprint(pre):
    return hashlib.sha256(json.dumps(pre,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def ids_hash(ids):
    return hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()


def write_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    # Atomic installation without overwriting immutable completed artifacts.
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
            f.flush(); os.fsync(f.fileno())
        os.link(temporary,path)
    finally:
        os.unlink(temporary)


def save_torch(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent); os.close(fd)
    try:
        torch.save(payload,temporary); os.link(temporary,path)
    finally: os.unlink(temporary)


def verify_preregistration(root,stage='search'):
    root=Path(root)
    pre=json.loads((root/'PREREGISTERED.json').read_text())
    required={'banks_root','datasets','seed','candidates','code_hashes','input_hashes'}
    if not required.issubset(pre): raise ValueError('incomplete preregistration')
    if tuple(pre['datasets'])!=DATASETS or pre['seed']!=42: raise ValueError('unexpected domains or seed')
    available={c['id']:c for c in candidate_grid()}
    if not pre['candidates'] or any(c!=available.get(c.get('id')) for c in pre['candidates']):
        raise ValueError('candidate outside finite preregistered engine grid')
    if len({c['id'] for c in pre['candidates']})!=len(pre['candidates']): raise ValueError('duplicate candidate')
    for category in ('code_hashes','input_hashes'):
        if not isinstance(pre[category],dict) or not pre[category]: raise ValueError('empty hash manifest')
        for path,expected in pre[category].items():
            if not Path(path).is_file() or sha256(path)!=expected: raise ValueError(f'{category} hash changed: {path}')
    banks=Path(pre['banks_root'])
    required_splits=('support','validation') if stage=='search' else ('support','validation','query')
    registered={str(Path(p).resolve()) for p in pre['input_hashes']}
    for d in DATASETS:
        for split in required_splits:
            if str((banks/d/f'{split}.pt').resolve()) not in registered: raise ValueError('bank missing from input hash manifest')
    return pre


def _assert_unfrozen(root):
    root=Path(root)
    if (root/'FROZEN.json').exists() or (root/'selection'/'FROZEN.json').exists():
        raise ValueError('experiment is frozen')


def _load_bank(banks_root,domain,split):
    if split not in ('support','validation','query'): raise ValueError('unsupported bank split')
    bank=torch.load(Path(banks_root)/domain/f'{split}.pt',map_location='cpu',weights_only=False)
    return validate_bank(bank,query=split=='query')


def _init_worker(banks_root,excluded):
    global _WORKER
    torch.set_num_threads(1)
    supports={d:_load_bank(banks_root,d,'support') for d in DATASETS}
    validations={d:_load_bank(banks_root,d,'validation') for d in excluded}
    for d in excluded:
        if np.intersect1d(array(supports[d]['node_ids']),array(validations[d]['node_ids'])).size:
            raise ValueError('support/validation ID overlap')
    _WORKER=dict(supports=supports,validations=validations,excluded=tuple(excluded),cache_key=None)


def _workspace(head,cfg):
    worker=_WORKER; key=(head,cfg['representation'],cfg['view'])
    if worker['cache_key']!=key:
        sources=[d for d in DATASETS if d not in worker['excluded']]
        transform=FeatureTransform.fit([worker['supports'][d] for d in sources],head,cfg['representation'],cfg['view'])
        # Cache one representation at a time per worker, never all validation matrices.
        worker['cache']=dict(transform=transform,sources=sources,
            support_x={d:transform.apply(b) for d,b in worker['supports'].items()},
            validation_x={d:transform.apply(b) for d,b in worker['validations'].items()})
        worker['cache_key']=key
    return worker['cache']


def fit_shared_and_basis(supports,sources,head,cfg,transform=None,features=None):
    transform=transform or FeatureTransform.fit([supports[d] for d in sources],head,cfg['representation'],cfg['view'])
    features=features or {d:transform.apply(supports[d]) for d in sources}
    xs=[]; ys=[]; fit_ids={}
    for d in sources:
        mask,y=binary_target(supports[d]['state'],head)
        xs.append(features[d][mask]); ys.append(y)
        fit_ids[d]=array(supports[d]['node_ids'])[mask].tolist()
    shared=fit_binary(xs,ys,cfg['l2'],cfg['pairwise_weight'])
    locals_=[fit_binary([x],[y],cfg['l2'],cfg['pairwise_weight']) for x,y in zip(xs,ys)]
    basis=adapter_basis(shared,locals_)
    return dict(shared=shared,basis=basis,local_models=locals_,transform=transform,
        source_domains=list(sources),source_fit_ids=fit_ids,
        transform_fit_ids={d:array(supports[d]['node_ids']).tolist() for d in sources})


def _compact_metrics(metrics):
    for fold in metrics['folds']:
        for key in ('calibration_ids','evaluation_ids'):
            ids=fold.pop(key); fold[key+'_sha256']=ids_hash(ids); fold[key+'_count']=len(ids)
    return metrics


def _fit_base(task):
    head,configs=task; cfg=configs[0]; cache=_workspace(head,cfg); worker=_WORKER
    fitted=fit_shared_and_basis(worker['supports'],cache['sources'],head,cfg,cache['transform'],cache['support_x'])
    shared,basis=fitted['shared'],fitted['basis']; held_cache={}
    for d in worker['excluded']:
        bank=worker['supports'][d]; sx=cache['support_x'][d]; vx=cache['validation_x'][d]
        mask,y=binary_target(bank['state'],head)
        held_cache[d]=dict(projection=(sx@basis)[mask],y=y,offset=(sx@shared['coef']+shared['bias'])[mask],
            validation_projection=vx@basis,validation_offset=vx@shared['coef']+shared['bias'],
            adaptation_ids=array(bank['node_ids'])[mask].tolist())
    rows=[]
    for c in configs:
        held={}; adaptations={}
        for d,data in held_cache.items():
            adapt=fit_adapter_projected(data['projection'],data['y'],data['offset'],c['adapter_rank'],c['adapter_prior'])
            adapt['requested_rank']=c['adapter_rank']; rank=adapt['rank']
            logits=data['validation_offset']+data['validation_projection'][:,:rank]@np.asarray(adapt['coordinates'])+adapt['bias']
            vb=worker['validations'][d]
            held[d]=_compact_metrics(reciprocal_metrics(logits,vb['state'],vb['node_ids'],d,head))
            adaptations[d]=adapt
        rows.append(dict(config=c,head=head,held=held,adaptations=adaptations,feature_count=len(cache['transform'].names)))
    return dict(head=head,base_id=cfg['base_id'],excluded=list(worker['excluded']),source_domains=cache['sources'],
        feature_names=cache['transform'].names,source_fit_ids=fitted['source_fit_ids'],transform_fit_ids=fitted['transform_fit_ids'],
        adaptation_ids={d:data['adaptation_ids'] for d,data in held_cache.items()},
        validation_ids_sha256={d:ids_hash(array(b['node_ids'])) for d,b in worker['validations'].items()},
        basis_rank=basis.shape[1],shared_fit={k:v for k,v in shared.items() if k!='coef'},rows=rows)


def _groups(pre):
    grouped={}
    for cfg in pre['candidates']: grouped.setdefault(cfg['base_id'],[]).append(cfg)
    return [(head,grouped[b]) for head in HEADS for b in sorted(grouped)]


def _expected_provenance(pre,pair,validation_domains=None):
    supports={d:_load_bank(pre['banks_root'],d,'support') for d in DATASETS}
    domains=pair if validation_domains is None else validation_domains
    validations={d:_load_bank(pre['banks_root'],d,'validation') for d in domains}
    sources=[d for d in DATASETS if d not in pair]; result={}
    for head in HEADS:
        fit_ids={d:array(b['node_ids'])[binary_target(b['state'],head)[0]].tolist() for d,b in supports.items()}
        held={}
        for d,b in validations.items():
            ids=array(b['node_ids']);state=array(b['state']);first,second=validation_halves(ids,d)
            folds=[]
            for cal,evaluation in ((first,second),(second,first)):
                cm,cy=binary_target(state[cal],head); em,ey=binary_target(state[evaluation],head)
                folds.append(dict(calibration_ids_sha256=ids_hash(ids[cal]),calibration_ids_count=int(cal.sum()),
                    evaluation_ids_sha256=ids_hash(ids[evaluation]),evaluation_ids_count=int(evaluation.sum()),
                    calibration_head_count=int(cm.sum()),evaluation_head_count=int(em.sum()),count=len(ey),
                    positive_count=int(ey.sum()),negative_count=int(len(ey)-ey.sum()),
                    calibration_counts=[int((cy==k).sum()) for k in (0,1)]))
            held[d]=dict(validation_ids_sha256=ids_hash(ids),folds=folds)
        result[head]=dict(source_domains=sources,source_fit_ids={d:fit_ids[d] for d in sources},
            transform_fit_ids={d:array(supports[d]['node_ids']).tolist() for d in sources},
            adaptation_ids={d:fit_ids[d] for d in pair},held=held)
    return result


def _metric_valid(value,name,count=None,two_class=None):
    if value is not None and (not isinstance(value,(int,float)) or not np.isfinite(value)):
        raise ValueError(f'cache provenance nonfinite {name}')
    if value is not None and (value<0 or (name in ('auc','brier') and value>1)):
        raise ValueError(f'cache provenance metric range: {name}')
    if count is not None:
        should_define=(two_class if name=='auc' else count>0)
        if (value is not None)!=should_define: raise ValueError(f'cache provenance undefined {name}')


def _check_held_metrics(metrics,expected):
    folds=metrics.get('folds',[])
    if len(folds)!=2: raise ValueError('cache provenance requires two reciprocal folds')
    for fold,want in zip(folds,expected['folds']):
        for key,value in want.items():
            actual=fold.get('calibration',{}).get('counts') if key=='calibration_counts' else fold.get(key)
            if actual!=value: raise ValueError(f'cache provenance fold mismatch: {key}')
        cal=fold['calibration']
        if not isinstance(cal.get('slope'),(int,float)) or not np.isfinite(cal['slope']) or not .05<=cal['slope']<=20:
            raise ValueError('cache provenance invalid calibration slope')
        if not isinstance(cal.get('bias'),(int,float)) or not np.isfinite(cal['bias']) or cal.get('strength')!=2.:
            raise ValueError('cache provenance invalid calibration parameters')
        two=fold['positive_count']>0 and fold['negative_count']>0
        for name in ('auc','bce','brier'): _metric_valid(fold.get(name),name,fold['count'],two)
    for key in ('count','positive_count','negative_count'):
        if metrics.get(key)!=sum(fold[key] for fold in folds): raise ValueError(f'cache provenance total mismatch: {key}')
    if metrics.get('valid_auc_folds')!=sum(fold['auc'] is not None for fold in folds):
        raise ValueError('cache provenance AUC fold count mismatch')
    for name in ('auc','bce','brier'):
        value=metrics.get(name);_metric_valid(value,name)
        values=[fold[name] for fold in folds if fold[name] is not None]
        want=float(np.mean(values)) if values and (name=='auc' or len(values)==2) else None
        if (value is None)!=(want is None) or (want is not None and not np.isclose(value,want,rtol=1e-12,atol=1e-12)):
            raise ValueError(f'cache provenance macro-fold mismatch: {name}')


def _check_cache(path,fp,head,configs,pair,expected):
    data=json.loads(Path(path).read_text())
    if data.get('preregistration_sha256')!=fp: raise ValueError('stale source cache fingerprint')
    if data.get('head')!=head or data.get('excluded')!=list(pair): raise ValueError('source cache provenance identity mismatch')
    if [r['config'] for r in data['rows']]!=configs: raise ValueError('source cache candidate mismatch')
    for key in ('source_domains','source_fit_ids','transform_fit_ids','adaptation_ids'):
        if data.get(key)!=expected[key]: raise ValueError(f'cache provenance mismatch: {key}')
    if set(data.get('validation_ids_sha256',{}))!=set(pair): raise ValueError('cache provenance held validation domains')
    for domain,want in expected['held'].items():
        if data['validation_ids_sha256'][domain]!=want['validation_ids_sha256']:
            raise ValueError('cache provenance validation ID hash mismatch')
    for row in data['rows']:
        if row.get('head')!=head or set(row.get('held',{}))!=set(pair): raise ValueError('cache provenance held row domains')
        for domain,want in expected['held'].items(): _check_held_metrics(row['held'][domain],want)
    return data


def _check_complete(directory,pre,pair,fp):
    state=json.loads((directory/'COMPLETE.json').read_text())
    expected={f"{head}/{configs[0]['base_id']}.json" for head,configs in _groups(pre)}
    if state.get('preregistration_sha256')!=fp or state.get('excluded')!=list(pair):
        raise ValueError('source COMPLETE identity mismatch')
    if state.get('base_files')!=len(expected) or state.get('configurations_per_head')!=len(pre['candidates']):
        raise ValueError('source COMPLETE count mismatch')
    if set(state.get('files',{}))!=expected: raise ValueError('source COMPLETE file set mismatch')
    actual={str(p.relative_to(directory)) for head in HEADS for p in (directory/head).glob('*.json')}
    if actual!=expected: raise ValueError('source COMPLETE actual file set mismatch')
    for path,digest in state['files'].items():
        if not (directory/path).is_file() or sha256(directory/path)!=digest: raise ValueError('completed source cache hash changed')
    return state


def run_pair(root,pair_index,workers=1):
    root=Path(root); _assert_unfrozen(root); pre=verify_preregistration(root,stage='search')
    pairs=list(itertools.combinations(DATASETS,2))
    if pair_index is None or not 0<=pair_index<len(pairs): raise ValueError('pair index must be 0..9')
    if workers<1: raise ValueError('workers must be positive')
    pair=pairs[pair_index]; destination=root/'source'/f'pair_{pair_index:02d}'; destination.mkdir(parents=True,exist_ok=True)
    fp=fingerprint(pre); tasks=_groups(pre); pending=[]; files=[]
    provenance=_expected_provenance(pre,pair)
    complete=destination/'COMPLETE.json'
    for head,configs in tasks:
        path=destination/head/(configs[0]['base_id']+'.json'); files.append(path)
        if path.exists(): _check_cache(path,fp,head,configs,pair,provenance[head])
        else: pending.append((head,configs))
    if complete.exists():
        return _check_complete(destination,pre,pair,fp)
    started=time.time()
    def store_result(result):
        result['preregistration_sha256']=fp
        write_json(destination/result['head']/(result['base_id']+'.json'),result)
        print(json.dumps(dict(pair=pair_index,head=result['head'],base=result['base_id'],elapsed_seconds=round(time.time()-started,2))),flush=True)
    if workers==1:
        _init_worker(pre['banks_root'],pair)
        for task in pending: store_result(_fit_base(task))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers,initializer=_init_worker,initargs=(pre['banks_root'],pair)) as pool:
            futures=[pool.submit(_fit_base,t) for t in pending]
            for future in as_completed(futures): store_result(future.result())
    result=dict(preregistration_sha256=fp,excluded=list(pair),base_files=len(files),configurations_per_head=len(pre['candidates']),
        files={str(p.relative_to(destination)):sha256(p) for p in files},elapsed_seconds=time.time()-started)
    write_json(complete,result)
    return result


def load_source_trials(root,target,head):
    root=Path(root); pre=json.loads((root/'PREREGISTERED.json').read_text()); fp=fingerprint(pre)
    if target not in DATASETS or head not in HEADS: raise ValueError('unknown target or head')
    result=[]
    for i,pair in enumerate(itertools.combinations(DATASETS,2)):
        if target not in pair: continue
        directory=root/'source'/f'pair_{i:02d}'
        complete=_check_complete(directory,pre,pair,fp)
        held=next(d for d in pair if d!=target)
        # Audit only the direction used for source selection. The discarded target
        # direction never causes target-validation deserialization before selection.
        # Source run/resume checks both directions against their complete banks.
        provenance=_expected_provenance(pre,pair,validation_domains=(held,))
        for _,configs in [t for t in _groups(pre) if t[0]==head]:
            path=directory/head/(configs[0]['base_id']+'.json')
            expected=complete['files'].get(str(path.relative_to(directory)))
            if not expected or sha256(path)!=expected: raise ValueError('completed source cache hash changed')
            cache=_check_cache(path,fp,head,configs,pair,provenance[head])
            held=next(d for d in pair if d!=target)
            expected_sources=[d for d in DATASETS if d not in pair]
            if cache['source_domains']!=expected_sources: raise ValueError('source training domains contaminated')
            result.extend(dict(config=row['config'],held={held:row['held'][held]}) for row in cache['rows'])
    selected=select_configuration(result,target)
    expected_held=set(DATASETS)-{target}
    if len(selected['trials'])!=len(pre['candidates']) or any(set(t['held'])!=expected_held for t in selected['trials']):
        raise ValueError('incomplete four-domain source selection')
    return result


def _predict_bank(bank,fitted,adaptation,chunk_size=8192):
    out=[]; n=len(bank['node_ids'])
    for start in range(0,n,chunk_size):
        small={k:(v[start:start+chunk_size] if k in ('candidate_g','candidate_l','symmetric_global') else v) for k,v in bank.items()}
        x=fitted['transform'].apply(small)
        out.append(predict_adapted(x,fitted['shared'],fitted['basis'],adaptation))
    return np.concatenate(out) if out else np.empty(0)


def run_target(root,target):
    root=Path(root); _assert_unfrozen(root)
    if target not in DATASETS: raise ValueError('unknown target')
    destination=root/'targets'/target
    if destination.exists() and any(destination.iterdir()): raise FileExistsError('target artifacts already exist')
    pre=verify_preregistration(root,stage='target'); fp=fingerprint(pre)
    selections={head:select_configuration(load_source_trials(root,target,head),target) for head in HEADS}
    configs={head:selections[head]['best']['config'] for head in HEADS}
    supports={d:_load_bank(pre['banks_root'],d,'support') for d in DATASETS}
    vb=_load_bank(pre['banks_root'],target,'validation'); ids=array(vb['node_ids']); z=array(vb['state'])
    if np.intersect1d(array(supports[target]['node_ids']),ids).size: raise ValueError('support/validation ID overlap')
    cal,policy=validation_halves(ids,target)
    models={}; adaptation={}; calibration={}; validation_raw={}; validation_final={}
    for head,cfg in configs.items():
        fitted=fit_shared_and_basis(supports,[d for d in DATASETS if d!=target],head,cfg)
        x=fitted['transform'].apply(supports[target]); mask,y=binary_target(supports[target]['state'],head)
        adapt=fit_adapter(x[mask],y,fitted['shared'],fitted['basis'],cfg['adapter_rank'],cfg['adapter_prior'])
        raw=_predict_bank(vb,fitted,adapt); cm,cy=binary_target(z[cal],head)
        calibrated=fit_affine(raw[cal][cm],cy)
        models[head]=fitted; adaptation[head]=adapt; calibration[head]=calibrated
        validation_raw[head]=raw; validation_final[head]=apply_affine(raw,calibrated)
    routefit=routing_boundary(validation_final['preference'][cal],z[cal]); threshold=routefit['threshold']
    validation_scores=score_payload(vb,validation_final['trust'],validation_final['preference'],threshold)
    validation_scores.update(state=vb['state'],calibration_mask=torch.from_numpy(cal),policy_mask=torch.from_numpy(policy))
    for head in HEADS: validation_scores['adapted_'+head+'_logit']=torch.from_numpy(validation_raw[head])
    score=array(validation_scores['selected_score'])[policy]; route=array(validation_scores['route'])[policy]
    correct=route==z[policy]
    policies={str(risk):fit_risk_policy(score,correct,risk=risk,min_accepted=20) for risk in (.05,.1,.15,.2)}
    top=fit_top_fraction(score,correct,ids[policy])
    # Query first deserialized only after source selection, adaptation, calibration,
    # routing boundary and acceptance policy are finalized.
    query=_load_bank(pre['banks_root'],target,'query')
    query_ids=array(query['node_ids'])
    if np.intersect1d(query_ids,np.r_[ids,array(supports[target]['node_ids'])]).size:
        raise ValueError('query overlaps fitting/calibration IDs')
    raw_query={head:_predict_bank(query,models[head],adaptation[head]) for head in HEADS}
    final_query={head:apply_affine(raw_query[head],calibration[head]) for head in HEADS}
    scores=score_payload(query,final_query['trust'],final_query['preference'],threshold)
    for head in HEADS: scores['adapted_'+head+'_logit']=torch.from_numpy(raw_query[head])
    serial_models={head:{**{k:v for k,v in model.items() if k!='transform'},'transform':model['transform'].state_dict(),
        'adaptation':adaptation[head],'calibration':calibration[head],'config':configs[head],
        'adaptation_ids':array(supports[target]['node_ids'])[binary_target(supports[target]['state'],head)[0]].tolist()} for head,model in models.items()}
    policy_metrics={}
    for head in HEADS:
        mask,y=binary_target(z[policy],head)
        policy_metrics[head]=dict(before_calibration=binary_metrics(validation_raw[head][policy][mask],y),
            after_calibration=binary_metrics(validation_final[head][policy][mask],y))
    selection=dict(target=target,preregistration_sha256=fp,configs=configs,adaptation=adaptation,calibration=calibration,
        routing=routefit,route_boundary=threshold,policies=policies,top_fraction_policy=top,
        calibration_ids=ids[cal].tolist(),policy_ids=ids[policy].tolist(),source_domains=[d for d in DATASETS if d!=target],
        validation_split='sha256(dataset|conflict|node_id)[0]&1; exact historical split-C algorithm',
        validation_metrics=policy_metrics,selected_by='four source-domain reciprocal-validation metrics; target never selects configurations',
        deviations=['independently optimized binary heads','Preference shared expert-shape terms allow expert identity bias; no strict swap equivariance',
            'source validation used for deployment-simulation configuration selection','top fraction is exploratory precision policy, not risk guarantee'])
    destination.mkdir(parents=True,exist_ok=True)
    save_torch(destination/'model.pt',dict(heads=serial_models,route_boundary=threshold,preregistration_sha256=fp))
    save_torch(destination/'validation_scores.pt',validation_scores)
    save_torch(destination/'scores.pt',scores)
    write_json(destination/'source_selection.json',selections)
    write_json(destination/'selection.json',selection)
    write_json(destination/'COMPLETE.json',dict(preregistration_sha256=fp,files={p.name:sha256(p) for p in destination.iterdir() if p.is_file()}))
    print(json.dumps(dict(target=target,configs={h:c['id'] for h,c in configs.items()},top_fraction=top['fraction'])),flush=True)
    return selection
