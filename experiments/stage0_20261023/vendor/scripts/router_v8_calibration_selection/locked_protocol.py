#!/usr/bin/env python3
"""Preregister, audit, freeze, evaluate once and report the improved V8 run.

This is the ONLY new experiment module allowed to load graph test labels.
All entrypoints, including reporting, must run inside a Slurm allocation.
"""
import argparse
import ast
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from eargtc.router_v8.artifacts import write_json, sha256, freeze_paths, verify_frozen, begin_once
from eargtc.router_v8.metrics import evaluate, selective_metrics
from eargtc.router_v8.deployment import validation_halves

DATASETS=('cora','citeseer','pubmed','arxiv','ogbn-products')
OLD=REPO/'runs/router_v8_trust_preference_20260906_v1'
SOURCE=REPO/'runs/router_v8_exploration_20260907_v2'
COVERAGE=[.01,.025,.05,.10,.15,.20,.25,.35,.50,.75,1.]


def arr(x):
    return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)


def exact_top_mask(score,ids,*,fraction=None,count=None):
    n=len(score)
    if count is None:
        f=Fraction(str(fraction)); count=(n*f.numerator+f.denominator-1)//f.denominator
    k=min(n,max(0,int(count)))
    order=np.lexsort((np.asarray(ids),-np.asarray(score)))
    mask=np.zeros(n,dtype=bool); mask[order[:k]]=True
    return mask


def selection_metrics(q,state,route,accept):
    return selective_metrics(q,state,np.where(accept,route,2))


def route_metrics(q,state,route,ids):
    """Probability metrics retain event semantics; routing metrics use actual route."""
    q=np.asarray(q,dtype=float); z=np.asarray(state); route=np.asarray(route)
    m=evaluate(q,z,ids)
    if not len(z):
        undefined=('nll','brier','macro_f1','balanced_accuracy','trust_auroc','trust_auprc',
            'neither_auroc','neither_auprc','preference_auroc','preference_auprc',
            'preference_accuracy','gnn_only_recall','llm_only_recall','full_conflict_pseudo_accuracy',
            'pair_oracle_accuracy','gnn_route_fraction','aurc','excess_aurc','preference_gnn_route_fraction',
            'preference_balanced_accuracy','overall_ece','selected_correctness_auroc',
            'selected_correctness_auprc','trust_bce','trust_brier','preference_bce','preference_brier')
        m.update({key:None for key in undefined})
        m.update(state_counts=[0,0,0],classwise_ece={k:None for k in ('G','L','N')},
            risk_at_coverage={str(f):None for f in COVERAGE})
        return m
    score=q[np.arange(len(q)),route]; correct=route==z; decisive=z!=2
    order=np.lexsort((ids,-score)); risk=np.cumsum(~correct[order])/np.arange(1,len(z)+1)
    errors=int((~correct).sum()); good=len(z)-errors
    ideal=np.r_[np.zeros(good),np.arange(1,errors+1)/np.arange(good+1,len(z)+1)]
    two=len(np.unique(correct))==2
    m.update(full_conflict_pseudo_accuracy=float(correct.mean()),
        preference_accuracy=float(correct[decisive].mean()) if decisive.any() else None,
        gnn_only_recall=float((route[z==0]==0).mean()) if (z==0).any() else None,
        llm_only_recall=float((route[z==1]==1).mean()) if (z==1).any() else None,
        gnn_route_fraction=float((route==0).mean()),
        preference_gnn_route_fraction=float((route[decisive]==0).mean()) if decisive.any() else None,
        preference_balanced_accuracy=float(np.mean([(route[z==k]==k).mean() for k in (0,1) if (z==k).any()])) if decisive.any() else None,
        aurc=float(risk.mean()),excess_aurc=float(risk.mean()-ideal.mean()),
        selected_correctness_auroc=float(roc_auc_score(correct,score)) if two else None,
        selected_correctness_auprc=float(average_precision_score(correct,score)) if two else None,
        risk_at_coverage={str(f):float(risk[np.flatnonzero(exact_top_mask(-np.arange(len(z)),np.arange(len(z)),fraction=f))[-1]]) for f in COVERAGE})
    c=q[:,:2].sum(1); r=np.divide(q[:,0],c,out=np.full(len(q),.5),where=c>0)
    for name,p,y in [('trust',c,decisive),('preference',r[decisive],z[decisive]==0)]:
        p=np.clip(p,1e-15,1-1e-15)
        m[name+'_bce']=float(-(y*np.log(p)+(~y)*np.log1p(-p)).mean()) if len(y) else None
        m[name+'_brier']=float(((p-y)**2).mean()) if len(y) else None
    return m


def validate_scores(query,scores):
    if any(k in query for k in ('state','labels','y','y_true','correctness')):
        raise ValueError('query contains labels')
    for k in ('node_ids','gnn_pred','llm_pred'):
        if not np.array_equal(arr(query[k]),arr(scores[k])): raise ValueError(f'query score identity: {k}')
    q=arr(scores['q']); route=arr(scores['route'])
    if q.shape!=(len(route),3) or not np.isfinite(q).all() or (q<0).any() or not np.allclose(q.sum(1),1,atol=1e-7):
        raise ValueError('invalid probabilities')
    if not np.isin(route,[0,1]).all(): raise ValueError('invalid route')
    if not np.allclose(arr(scores['selected_score']),q[np.arange(len(q)),route.astype(int)],rtol=1e-7,atol=1e-9):
        raise ValueError('selected reliability does not match actual route')


def code_paths():
    # Include local modules imported by the graph-unpickling compatibility helper,
    # applicable package initializers, and the complete recursive router namespaces.
    paths={REPO/'eargtc/__init__.py',REPO/'eargtc/data.py',REPO/'eargtc/utils.py'}
    for directory in ('eargtc/router_v8_calibration_selection','scripts/router_v8_calibration_selection','eargtc/router_v8_exploration','scripts/router_v8_exploration','eargtc/router_v8'):
        paths.update((REPO/directory).rglob('*.py'))
    for package in ('scripts','scripts/router_v8_calibration_selection'):
        initializer=REPO/package/'__init__.py'
        if initializer.exists(): paths.add(initializer)
    return sorted(paths)


def verify_code_manifest(pre):
    expected={str(p.resolve()) for p in code_paths()}
    recorded={str(Path(p).resolve()) for p in pre['code_hashes']}
    if expected!=recorded: raise ValueError('preregistered code file set changed')
    for path,digest in pre['code_hashes'].items():
        if not Path(path).is_file() or sha256(path)!=digest: raise ValueError(f'code hash changed: {path}')


def verify_run_frozen(root):
    root=Path(root)
    frozen=verify_frozen(root/'selection/FROZEN.json')
    verify_code_manifest(json.loads((root/'PREREGISTERED.json').read_text()))
    return frozen


def verify_original_banks(banks):
    """Provenance-only preflight: hash bytes, never unpickle graph labels."""
    verify_frozen(OLD/'selection/FROZEN.json')
    checks={}
    for domain in DATASETS:
        folder=Path(banks)/domain
        manifest=json.loads((folder/'manifest.json').read_text())
        for split in ('support','validation','query'):
            expected=manifest.get('bank_sha256',{}).get(split)
            if not expected or sha256(folder/f'{split}.pt')!=expected:
                raise ValueError(f'original bank hash mismatch: {domain}/{split}')
        graph=str(Path(manifest['graph_path']).resolve())
        inputs={str(Path(p).resolve()):h for p,h in manifest['input_hashes'].items()}
        if graph not in inputs: raise ValueError(f'graph path is not registered: {domain}')
        if sha256(graph)!=inputs[graph]: raise ValueError(f'original graph hash mismatch: {domain}')
        checks[domain]=dict(original_bank_hashes_verified=True,graph_registered=True)
    return checks


def preregister(root):
    from eargtc.router_v8_exploration.experiment import candidate_grid
    dest=root/'PREREGISTERED.json'
    if dest.exists(): raise FileExistsError(dest)
    paths=[OLD/'banks'/d/f for d in DATASETS for f in ('support.pt','validation.pt','query.pt','manifest.json')]+[OLD/'selection/FROZEN.json']
    source_frozen=verify_frozen(SOURCE/'selection/FROZEN.json')
    paths += [SOURCE/'selection/FROZEN.json',SOURCE/'PREREGISTERED.json'] + [Path(p) for p in source_frozen['hashes']]
    pre=dict(source_root=str(SOURCE.resolve()),source_frozen_sha256=sha256(SOURCE/'selection/FROZEN.json'),primary_protocol='target_calibration_selection',reference_protocol='frozen_source_only_round2',
        shortlist=dict(families=['representation','view'],per_family=['minimum_source_bce','maximum_source_auc_within_family_best_bce_plus_0.05'],include_source_global_best=True,max_per_head=25),
        endpoint_selection=dict(calibration_only=True,small_calibration_max=256,small_crossfit='LOO',large_crossfit_folds=5,min_per_class=5,bce_tolerance=.05,ranking='raw_adapted_calibration_auroc'),
        schema='router_v8_calibration_selection_v3',seed=42,shots=5,datasets=list(DATASETS),banks_root=str((OLD/'banks').resolve()),
        candidates=candidate_grid(),coverage=COVERAGE,top_k=[20,50,100,200,500,1000],
        code_hashes={str(p.resolve()):sha256(p) for p in code_paths()},
        input_hashes={str(p.resolve()):sha256(p) for p in paths},
        protocol_sha256=sha256(REPO/'docs/router_v8_calibration_selection_work/PLAN.md'),
        selection='source-only shortlist then target calibration crossfit BCE<=minimum+0.05, max raw calibration AUC; scarce class keeps source-global endpoint',
        source_supervision='three source corrected-OOF supports; held source support only adapter; held validation halves reciprocal calibration/evaluation',
        target_roles='source-derived finite shortlist; own support adaptation; calibration half crossfit endpoint selection/probability/route; disjoint policy half acceptance only',
        rank_adaptation='source-local coefficient deviation SVD; rank 0/1/2; mean BCE+prior/2 parameter norm',
        test_contract='all five frozen before one evaluator graph-label read; historical benchmark reuse; no test-selected retry',
        protocol_changes=['source validation used for selection/calibration simulation, never shared fit',
            'separate head objectives and representations; direct expert-context preference terms permitted',
            'low-rank support adapter; calibration-half accuracy routing boundary; validation-selected top fraction',
            'reuse frozen round2 full source grid; calibration-half model selection is an explicit additional protocol, comparable to v6 validation endpoint privilege',
            'round1 development validation observed; sequential benchmark development, no new test labels observed'],
        exploration='finite preregistered search; no claim of global optimum; full baselines/ablations/multiseed deferred')
    write_json(pre,dest)
    print(json.dumps(dict(preregistered=str(dest),configs_per_head=len(pre['candidates']),heads=2)),flush=True)


def audit_inputs(root):
    pre=json.loads((root/'PREREGISTERED.json').read_text()); banks=Path(pre['banks_root']); checks={}
    verify_code_manifest(pre)
    provenance=verify_original_banks(banks)
    for p in (REPO/'eargtc/router_v8_calibration_selection').rglob('*.py'):
        for node in ast.walk(ast.parse(p.read_text())):
            names=[node.module or ''] if isinstance(node,ast.ImportFrom) else [n.name for n in node.names] if isinstance(node,ast.Import) else []
            if any(n.startswith(('eargtc.router_bc','eargtc.router_v7','eargtc.router_a20')) for n in names): raise ValueError(f'legacy import {p}')
    for p,h in pre['input_hashes'].items():
        if sha256(p)!=h: raise ValueError(f'input hash changed {p}')
    for d in DATASETS:
        folder=banks/d; man=json.loads((folder/'manifest.json').read_text()); split={k:set(v) for k,v in man['split_ids'].items()}
        assert not (split['train']&split['validation'] or split['train']&split['test'] or split['validation']&split['test'])
        assert split['train']|split['validation']|split['test']==set(range(man['num_nodes']))
        ss=torch.load(folder/'support.pt',weights_only=False); vv=torch.load(folder/'validation.pt',weights_only=False); qq=torch.load(folder/'query.pt',weights_only=False)
        for bank,part in [(ss,'train'),(vv,'validation'),(qq,'test')]: assert set(bank['node_ids'].tolist())<=split[part]
        assert 'state' not in qq
        for f in man['oof_audit']['folds']:
            fit,held=set(f['prototype_fit_ids']),set(f['holdout_ids'])
            assert not fit&held and fit|held==split['train']
        assert set(man['prototype_fit']['validation_and_query_fit_ids'])==split['train']
        predpath=next(p for p in man['input_hashes'] if Path(p).name=='predictions.pt')
        pred=torch.load(predpath,weights_only=False); g=pred['gnn_logits'].float().softmax(1).argmax(1); l=pred['llm_logits'].float().softmax(1).argmax(1)
        tids=torch.tensor(sorted(split['test'])); conflict=tids[g[tids]!=l[tids]]
        assert torch.equal(conflict,qq['node_ids']) and torch.equal(g[conflict],qq['gnn_pred']) and torch.equal(l[conflict],qq['llm_pred'])
        ca,po=validation_halves(arr(vv['node_ids']),d)
        checks[d]=dict(support_states=np.bincount(arr(ss['state']),minlength=3).tolist(),support_ids=ss['node_ids'].tolist(),
            validation_count=len(vv['node_ids']),calibration_count=int(ca.sum()),policy_count=int(po.sum()),
            test_conflict_count=len(qq['node_ids']),historic_split_and_expert_identity=True,prototype_fold_exclusion=True)
    write_json(dict(all_checks_pass=True,checks=checks,no_legacy_imports=True,original_bank_provenance=provenance,original_frozen_sha256=sha256(OLD/'selection/FROZEN.json')),root/'PREFLIGHT_AUDIT.json')
    print(json.dumps(checks),flush=True)


def verify_target_complete(folder,pre):
    from eargtc.router_v8_exploration.experiment import fingerprint
    required={'selection.json','source_selection.json','target_selection.json','model.pt','scores.pt','validation_scores.pt'}
    complete=json.loads((folder/'COMPLETE.json').read_text())
    expected=fingerprint(pre)
    if complete.get('preregistration_sha256')!=expected: raise ValueError('target COMPLETE preregistration mismatch')
    if set(complete.get('files',{}))!=required: raise ValueError('target COMPLETE file inventory mismatch')
    for name,digest in complete['files'].items():
        if sha256(folder/name)!=digest: raise ValueError('target COMPLETE artifact hash mismatch: '+name)
    selection=json.loads((folder/'selection.json').read_text())
    if selection.get('preregistration_sha256')!=expected: raise ValueError('target selection preregistration mismatch')
    return complete


def freeze_all(root):
    prepath=root/'PREREGISTERED.json'; pre=json.loads(prepath.read_text()); banks=Path(pre['banks_root']); paths=[prepath,root/'PREFLIGHT_AUDIT.json']
    verify_code_manifest(pre)
    verify_original_banks(banks)
    audit=json.loads((root/'PREFLIGHT_AUDIT.json').read_text())
    if not audit.get('all_checks_pass') or audit.get('original_frozen_sha256')!=sha256(OLD/'selection/FROZEN.json'):
        raise ValueError('preflight audit missing or stale')
    for group in ('code_hashes','input_hashes'):
        for p,h in pre[group].items():
            if sha256(p)!=h: raise ValueError(f'preregistered hash changed: {p}')
            paths.append(Path(p))
    plan=REPO/'docs/router_v8_calibration_selection_work/PLAN.md'
    if sha256(plan)!=pre['protocol_sha256']: raise ValueError('preregistered protocol changed')
    paths.append(plan)
    for d in DATASETS:
        folder=root/'targets'/d; verify_target_complete(folder,pre)
        paths.append(folder/'COMPLETE.json'); s=json.loads((folder/'selection.json').read_text())
        assert s['target']==d
        v=torch.load(banks/d/'validation.pt',weights_only=False); ca,po=validation_halves(arr(v['node_ids']),d)
        assert set(s['calibration_ids'])==set(arr(v['node_ids'])[ca].tolist())
        assert set(s['policy_ids'])==set(arr(v['node_ids'])[po].tolist())
        query=torch.load(banks/d/'query.pt',weights_only=False); score=torch.load(folder/'scores.pt',weights_only=False)
        validate_scores(query,score)
        paths.extend(folder/f for f in ('selection.json','source_selection.json','target_selection.json','model.pt','scores.pt','validation_scores.pt'))
        man=json.loads((banks/d/'manifest.json').read_text())
        for p,h in man['input_hashes'].items():
            if sha256(p)!=h: raise ValueError(f'upstream changed {p}')
            paths.append(Path(p))
    source=Path(pre['source_root']); verify_frozen(source/'selection/FROZEN.json')
    paths.extend((source/'source').rglob('*.json'))
    if len(list((source/'source').glob('pair_*/COMPLETE.json')))!=10: raise ValueError('source search not complete')
    freeze_paths(paths,root/'selection/FROZEN.json',dict(datasets=list(DATASETS),seed=42,test_labels_accessed=False,protocol='target calibration endpoint selection; source-only reference predeclared'))
    print('All five datasets frozen before test label access.',flush=True)


def dataset_result(scores,state,selection,pre,query):
    """Complete metric bundle, shared with the post-export label-free audit."""
    q=arr(scores['q']); route=arr(scores['route']).astype(int)
    ids=arr(scores['node_ids']); score=arr(scores['selected_score']); state=np.asarray(state)
    metrics=route_metrics(q,state,route,ids)
    policies={k:selection_metrics(q,state,route,np.zeros(len(q),bool) if p['reject_all'] else score>=p['threshold']) for k,p in selection['policies'].items()}
    fraction=selection['top_fraction_policy']['fraction']
    accepted=exact_top_mask(score,ids,fraction=fraction) if fraction is not None else np.zeros(len(q),bool)
    chosen=selection_metrics(q,state,route,accepted)
    chosen['validation_selected_fraction']=fraction
    points={f'top_fraction_{f}':selection_metrics(q,state,route,exact_top_mask(score,ids,fraction=f)) for f in pre['coverage']}
    points.update({f'top_k_{k}':selection_metrics(q,state,route,exact_top_mask(score,ids,count=k)) for k in pre['top_k']})
    expert_confidence=np.where(route==0,arr(query['candidate_g'])[:,0],arr(query['candidate_l'])[:,2])
    control={f'top_fraction_{f}':selection_metrics(q,state,route,exact_top_mask(expert_confidence,ids,fraction=f)) for f in (.05,.1)}
    result=dict(metrics=metrics,policies=policies,chosen_top_fraction=chosen,operating_points=points,
        same_route_expert_confidence_control=control,configs=selection['configs'])
    return result,accepted


def audit_exported_dataset(scores,decisions,selection,saved,pre,query):
    validate_scores(query,scores)
    for key,value in scores.items():
        if key not in decisions or not np.array_equal(arr(value),arr(decisions[key])):
            raise ValueError(f'exported frozen score changed: {key}')
    state=arr(decisions['state'])
    if state.shape!=(len(scores['node_ids']),) or not np.isin(state,[0,1,2]).all():
        raise ValueError('invalid exported state')
    expected,accepted=dataset_result(scores,state,selection,pre,query)
    if expected!=saved: raise ValueError('exported metric bundle mismatch')
    route=arr(scores['route']).astype(int)
    if not np.array_equal(arr(decisions['selected_action']),np.where(accepted,route,2)):
        raise ValueError('exported selected action mismatch')
    pseudo=np.where(accepted,np.where(route==0,arr(scores['gnn_pred']),arr(scores['llm_pred'])),-1)
    if not np.array_equal(arr(decisions['pseudo_label']),pseudo): raise ValueError('exported pseudo label mismatch')
    return dict(saved_metrics_recomputed=True,wilson_and_fixed_points_recomputed=True,
        confidence_control_recomputed=True,selected_actions_and_pseudo_labels_recomputed=True)


def evaluate_locked(root):
    freeze=root/'selection/FROZEN.json'; verify_run_frozen(root)
    pre=json.loads((root/'PREREGISTERED.json').read_text()); banks=Path(pre['banks_root'])
    out=root/'test'; out.mkdir(exist_ok=True); begin_once(out/'EVALUATION_STARTED')
    # No test label loader anywhere above this one-shot token.
    from eargtc.data import _install_pyg_optional_extension_stubs
    _install_pyg_optional_extension_stubs()
    results={}; source_reference={}
    for d in DATASETS:
        man=json.loads((banks/d/'manifest.json').read_text()); sel=json.loads((root/'targets'/d/'selection.json').read_text())
        scores=torch.load(root/'targets'/d/'scores.pt',map_location='cpu',weights_only=False)
        graph=torch.load(man['graph_path'],map_location='cpu',weights_only=False)
        ids=arr(scores['node_ids']); g=arr(scores['gnn_pred']); l=arr(scores['llm_pred']); y=arr(graph.y).reshape(-1)[ids]
        state=np.where(y==g,0,np.where(y==l,1,2)); route=arr(scores['route']).astype(int)
        query=torch.load(banks/d/'query.pt',weights_only=False)
        result,accepted=dataset_result(scores,state,sel,pre,query)
        # Ensure undefined populations can be serialized before saving partial output.
        json.dumps(result,allow_nan=False)
        results[d]=result
        ref_scores=torch.load(Path(pre['source_root'])/'targets'/d/'scores.pt',map_location='cpu',weights_only=False)
        ref_selection=json.loads((Path(pre['source_root'])/'targets'/d/'selection.json').read_text())
        validate_scores(query,ref_scores)
        source_reference[d],_=dataset_result(ref_scores,state,ref_selection,pre,query)
        pseudo=np.where(accepted,np.where(route==0,g,l),-1)
        torch.save(dict(**scores,state=torch.from_numpy(state),selected_action=torch.from_numpy(np.where(accepted,route,2)),pseudo_label=torch.from_numpy(pseudo)),out/f'{d}_decisions.pt')
        write_json(result,out/f'{d}_metrics.json')
        metrics=result['metrics']
        print(json.dumps(dict(dataset=d,trust_auc=metrics['trust_auroc'],pref_auc=metrics['preference_auroc'],all_conflict_accuracy=metrics['full_conflict_pseudo_accuracy'],chosen_top_fraction=result['chosen_top_fraction'])),flush=True)
        del graph
    write_json(results,out/'metrics.json')
    write_json(source_reference,out/'source_only_reference_metrics.json')
    write_json(dict(source_only_reference_metrics_sha256=sha256(out/'source_only_reference_metrics.json'),complete=True,datasets=list(DATASETS),frozen_sha256=sha256(freeze),metrics_sha256=sha256(out/'metrics.json')),out/'COMPLETE.json')


def _verify_result_completion(root):
    complete=json.loads((root/'test/COMPLETE.json').read_text())
    if not complete.get('complete') or complete.get('datasets')!=list(DATASETS): raise ValueError('incomplete results')
    if complete.get('metrics_sha256')!=sha256(root/'test/metrics.json') or complete.get('frozen_sha256')!=sha256(root/'selection/FROZEN.json'):
        raise ValueError('result hash or frozen binding changed')
    if complete.get('source_only_reference_metrics_sha256')!=sha256(root/'test/source_only_reference_metrics.json'):
        raise ValueError('source-only reference result hash changed')
    return complete


def final_audit(root):
    verify_run_frozen(root); _verify_result_completion(root)
    pre=json.loads((root/'PREREGISTERED.json').read_text())
    metrics=json.loads((root/'test/metrics.json').read_text()); checks={}
    result_paths=[root/'test/metrics.json',root/'test/COMPLETE.json',root/'test/source_only_reference_metrics.json']
    source_reference=json.loads((root/'test/source_only_reference_metrics.json').read_text())
    reference_paths=[OLD/'selection/FROZEN.json']
    for d in DATASETS:
        s=torch.load(root/'test'/f'{d}_decisions.pt',weights_only=False)
        scores=torch.load(root/'targets'/d/'scores.pt',weights_only=False)
        selection=json.loads((root/'targets'/d/'selection.json').read_text())
        query=torch.load(Path(pre['banks_root'])/d/'query.pt',weights_only=False)
        checks[d]=audit_exported_dataset(scores,s,selection,metrics[d],pre,query)
        ref_scores=torch.load(Path(pre['source_root'])/'targets'/d/'scores.pt',weights_only=False)
        ref_selection=json.loads((Path(pre['source_root'])/'targets'/d/'selection.json').read_text())
        validate_scores(query,ref_scores)
        expected_reference,_=dataset_result(ref_scores,arr(s['state']),ref_selection,pre,query)
        if expected_reference!=source_reference[d]: raise ValueError('source-only reference metrics mismatch')
        checks[d]['source_only_reference_recomputed']=True
        individual=root/'test'/f'{d}_metrics.json'
        if json.loads(individual.read_text())!=metrics[d]: raise ValueError('individual metric mismatch')
        old_path=OLD/'test'/f'{d}_decisions.pt'; old=torch.load(old_path,weights_only=False)
        for k in ('node_ids','gnn_pred','llm_pred','state'):
            if not np.array_equal(arr(s[k]),arr(old[k])): raise ValueError(f'original conflict identity changed: {d}/{k}')
        checks[d]['original_v8_conflict_identity']=True
        result_paths.extend([root/'test'/f'{d}_decisions.pt',individual]);reference_paths.append(old_path)
    # All original frozen V8 dependencies remain byte-identical; comparison report
    # inputs created after the old freeze are explicitly bound in this final audit.
    verify_frozen(OLD/'selection/FROZEN.json')
    if DATASETS: reference_paths.append(OLD/'test/metrics.json')
    write_json(dict(all_checks_pass=True,checks=checks,original_v8_unchanged=True,
        frozen_sha256=sha256(root/'selection/FROZEN.json'),metrics_sha256=sha256(root/'test/metrics.json'),
        result_hashes={str(p.resolve()):sha256(p) for p in result_paths},
        reference_hashes={str(p.resolve()):sha256(p) for p in reference_paths}),root/'FINAL_AUDIT.json')
    print('Final metrics, operating points, pseudo labels, population and frozen identity audit passed.',flush=True)


def verify_report_ready(root):
    verify_run_frozen(root);_verify_result_completion(root)
    audit=json.loads((root/'FINAL_AUDIT.json').read_text())
    if not audit.get('all_checks_pass') or set(audit.get('checks',{}))!=set(DATASETS):
        raise ValueError('successful final audit required')
    if audit.get('frozen_sha256')!=sha256(root/'selection/FROZEN.json') or audit.get('metrics_sha256')!=sha256(root/'test/metrics.json'):
        raise ValueError('final audit binding changed')
    for group in ('result_hashes','reference_hashes'):
        if not audit.get(group): raise ValueError('final audit has no bound file hashes')
        for path,digest in audit[group].items():
            if not Path(path).is_file() or sha256(path)!=digest: raise ValueError(f'final audit hash changed: {path}')
    verify_frozen(OLD/'selection/FROZEN.json')
    return audit


def fmt(x,pct=False):
    return '—' if x is None else (f'{100*x:.2f}%' if pct else f'{x:.4f}')


def report(root):
    verify_report_ready(root)
    results=json.loads((root/'test/metrics.json').read_text()); pre=json.loads((root/'PREREGISTERED.json').read_text())
    old=json.loads((OLD/'test/metrics.json').read_text())
    text=['# Router-v8：source 筛选与目标校准集交叉验证选择双头端点','',
        '完成五图、固定历史 5-shot seed42 的连续探索。第一轮和第二轮 source-only 开发端点均封存，未分别打开新 test。本轮主协议在测试前固定为目标 calibration 端点选择；第二轮 source-only 端点仅作预先声明的参考，同一次读取测试标签进行评价。开发 validation 已观察，不是全新未见数据。这里“最优”指 source 指标构成有限 shortlist 后，按目标 calibration 的 crossfit BCE 门限和固定原始 logit AUROC 选出的配置，未按 test 选择候选或重试。完整 baselines、消融与多 seed 依用户要求暂缓。','',
        '## 协议与输入','',
        '- 复用原 V8 corrected structural-OOF 证据、GNN/LLM 专家、SSL 原型及历史 train/validation/test IDs；test 为 train/validation 的升序补集。',
        '- 对每个最终目标单独 source-LODO：主体只在其余四图中的三图 OOF support 训练，第四图用自己的 OOF support 适配，在其 validation 的历史 hash 两半上交叉校准/评价，反向再做一次。两个半区 AUROC 分别计算后平均，避免跨校准器偏移的混淆。',
        '- 此轮明确改变首版“仅源 support 选模”协议：源 validation 允许评价和模拟校准，未进入共享主体训练。直接对旧 OOF 表逐行 LOO 无法排除其他行专家或原型依赖被留出标签，因此采用独立 validation 评价完整部署。最终目标 calibration half 现在允许从 source 派生有限候选中选择端点；该权限变化单独声明，policy half 不参与选头。',
        '- Trust 学习 any-correct；Preference 仅在 G/L decisive 行学习 GNN-only。分别使用域等权 BCE、可选同域 pairwise ranking 和 L2，随后组合 q=(c*r,c*(1-r),1-c)。这不是首版共享梯度/三事件 NLL 的完全同一实现。',
        '- 表示包含概率核心、上下文交互、可选结构/原型，以及基于固定 cutpoints 的浅层 additive stumps。Preference 允许直接专家分布形状，不再受共享列相减抵消限制；因此不声称完整候选交换等变。standard/rank 仅拟合源 support；support_rank 使用各域自己的 support 为 ECDF reference，query 不参与统计拟合。',
        '- 目标 support 学习 source-local 系数偏差 SVD 的 rank0/1/2 adapter 和 bias。目标 calibration half 先用内部 LOO（不超过256个conflict）或确定性五折交叉验证选择端点，再拟合正斜率概率校准及准确率路由阈值；独立 policy half 只选择接受比例/Wilson 阈值。接受分数取实际被选专家的 q。',
        '- 五图模型、分数、路由、policy、源结果、代码及上游数据哈希全部冻结后，测试标签只由单次 evaluator 读取。历史测试已见，本轮是复用 benchmark，不能声称完全未见测试或无偏算法级探索。','',
        f"复用第二轮每头 {len(pre['candidates'])} 配置的 source-LODO。每个表示/变换 family 取 source 最低 BCE 与 family-BCE门限内最高 AUROC 候选，加入 source-global 最佳，去重后每头至多25个。目标 calibration 选择规则：honest crossfit BCE≤最小BCE+0.05，再最大固定原始logit calibration AUROC；任一类不足5行则保留source-global候选。最终仅对选定头在完整calibration half拟合affine。所有数值工作经 Slurm。",'',
        '## 双模块与全部 disagreement 路由','',
        '| 数据集 | Trust AUROC | Preference AUROC | Preference G/L accuracy | 全 disagreement pseudo accuracy | G-only recall | L-only recall |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for d in DATASETS:
        m=results[d]['metrics']; text.append('| '+d+' | '+' | '.join(fmt(m[k],pct) for k,pct in [('trust_auroc',False),('preference_auroc',False),('preference_accuracy',True),('full_conflict_pseudo_accuracy',True),('gnn_only_recall',True),('llm_only_recall',True)])+' |')
    reference=json.loads((root/'test/source_only_reference_metrics.json').read_text())
    text += ['','## 预先声明的 source-only 参考','',
        '主协议在test前已固定；下面同时报告已冻结source-only版本，不根据test优劣在两者之间选择。差异包括目标校准集端点选择权限，不能解释为纯架构增益。','',
        '| 数据集 | Source-only Trust AUC | 主协议 Trust AUC | Source-only Preference AUC | 主协议 Preference AUC | Source-only 全冲突accuracy | 主协议全冲突accuracy |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for d in DATASETS:
        a=reference[d]['metrics'];b=results[d]['metrics']
        text.append('| '+d+' | '+' | '.join([fmt(a['trust_auroc']),fmt(b['trust_auroc']),fmt(a['preference_auroc']),fmt(b['preference_auroc']),fmt(a['full_conflict_pseudo_accuracy'],True),fmt(b['full_conflict_pseudo_accuracy'],True)])+' |')
    text+=['','## validation 选择的伪标签输出','',
        '比例只在 policy validation 上选择：至少10行，候选中接近最好 precision（容差1个百分点）的最大比例。下表不是从 test 曲线挑出的最优点，也不具备 test 风险保证。','',
        '| 数据集 | 选定比例 | 接受数 | disagreement pseudo accuracy | 接受 neither 错误 | 接受选错专家错误 | Wilson10% 接受数 / accuracy |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for d in DATASETS:
        p=results[d]['chosen_top_fraction']; e=p['error_decomposition']; w=results[d]['policies']['0.1']
        text.append(f"| {d} | {fmt(p['validation_selected_fraction'],True)} | {p['accepted_count']} | {fmt(p['pseudo_accuracy'],True)} | {e['accepted_neither']} | {e['accepted_wrong_expert']} | {w['accepted_count']} / {fmt(w['pseudo_accuracy'],True)} |")
    text+=['','Wilson 为 validation 上单侧95%误差上界≤10%、最少20行；不可行则 reject-all，accuracy 未定义。','',
        '## 固定覆盖率与首版 V8 比较','',
        '| 数据集 | 新 Top5% | 旧 Top5% | 新 Top10% | 旧 Top10% | 新 Top25% | 新 Top50% |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for d in DATASETS:
        p=results[d]['operating_points']; o=old[d]['operating_points']
        cells=[p['top_fraction_0.05'],o['top_fraction_0.05'],p['top_fraction_0.1'],o['top_fraction_0.1'],p['top_fraction_0.25'],p['top_fraction_0.5']]
        text.append('| '+d+' | '+' | '.join(f"{fmt(v['pseudo_accuracy'],True)} / {v['accepted_count']}" for v in cells)+' |')
    text+=['','以上在完全相同 disagreement 人口与精确相同接受数上比较。路由改变也会改变 selected-correctness target，因此不是纯 Trust 消融。','',
        '## 源选择与目标开发结果','']
    for d in DATASETS:
        sel=json.loads((root/'targets'/d/'selection.json').read_text()); src=json.loads((root/'targets'/d/'source_selection.json').read_text())
        text += [f'### {d}','',f"最终配置：`{json.dumps(sel['configs'],ensure_ascii=False,sort_keys=True)}`。",'',
            f"选择详情：`targets/{d}/source_selection.json`；适配、校准及 policy 独立开发结果：`targets/{d}/selection.json`。",'']
    text+=['## 限制与解释','',
        '本轮同时改变源开发标签角色、表示、任务优化、适配、目标校准集端点选择与路由策略；不能把提高归因于单个机制。目标calibration选择属于validation模型选择，不是仅support训练的few-shot Router；专家本身仍维持原5-shot设置。低样本源 support、数据集间迁移与类不平衡仍可能造成部分图退步。Trust AUROC 衡量是否至少一个专家正确，与输出伪标签准确率有不同含义；两者均须报告。',
        '所有预定覆盖率/固定 k、BCE/Brier/AUPRC、两侧 recall、AURC、same-route expert-confidence 轻量对照与错误分解见 test/metrics.json。没有利用 test 类别频率或长尾统计拟合权重。冻结专家的既有 transductive/validation 配置选择仍是继承的限制。',
        '审计见 PREFLIGHT_AUDIT.json、FINAL_AUDIT.json、selection/FROZEN.json；作业与资源记录另附。','']
    (root/'REPORT.md').write_text('\n'.join(text))
    print(str(root/'REPORT.md'),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preregister','audit','freeze','evaluate','final-audit','report']);parser.add_argument('--root',type=Path,required=True)
    a=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Run numerical work through Slurm')
    torch.set_num_threads(1)
    {'preregister':preregister,'audit':audit_inputs,'freeze':freeze_all,'evaluate':evaluate_locked,'final-audit':final_audit,'report':report}[a.stage](a.root.resolve())

if __name__=='__main__': main()
