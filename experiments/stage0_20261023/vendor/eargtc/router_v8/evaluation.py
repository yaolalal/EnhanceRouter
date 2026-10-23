"""All-five freeze and one-shot locked evaluation. Only this module reads test labels."""
import json
from pathlib import Path
import numpy as np
import torch
from eargtc.data import _install_pyg_optional_extension_stubs
from .artifacts import write_json,sha256,freeze_paths,verify_frozen,begin_once
from .experiment import DATASETS
from .metrics import evaluate,selective_metrics
from .policy import apply_policy,top_mask


def freeze_all(root):
    root=Path(root); pre_path=root/'PREREGISTERED.json'
    pre=json.loads(pre_path.read_text()); paths=[pre_path]
    for filename,expected in pre['code_hashes'].items():
        if sha256(filename)!=expected: raise ValueError(f'preregistered code changed: {filename}')
        paths.append(Path(filename))
    for name,expected in pre['input_hashes'].items():
        if sha256(name)!=expected: raise ValueError(f'preregistered input hash mismatch: {name}')
        paths.append(Path(name))
    for d in DATASETS:
        directory=root/'targets'/d
        record=json.loads((directory/'selection.json').read_text())
        if record['target']!=d: raise ValueError('target mismatch')
        if set(record['calibration_ids']) & set(record['policy_ids']): raise ValueError('validation overlap')
        paths.extend(directory/f for f in ('model.pt','scores.pt','validation_scores.pt','source_selection.json','selection.json'))
        query=torch.load(root/'banks'/d/'query.pt',map_location='cpu',weights_only=False)
        scores=torch.load(directory/'scores.pt',map_location='cpu',weights_only=False)
        if any(k in query for k in ('state','labels','y','y_true','correctness')): raise ValueError('query contains labels')
        for k in ('node_ids','gnn_pred','llm_pred'):
            if not torch.equal(query[k],scores[k]): raise ValueError(f'query/score mismatch: {d}:{k}')
        if (query['gnn_pred']==query['llm_pred']).any(): raise ValueError('agreement in conflict bank')
        q=scores['q'].numpy()
        if not np.isfinite(q).all() or (q<0).any() or not np.allclose(q.sum(1),1,atol=1e-7):
            raise ValueError('invalid event probabilities')
        manifest=json.loads((root/'banks'/d/'manifest.json').read_text())
        if str(Path(manifest['graph_path']).resolve()) not in manifest['input_hashes']:
            raise ValueError('graph label artifact is not hashed')
        for filename,expected in manifest['input_hashes'].items():
            if sha256(filename)!=expected: raise ValueError(f'upstream input changed: {filename}')
            paths.append(Path(filename))
    repo=Path(__file__).resolve().parents[2]
    paths.extend((repo/'eargtc'/'router_v8').glob('*.py'))
    paths.extend((repo/'scripts'/'router_v8').glob('*.py'))
    freeze_paths(paths,root/'selection'/'FROZEN.json',dict(datasets=DATASETS,seed=42,test_labels_accessed=False))


def locked_evaluate(root):
    root=Path(root); frozen=root/'selection'/'FROZEN.json'
    verify_frozen(frozen)
    pre=json.loads((root/'PREREGISTERED.json').read_text())
    output=root/'test'; output.mkdir(parents=True,exist_ok=True)
    # Claim token precedes reading any graph labels. A partial failure may not silently rerun.
    begin_once(output/'EVALUATION_STARTED')
    results={}
    _install_pyg_optional_extension_stubs()
    for d in DATASETS:
        manifest=json.loads((root/'banks'/d/'manifest.json').read_text())
        record=json.loads((root/'targets'/d/'selection.json').read_text())
        scores=torch.load(root/'targets'/d/'scores.pt',map_location='cpu',weights_only=False)
        graph=torch.load(manifest['graph_path'],map_location='cpu',weights_only=False)
        ids=scores['node_ids'].long(); g=scores['gnn_pred'].long(); l=scores['llm_pred'].long()
        labels=graph.y.reshape(-1).long()[ids]
        state=torch.where(labels==g,0,torch.where(labels==l,1,2)).numpy()
        q=scores['q'].numpy(); ids_np=ids.numpy(); route=q[:,:2].argmax(1); score=q[:,:2].max(1)
        metric=evaluate(q,state,ids_np)
        policies={key:selective_metrics(q,state,apply_policy(q,policy)) for key,policy in record['policies'].items()}
        points={}
        def collect(name,accept):
            points[name]=selective_metrics(q,state,np.where(accept,route,2))
        for f in pre['coverage']:
            collect(f'top_fraction_{f}',top_mask(score,ids_np,fraction=f))
        for k in pre['top_k']:
            collect(f'top_k_{k}',top_mask(score,ids_np,count=k))
        for threshold in pre['thresholds']:
            collect(f'threshold_{threshold}',score>=threshold)
        for rho in pre['costs']:
            accept=score>=1-rho; collect(f'reject_cost_{rho}',accept)
            # Guide's reporting utility is distinct from decision loss (0/1/rho).
            points[f'reject_cost_{rho}']['utility']=float(np.where(accept,np.where(route==state,1.,-1.),-rho).mean())
        primary=policies['0.1']; validation=record['validation_metrics']['policies']['0.1']
        gap=dict(coverage=primary['coverage']-validation['coverage'],
                 risk=primary['risk']-validation['risk'] if primary['risk'] is not None and validation['risk'] is not None else None)
        results[d]=dict(metrics=metric,policies=policies,operating_points=points,
                        validation_to_test_gap=gap,config=record['config'],negative_slopes=record['negative_slopes'])
        actions=apply_policy(q,record['policies']['0.1'])
        pseudo=np.where(actions==0,g.numpy(),np.where(actions==1,l.numpy(),-1))
        torch.save(dict(**scores,state=torch.from_numpy(state),primary_action=torch.from_numpy(actions),
                        pseudo_label=torch.from_numpy(pseudo)),output/f'{d}_decisions.pt')
        write_json(results[d],output/f'{d}_metrics.json')
        print(json.dumps(dict(dataset=d,trust_auc=metric['trust_auroc'],preference_auc=metric['preference_auroc'],
            full_conflict_pseudo_accuracy=metric['full_conflict_pseudo_accuracy'],primary=primary)),flush=True)
    write_json(results,output/'metrics.json')
    write_json(dict(complete=True,datasets=DATASETS,frozen_sha256=sha256(frozen),
        metrics_sha256=sha256(output/'metrics.json')),output/'COMPLETE.json')
    return results
