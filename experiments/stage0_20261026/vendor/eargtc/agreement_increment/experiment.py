"""Calibration-only search and policy-only finalization of independent selectors."""
import gc
import os
import platform
import time
from pathlib import Path
import joblib
import numpy as np
import torch

from .artifacts import (DATASETS, code_files, read_json, write_json, sha256, verify_record,
                        require_slurm, preservation_inventory)
from .policy import calibration_folds, selection_metrics, selection_key, select_policy, top_mask, FRACTIONS


def features(bank, mask=None):
    return {k: bank[k].numpy() if mask is None else bank[k].numpy()[mask] for k in ('tab', 'g', 'l')}


def _scope_config(base, scope):
    return dict(id=f"{base['id']}__{scope}", model=base, scope=scope)


def register(root):
    require_slurm()
    from .models import candidate_configs
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    if (root/'CONFIG.json').exists():
        raise FileExistsError('configuration already registered')
    configs = []
    for base in candidate_configs():
        scopes = ['target'] if base['family'] == 'baseline' else ['target', 'pooled']
        configs.extend(_scope_config(base, scope) for scope in scopes)
    ids = [c['id'] for c in configs]
    if not configs or len(ids) != len(set(ids)): raise ValueError('empty or duplicate candidate IDs')
    config = dict(schema='agreement_increment_exploration_v1', datasets=DATASETS, seed=42,
        candidates=configs, fractions=FRACTIONS, min_policy_accepted=20,
        selection='calibration fivefold OOF mean precision at eligible 5/10/20% then BCE then ID',
        small_calibration_fallback='no eligible ranking point: BCE then ID',
        acceptance='policy highest precision then Wilson lower bound then greater fraction',
        pooled_weight='target calibration 0.5; other four validation domains share 0.5 equally',
        model_scores='ranking scores; no claim of calibrated correctness probabilities',
        label_roles={'source':'other four official validation agreement',
                     'calibration':'target model/hyperparameter selection and refit',
                     'policy':'target acceptance fraction only, after model choice',
                     'query':'features for inference only; truth in final evaluation only'},
        independent_of_v4=True, query_consistency_training=False,
        limitations=['historical benchmark has been previously evaluated',
                    'OOF hyperparameter search is development selection, not unbiased evaluation',
                    'policy precision is optimized over a fixed grid and is not an accuracy guarantee',
                    'graph rows are correlated; binomial intervals are descriptive'],
        slurm_job_id=os.environ['SLURM_JOB_ID'])
    write_json(root/'CONFIG.json', config)
    preservation_inventory(root)
    files = code_files()+[root/'CONFIG.json']
    for dataset in DATASETS:
        files.extend(root/'banks'/dataset/name for name in ('validation.pt','query.pt','manifest.json'))
        metadata_path = Path(__file__).resolve().parents[2]/'runs/joint_direct_numbered_router_export_20260813_v1'/dataset/'safe_input_metadata.json'
        metadata = read_json(metadata_path)
        graph_path = Path(metadata['graph_path'])
        if sha256(graph_path) != metadata['graph_path_sha256']:
            raise ValueError('historical graph hash mismatch before registration')
        files.extend([metadata_path,graph_path])
    write_json(root/'REGISTRATION.json', dict(files={str(p):sha256(p) for p in files},
        slurm_job_id=os.environ['SLURM_JOB_ID'], python=platform.python_version(),
        torch=torch.__version__, test_labels_accessed=False))
    print(f'registered {len(configs)} candidates', flush=True)


def _validate_open(root):
    if (root/'FROZEN.json').exists() or (root/'evaluation/STARTED.json').exists():
        raise RuntimeError('training/selection forbidden after freeze or evaluation')
    verify_record(read_json(root/'REGISTRATION.json'))


def _training_data(cal_bank, source_banks, mask, scope):
    target = features(cal_bank, mask)
    y = cal_bank['target'].numpy()[mask].astype(bool)
    if scope == 'target':
        return target, y, np.ones(len(y), float)
    nonempty = [b for b in source_banks if len(b['ids'])]
    if not nonempty: return target, y, np.ones(len(y), float)
    pieces = [target]+[features(b) for b in nonempty]
    labels = [y]+[b['target'].numpy().astype(bool) for b in nonempty]
    weights = [np.full(len(y), .5/max(len(y), 1))]
    weights += [np.full(len(z), .5/len(nonempty)/len(z)) for z in labels[1:]]
    joined = {k:np.concatenate([p[k] for p in pieces], axis=0) for k in target}
    weight = np.concatenate(weights); weight *= len(weight)/weight.sum()
    return joined, np.concatenate(labels), weight


def _load_training(root, dataset):
    from .data import load_bank
    target = load_bank(root, dataset, 'validation')
    cal = target['cal_mask'].numpy().astype(bool)
    cal_bank = {k: v[cal] if isinstance(v, torch.Tensor) and v.ndim else v for k,v in target.items()}
    # Only the calibration slice is returned to search; target policy labels are discarded here.
    sources = [load_bank(root, d, 'validation') for d in DATASETS if d != dataset]
    return cal_bank, sources


def search(root, dataset, device='cpu'):
    require_slurm(); root = Path(root).resolve(); _validate_open(root)
    from .models import fit_predict
    dest = root/'targets'/dataset; dest.mkdir(parents=True, exist_ok=True)
    if (dest/'selection.json').exists() or (dest/'POLICY_STARTED.json').exists():
        raise RuntimeError('cannot search after target policy access')
    if (dest/'SEARCH_SEALED.json').exists():
        verify_record(read_json(dest/'SEARCH_SEALED.json'))
        print(f'{dataset} search already complete and verified', flush=True)
        return
    config = read_json(root/'CONFIG.json')
    cal, sources = _load_training(root, dataset)
    ids = cal['ids'].numpy(); y = cal['target'].numpy().astype(bool)
    folds = calibration_folds(ids, dataset)
    source_info = {b['dataset']:dict(count=len(b['ids']),
                    bank_sha256=sha256(root/'banks'/b['dataset']/'validation.pt')) for b in sources}
    write_json(dest/'cv_manifest.json', dict(dataset=dataset, calibration_ids=ids.tolist(),
        folds=folds.tolist(), sources=source_info, target_policy_used=False,
        target_query_labels_used=False, registration_sha256=sha256(root/'REGISTRATION.json')))
    trials = []; outputs = {}
    cache = dest/'cv'; cache.mkdir(exist_ok=True)
    # Fold training data are shared across candidate architectures and never contain target holdouts.
    training = {(scope, int(f)): _training_data(cal, sources, folds != f, scope)
                for scope in ('target', 'pooled') for f in np.unique(folds)}
    predicts = {int(f):features(cal, folds == f) for f in np.unique(folds)}
    for number, candidate in enumerate(config['candidates']):
        cid = candidate['id']; record_path = cache/f'{cid}.json'; score_path = cache/f'{cid}.npz'
        if record_path.exists() and score_path.exists():
            row = read_json(record_path)
            oof = np.load(score_path)['score']
            if row['registration_sha256'] != sha256(root/'REGISTRATION.json') or oof.shape != y.shape:
                raise ValueError('incompatible resumed trial')
            if row['oof_sha256'] != sha256(score_path) or row['id'] != cid or row['model'] != candidate['model'] or row['scope'] != candidate['scope']:
                raise ValueError('resumed OOF scores/config changed')
            if row['metrics'] != selection_metrics(oof, y, ids):
                raise ValueError('resumed metrics disagree with current calibration labels')
        else:
            started = time.monotonic(); oof = np.empty(len(ids), dtype=float)
            for f in np.unique(folds):
                train_x, train_y, weights = training[(candidate['scope'], int(f))]
                predicted, model = fit_predict(candidate['model'], train_x, train_y,
                                              predicts[int(f)], sample_weight=weights,
                                              seed=config['seed'], device=device)
                oof[folds == f] = predicted
                del model
            row = dict(candidate, metrics=selection_metrics(oof, y, ids),
                       seconds=time.monotonic()-started,
                       registration_sha256=sha256(root/'REGISTRATION.json'))
            np.savez_compressed(score_path, score=oof)
            row['oof_sha256'] = sha256(score_path)
            write_json(record_path, row)
        trials.append(row); outputs[cid] = oof
        write_json(dest/'search_progress.json', dict(completed=len(trials), total=len(config['candidates']),
                   latest=row, current_best=min(trials, key=selection_key)))
        print(f"{dataset} {number+1}/{len(config['candidates'])} {cid} top={row['metrics']['top_precision']} bce={row['metrics']['bce']:.5f}", flush=True)
    chosen = min(trials, key=selection_key)
    write_json(dest/'trials.json', trials)
    np.savez_compressed(dest/'oof.npz', ids=ids, target=y, folds=folds, **outputs)
    write_json(dest/'chosen.json', dict(chosen=chosen, trial_count=len(trials),
               calibration_ids=ids.tolist(), target_policy_used=False, query_truth_used=False))
    write_json(dest/'SEARCH_SEALED.json', dict(files={str(dest/name):sha256(dest/name)
        for name in ('chosen.json','trials.json','oof.npz','cv_manifest.json')},
        registration_sha256=sha256(root/'REGISTRATION.json'), query_truth_used=False, policy_used=False))
    print(f"CHOSEN {dataset} {chosen['id']}", flush=True)


def finalize(root, dataset, device='cpu'):
    require_slurm(); root = Path(root).resolve(); _validate_open(root)
    from .data import load_bank
    from .models import fit_predict, predict_model, candidate_configs
    dest = root/'targets'/dataset
    if (dest/'selection.json').exists(): raise FileExistsError('target already finalized')
    seal = read_json(dest/'SEARCH_SEALED.json'); verify_record(seal)
    if seal['registration_sha256'] != sha256(root/'REGISTRATION.json'):
        raise ValueError('search registration changed')
    chosen = read_json(dest/'chosen.json')['chosen']
    trials = read_json(dest/'trials.json')
    registered = read_json(root/'CONFIG.json')['candidates']
    if {r['id'] for r in trials} != {r['id'] for r in registered} or chosen != min(trials,key=selection_key):
        raise ValueError('chosen model is not the preregistered calibration winner')
    marker = dict(search_seal_sha256=sha256(dest/'SEARCH_SEALED.json'),
                  registration_sha256=sha256(root/'REGISTRATION.json'), chosen_id=chosen['id'])
    if (dest/'POLICY_STARTED.json').exists():
        if read_json(dest/'POLICY_STARTED.json') != marker:
            raise ValueError('resumed finalization changed its pre-policy model or inputs')
    else:
        write_json(dest/'POLICY_STARTED.json', marker)
    cal, sources = _load_training(root, dataset)
    tx, ty, weights = _training_data(cal, sources, np.ones(len(cal['ids']), bool), chosen['scope'])
    validation = load_bank(root, dataset, 'validation')
    vscore, model = fit_predict(chosen['model'], tx, ty, features(validation),
                               sample_weight=weights, seed=42, device=device)
    del tx, ty, weights, sources
    gc.collect()
    query = load_bank(root, dataset, 'query')
    qscore = np.asarray(predict_model(model, features(query)), float)
    pm = validation['policy_mask'].numpy().astype(bool)
    policy = select_policy(np.asarray(vscore)[pm], validation['target'].numpy()[pm], validation['ids'].numpy()[pm])
    keep = top_mask(qscore, query['ids'].numpy(), policy['fraction'])
    baselines = {}
    # Fixed, preregistered feature baselines are evaluated, never chosen with query truth.
    for c in candidate_configs():
        if c['family'] == 'baseline':
            score, _ = fit_predict(c, features(cal), cal['target'].numpy(), features(query), seed=42, device='cpu')
            baselines[c['id']] = torch.as_tensor(np.asarray(score, float))
    joblib.dump(model, dest/'model.joblib', compress=3)
    scores = dict(node_ids=query['ids'], gnn_pred=query['gnn_pred'], llm_pred=query['llm_pred'],
                  selected_score=torch.as_tensor(qscore), baselines=baselines)
    torch.save(scores, dest/'scores.pt')
    torch.save(dict(node_ids=validation['ids'], score=torch.as_tensor(np.asarray(vscore,float)),
               target=validation['target'], cal_mask=validation['cal_mask'], policy_mask=validation['policy_mask']),
               dest/'validation_scores.pt')
    output = root/'pipeline'; output.mkdir(exist_ok=True)
    pipeline = dict(schema='agreement_increment_pipeline_v1', dataset=dataset, node_ids=query['ids'],
        gnn_pred=query['gnn_pred'], llm_pred=query['llm_pred'], selected_score=torch.as_tensor(qscore),
        accepted_mask=torch.as_tensor(keep),
        pseudo_label=torch.where(torch.as_tensor(keep), query['gnn_pred'], -1))
    torch.save(pipeline, output/f'{dataset}.pt')
    record = dict(dataset=dataset, chosen=chosen, policy=policy,
        calibration_ids=validation['ids'][validation['cal_mask']].tolist(),
        policy_ids=validation['ids'][validation['policy_mask']].tolist(),
        query_count=len(query['ids']), accepted_count=int(keep.sum()),
        independent_of_v4=True, query_truth_used=False,
        source_datasets=[d for d in DATASETS if d != dataset] if chosen['scope']=='pooled' else [],
        model_sha256=sha256(dest/'model.joblib'), score_sha256=sha256(dest/'scores.pt'),
        pipeline_sha256=sha256(output/f'{dataset}.pt'), slurm_job_id=os.environ['SLURM_JOB_ID'])
    write_json(dest/'selection.json', record)
    print(f"FINALIZED {dataset} {chosen['id']} fraction={policy['fraction']} policy_precision={policy['precision']} count={int(keep.sum())}", flush=True)
