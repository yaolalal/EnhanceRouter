"""Source-derived finite endpoint selection using target calibration labels only.

Immutable v2 models/features/adapters are reused. Policy labels are accessed only
once endpoint choice, final affine calibration and routing have been finalized.
Query is not deserialized until acceptance policy fitting is complete.
"""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from eargtc.router_v8_exploration import experiment as v2
from eargtc.router_v8_exploration.core import (
    HEADS, array, binary_target, complexity, select_configuration,
    validation_halves, fit_affine, apply_affine, binary_metrics, fit_adapter,
    routing_boundary, score_payload, fit_risk_policy, fit_top_fraction,
)
from eargtc.router_v8.artifacts import verify_frozen

DATASETS = v2.DATASETS
load_source_trials = v2.load_source_trials
_load_bank = v2._load_bank
sha256 = v2.sha256
fingerprint = v2.fingerprint
write_json = v2.write_json
save_torch = v2.save_torch
ids_hash = v2.ids_hash
_TARGET_WORKER = None


def _finite_metrics(row, names):
    for name in names:
        value = row.get(name)
        if value is not None and (not isinstance(value, (int, float)) or not np.isfinite(value)):
            raise ValueError(f'nonfinite metric: {name}')


def source_shortlist(selection):
    """At most two source choices per representation/view, plus source-global."""
    families = {}
    for row in selection['trials']:
        _finite_metrics(row, ('macro_bce', 'macro_auc', 'macro_brier'))
        cfg = row['config']
        families.setdefault((cfg['representation'], cfg['view']), []).append(row)
    if len(families) > 12:
        raise ValueError('source shortlist exceeds preregistered families')
    chosen = {}
    for key in sorted(families):
        usable = [r for r in families[key] if r['macro_bce'] is not None]
        if not usable:
            continue
        minimum = min(usable, key=lambda r: (r['macro_bce'], complexity(r['config']), r['config']['id']))
        eligible = [r for r in usable if r['macro_bce'] <= minimum['macro_bce'] + .05 + 1e-12]
        ranked = min(eligible, key=lambda r: (
            -(r['macro_auc'] if r['macro_auc'] is not None else -1.),
            r['macro_bce'], complexity(r['config']), r['config']['id']))
        for row in (minimum, ranked):
            chosen[row['config']['id']] = row
    best = selection['best']
    if best['config']['id'] not in {r['config']['id'] for r in selection['trials']}:
        raise ValueError('source-global candidate absent from trials')
    chosen[best['config']['id']] = best
    if len(chosen) > 25:
        raise ValueError('shortlist limit exceeded')
    return [chosen[key] for key in sorted(chosen)]


def calibration_folds(node_ids, dataset):
    """Label-blind full-conflict folds, before either binary head's masking."""
    ids = array(node_ids)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('calibration IDs must be unique and one-dimensional')
    if len(ids) <= 256:
        return [ids == ident for ident in np.sort(ids)]
    assignments = np.array([int.from_bytes(hashlib.sha256(
        f'{dataset}|v8-calibration-selection|{int(ident)}'.encode()).digest(), 'big') % 5 for ident in ids])
    if len(np.unique(assignments)) != 5:
        # Label-blind fallback also invariant to incoming row order.
        assignments[np.argsort(ids, kind='stable')] = np.arange(len(ids)) % 5
    return [assignments == fold for fold in range(5)]


def crossfit_calibration(raw_logits, state, node_ids, dataset, head):
    """Honest per-row affine probabilities, with fixed-model raw-logit AUROC."""
    raw, state, ids = np.asarray(raw_logits, float), array(state), array(node_ids)
    if raw.ndim != 1 or raw.shape != state.shape or raw.shape != ids.shape or not np.isfinite(raw).all():
        raise ValueError('invalid finite crossfit inputs')
    if not np.isin(state, [0, 1, 2]).all():
        raise ValueError('invalid calibration state')
    full_folds = calibration_folds(ids, dataset)
    decisive, y = binary_target(state, head)
    logits = np.full(len(ids), np.nan)
    audit = []
    for index, held in enumerate(full_folds):
        fit_mask, fit_y = binary_target(state[~held], head)
        fit_ids = ids[~held][fit_mask]
        fit = fit_affine(raw[~held][fit_mask], fit_y)
        evaluation = held & decisive
        logits[evaluation] = apply_affine(raw[evaluation], fit)
        audit.append(dict(fold=index, holdout_ids=ids[held].tolist(),
            evaluation_ids=ids[evaluation].tolist(), fit_ids=fit_ids.tolist(),
            holdout_ids_sha256=ids_hash(ids[held]), fit_ids_sha256=ids_hash(fit_ids),
            evaluation_ids_sha256=ids_hash(ids[evaluation]), calibration=fit))
    honest = logits[decisive]
    if not np.isfinite(honest).all():
        raise ValueError('crossfit logits must be finite with complete coverage')
    measured = binary_metrics(honest, y)
    raw_measured = binary_metrics(raw[decisive], y)
    summary = dict(count=len(y), positive_count=measured['positive_count'],
        negative_count=measured['negative_count'], full_calibration_count=len(ids),
        raw_auc=raw_measured['auc'], raw_bce=raw_measured['bce'], raw_brier=raw_measured['brier'],
        crossfit_bce=measured['bce'], crossfit_brier=measured['brier'],
        fold_count=len(full_folds), method='leave_one_out' if len(ids) <= 256 else 'hash_fivefold',
        calibration_ids_sha256=ids_hash(ids), evaluation_ids_sha256=ids_hash(ids[decisive]))
    return dict(summary=summary, oof_logits=honest.tolist(), oof_probabilities=expit(honest).tolist(),
        evaluation_ids=ids[decisive].tolist(), folds=audit)


def select_target_candidate(rows, source_global_id):
    """BCE gate, fixed-raw ranking, then probability/source/complexity ties."""
    if not rows or len(rows) > 25 or len({r['config']['id'] for r in rows}) != len(rows):
        raise ValueError('invalid bounded candidate table')
    by_id = {r['config']['id']: r for r in rows}
    if source_global_id not in by_id:
        raise ValueError('source-global candidate missing')
    for row in rows:
        _finite_metrics(row['calibration'], ('crossfit_bce', 'crossfit_brier', 'raw_auc'))
        _finite_metrics(row['source'], ('macro_auc', 'macro_bce'))
    counts = {(r['calibration']['negative_count'], r['calibration']['positive_count']) for r in rows}
    if len(counts) != 1:
        raise ValueError('candidate calibration populations differ')
    valid = [r for r in rows if r['calibration']['crossfit_bce'] is not None]
    minimum = min((r['calibration']['crossfit_bce'] for r in valid), default=None)
    eligible = [r for r in valid if r['calibration']['crossfit_bce'] <= minimum + .05 + 1e-12]
    if min(next(iter(counts))) < 5:
        best = by_id[source_global_id]
        reason = 'scarce_calibration_class_source_global'
    else:
        if not eligible or any(r['calibration']['raw_auc'] is None for r in eligible):
            raise ValueError('no eligible finite target calibration ranking')
        best = min(eligible, key=lambda r: (-r['calibration']['raw_auc'],
            r['calibration']['crossfit_bce'],
            -(r['source']['macro_auc'] if r['source']['macro_auc'] is not None else -1.),
            complexity(r['config']), r['config']['id']))
        reason = 'calibration_raw_auc_with_crossfit_bce_gate'
    return dict(best=best, candidates=sorted(rows, key=lambda r: r['config']['id']),
        reason=reason, source_global_id=source_global_id, candidate_count=len(rows),
        best_crossfit_bce=minimum, bce_tolerance=.05,
        eligible_ids=[r['config']['id'] for r in eligible],
        selection='calibration crossfit BCE <= minimum+0.05; raw AUC, crossfit BCE, source AUC, complexity, ID')


def verify_preregistration(root, stage='target'):
    """Audit declared code/banks and the complete immutable v2 source freeze."""
    pre = v2.verify_preregistration(root, stage=stage)
    if 'source_root' not in pre:
        raise ValueError('source_root missing from preregistration')
    source = Path(pre['source_root']).resolve()
    freeze = source / 'selection/FROZEN.json'
    declared = {str(Path(p).resolve()): digest for p, digest in pre['input_hashes'].items()}
    expected = declared.get(str(freeze))
    if not expected or sha256(freeze) != expected:
        raise ValueError('declared source frozen hash mismatch')
    if pre.get('source_frozen_sha256', expected) != expected:
        raise ValueError('source_frozen_sha256 disagrees with input manifest')
    frozen = verify_frozen(freeze)
    if not frozen.get('hashes'):
        raise ValueError('source freeze is empty')
    source_pre = v2.verify_preregistration(source, stage='target')
    if Path(source_pre['banks_root']).resolve() != Path(pre['banks_root']).resolve():
        raise ValueError('source and target banks differ')
    if source_pre['candidates'] != pre['candidates']:
        raise ValueError('source candidate grid differs from preregistered target reuse')
    required = [source / 'PREREGISTERED.json']
    required += [source / 'source' / f'pair_{i:02d}' / 'COMPLETE.json' for i in range(10)]
    required += [source / 'targets' / d / 'selection.json' for d in DATASETS]
    recorded = {str(Path(p).resolve()) for p in frozen['hashes']}
    if not {str(p) for p in required}.issubset(recorded):
        raise ValueError('source freeze missing complete source/target artifacts')
    return pre


def _init_target_worker(banks_root, target):
    global _TARGET_WORKER
    torch.set_num_threads(1)
    supports = {d: _load_bank(banks_root, d, 'support') for d in DATASETS}
    validation = _load_bank(banks_root, target, 'validation')
    _TARGET_WORKER = dict(supports=supports, validation=validation, target=target)


def _fit_target_candidate(task):
    head, source = task
    worker = _TARGET_WORKER
    target, supports, validation = worker['target'], worker['supports'], worker['validation']
    cfg = source['config']
    fitted = v2.fit_shared_and_basis(supports, [d for d in DATASETS if d != target], head, cfg)
    x = fitted['transform'].apply(supports[target], reference_bank=supports[target])
    mask, y = binary_target(supports[target]['state'], head)
    adapt = fit_adapter(x[mask], y, fitted['shared'], fitted['basis'], cfg['adapter_rank'], cfg['adapter_prior'])
    raw = v2._predict_bank(validation, fitted, adapt, reference_bank=supports[target])
    ids = array(validation['node_ids']); cal, _ = validation_halves(ids, target)
    crossfit = crossfit_calibration(raw[cal], array(validation['state'])[cal], ids[cal], target, head)
    diagnostic = dict(config=cfg, source={k: source.get(k) for k in ('macro_auc', 'macro_bce', 'macro_brier')},
        calibration=crossfit['summary'], folds=crossfit['folds'],
        oof_logits=crossfit['oof_logits'], oof_probabilities=crossfit['oof_probabilities'],
        evaluation_ids=crossfit['evaluation_ids'])
    return dict(head=head, model=fitted, adaptation=adapt, validation_raw=raw, diagnostic=diagnostic)


def run_target(root, target, workers=2):
    root = Path(root)
    v2._assert_unfrozen(root)
    if target not in DATASETS or workers < 1:
        raise ValueError('unknown target or invalid worker count')
    destination = root / 'targets' / target
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('target artifacts already exist')
    pre = verify_preregistration(root); fp = fingerprint(pre)
    # Directory creation is an exclusive target writer guard, before any fitting.
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=True)
    with (destination / 'TARGET_STARTED').open('x') as handle:
        handle.write(fp + '\n')
    selections = {h: select_configuration(load_source_trials(pre['source_root'], target, h), target) for h in HEADS}
    for selected in selections.values():
        selected['shortlist'] = source_shortlist(selected)
    supports = {d: _load_bank(pre['banks_root'], d, 'support') for d in DATASETS}
    vb = _load_bank(pre['banks_root'], target, 'validation')
    ids, z = array(vb['node_ids']), array(vb['state'])
    if np.intersect1d(array(supports[target]['node_ids']), ids).size:
        raise ValueError('support/validation ID overlap')
    cal, policy = validation_halves(ids, target)
    tasks = [(head, source) for head in HEADS for source in selections[head]['shortlist']]
    results = {h: {} for h in HEADS}
    def collect(result):
        results[result['head']][result['diagnostic']['config']['id']] = result
        print(json.dumps(dict(target=target, head=result['head'], candidate=result['diagnostic']['config']['id'],
            crossfit_bce=result['diagnostic']['calibration']['crossfit_bce'])), flush=True)
    if workers == 1:
        global _TARGET_WORKER
        _TARGET_WORKER = dict(supports=supports, validation=vb, target=target)
        for task in tasks:
            collect(_fit_target_candidate(task))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_target_worker,
                initargs=(pre['banks_root'], target)) as pool:
            for future in as_completed([pool.submit(_fit_target_candidate, task) for task in tasks]):
                collect(future.result())
    target_selection = {h: select_target_candidate(
        [r['diagnostic'] for r in results[h].values()], selections[h]['best']['config']['id']) for h in HEADS}
    configs = {h: target_selection[h]['best']['config'] for h in HEADS}
    models, adaptation, calibration, validation_raw, validation_final = {}, {}, {}, {}, {}
    for head, cfg in configs.items():
        selected = results[head][cfg['id']]
        raw = selected['validation_raw']; cm, cy = binary_target(z[cal], head)
        models[head], adaptation[head] = selected['model'], selected['adaptation']
        calibration[head] = fit_affine(raw[cal][cm], cy)
        validation_raw[head] = raw
        validation_final[head] = apply_affine(raw, calibration[head])
    del results
    routefit = routing_boundary(validation_final['preference'][cal], z[cal])
    threshold = routefit['threshold']
    validation_scores = score_payload(vb, validation_final['trust'], validation_final['preference'], threshold)
    validation_scores.update(state=vb['state'], calibration_mask=torch.from_numpy(cal), policy_mask=torch.from_numpy(policy))
    for head in HEADS:
        validation_scores['adapted_' + head + '_logit'] = torch.from_numpy(validation_raw[head])
    score, route = array(validation_scores['selected_score'])[policy], array(validation_scores['route'])[policy]
    correct = route == z[policy]
    policies = {str(risk): fit_risk_policy(score, correct, risk=risk, min_accepted=20) for risk in (.05, .1, .15, .2)}
    top = fit_top_fraction(score, correct, ids[policy])
    query = _load_bank(pre['banks_root'], target, 'query')
    if np.intersect1d(array(query['node_ids']), np.r_[ids, array(supports[target]['node_ids'])]).size:
        raise ValueError('query overlaps fitting/calibration IDs')
    raw_query = {h: v2._predict_bank(query, models[h], adaptation[h], reference_bank=supports[target]) for h in HEADS}
    final_query = {h: apply_affine(raw_query[h], calibration[h]) for h in HEADS}
    scores = score_payload(query, final_query['trust'], final_query['preference'], threshold)
    for head in HEADS:
        scores['adapted_' + head + '_logit'] = torch.from_numpy(raw_query[head])
    serial_models = {head: {
        **{k: v for k, v in model.items() if k != 'transform'}, 'transform': model['transform'].state_dict(),
        'adaptation': adaptation[head], 'calibration': calibration[head], 'config': configs[head],
        'normalization_reference_ids': {**model['normalization_reference_ids'],
            **({target: array(supports[target]['node_ids']).tolist()} if configs[head]['view'] == 'support_rank' else {})},
        'adaptation_ids': array(supports[target]['node_ids'])[binary_target(supports[target]['state'], head)[0]].tolist()
    } for head, model in models.items()}
    metrics = {}
    for head in HEADS:
        mask, y = binary_target(z[policy], head)
        metrics[head] = dict(before_calibration=binary_metrics(validation_raw[head][policy][mask], y),
            after_calibration=binary_metrics(validation_final[head][policy][mask], y))
    summaries = {}
    for head, chosen in target_selection.items():
        measured = chosen["best"]["calibration"]
        summaries[head] = dict(shortlist_count=chosen["candidate_count"],
            method=("source_fallback" if chosen["reason"] == "scarce_calibration_class_source_global"
                else "loo" if measured["method"] == "leave_one_out" else "fivefold"),
            calibration_auc=measured["raw_auc"], crossfit_bce=measured["crossfit_bce"],
            crossfit_brier=measured["crossfit_brier"], positive_count=measured["positive_count"],
            negative_count=measured["negative_count"], selected_id=configs[head]["id"])
    selection = dict(calibration_selection_summary=summaries, target=target, protocol='target_calibration_selection', preregistration_sha256=fp,
        configs=configs, adaptation=adaptation, calibration=calibration, routing=routefit, route_boundary=threshold,
        policies=policies, top_fraction_policy=top, calibration_ids=ids[cal].tolist(), policy_ids=ids[policy].tolist(),
        source_domains=[d for d in DATASETS if d != target], source_root=str(pre['source_root']),
        validation_split='sha256(dataset|conflict|node_id)[0]&1; exact historical split-C algorithm',
        validation_metrics=metrics, selected_by='calibration raw AUC within crossfit BCE gate over source-only shortlist',
        label_roles=dict(source='four-source support model; source reciprocal-validation shortlist',
            support='target adapter and own-support normalization references',
            calibration='finite endpoint selection; final affine calibration and routing',
            policy='acceptance policies and reused developer evaluation only', query='label-free inference'),
        deviations=['target calibration selects finite source-derived endpoint: protocol change',
            'policy is reused benchmark development data, not a pristine holdout',
            'positive affine crossfit BCE/Brier; ranking measured on fixed raw logits',
            'top fraction is exploratory precision policy, not risk guarantee'])
    save_torch(destination / 'model.pt', dict(heads=serial_models, route_boundary=threshold, preregistration_sha256=fp))
    save_torch(destination / 'validation_scores.pt', validation_scores)
    save_torch(destination / 'scores.pt', scores)
    write_json(destination / 'source_selection.json', selections)
    write_json(destination / 'target_selection.json', target_selection)
    write_json(destination / 'selection.json', selection)
    files = ('model.pt', 'validation_scores.pt', 'scores.pt', 'source_selection.json', 'target_selection.json', 'selection.json')
    write_json(destination / 'COMPLETE.json', dict(preregistration_sha256=fp, protocol='target_calibration_selection',
        files={name: sha256(destination / name) for name in files}))
    print(json.dumps(dict(target=target, configs={h: c['id'] for h, c in configs.items()}, top_fraction=top['fraction'])), flush=True)
    return selection
