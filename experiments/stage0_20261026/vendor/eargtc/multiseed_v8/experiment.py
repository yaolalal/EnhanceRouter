"""Finite, label-isolated source screening and target parameter crossfitting.

Only immutable v2 metric/calibration/policy functions are shared with older runs.
No old target model, prediction, support label or query label enters fitting.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
import sklearn
from sklearn.linear_model import LogisticRegression
from scipy.special import logit

from eargtc.router_v8_exploration.core import (
    HEADS, array, binary_target, binary_metrics, fit_affine, apply_affine,
    routing_boundary, score_payload, fit_risk_policy, fit_top_fraction,
)
from eargtc.router_v8_learned.data import load_bank, take
from eargtc.router_v8_learned.model import neural_grid, fit_head, predict_head, feature_importance

DATASETS = ('cora', 'citeseer', 'pubmed', 'arxiv', 'ogbn-products')
PROBABILITY_INDICES = list(range(16)) + list(range(27, 43)) + list(range(54, 63))
TARGET_FILES = ('model.pt', 'scores.pt', 'validation_scores.pt', 'selection.json',
                'source_selection.json', 'target_selection.json')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(pre):
    return hashlib.sha256(json.dumps(pre, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def ids_hash(ids):
    return hashlib.sha256(np.asarray(array(ids), dtype='<i8').tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    # Exclusive final creation means a failed stage never silently overwrites.
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def save_torch(path, value):
    with Path(path).open('xb') as handle:
        torch.save(value, handle)


def verify_preregistration(root, stage):
    from .protocol import verify_preregistration as verify
    return verify(root, stage=stage)


def _start(root, directory, stage, fp):
    if (Path(root) / 'selection' / 'FROZEN.json').exists():
        raise RuntimeError('run is frozen')
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f'partial or complete stage already exists: {directory}')
    with (directory / f'{stage.upper()}_STARTED').open('x') as handle:
        handle.write(fp + '\n')


def _complete(directory, fp, files, **identity):
    write_json(directory / 'COMPLETE.json', dict(preregistration_sha256=fp,
        files={name: sha256(directory / name) for name in files}, **identity))


def _check_complete(directory, fp, files, **identity):
    record = json.loads((directory / 'COMPLETE.json').read_text())
    if record.get('preregistration_sha256') != fp or any(record.get(k) != v for k, v in identity.items()):
        raise ValueError('source COMPLETE identity mismatch')
    if set(record.get('files', {})) != set(files):
        raise ValueError('source COMPLETE file inventory mismatch')
    expected = set(files) | {'COMPLETE.json', 'SOURCE_STARTED'}
    if {p.name for p in directory.iterdir()} != expected:
        raise ValueError('source directory inventory mismatch')
    for name, digest in record['files'].items():
        if sha256(directory / name) != digest:
            raise ValueError(f'changed source artifact: {name}')
    return record


def training_bank(bank):
    """Explicitly authorized row subset; policy-role flags never enter trainer."""
    return {k: v for k, v in bank.items() if k not in ('cal_mask', 'policy_mask')}


def label_free(bank):
    return {k: v for k, v in bank.items() if k not in ('state', 'cal_mask', 'policy_mask')}


def calibration_folds(ids, dataset):
    """Five deterministic label-blind folds over ALL calibration conflicts."""
    ids = np.asarray(array(ids))
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('calibration IDs must be unique and one-dimensional')
    keys = [hashlib.sha256(f'{dataset}|v8-learned-fivefold|{int(i)}'.encode()).digest() for i in ids]
    order = sorted(range(len(ids)), key=lambda j: (keys[j], int(ids[j])))
    assignment = np.empty(len(ids), dtype=int)
    assignment[order] = np.arange(len(ids)) % 5
    return [assignment == k for k in range(5)]


def _mean(values, require_all=False):
    if any(v is not None and not np.isfinite(v) for v in values):
        raise ValueError('nonfinite selection metric')
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid and (not require_all or len(valid) == len(values)) else None


def complexity(cfg):
    kind = cfg.get('kind', 'neural')
    if kind == 'ridge':
        return (0, 0 if cfg.get('training_scope') == 'target' else 1, {'prob': 0, 'top16': 1, 'all': 2}[cfg['feature_mode']], cfg['C'])
    if kind == 'hist':
        return (1, cfg['max_depth'], {'prob': 0, 'top16': 1, 'all': 2}[cfg['feature_mode']])
    if kind == 'trees':
        return (2, cfg['max_depth'], {'prob': 0, 'top16': 1, 'all': 2}[cfg['feature_mode']])
    return (3, cfg.get('target_steps', 0), {'none': 0, 'gnn': 1, 'llm': 2, 'both': 3}[cfg['embedding_mode']],
            cfg['width'], {'prob': 0, 'all': 1, 'gated': 2}[cfg['feature_mode']])


def _source_best(rows):
    usable = [r for r in rows if r['macro_bce'] is not None]
    if not usable:
        raise ValueError('no usable source BCE')
    minimum = min(r['macro_bce'] for r in usable)
    eligible = [r for r in usable if r['macro_bce'] <= minimum + .05 + 1e-12]
    return min(eligible, key=lambda r: (-(r['macro_auc'] if r['macro_auc'] is not None else -1),
        r['macro_bce'], complexity(r['config']), r['config']['id']))


def source_shortlist(rows):
    best = _source_best(rows)
    selected = {best['config']['id']: best}
    for family in ('none', 'gnn', 'llm', 'both'):
        family_rows = [r for r in rows if r['config']['embedding_mode'] == family]
        if any(r['macro_bce'] is not None for r in family_rows):
            winner = _source_best(family_rows)
            selected[winner['config']['id']] = winner
    return dict(best=best, shortlist=[selected[k] for k in sorted(selected)], trials=rows,
                bce_tolerance=.05, selection='global and per embedding: BCE gate, AUC, BCE, complexity, ID')


def run_source(root, target, held, device='cpu', base_reference=None):
    root = Path(root)
    if target not in DATASETS or held not in DATASETS or target == held:
        raise ValueError('invalid source roles')
    pre = verify_preregistration(root, 'source'); fp = fingerprint(pre)
    destination = root / 'source' / f'{target}__{held}'
    configs = pre['neural_grid']
    files = [f'{h}__{c["id"]}.{ext}' for h in HEADS for c in configs for ext in ('json', 'pt')]
    if (destination / 'COMPLETE.json').exists():
        return validate_source_artifacts(root, target, held, pre)
    reference_directory = None
    if base_reference is not None:
        reference_directory = Path(base_reference).resolve()
        expected_reference = (root / 'source' / f'{held}__{target}').resolve()
        if reference_directory != expected_reference:
            raise ValueError('base reference must be the same-run reverse source case')
        validate_source_artifacts(root, held, target, pre, _allow_reference=False)
    _start(root, destination, 'source', fp)
    sources = [d for d in DATASETS if d not in (target, held)]
    banks = [training_bank(load_bank(pre['banks_root'], d, 'validation')) for d in sources] if reference_directory is None else []
    unlabeled = [load_bank(pre['banks_root'], d, 'unlabeled') for d in sources] if reference_directory is None else []
    hb = load_bank(pre['banks_root'], held, 'validation')
    hu = load_bank(pre['banks_root'], held, 'unlabeled')
    halves = [array(hb['cal_mask']).astype(bool), array(hb['policy_mask']).astype(bool)]
    if np.any(halves[0] & halves[1]) or not np.all(halves[0] | halves[1]):
        raise ValueError('invalid held halves')
    for head in HEADS:
        for cfg in configs:
            model_name = f'{head}__{cfg["id"]}.pt'
            reference = None
            if reference_directory is None:
                base = fit_head(banks, head, cfg, device=device, steps=cfg['source_steps'], unlabeled=unlabeled)
                save_torch(destination / model_name, base)
            else:
                reference_path = reference_directory / model_name
                base = torch.load(reference_path, map_location='cpu', weights_only=False)
                if base['head'] != head or base['config'] != cfg:
                    raise ValueError('reused base head/config mismatch')
                reference = dict(path=str(reference_path), sha256=sha256(reference_path),
                                 target=held, held=target, equivalence='identical ordered three-domain source fit')
                with reference_path.open('rb') as incoming, (destination / model_name).open('xb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                if sha256(destination / model_name) != reference['sha256']:
                    raise ValueError('reused checkpoint bytes changed')
            fold_rows = []
            for index, adapt_mask in enumerate(halves):
                evaluation = ~adapt_mask
                adapted = fit_head([training_bank(take(hb, adapt_mask))], head, cfg, device=device,
                                   initial=base, steps=cfg['adapt_steps'], unlabeled=[hu])
                eb = take(hb, evaluation)
                raw = predict_head(adapted, label_free(eb), device=device)
                mask, y = binary_target(eb['state'], head)
                metrics = binary_metrics(raw[mask], y)
                fold_rows.append(dict(fold=index, metrics=metrics,
                    adaptation_ids=array(hb['ids'])[adapt_mask].tolist(),
                    evaluation_ids=array(eb['ids']).tolist(),
                    evaluation_head_ids=array(eb['ids'])[mask].tolist(),
                    training_audit=adapted['training_audit']))
            metrics = {key: _mean([r['metrics'][key] for r in fold_rows], require_all=key != 'auc')
                       for key in ('bce', 'auc', 'brier')}
            record = dict(target=target, held=held, head=head, config=cfg, source_domains=sources,
                preregistration_sha256=fp, source_training_audit=base['training_audit'],
                metrics=metrics, folds=fold_rows, valid_auc_folds=sum(r['metrics']['auc'] is not None for r in fold_rows))
            if reference is not None:
                record['base_reference'] = reference
            write_json(destination / f'{head}__{cfg["id"]}.json', record)
            print(json.dumps(dict(stage='source', target=target, held=held, head=head,
                                  candidate=cfg['id'], metrics=metrics)), flush=True)
    _complete(destination, fp, files, target=target, held=held)
    return json.loads((destination / 'COMPLETE.json').read_text())


def run_source_pair(root, a, b, device='cpu'):
    """Two directed cases; the identical ordered three-domain base is fitted once."""
    forward = run_source(root, a, b, device=device)
    reference = Path(root) / 'source' / f'{a}__{b}'
    reverse = run_source(root, b, a, device=device, base_reference=reference)
    return dict(forward=forward, reverse=reverse)


def _expected_ids(bank, head, mask=None):
    ids, state = array(bank['ids']), array(bank['state'])
    if mask is not None:
        ids, state = ids[mask], state[mask]
    selected, _ = binary_target(state, head)
    return ids.tolist(), ids[selected].tolist()


def _check_training_audit(audit, inputs, supervised, transforms, unlabeled, cfg, stage):
    expected = dict(input_ids=inputs, supervised_ids=supervised, transform_fit_ids=transforms,
                    unlabeled_ids=unlabeled, seed=42, stage=stage)
    for key, value in expected.items():
        if audit.get(key) != value:
            raise ValueError(f'source provenance {stage} {key} mismatch')
    requested = cfg['source_steps' if stage == 'source' else 'adapt_steps']
    actual = requested if any(supervised.values()) else 0
    if audit.get('requested_steps') != requested or audit.get('actual_steps') != actual:
        raise ValueError('source provenance training step mismatch')
    teacher = 'source_training_ema' if stage == 'source' else 'fixed_source'
    if audit.get('teacher_kind') != teacher:
        raise ValueError('source provenance teacher mismatch')


def validate_source_artifacts(root, target, held, pre, _allow_reference=True):
    """Bind cache provenance to actual bank IDs, not self-reported role names."""
    if target not in DATASETS or held not in DATASETS or target == held:
        raise ValueError('invalid source roles')
    directory = Path(root) / 'source' / f'{target}__{held}'
    fp, configs = fingerprint(pre), pre['neural_grid']
    files = [f'{h}__{c["id"]}.{ext}' for h in HEADS for c in configs for ext in ('json', 'pt')]
    complete = _check_complete(directory, fp, files, target=target, held=held)
    sources = [d for d in DATASETS if d not in (target, held)]
    inputs, supervised, unlabeled = {}, {h: {} for h in HEADS}, {}
    for domain in sources:
        bank = load_bank(pre['banks_root'], domain, 'validation')
        for head in HEADS:
            inputs[domain], supervised[head][domain] = _expected_ids(bank, head)
        del bank
        query = load_bank(pre['banks_root'], domain, 'unlabeled')
        unlabeled[domain] = sorted(array(query['ids']).tolist())
        if len(unlabeled[domain]) > 4096 or set(unlabeled[domain]) & set(inputs[domain]):
            raise ValueError('invalid source unlabeled population')
        del query
    bank = load_bank(pre['banks_root'], held, 'validation')
    held_ids = array(bank['ids']); held_state = array(bank['state'])
    halves = [array(bank['cal_mask']).astype(bool), array(bank['policy_mask']).astype(bool)]
    if np.any(halves[0] & halves[1]) or not np.all(halves[0] | halves[1]):
        raise ValueError('invalid held halves')
    held_query = load_bank(pre['banks_root'], held, 'unlabeled')
    held_unlabeled = {held: sorted(array(held_query['ids']).tolist())}
    if len(held_unlabeled[held]) > 4096 or set(held_unlabeled[held]) & set(held_ids.tolist()):
        raise ValueError('invalid held unlabeled population')
    del held_query
    reference_checked = False
    for head in HEADS:
        for cfg in configs:
            row = json.loads((directory / f'{head}__{cfg["id"]}.json').read_text())
            if (row.get('config') != cfg or row.get('target') != target or row.get('held') != held
                    or row.get('head') != head or row.get('preregistration_sha256') != fp
                    or row.get('source_domains') != sources):
                raise ValueError('source trial role/configuration mismatch')
            _check_training_audit(row['source_training_audit'], inputs, supervised[head], supervised[head],
                                  unlabeled, cfg, 'source')
            base = torch.load(directory / f'{head}__{cfg["id"]}.pt', map_location='cpu', weights_only=False)
            if (base.get('head') != head or base.get('config') != cfg
                    or base.get('training_audit') != row['source_training_audit']):
                raise ValueError('source checkpoint head/config/audit mismatch')
            if 'base_reference' in row:
                if not _allow_reference:
                    raise ValueError('source base reference chains/cycles are forbidden')
                reference = row['base_reference']
                expected_path = (Path(root) / 'source' / f'{held}__{target}' / f'{head}__{cfg["id"]}.pt').resolve()
                if (Path(reference.get('path', '')).resolve() != expected_path
                        or reference.get('target') != held or reference.get('held') != target):
                    raise ValueError('source base reference role/path mismatch')
                if not reference_checked:
                    validate_source_artifacts(root, held, target, pre, _allow_reference=False)
                    reference_checked = True
                if (reference.get('sha256') != sha256(expected_path)
                        or reference['sha256'] != complete['files'][f'{head}__{cfg["id"]}.pt']):
                    raise ValueError('source base reference checkpoint hash mismatch')
            if len(row.get('folds', [])) != 2:
                raise ValueError('source reciprocal fold count mismatch')
            for index, mask in enumerate(halves):
                fold = row['folds'][index]
                adaptation_ids, adaptation_supervised = _expected_ids(bank, head, mask)
                evaluation_ids, evaluation_supervised = _expected_ids(bank, head, ~mask)
                if (fold.get('fold') != index or fold.get('adaptation_ids') != adaptation_ids
                        or fold.get('evaluation_ids') != evaluation_ids
                        or fold.get('evaluation_head_ids') != evaluation_supervised):
                    raise ValueError('source reciprocal half IDs mismatch')
                _check_training_audit(fold['training_audit'], {held: adaptation_ids},
                    {held: adaptation_supervised}, supervised[head], held_unlabeled, cfg, 'adaptation')
                _, y = binary_target(held_state[~mask], head)
                expected_counts = dict(count=len(y), positive_count=int(y.sum()), negative_count=int(len(y) - y.sum()))
                if any(fold['metrics'].get(k) != v for k, v in expected_counts.items()):
                    raise ValueError('source fold class count mismatch')
                for metric in ('bce', 'auc', 'brier'):
                    value = fold['metrics'].get(metric)
                    defined = len(y) > 0 if metric != 'auc' else len(np.unique(y)) == 2
                    if defined != (value is not None) or (value is not None and not np.isfinite(value)):
                        raise ValueError('source fold metric invalid')
            recomputed = {key: _mean([f['metrics'][key] for f in row['folds']], require_all=key != 'auc')
                          for key in ('bce', 'auc', 'brier')}
            if row.get('metrics') != recomputed or row.get('valid_auc_folds') != sum(f['metrics']['auc'] is not None for f in row['folds']):
                raise ValueError('source reciprocal metric aggregation mismatch')
    return complete


def load_source_trials(root, target, head, pre, validated=False):
    fp = fingerprint(pre)
    configs = pre['neural_grid']
    expected = [f'{h}__{c["id"]}.{ext}' for h in HEADS for c in configs for ext in ('json', 'pt')]
    aggregates = {c['id']: dict(config=c, held={}) for c in configs}
    for held in DATASETS:
        if held == target:
            continue
        directory = Path(root) / 'source' / f'{target}__{held}'
        if not validated:
            validate_source_artifacts(root, target, held, pre)
        for cfg in configs:
            row = json.loads((directory / f'{head}__{cfg["id"]}.json').read_text())
            if (row['config'] != cfg or row['target'] != target or row['held'] != held or row['head'] != head
                    or row['preregistration_sha256'] != fp
                    or set(row['source_domains']) != set(DATASETS) - {target, held}):
                raise ValueError('source trial role/configuration mismatch')
            aggregates[cfg['id']]['held'][held] = row['metrics']
    result = []
    for row in aggregates.values():
        values = list(row['held'].values())
        result.append(dict(**row, **{'macro_' + k: _mean([m[k] for m in values], require_all=k != 'auc')
                                    for k in ('bce', 'auc', 'brier')}))
    return result


def tabular_grid():
    modes = ('prob', 'all', 'top16')
    return [dict(id=f'ridge_{scope}_{mode}_C{c:g}', kind='ridge', training_scope=scope,
                 feature_mode=mode, C=c)
            for scope, cs in (('source_target', (.1, 1., 10.)), ('target', (.01, .1, 1., 10.)))
            for mode in modes for c in cs] + [
            dict(id=f'hist_{mode}_depth{depth}', kind='hist', training_scope='target',
                 feature_mode=mode, max_depth=depth, max_iter=100, learning_rate=.05,
                 l2_regularization=1., min_samples_leaf=10)
            for mode in modes for depth in (2, 3)] + [
            dict(id=f'trees_{mode}_depth{depth}', kind='trees', training_scope='target',
                 feature_mode=mode, max_depth=depth, n_estimators=200, min_samples_leaf=5)
            for mode in modes for depth in (1, 2, 4)]


def _uses_sources(cfg):
    return cfg['kind'] == 'ridge' and cfg.get('training_scope', 'source_target') == 'source_target'


def fit_tabular(banks, head, cfg):
    """All supervised selection/scaling sees exclusively supplied training rows."""
    if not _uses_sources(cfg) and len(banks) != 1:
        raise ValueError('target-only tabular candidate requires exactly one target training bank')
    xs, ys, audits = [], [], {}
    for bank in banks:
        mask, y = binary_target(bank['state'], head)
        xs.append(np.asarray(array(bank['tab']), dtype=float)[mask]); ys.append(y)
        domain = bank.get('dataset', bank.get('domain', 'unknown'))
        audits[domain] = array(bank['ids'])[mask].tolist()
    x = np.concatenate(xs); y = np.concatenate(ys)
    active = sum(bool(len(v)) for v in ys)
    weights = np.concatenate([np.full(len(v), len(y) / (active * len(v))) if len(v) else np.empty(0) for v in ys])
    mean = np.average(x, axis=0, weights=weights) if len(x) else np.zeros(70)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights)) if len(x) else np.ones(70)
    scale[scale < 1e-8] = 1.
    scaled = np.clip((x - mean) / scale, -8, 8)
    indices = np.arange(70)
    if cfg['feature_mode'] == 'prob':
        indices = np.array(PROBABILITY_INDICES)
    elif cfg['feature_mode'] == 'top16':
        correlations = np.abs(scaled.T @ (weights * (y - np.average(y, weights=weights)))) if len(y) else np.zeros(70)
        indices = np.lexsort((np.arange(70), -correlations))[:16]
    fitted = dict(kind=cfg['kind'], head=head, config=cfg, mean=mean, scale=scale,
                  indices=indices, sklearn_version=sklearn.__version__,
                  training_audit=dict(training_ids=audits, feature_fit_ids=audits, seed=42,
                      domain_loss_mass={name: 1 / active for name, values in audits.items() if values},
                      sample_weight_sum=float(weights.sum()), normalization='equal-domain supervised rows'))
    if len(np.unique(y)) < 2:
        fitted['constant_logit'] = float(logit((y.sum() + .5) / (len(y) + 1)))
    elif cfg['kind'] == 'ridge':
        estimator = LogisticRegression(C=cfg['C'], solver='lbfgs', max_iter=2000, random_state=42)
        estimator.fit(scaled[:, indices], y, sample_weight=weights)
        fitted.update(coef=estimator.coef_[0], intercept=float(estimator.intercept_[0]))
    elif cfg['kind'] == 'hist':
        estimator = HistGradientBoostingClassifier(max_iter=100, learning_rate=.05,
            max_depth=cfg['max_depth'], l2_regularization=1., min_samples_leaf=10,
            early_stopping=False, random_state=42)
        estimator.fit(scaled[:, indices], y)
        fitted['estimator'] = estimator
    else:
        estimator = ExtraTreesClassifier(n_estimators=200, max_depth=cfg['max_depth'],
            min_samples_leaf=5, random_state=42, n_jobs=1)
        estimator.fit(scaled[:, indices], y)
        # Arrays, not pickled sklearn objects, keep checkpoints portable.
        fitted['trees'] = [dict(children_left=t.tree_.children_left, children_right=t.tree_.children_right,
            feature=t.tree_.feature, threshold=t.tree_.threshold, value=t.tree_.value[:, 0, :])
            for t in estimator.estimators_]
    return fitted


def predict_tabular(fitted, bank):
    x = np.clip((np.asarray(array(bank['tab']), float) - fitted['mean']) / fitted['scale'], -8, 8)
    x = x[:, fitted['indices']]
    if 'constant_logit' in fitted:
        return np.full(len(x), fitted['constant_logit'], dtype=float)
    if fitted['kind'] == 'ridge':
        return x @ fitted['coef'] + fitted['intercept']
    if fitted['kind'] == 'hist':
        if fitted['sklearn_version'] != sklearn.__version__:
            raise ValueError('HGB checkpoint sklearn version mismatch')
        return np.asarray(fitted['estimator'].decision_function(x), dtype=float)
    p = np.zeros(len(x))
    for tree in fitted['trees']:
        nodes = np.zeros(len(x), dtype=int)
        active = tree['children_left'][nodes] >= 0
        while active.any():
            rows = np.flatnonzero(active); current = nodes[rows]
            left = x[rows, tree['feature'][current]] <= tree['threshold'][current]
            nodes[rows] = np.where(left, tree['children_left'][current], tree['children_right'][current])
            active = tree['children_left'][nodes] >= 0
        value = tree['value'][nodes]
        p += value[:, 1] / value.sum(axis=1)
    return logit(np.clip(p / len(fitted['trees']), 1e-7, 1 - 1e-7))


def _predict(fitted, bank, device):
    bank = label_free(bank)
    return predict_head(fitted, bank, device=device) if fitted.get('kind', 'neural') == 'neural' else predict_tabular(fitted, bank)


def crossfit_candidate(cal_bank, sources, head, cfg, initial=None, unlabeled=None, device='cpu'):
    """Own held rows are excluded before masking, fitting and feature selection."""
    ids = array(cal_bank['ids']); state = array(cal_bank['state'])
    folds = calibration_folds(ids, cal_bank.get('dataset', cal_bank.get('domain', 'unknown')))
    raw = np.full(len(ids), np.nan); records = []
    for index, held in enumerate(folds):
        training = training_bank(take(cal_bank, ~held))
        if cfg.get('kind', 'neural') == 'neural':
            if initial is None:
                raise ValueError('neural CV requires independently source-fitted initial')
            fitted = initial if cfg['target_steps'] == 0 else fit_head([training], head, initial['config'],
                device=device, initial=initial, steps=cfg['target_steps'], unlabeled=unlabeled)
        else:
            fitted = fit_tabular(sources + [training] if _uses_sources(cfg) else [training], head, cfg)
        raw[held] = _predict(fitted, take(cal_bank, held), device)
        hm, hy = binary_target(state[held], head)
        metrics = binary_metrics(raw[held][hm], hy)
        fit_mask, _ = binary_target(state[~held], head)
        records.append(dict(fold=index, holdout_ids=ids[held].tolist(),
            evaluation_ids=ids[held][hm].tolist(), target_training_ids=ids[~held][fit_mask].tolist()
                if cfg.get('target_steps', 1) != 0 else [],
            metrics=metrics, training_audit=fitted['training_audit']))
    if not np.isfinite(raw).all():
        raise ValueError('incomplete or nonfinite OOF predictions')
    mask, y = binary_target(state, head)
    pooled = binary_metrics(raw[mask], y)
    summary = dict(count=pooled['count'], positive_count=pooled['positive_count'], negative_count=pooled['negative_count'],
        crossfit_bce=pooled['bce'], crossfit_brier=pooled['brier'], pooled_oof_auc=pooled['auc'],
        mean_fold_auc=_mean([r['metrics']['auc'] for r in records]), valid_auc_folds=sum(r['metrics']['auc'] is not None for r in records),
        fold_count=5, full_calibration_count=len(ids), method='parameter_hash_fivefold', calibration_ids_sha256=ids_hash(ids))
    return dict(config=cfg, calibration=summary, folds=records, oof_raw_logits=raw.tolist(),
                calibration_ids=ids.tolist(), evaluation_ids=ids[mask].tolist())


def select_target_candidate(rows, source_global_id):
    if not rows or len({r['config']['id'] for r in rows}) != len(rows):
        raise ValueError('empty/duplicate target candidates')
    by_id = {r['config']['id']: r for r in rows}
    if source_global_id not in by_id:
        raise ValueError('source-global fallback is missing')
    counts = {(r['calibration']['positive_count'], r['calibration']['negative_count']) for r in rows}
    if len(counts) != 1:
        raise ValueError('candidate populations differ')
    for row in rows:
        for name in ('crossfit_bce', 'crossfit_brier', 'mean_fold_auc', 'pooled_oof_auc'):
            value = row['calibration'][name]
            if value is not None and not np.isfinite(value):
                raise ValueError('nonfinite target selection metric')
    valid = [r for r in rows if r['calibration']['crossfit_bce'] is not None]
    minimum = min((r['calibration']['crossfit_bce'] for r in valid), default=None)
    eligible = [r for r in valid if r['calibration']['crossfit_bce'] <= minimum + .05 + 1e-12]
    if min(next(iter(counts))) < 5 or not any(r['calibration']['mean_fold_auc'] is not None for r in eligible):
        best = by_id[source_global_id]; reason = 'scarce_calibration_class_source_global'
    else:
        eligible = [r for r in eligible if r['calibration']['mean_fold_auc'] is not None]
        best = min(eligible, key=lambda r: (-r['calibration']['mean_fold_auc'], r['calibration']['crossfit_bce'],
                                          complexity(r['config']), r['config']['id']))
        reason = 'calibration_mean_valid_fold_auc_with_oof_bce_gate'
    return dict(best=best, candidates=rows, source_global_id=source_global_id, reason=reason,
        candidate_count=len(rows), best_crossfit_bce=minimum, bce_tolerance=.05,
        eligible_ids=[r['config']['id'] for r in eligible])


def finish_calibration(cal_bank, chosen):
    """Affine and route see selected OOF raw scores only, never in-sample refit scores."""
    state = array(cal_bank['state']); calibration = {}; oof_final = {}
    for head in HEADS:
        mask, y = binary_target(state, head)
        raw = np.asarray(chosen[head]['best']['oof_raw_logits'])
        calibration[head] = fit_affine(raw[mask], y)
        oof_final[head] = apply_affine(raw, calibration[head])
    return calibration, routing_boundary(oof_final['preference'], state)


def _score_bank(bank, logits, boundary):
    compatible = dict(node_ids=bank['ids'], gnn_pred=bank['gnn_pred'], llm_pred=bank['llm_pred'])
    return score_payload(compatible, logits['trust'], logits['preference'], boundary)


def run_target(root, target, device='cpu'):
    root = Path(root)
    if target not in DATASETS:
        raise ValueError('unknown target')
    pre = verify_preregistration(root, 'target'); fp = fingerprint(pre)
    destination = root / 'targets' / target
    _start(root, destination, 'target', fp)
    for held in DATASETS:
        if held != target:
            validate_source_artifacts(root, target, held, pre)
    selections = {h: source_shortlist(load_source_trials(root, target, h, pre, True)) for h in HEADS}
    source_domains = [d for d in DATASETS if d != target]
    sources = [training_bank(load_bank(pre['banks_root'], d, 'validation')) for d in source_domains]
    source_u = [load_bank(pre['banks_root'], d, 'unlabeled') for d in source_domains]
    vb = load_bank(pre['banks_root'], target, 'validation')
    cal = array(vb['cal_mask']).astype(bool); policy = array(vb['policy_mask']).astype(bool)
    if np.any(cal & policy) or not np.all(cal | policy):
        raise ValueError('invalid target halves')
    cb = training_bank(take(vb, cal))
    target_u = [load_bank(pre['banks_root'], target, 'unlabeled')]
    target_selections, chosen_models = {}, {}
    for head in HEADS:
        rows, bases = [], {}
        for selected in selections[head]['shortlist']:
            cfg = selected['config']
            base = fit_head(sources, head, cfg, device=device, steps=cfg['source_steps'], unlabeled=source_u)
            bases[cfg['id']] = base
            for steps in (0, cfg['adapt_steps']):
                candidate = dict(cfg, kind='neural', base_id=cfg['id'], id=f'{cfg["id"]}_target{steps}', target_steps=steps)
                result = crossfit_candidate(cb, sources, head, candidate, initial=base, unlabeled=target_u, device=device)
                result['source'] = {k: selected[k] for k in ('macro_auc', 'macro_bce', 'macro_brier')}
                rows.append(result)
                print(json.dumps(dict(stage='target_cv', target=target, head=head, candidate=candidate['id'], metrics=result['calibration'])), flush=True)
        for candidate in tabular_grid():
            result = crossfit_candidate(cb, sources, head, candidate, device='cpu')
            rows.append(result)
            print(json.dumps(dict(stage='target_cv', target=target, head=head, candidate=candidate['id'], metrics=result['calibration'])), flush=True)
        fallback = selections[head]['best']['config']['id'] + '_target0'
        chosen = select_target_candidate(rows, fallback)
        target_selections[head] = chosen
        cfg = chosen['best']['config']
        if cfg.get('kind') == 'neural':
            base = bases[cfg['base_id']]
            fitted = base if cfg['target_steps'] == 0 else fit_head([cb], head, base['config'], device=device,
                initial=base, steps=cfg['target_steps'], unlabeled=target_u)
            fitted = dict(fitted, feature_importance=feature_importance(fitted))
        else:
            fitted = fit_tabular(sources + [cb] if _uses_sources(cfg) else [cb], head, cfg)
        chosen_models[head] = fitted
    calibration, routefit = finish_calibration(cb, target_selections)
    threshold = routefit['threshold']
    raw_validation = {h: _predict(chosen_models[h], vb, device) for h in HEADS}
    final_validation = {h: apply_affine(raw_validation[h], calibration[h]) for h in HEADS}
    vs = _score_bank(vb, final_validation, threshold)
    vs.update(state=vb['state'], calibration_mask=torch.as_tensor(cal), policy_mask=torch.as_tensor(policy))
    for h in HEADS:
        vs['adapted_' + h + '_logit'] = torch.as_tensor(raw_validation[h])
    ids, state = array(vb['ids']), array(vb['state'])
    score = array(vs['selected_score'])[policy]; correct = array(vs['route'])[policy] == state[policy]
    policies = {str(r): fit_risk_policy(score, correct, risk=r, min_accepted=20) for r in (.05, .1, .15, .2)}
    top = fit_top_fraction(score, correct, ids[policy])
    query = load_bank(pre['banks_root'], target, 'query')
    if np.intersect1d(array(query['ids']), ids).size:
        raise ValueError('query overlaps calibration/policy')
    raw_query = {h: _predict(chosen_models[h], query, device) for h in HEADS}
    final_query = {h: apply_affine(raw_query[h], calibration[h]) for h in HEADS}
    scores = _score_bank(query, final_query, threshold)
    for h in HEADS:
        scores['adapted_' + h + '_logit'] = torch.as_tensor(raw_query[h])
    metrics = {}
    for h in HEADS:
        mask, y = binary_target(state[policy], h)
        metrics[h] = dict(before_calibration=binary_metrics(raw_validation[h][policy][mask], y),
                          after_calibration=binary_metrics(final_validation[h][policy][mask], y))
    configs = {h: target_selections[h]['best']['config'] for h in HEADS}
    selection = dict(target=target, protocol='router_v8_learned_v4', preregistration_sha256=fp,
        configs=configs, calibration=calibration, routing=routefit, route_boundary=threshold,
        policies=policies, top_fraction_policy=top, calibration_ids=ids[cal].tolist(), policy_ids=ids[policy].tolist(),
        source_domains=source_domains, validation_metrics=metrics,
        calibration_selection_summary={h: dict(target_selections[h]['best']['calibration'],
            selected_id=configs[h]['id'], shortlist_count=target_selections[h]['candidate_count'],
            selection_reason=target_selections[h]['reason']) for h in HEADS},
        selected_by='honest target parameter-fivefold BCE gate; mean valid-fold AUC, BCE, complexity, stable ID',
        label_roles=dict(source='other four official validation domains', calibration='fivefold parameter selection, refit, OOF affine and route',
            policy='acceptance and reused developer metrics only', query='label-free inference and deterministic unlabeled consistency subset'),
        deviations=['expanded validation supervision: not five-shot-only Router training',
            'unlabeled query consistency is transductive', 'selected internal CV is not unbiased whole-procedure performance',
            'OOF target fits use about 80% of calibration rows while final refits use 100%; OOF affine and route are transported to a changed model score distribution, with potential calibration/routing shift',
            'upstream validation checkpoint dependence and Arxiv raw LLM parity limitation retained',
            'benchmark policy is reused development data; historical benchmark test results were already known'])
    save_torch(destination / 'model.pt', dict(heads={h: dict(chosen_models[h], calibration=calibration[h], selected_config=configs[h]) for h in HEADS},
                                            route_boundary=threshold, preregistration_sha256=fp))
    save_torch(destination / 'scores.pt', scores)
    save_torch(destination / 'validation_scores.pt', vs)
    write_json(destination / 'selection.json', selection)
    write_json(destination / 'source_selection.json', selections)
    write_json(destination / 'target_selection.json', target_selections)
    _complete(destination, fp, TARGET_FILES, target=target, protocol='router_v8_learned_v4')
    return selection
