"""Bounded, separately registered expansions of Router-v8(v7).

Each stage returns additions only; the driver retains earlier winners as fallbacks.
No view/environment variable enters features, sample probabilities, or losses.
"""
from copy import deepcopy
import numpy as np
from scipy.special import logit
from eargtc.router_aug_search import v7_engine as original
from eargtc.router_v8_v7_explore import core_v2 as historical


def _neural_cfg(mode, family='mlp', variant='plain'):
    return dict(family=family, mode=mode, scope='transfer', rank=False,
                variant=variant, seed=42, seeds=[42, 137, 2026],
                steps=280, adapt_steps=60, pretrain_steps=80)


def candidate_grid(dataset: str, head: str, stage='capacity') -> list[dict]:
    if dataset not in original.DATASETS or head not in original.HEADS:
        raise ValueError((dataset, head))
    if stage == 'original':
        return original.candidate_grid(dataset, head)
    rows = []
    if stage == 'capacity':
        # Six individual deviations from historical plain MLP, never combined.
        for mode in ('compact', 'context', 'embed'):
            for delta in ({'width': 64}, {'width': 128}, {'dropout': 0.},
                          {'dropout': .3}, {'weight_decay': .01}, {'weight_decay': .15}):
                rows.append(dict(_neural_cfg(mode, 'neural'), **delta))
            for c in (.03, .3, 3., 30.):
                rows.append(dict(family='lr', mode=mode, scope='pooled', rank=False, C=c, seed=42))
        # Tree capacity appropriate for the larger six-view bank.
        for mode in ('compact', 'context', 'full'):
            for depth, leaf in ((12, 3), (8, 6), (8, 12)):
                rows.append(dict(family='trees', mode=mode, scope='pooled', rank=False,
                                 depth=depth, leaf=leaf, seed=42))
    elif stage == 'nonlinear':
        for mode in ('compact', 'context', 'full'):
            for scope in ('pooled', 'target'):
                for depth in (3, 6):
                    rows.append(dict(family='hgb', mode=mode, scope=scope, rank=False,
                                     depth=depth, max_iter=160, learning_rate=.05,
                                     min_samples_leaf=12, l2_regularization=1., seed=42))
        for mode in ('compact', 'context', 'embed'):
            for scope in ('pooled', 'target'):
                for c in (.1, 1., 10.):
                    rows.append(dict(family='svc', mode=mode, scope=scope, rank=False,
                                     C=c, gamma='scale', seed=42))
    elif stage == 'training':
        for mode in ('compact', 'context', 'embed'):
            for variant in ('plain', 'rank', 'aug', 'ssl'):
                cfg = _neural_cfg(mode, variant=variant)
                cfg.update(steps=560, adapt_steps=120)
                rows.append(cfg)
            for variant in ('plain', 'rank'):
                rows.append(_neural_cfg(mode, 'residual', variant))
            cfg = _neural_cfg(mode)
            cfg['seeds'] = [42, 137, 2026, 2027, 2028]
            rows.append(cfg)
        if head != 'agreement':
            from .joint_engine import candidate_grid as joint_grid
            rows.extend(joint_grid(dataset, head))
    else:
        raise ValueError(f'Unknown expanded v7 stage: {stage}')
    for i, cfg in enumerate(rows):
        cfg['id'] = f'v7_{stage}_{i:03d}_{cfg["family"]}_{cfg["mode"]}_{cfg["scope"]}'
        cfg['stage'] = stage
    return rows


def _network(width, cfg):
    import torch.nn as nn
    hidden = cfg.get('width', 32)
    dropout = cfg.get('dropout', .15)
    layers = [nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)]
    if cfg['family'] == 'residual':
        class Residual(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                           nn.Dropout(dropout), nn.Linear(hidden, hidden))
            def forward(self, value):
                return value + self.block(value)
        layers.extend([Residual(), nn.GELU()])
    layers.extend([nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)])
    return nn.Sequential(*layers)


def _fit_neural(x, y, w, domains, cfg):
    """Historical sampling/loss/transfer recipe with one architecture knob."""
    import torch
    from torch.nn import functional as F
    tx = torch.tensor(x, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    source = np.flatnonzero(domains != cfg['target_domain'])
    target = np.flatnonzero(domains == cfg['target_domain'])
    if not len(source):
        source = np.arange(len(x))
    states = []
    for seed in cfg['seeds']:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = _network(x.shape[1], cfg)
        opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=cfg.get('weight_decay', .05))
        stages = [(source, cfg['steps'], .002)]
        adapted = bool(len(target) and len(np.unique(y[target])) == 2)
        if adapted:
            stages.append((target, cfg['adapt_steps'], .0005))
        for eligible, steps, lr in stages:
            for group in opt.param_groups:
                group['lr'] = lr
            p = w[eligible] / w[eligible].sum()
            for _ in range(steps):
                ids = rng.choice(eligible, 128, p=p)
                a, b, dd = tx[ids], ty[ids], domains[ids]
                z = net(a).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(z, b)
                if cfg['variant'] == 'rank':
                    pairs = np.arange(len(ids))
                    for d in np.unique(dd):
                        jj = np.flatnonzero(dd == d)
                        pairs[jj] = rng.permutation(jj)
                    delta = b - b[pairs]
                    valid = delta != 0
                    if valid.any():
                        loss = loss + .35 * F.softplus(-(z-z[pairs])[valid]*delta[valid]).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
                opt.step()
        states.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    return dict(kind='expanded_neural', states=states, config=deepcopy(cfg), target_adapted=adapted)


def _fit_estimator(x, y, w, domains, cfg):
    if len(np.unique(y)) < 2:
        return dict(kind='constant', logit=float(logit((y.sum()+.5)/(len(y)+1))))
    family = cfg['family']
    if family in ('lr', 'trees', 'mlp'):
        return historical.fit_estimator(x, y, w, domains, cfg)
    if family in ('neural', 'residual'):
        return _fit_neural(x, y, w, domains, cfg)
    if family == 'hgb':
        from sklearn.ensemble import HistGradientBoostingClassifier
        estimator = HistGradientBoostingClassifier(max_iter=cfg['max_iter'],
            max_depth=cfg['depth'], learning_rate=cfg['learning_rate'],
            min_samples_leaf=cfg['min_samples_leaf'], l2_regularization=cfg['l2_regularization'],
            early_stopping=False, random_state=cfg['seed'])
    elif family == 'svc':
        from sklearn.svm import SVC
        estimator = SVC(C=cfg['C'], gamma=cfg['gamma'], kernel='rbf',
                        random_state=cfg['seed'], cache_size=512)  # default disables probability calibration
    else:
        raise ValueError(f'Unknown expanded model family: {family}')
    estimator.fit(x, y.astype(int), sample_weight=w)
    return dict(kind='sklearn', estimator=estimator)


def _predict_estimator(model, x):
    if model['kind'] != 'expanded_neural':
        return historical.predict_estimator(model, x)
    import torch
    if not len(x):
        return np.empty(0, float)
    net = _network(x.shape[1], model['config'])
    scores = []
    for state in model['states']:
        net.load_state_dict(state)
        net.eval()
        parts = []
        with torch.inference_mode():
            for i in range(0, len(x), 4096):
                parts.append(net(torch.tensor(x[i:i+4096], dtype=torch.float32)).squeeze(1).numpy())
        scores.append(np.concatenate(parts))
    return np.mean(scores, axis=0)


def fit_candidate(cfg, head, banks: dict, target: str) -> dict:
    original._slurm()
    if cfg.get('family') == 'joint_shared':
        from .joint_engine import fit_candidate as fit_joint
        return fit_joint(cfg, head, banks, target)
    if 'stage' not in cfg or cfg.get('stage') == 'original':
        return original.fit_candidate(cfg, head, banks, target)
    if head not in original.HEADS or target not in original.DATASETS:
        raise ValueError((target, head))
    if set(banks) != set(original.DATASETS):
        raise ValueError('Historical v7 requires exactly the five training datasets')
    for dataset, bank in banks.items():
        original._check_bank(bank, dataset)
    cfg = deepcopy(cfg)
    scope = cfg['scope']
    ordered = [banks[d] for d in original.DATASETS if d != target] + [banks[target]]
    if scope not in ('source', 'target', 'pooled', 'transfer'):
        raise ValueError(f'Unknown training scope: {scope}')
    use = ordered[:-1] if scope == 'source' else ordered[-1:] if scope == 'target' else ordered
    xs, ys, ws, ds = [], [], [], []
    fitids, parentids, featureids, domain_masses = {}, {}, {}, {}
    for j, bank in enumerate(use):
        labels = np.asarray(bank['target'] if head == 'agreement' else bank['state'])
        if labels.shape != (len(bank['ids']),):
            raise ValueError('Training labels must align with sample IDs')
        if not np.isin(labels, (0, 1) if head == 'agreement' else (0, 1, 2)).all():
            raise ValueError('Unexpected training target/state')
        mask = labels != 2 if head == 'preference' else np.ones(len(labels), bool)
        y = labels[mask] == 0 if head == 'preference' else labels[mask] != 2 if head == 'trust' else labels[mask].astype(bool)
        if not len(y):
            continue
        xs.append(historical.raw_features(bank, cfg['mode'])[mask])
        ys.append(y)
        ds.append(np.full(len(y), j))
        dataset = bank['dataset']
        fitids[dataset] = np.asarray(bank['ids'])[mask].tolist()
        parentids[dataset] = np.asarray(bank['node_ids'])[mask].tolist()
        featureids[dataset] = np.asarray(bank['ids']).tolist()
        mass = (.25 if dataset == target else .75/4) if scope in ('pooled', 'transfer') else 1/len(use)
        domain_masses[dataset] = mass
        ws.append(np.full(len(y), mass/len(y)))
    if not xs:
        raise ValueError('No eligible training rows for requested head')
    x, y, w, domains = map(np.concatenate, (xs, ys, ws, ds))
    w *= len(w)/w.sum()
    transform = historical.Transform.fit(x, w, cfg['rank'])
    xx = transform.apply(x)
    cfg['target_domain'] = len(use)-1 if scope == 'transfer' else -1
    estimator = _fit_estimator(xx, y, w, domains, cfg)
    train_z = _predict_estimator(estimator, xx)
    center = float(np.average(train_z, weights=w))
    scale = max(float(np.sqrt(np.average((train_z-center)**2, weights=w))), .1)
    audit = dict(fit_sample_ids=fitids, fit_parent_ids=parentids,
        transform_fit_sample_ids=deepcopy(fitids), transform_fit_parent_ids=deepcopy(parentids),
        raw_feature_sample_ids=featureids, normalization_fit_sample_ids=deepcopy(fitids),
        domain_masses=domain_masses, fit_rows=len(y), target_domain=cfg['target_domain'],
        validation_training_rows=0, query_training_rows=0, validation_loss_evaluations=0,
        environment_features=False, view_weights='all unit; historical dataset mass retained',
        early_stopping=False)
    return dict(version='Router-v8(v7)', head=head, target=target, config=cfg,
        transform=transform, estimator=estimator, mode=cfg['mode'], center=center, scale=scale,
        fit_ids=fitids, fit_audit=audit)


def predict_candidate(model, bank) -> np.ndarray:
    original._slurm()
    if model['estimator'].get('kind') == 'joint_shared':
        from .joint_engine import predict_candidate as predict_joint
        return predict_joint(model, bank)
    x = historical.raw_features(bank, model['mode'])
    return np.asarray(_predict_estimator(model['estimator'], model['transform'].apply(x)), float).reshape(-1)


def describe_space():
    return dict(version='Router-v8(v7)', stages={
        'capacity': dict(count=39, reason='Single-factor MLP width/dropout/weight decay, finer LR C, and tree depth/leaf size.'),
        'nonlinear': dict(count=30, reason='Independent HGB and RBF-SVC families on existing gold-fitted transforms.'),
        'training': dict(count=21, disagreement_count=27, agreement_count=21, joint_additions=6, reason='Longer historical training, residual architectures, five-seed averaging; six shared-trunk joint-head candidates on disagreement only.')},
        fallback='Driver retains sealed original and preceding-stage winners.',
        validation_fitting=False, view_conditioning=False)
