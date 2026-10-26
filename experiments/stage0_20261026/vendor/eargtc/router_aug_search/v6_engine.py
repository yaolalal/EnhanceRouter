"""Exact historical Router-v8(v6) catalogs and training-only augmented refits.

Catalog construction reads recipe JSON, never fitted checkpoints or scores. The
three inherited agreement branches keep their one historical recipe. Numerical
entry points require Slurm; callers authenticate gold parents before supplying
prepared banks. Observation IDs, parent IDs and environments remain audit data.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FINAL = REPO / 'runs/router_v8_v6_20260910'
EXP04 = REPO / 'runs/router_v8_v6_exp04_superseded_20260909'
EXP03 = REPO / 'runs/router_v8_v6_exp03_superseded_20260909'
GOLD = REPO / 'runs/router_v8_v6_exp02_superseded_20260909'
DATASETS = ('cora', 'citeseer', 'pubmed', 'arxiv', 'ogbn-products')
HEADS = ('trust', 'preference', 'agreement')
_SOURCE_CACHE = OrderedDict()
_SOURCE_CACHE_LIMIT = 32


def _read(path):
    return json.loads(path.read_text())


def _require_slurm():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Numerical v6 fitting/prediction requires Slurm')


def _identity(dataset, head):
    if dataset not in DATASETS or head not in HEADS:
        raise ValueError('Unknown dataset or head')


def _normalize(cfg):
    cfg = copy.deepcopy(cfg)
    cfg.pop('id', None)
    if cfg['kind'] == 'neural':
        cfg['consistency'] = 0.
    elif cfg.get('scope') is None:
        cfg['scope'] = 'pooled' if cfg.get('training_scope') == 'source_target' else 'target'
    return cfg


def incumbent_config(dataset: str, head: str) -> dict:
    """Return the historical refit incumbent, without loading its old weights."""
    _identity(dataset, head)
    if head == 'agreement':
        old = _read(GOLD / 'targets' / dataset / 'selection.json')['configs']['agreement']
        return dict(id='000_incumbent', config=copy.deepcopy(old['model']), scope=old['scope'])
    cfg = _normalize(_read(EXP04 / 'targets' / dataset / 'selection.json')['chosen'][head]['config'])
    return dict(cfg, id='incumbent')


def final_config(dataset: str, head: str) -> dict:
    """Return the final v6 winner recipe (distinct from its search incumbent)."""
    _identity(dataset, head)
    branch = 'agreement' if head == 'agreement' else 'disagreement'
    path = FINAL / 'branches' / dataset / branch / 'selection.json'
    if not path.is_file():
        return incumbent_config(dataset, head)
    selection = _read(path)
    if head == 'agreement':
        row = selection['chosen']
        return dict(id=row['id'], config=copy.deepcopy(row['config']), scope=row['scope'])
    chosen_id = selection['selected'][head + '_id']
    return copy.deepcopy(next(row['config'] for row in selection['shortlists'][head] if row['id'] == chosen_id))


def final_policy_recipe(dataset: str) -> dict:
    """Expose finite historical post-calibration constants, never a fitted affine.

    Final-v6 saved composed affine maps. Factor out its gold-OOF affine solely
    to recover the validation-selected post-transform recipe, which callers may
    apply to their freshly cross-fitted augmented calibration. This does not
    reuse the old supervised calibration slope/bias as a fitted transform.
    """
    _require_slurm()
    _identity(dataset, 'trust')
    path = FINAL / 'branches' / dataset / 'disagreement' / 'selection.json'
    if path.is_file():
        selection = _read(path)
        chosen = selection['selected']
        if chosen['id'] != '000_exact_incumbent':
            post = {}
            for head in ('trust', 'preference'):
                gold = next(row['affine'] for row in selection['gold_calibration']
                            if row['head'] == head and row['id'] == chosen[head + '_id'])
                combined = chosen[head + '_affine']
                if gold['slope'] == 0:
                    raise ValueError('Cannot recover post-recipe from zero gold slope')
                slope = combined['slope'] / gold['slope']
                post[head] = dict(slope=slope, bias=combined['bias'] - slope * gold['bias'])
            return dict(post=post, boundary=chosen['boundary'], fraction=selection['acceptance']['fraction'],
                        source=str(path), recipe_only=True, reuse_gold_affine=False)
    # Inherited branches and the exact old-incumbent recipe already store the
    # post maps separately from gold-fitted calibration.
    inherited_path = EXP04 / 'targets' / dataset / 'selection.json'
    selection = _read(inherited_path)
    old = selection['post']
    return dict(post={h: copy.deepcopy(old[h]) for h in ('trust', 'preference')},
                boundary=old['route']['threshold'], fraction=selection['acceptance']['fraction'],
                source=str(inherited_path), recipe_only=True, reuse_gold_affine=False)


def candidate_grid(dataset: str, head: str) -> list[dict]:
    """Preserve the final branch registry or its actual inherited branch trials.

    Refinements in exp03 were adaptive. Their *realized per-dataset recipes*,
    inherited by exp04, are reused, rather than recombining their hyperparameters.
    The historical query-consistency option is disabled as in final v6.
    """
    _identity(dataset, head)
    final = _read(FINAL / 'REGISTRATION.json')
    if head == 'agreement':
        rows = [incumbent_config(dataset, head)]
        if dataset in ('arxiv', 'ogbn-products'):
            rows.extend(copy.deepcopy(final['agreement_configs']))
        return rows
    if dataset in final['disagreement_configs']:
        return copy.deepcopy(final['disagreement_configs'][dataset][head])
    old = _read(EXP04 / 'search' / dataset / head / 'trials.json')
    recipes = [incumbent_config(dataset, head)] + [row['config'] for row in old]
    seen, result = set(), []
    for recipe in recipes:
        cfg = _normalize(recipe)
        fingerprint = json.dumps(cfg, sort_keys=True)
        if fingerprint not in seen:
            seen.add(fingerprint)
            cfg['id'] = 'incumbent' if not result else f'candidate_{len(result):03d}'
            result.append(cfg)
    return result


def selection_incumbent_id(dataset: str, head: str) -> str:
    """Map inherited exp04's AUROC-floor incumbent (the exp03 winner) to its ID.

    This is different from the exp04 winner retained at catalog position zero.
    The final v6 citeseer/arxiv protocol does not use this inherited floor.
    """
    _identity(dataset, head)
    if head == 'agreement' or dataset not in ('cora', 'pubmed', 'ogbn-products'):
        raise ValueError('The exp03 floor applies only to inherited disagreement searches')
    old = _read(EXP03 / 'targets' / dataset / 'selection.json')['chosen'][head]['config']
    recipe = _normalize(old)
    matches = [cfg['id'] for cfg in candidate_grid(dataset, head) if _normalize(cfg) == recipe]
    if len(matches) != 1:
        raise ValueError('Historical exp03 floor recipe is missing or ambiguous in normalized catalog')
    return matches[0]


def describe_space(dataset: str, head: str) -> dict:
    """Catalog lineage and immutable recipe-source hashes for preregistration."""
    rows = candidate_grid(dataset, head)
    paths = [FINAL / 'REGISTRATION.json']
    if head == 'agreement':
        paths.append(GOLD / 'targets' / dataset / 'selection.json')
        origin = 'final_v6_agreement_search' if dataset in ('arxiv', 'ogbn-products') else 'inherited_exp02_fixed_agreement'
    else:
        paths += [EXP04 / 'targets' / dataset / 'selection.json', EXP04 / 'search' / dataset / head / 'trials.json']
        origin = 'final_v6_disagreement_search' if dataset in ('citeseer', 'arxiv') else 'inherited_exp04_disagreement_search'
        if dataset in ('cora', 'pubmed', 'ogbn-products'):
            paths.append(EXP03 / 'targets' / dataset / 'selection.json')
    return dict(dataset=dataset, head=head, candidate_count=len(rows), origin=origin,
        catalog_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        recipe_source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        consistency=0, incumbent='refit historical recipe on supplied augmented training banks',
        agreement_incumbent_alias_retained=head == 'agreement' and len(rows) > 1,
        notes='Actual per-dataset lineage; no Cartesian expansion of adaptive exp03 refinements. No historical fitted weights or affine calibration loaded.')


def _prepare(banks, target, head):
    import torch
    if target not in banks or set(banks) - set(DATASETS):
        raise ValueError('Missing target or unknown training dataset')
    clean = {}
    for dataset, bank in banks.items():
        if bank.get('dataset') != dataset or bank.get('split') != 'train':
            raise ValueError('All supplied banks must identify their dataset and train split')
        needed = ('ids', 'node_ids', 'environment_index', 'weight', 'tab', 'g', 'l',
                  'target' if head == 'agreement' else 'state')
        if any(k not in bank for k in needed):
            raise ValueError('Prepared bank is missing features, targets or provenance')
        n = len(bank['ids'])
        for key in ('ids', 'node_ids', 'environment_index', 'weight'):
            if torch.as_tensor(bank[key]).shape != (n,):
                raise ValueError('Invalid observation provenance shape')
        ids = torch.as_tensor(bank['ids'])
        parents = torch.as_tensor(bank['node_ids'])
        env = torch.as_tensor(bank['environment_index'])
        if ids.dtype != torch.int64 or parents.dtype != torch.int64 or ids.unique().numel() != n:
            raise ValueError('Sample IDs must be unique int64 values; parents must be int64')
        if (ids < 0).any() or (parents < 0).any() or ((env < 0) | (env >= 6)).any():
            raise ValueError('Invalid sample, parent or six-view environment identity')
        if not torch.equal(torch.as_tensor(bank['weight']), torch.ones(n)):
            raise ValueError('Every augmented observation must have unit weight')
        if any(k in bank for k in ('gold_target', 'gold_labels', 'labels', 'query_truth', 'validation_truth')):
            raise ValueError('Raw truth is not a model input')
        label_key = 'target' if head == 'agreement' else 'state'
        labels = torch.as_tensor(bank[label_key])
        valid = (labels == 0) | (labels == 1)
        if head != 'agreement':
            valid |= labels == 2
        if labels.shape != (n,) or not valid.all():
            raise ValueError('Invalid branch supervision')
        for key, width in (('tab', 43 if head == 'agreement' else 70), ('g', 128), ('l', 4096)):
            if torch.as_tensor(bank[key]).shape != (n, width):
                raise ValueError('Invalid historical feature dimensions')
        # The historical neural validator rejects augmentation/audit-only keys.
        clean[dataset] = {k: bank[k] for k in ('dataset', 'split', 'ids', 'tab', 'g', 'l', label_key)}
    return clean


def _parent_ids(ids_by_dataset, banks):
    result = {}
    for dataset, ids in ids_by_dataset.items():
        mapping = dict(zip(banks[dataset]['ids'].tolist(), banks[dataset]['node_ids'].tolist()))
        result[dataset] = [int(mapping[int(i)]) for i in ids]
    return result


def _audit(stages, banks, *, mass=None):
    """Union source and adaptation parameter IDs; retain true transform scope."""
    input_ids, supervised, transforms = {}, {}, {}
    for stage in stages:
        for dest, names in ((input_ids, ('input_ids', 'training_ids')),
                            (supervised, ('supervised_ids', 'training_ids')),
                            (transforms, ('transform_fit_ids', 'feature_fit_ids'))):
            values = next((stage[k] for k in names if k in stage), {})
            for dataset, ids in values.items():
                dest[dataset] = list(dict.fromkeys(dest.get(dataset, []) + list(ids)))
    return dict(input_ids=input_ids, supervised_ids=supervised, transform_fit_ids=transforms,
        input_parent_ids=_parent_ids(input_ids, banks),
        supervised_parent_ids=_parent_ids(supervised, banks),
        transform_fit_parent_ids=_parent_ids(transforms, banks),
        domain_loss_mass=mass if mass is not None else copy.deepcopy(stages[-1].get('domain_loss_mass', {})),
        stages=copy.deepcopy(stages), augmented_observations_equal_weight=True,
        environment_features=False, validation_training_rows=0, validation_transform_rows=0,
        query_training_rows=0, calibration='none; caller owns parent-group OOF')


def _source_key(cfg, head, sources, banks):
    """Content-hash every source input and target, not mutable object pointers.

    Rehashing detects in-place edits (including edits through NumPy aliases) and
    prevents source-fold or dataset substitutions. Target-only OOF changes do
    not invalidate a source fit. Environment metadata never enters the key.
    """
    import numpy as np
    source_cfg = {k: v for k, v in cfg.items() if k not in ('id', 'target_steps')}
    digest = hashlib.sha256(json.dumps(dict(head=head, config=source_cfg), sort_keys=True).encode())
    for dataset in sources:
        digest.update(dataset.encode())
        for key in ('ids', 'node_ids', 'tab', 'g', 'l', 'state'):
            value = np.ascontiguousarray(np.asarray(banks[dataset][key]))
            digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
            digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def fit_candidate(cfg: dict, head: str, banks: dict[str, dict], target: str) -> dict:
    """Refit a recipe using training banks only; no validation argument exists."""
    _require_slurm()
    _identity(target, head)
    cfg = copy.deepcopy(cfg)
    prepared = _prepare(banks, target, head)
    sources = [d for d in DATASETS if d != target]
    scope = cfg.get('scope')
    if head != 'agreement' and cfg['kind'] == 'neural':
        if float(cfg.get('consistency', 0)) != 0:
            raise ValueError('Unlabeled consistency is outside the augmented training protocol')
        if any(d not in prepared for d in sources):
            raise ValueError('Historical neural source scope requires all other datasets')
        from eargtc.router_v8_gold_training.model import fit_head
        cache_key = _source_key(cfg, head, sources, banks)
        cache_hit = cache_key in _SOURCE_CACHE
        if cache_hit:
            _SOURCE_CACHE.move_to_end(cache_key)
        else:
            source = fit_head([prepared[d] for d in sources], head, cfg, device='cpu',
                              steps=cfg['source_steps'], unlabeled=[])
            _SOURCE_CACHE[cache_key] = copy.deepcopy(source)
            if len(_SOURCE_CACHE) > _SOURCE_CACHE_LIMIT:
                _SOURCE_CACHE.popitem(last=False)
        fitted = copy.deepcopy(_SOURCE_CACHE[cache_key])
        fitted['config'] = copy.deepcopy(cfg)
        stages = [fitted['training_audit']]
        if cfg['target_steps']:
            fitted = fit_head([prepared[target]], head, cfg, device='cpu', initial=fitted,
                              steps=cfg['target_steps'], unlabeled=[])
            stages.append(fitted['training_audit'])
        audit = _audit(stages, banks)
        audit.update(source_cache_hit=cache_hit, source_cache_key=cache_key)
    else:
        if scope not in ('source', 'pooled', 'target'):
            raise ValueError('Historical scope must be source, pooled or target')
        # Order matches the original agreement/disagreement training recipes.
        selected = ([target] if scope == 'target' else sources if scope == 'source'
                    else [target] + sources if head == 'agreement' else sources + [target])
        if any(d not in prepared for d in selected):
            raise ValueError('Missing historical source training dataset')
        if head == 'agreement':
            import numpy as np
            from eargtc.agreement_increment.models import fit_predict
            selected = [d for d in selected if len(prepared[d]['ids'])]
            if not selected:
                raise ValueError('Agreement recipe has no training observations')
            xs = {k: np.concatenate([np.asarray(prepared[d][k]) for d in selected]) for k in ('tab', 'g', 'l')}
            y = np.concatenate([np.asarray(prepared[d]['target'], dtype=bool) for d in selected])
            mass = {d: (.5 if d == target else .5 / len(sources)) if scope == 'pooled'
                    else 1 / len(selected) for d in selected}
            weight = np.concatenate([np.full(len(prepared[d]['ids']), mass[d] / len(prepared[d]['ids'])) for d in selected])
            weight *= len(weight) / weight.sum()
            # Even the prediction argument is training-only. Validation/query are
            # separately predicted after this function returns its fitted artifact.
            _, fitted = fit_predict(cfg['config'], xs, y, xs, sample_weight=weight, seed=42, device='cpu')
            ids = {d: prepared[d]['ids'].tolist() for d in selected}
            supervised = {} if fitted['kind'] == 'baseline' else ids
            transform_ids = {} if fitted['kind'] in ('baseline', 'constant') else ids
            stage = dict(fitted.get('training_audit', {}), input_ids=ids,
                         supervised_ids=supervised, transform_fit_ids=transform_ids,
                         domain_loss_mass=mass)
            audit = _audit([stage], banks, mass=mass)
        else:
            from eargtc.router_v8_learned.experiment import fit_tabular
            fitted = fit_tabular([prepared[d] for d in selected], head, cfg)
            stage = dict(fitted['training_audit'], input_ids={d: prepared[d]['ids'].tolist() for d in selected})
            audit = _audit([stage], banks)
    return dict(schema='router_aug_v6_candidate_v1', version='v6', head=head, target=target,
                config=cfg, fitted=fitted, training_audit=audit)


def predict_candidate(model: dict, bank: dict):
    """Return raw disagreement logits or original agreement probability scores."""
    _require_slurm()
    import numpy as np
    if model.get('schema') != 'router_aug_v6_candidate_v1':
        raise ValueError('Unknown v6 fitted artifact schema')
    fitted = model['fitted']
    features = {k: bank[k] for k in ('tab', 'g', 'l')}
    if model['head'] == 'agreement':
        from eargtc.agreement_increment.models import predict_model
        result = predict_model(fitted, features)
    elif fitted.get('kind', 'neural') == 'neural':
        from eargtc.router_v8_gold_training.model import predict_head
        result = predict_head(fitted, dict(features, ids=bank['ids']), device='cpu')
    else:
        from eargtc.router_v8_learned.experiment import predict_tabular
        result = predict_tabular(fitted, features)
    result = np.asarray(result, dtype=np.float64)
    if result.shape != (len(bank['ids']),) or not np.isfinite(result).all():
        raise ValueError('Invalid raw candidate predictions')
    return result
