"""Replay/freeze two faithful Router-v8 endpoints without opening query truth."""
import os
from pathlib import Path

import joblib
import numpy as np
import torch

from eargtc.router_v8_exploration.core import apply_affine, score_payload, top_mask
from eargtc.router_v8_learned.data import load_bank as load_disagreement
from eargtc.router_v8_learned.experiment import _predict, fingerprint, TARGET_FILES
from eargtc.agreement_increment.data import load_bank as load_agreement
from eargtc.agreement_increment.experiment import features
from eargtc.agreement_increment.models import predict_model
from eargtc.agreement_increment.policy import top_mask as agreement_top_mask
from .protocol import DATASETS, roots, read, write, sha256, check_hashes, require_slurm, verify_preregistration


def _replay_disagreement(root, dataset, pre, device):
    dest = root / 'targets' / dataset
    complete = read(dest / 'COMPLETE.json')
    if complete['preregistration_sha256'] != fingerprint(pre) or set(complete['files']) != set(TARGET_FILES):
        raise ValueError('Disagreement target completion differs from registration')
    check_hashes({str(dest / name): digest for name, digest in complete['files'].items()})
    model = torch.load(dest / 'model.pt', map_location='cpu', weights_only=False)
    selection = read(dest / 'selection.json')
    payloads = {}
    for split, filename in (('validation', 'validation_scores.pt'), ('query', 'scores.pt')):
        bank = load_disagreement(root, dataset, split)
        raw = {h: _predict(model['heads'][h], bank, device) for h in ('trust', 'preference')}
        calibrated = {h: apply_affine(raw[h], selection['calibration'][h]) for h in raw}
        expected = score_payload(dict(node_ids=bank['ids'], gnn_pred=bank['gnn_pred'],
                                      llm_pred=bank['llm_pred']), calibrated['trust'],
                                 calibrated['preference'], selection['route_boundary'])
        saved = torch.load(dest / filename, map_location='cpu', weights_only=True)
        for key, value in expected.items():
            if key in ('node_ids', 'gnn_pred', 'llm_pred', 'route'):
                if not torch.equal(value, saved[key]):
                    raise ValueError(f'Disagreement discrete replay differs: {key}')
            elif not torch.allclose(value, saved[key], atol=5e-5, rtol=1e-5):
                raise ValueError(f'Disagreement score replay differs: {key}')
        payloads[split] = saved
    policy = selection['top_fraction_policy']
    query = payloads['query']
    accepted = np.zeros(len(query['node_ids']), dtype=bool) if policy['reject_all'] else top_mask(
        query['selected_score'].numpy(), query['node_ids'].numpy(), policy['fraction'])
    return payloads, torch.from_numpy(accepted), selection


def _replay_agreement(root, dataset):
    dest = root / 'targets' / dataset
    selection = read(dest / 'selection.json')
    check_hashes({str(dest / 'model.joblib'): selection['model_sha256'],
                  str(dest / 'scores.pt'): selection['score_sha256'],
                  str(root / 'pipeline' / f'{dataset}.pt'): selection['pipeline_sha256']})
    model = joblib.load(dest / 'model.joblib')
    payloads = {}
    for split, filename, score_key in (('validation', 'validation_scores.pt', 'score'),
                                        ('query', 'scores.pt', 'selected_score')):
        bank = load_agreement(root, dataset, split)
        saved = torch.load(dest / filename, map_location='cpu', weights_only=True)
        predicted = np.asarray(predict_model(model, features(bank)), float)
        if (not torch.equal(saved['node_ids'], bank['ids'])
                or not np.allclose(predicted, saved[score_key].numpy(), rtol=1e-5, atol=5e-5)):
            raise ValueError('Agreement frozen model replay differs')
        payloads[split] = dict(node_ids=bank['ids'], selected_score=saved[score_key],
                               gnn_pred=bank['gnn_pred'], llm_pred=bank['llm_pred'])
    q = payloads['query']
    accepted = torch.from_numpy(agreement_top_mask(q['selected_score'].numpy(), q['node_ids'].numpy(),
                                                   selection['policy']['fraction']))
    pipeline = torch.load(root / 'pipeline' / f'{dataset}.pt', map_location='cpu', weights_only=True)
    if (not torch.equal(pipeline['node_ids'], q['node_ids'])
            or not torch.equal(pipeline['accepted_mask'], accepted)):
        raise ValueError('Agreement acceptance replay differs')
    return payloads, accepted, selection


def export(run_root, split_seed, device='cpu'):
    require_slurm()
    seed, disagreement, agreement = roots(run_root, split_seed)
    if (seed / 'router_work/V8_FROZEN.json').exists():
        raise FileExistsError('V8 exports already frozen')
    pre = verify_preregistration(disagreement)
    check_hashes(read(agreement / 'REGISTRATION.json')['files'])
    files = {}
    audits = {}
    # Require all five selections before exporting any endpoint.
    for dataset in DATASETS:
        for root in (disagreement, agreement):
            if not (root / 'targets' / dataset / 'selection.json').is_file():
                raise FileNotFoundError(f'Missing five-domain selection: {root}/{dataset}')
    for dataset in DATASETS:
        raw = torch.load(seed / 'upstream' / dataset / 'pipeline/predictions.pt',
                         map_location='cpu', weights_only=True)
        g, l = raw['gnn_logits'].argmax(1), raw['llm_logits'].argmax(1)
        n = len(g)
        split = read(Path(run_root).resolve() / 'splits' / f'{dataset}_5shot_seed{split_seed}.json')
        query_ids = torch.tensor(split['unlabeled_ids'], dtype=torch.long)
        valid_ids = torch.tensor(split['valid_ids'], dtype=torch.long)
        dscores, dkeep, dselection = _replay_disagreement(disagreement, dataset, pre, device)
        ascores, akeep, aselection = _replay_agreement(agreement, dataset)
        common = dict(node_ids=torch.arange(n), choose_g=torch.ones(n, dtype=torch.bool),
            trust=torch.zeros(n, dtype=torch.float32), accepted=torch.zeros(n, dtype=torch.bool),
            selected_pred=g.clone(), agreement=g == l,
            selected_score=torch.zeros(n, dtype=torch.float64))
        for name, scores in dscores.items():
            ids = scores['node_ids']
            expected = valid_ids if name == 'validation' else query_ids
            expected = expected[g[expected] != l[expected]]
            if not torch.equal(ids, expected):
                raise ValueError('Disagreement row identity differs from new split')
            if not torch.equal(scores['gnn_pred'], g[ids]) or not torch.equal(scores['llm_pred'], l[ids]):
                raise ValueError('Disagreement expert identities changed')
            common['choose_g'][ids] = scores['route'] == 0
            common['trust'][ids] = scores['selected_score'].float()
            common['selected_score'][ids] = scores['selected_score'].double()
        common['selected_pred'] = torch.where(common['choose_g'], g, l)
        common['accepted'][dscores['query']['node_ids']] = dkeep
        endpoints = {'r8_disagreement': common, 'r8_full': {k: v.clone() for k, v in common.items()}}
        full = endpoints['r8_full']
        for name, scores in ascores.items():
            ids = scores['node_ids']
            expected = valid_ids if name == 'validation' else query_ids
            expected = expected[g[expected] == l[expected]]
            if (not torch.equal(ids, expected) or not torch.equal(scores['gnn_pred'], g[ids])
                    or not torch.equal(scores['llm_pred'], l[ids])):
                raise ValueError('Agreement identity differs from new split/experts')
            full['selected_score'][ids] = scores['selected_score'].double()
        full['accepted'][ascores['query']['node_ids']] = akeep
        for key in ('node_ids', 'choose_g', 'trust', 'selected_pred', 'agreement'):
            if not torch.equal(common[key], full[key]):
                raise ValueError(f'Full endpoint altered fixed Joint feature {key}')
        conflict = g != l
        for key in common:
            if not torch.equal(common[key][conflict], full[key][conflict]):
                raise ValueError(f'Full endpoint altered conflict field {key}')
        for rid, payload in endpoints.items():
            if (not bool(torch.isfinite(payload['trust']).all())
                    or bool(((payload['trust'] < 0) | (payload['trust'] > 1)).any())
                    or not bool(torch.isin(payload['accepted'].nonzero().flatten(), query_ids).all())):
                raise ValueError('Unsafe exported trust or acceptance population')
            out = seed / 'router' / rid / dataset
            out.mkdir(parents=True, exist_ok=True)
            with (out / 'predictions.pt').open('xb') as handle:
                torch.save(payload, handle)
            metadata = dict(dataset=dataset, split_seed=split_seed, model_seed=42, router_id=rid,
                algorithm='frozen learned v4' if rid == 'r8_disagreement' else 'frozen learned v4 + independent agreement v5',
                disagreement_selection=dselection,
                agreement_selection=aselection if rid == 'r8_full' else None,
                trust_definition='v4 selected_score=q_route on conflict; zero on agreement in both endpoints',
                agreement_score_definition='full selected_score agreement rows carry original selector ranking score; Joint trust unchanged',
                cross_branch_score_calibration=False, conflict_fields_identical=True,
                accepted_query_only=True, query_truth_read=False,
                source_registrations={str(disagreement / 'PREREGISTERED.json'): sha256(disagreement / 'PREREGISTERED.json'),
                                      str(agreement / 'REGISTRATION.json'): sha256(agreement / 'REGISTRATION.json')},
                predictions_sha256=sha256(out / 'predictions.pt'))
            write(out / 'selection.json', metadata)
            for filename in ('predictions.pt', 'selection.json'):
                files[str(out / filename)] = sha256(out / filename)
        audits[dataset] = dict(conflict_replay=True, agreement_replay=True, conflict_union_exact=True,
            disagreement_accepted=int(dkeep.sum()), agreement_accepted=int(akeep.sum()))
    for root in (disagreement, agreement):
        for path in sorted((root / 'targets').rglob('*')):
            if path.is_file():
                files[str(path)] = sha256(path)
    for path in (disagreement / 'PREREGISTERED.json', agreement / 'REGISTRATION.json', agreement / 'CONFIG.json'):
        files[str(path)] = sha256(path)
    record = dict(schema='multiseed_v8_pretest_freeze_v1', files=files, datasets=list(DATASETS),
        split_seed=split_seed, model_seed=42, audits=audits, test_labels_accessed=False,
        slurm_job_id=os.environ['SLURM_JOB_ID'])
    # Both original stage guards now reject further training or policy changes.
    write(disagreement / 'selection/FROZEN.json', record)
    write(agreement / 'FROZEN.json', record)
    write(seed / 'router_work/V8_FROZEN.json', record)
    return record
