"""Rebuild unchanged 70/43-column evidence with split-safe supervision only."""
from pathlib import Path
import os
import shutil
import uuid

import torch

from eargtc.router_v8.evidence import probability_evidence, structural_evidence
from eargtc.router_v8.prototypes import fit_class_prototypes, prototype_evidence
from eargtc.router_v8_learned.data import (
    validate_bank as validate_disagreement, take, _historical_masks,
    _deterministic_subset_indices, _hidden_row_hash,
)
from eargtc.agreement_increment.data import validate_bank as validate_agreement
from eargtc.router_bc_split_c.calibration import deterministic_calibration_threshold_split
from .protocol import REPO, DATASETS, roots, require_slurm, read, write, sha256, check_hashes


def _safe_load(path, keys):
    value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'Unsafe schema at {path}: expected only {keys}')
    return value


def _hidden(dataset, n, inputs):
    root = REPO / 'runs/router_v7_decision_llm_hidden_20260827_v1'
    hidden_path = root / dataset / 'hidden.pt'
    # AUDIT alone has no content binding; retain the archived safe-export hash binding.
    binding = REPO / 'runs/joint_cross_repair_export_20260906_v2' / dataset
    complete = read(binding / 'complete.json')
    metadata = read(binding / 'metadata.json')
    if (complete.get('dataset') != dataset or complete.get('test_labels_exported') is not False
            or sha256(binding / 'metadata.json') != complete['artifacts']['metadata.json']):
        raise ValueError('Historical hidden provenance record failed')
    expected = metadata['source_hashes'].get(str(hidden_path))
    if metadata['llm_hidden_path'] != str(hidden_path) or sha256(hidden_path) != expected:
        raise ValueError('LLM hidden bytes differ from historical binding')
    audit = read(root / 'AUDIT.json')['datasets'][dataset]
    if (audit['status'] != 'pass' or audit['rows'] != n or audit['width'] != 4096
            or audit['dtype'] != 'float16'):
        raise ValueError('LLM hidden audit failed')
    value = _safe_load(hidden_path, ('node_ids', 'hidden', 'metadata'))
    if (not torch.equal(value['node_ids'], torch.arange(n))
            or value['hidden'].shape != (n, 4096) or value['hidden'].dtype != torch.float16
            or value['metadata']['dataset'] != dataset
            or value['metadata']['row_order_hash'] != _hidden_row_hash(value['node_ids'])):
        raise ValueError('LLM hidden row identity failed')
    for path in (hidden_path, root / 'AUDIT.json', binding / 'complete.json', binding / 'metadata.json'):
        inputs[str(path)] = expected if path == hidden_path else sha256(path)
    return value['hidden']


def prepare(run_root, split_seed, dataset):
    require_slurm()
    if dataset not in DATASETS:
        raise ValueError('Unknown dataset')
    seed, disagreement_root, agreement_root = roots(run_root, split_seed)
    existing = [root / 'banks' / dataset for root in (disagreement_root, agreement_root)]
    if all((p / 'manifest.json').is_file() for p in existing):
        for directory in existing:
            manifest = read(directory / 'manifest.json')
            if manifest['split_seed'] != split_seed or manifest['dataset'] != dataset:
                raise ValueError('Existing bank identity changed')
            check_hashes(manifest['input_hashes'])
            check_hashes({str(directory / name): digest for name, digest in manifest['bank_hashes'].items()})
        return dict(dataset=dataset, split_seed=split_seed, already_prepared_verified=True)
    if any(p.exists() for p in existing):
        raise FileExistsError('Incomplete existing bank; refusing to overwrite')
    upstream = seed / 'upstream' / dataset
    split_path = Path(run_root).resolve() / 'splits' / f'{dataset}_5shot_seed{split_seed}.json'
    paths = [split_path, upstream / 'pipeline/predictions.pt', upstream / 'supervision.pt',
             upstream / 'graph.pt', upstream / 'embeddings.pt']
    completion = read(upstream / 'COMPLETE.json')
    if (completion.get('dataset') != dataset or completion.get('split_seed') != split_seed
            or completion.get('all_checks_pass') is not True or completion.get('test_truth_exported') is not False):
        raise ValueError('Upstream completion failed identity or label safety')
    check_hashes({str(upstream / name): digest for name, digest in completion['files'].items()})
    paths.append(upstream / 'COMPLETE.json')
    inputs = {str(p): sha256(p) for p in paths}
    split = read(split_path)
    n = int(split['num_nodes'])
    ids = torch.arange(n)
    support = torch.tensor(split['support_ids'], dtype=torch.long)
    valid = torch.tensor(split['valid_ids'], dtype=torch.long)
    query = torch.tensor(split['unlabeled_ids'], dtype=torch.long)
    owner = torch.zeros(n, dtype=torch.bool)
    for part in (support, valid, query):
        if (part.unique().numel() != part.numel() or bool(((part < 0) | (part >= n)).any())
                or bool(owner[part].any())):
            raise ValueError('Split IDs overlap, repeat, or leave node universe')
        owner[part] = True
    if not bool(owner.all()):
        raise ValueError('Support/validation/query must partition all nodes')
    supervision = _safe_load(upstream / 'supervision.pt',
                             ('support_ids', 'support_targets', 'valid_ids', 'valid_targets'))
    if (not torch.equal(support, supervision['support_ids'])
            or not torch.equal(valid, supervision['valid_ids'])):
        raise ValueError('Supervision identity differs from outer split')
    raw = _safe_load(upstream / 'pipeline/predictions.pt', ('gnn_logits', 'llm_logits'))
    graph = _safe_load(upstream / 'graph.pt', ('edge_index', 'num_nodes'))
    if int(graph['num_nodes']) != n:
        raise ValueError('Graph node universe differs from split')
    embedding = _safe_load(upstream / 'embeddings.pt', ('embeddings', 'node_ids'))
    if not torch.equal(embedding['node_ids'], ids):
        raise ValueError('Upstream embedding row identity differs')
    g = embedding['embeddings']
    pg, pl = raw['gnn_logits'].float().softmax(1), raw['llm_logits'].float().softmax(1)
    if pg.shape != pl.shape or pg.shape[0] != n or g.shape != (n, 128):
        raise ValueError('Upstream embedding/logit dimensions differ from historical architecture')
    pred_g, pred_l = pg.argmax(1), pl.argmax(1)
    hidden = _hidden(dataset, n, inputs)
    ssl_path = REPO / 'runs/gcn_ssl_v2_selected_20260623_v1' / dataset / 'encoder_and_embeddings.pt'
    ssl = torch.load(ssl_path, map_location='cpu', weights_only=True, mmap=True)['embeddings']
    ssl_manifest = REPO / 'runs/router_v8_trust_preference_20260906_v1/banks' / dataset / 'manifest.json'
    ssl_expected = read(ssl_manifest)['input_hashes'][str(ssl_path)]
    if sha256(ssl_path) != ssl_expected:
        raise ValueError('Prototype SSL bytes differ from frozen v8 provenance')
    inputs[str(ssl_path)] = ssl_expected
    inputs[str(ssl_manifest)] = sha256(ssl_manifest)
    if not torch.equal(g, ssl):
        raise ValueError('Raw GNN embedding no longer matches verified historical SSL source')
    if ssl.shape != (n, 128) or ssl.dtype != torch.float32:
        raise ValueError('Prototype embedding identity changed')
    labels = torch.full((n,), -1, dtype=torch.long)
    labels[support] = supervision['support_targets'].long()
    labels[valid] = supervision['valid_targets'].long()
    if (bool(((labels[support] < 0) | (labels[support] >= pg.shape[1])).any())
            or bool(((labels[valid] < 0) | (labels[valid] >= pg.shape[1])).any())):
        raise ValueError('Allowed supervision labels outside class space')
    probability = probability_evidence(pg, pl, pred_g, pred_l)
    structure = structural_evidence(edge_index=graph['edge_index'], full_gnn_pred=pred_g,
        row_ids=ids, candidate_g=pred_g, candidate_l=pred_l, num_nodes=n, num_classes=pg.shape[1])
    fitted = fit_class_prototypes(ssl, support, labels[support], num_classes=pg.shape[1])
    prototype = prototype_evidence(embeddings=ssl, row_ids=ids, candidate_g=pred_g,
                                   candidate_l=pred_l, fitted=fitted)
    cg = torch.cat((probability[0], structure[0], prototype[0]), 1)
    cl = torch.cat((probability[1], structure[1], prototype[1]), 1)
    global_ = torch.cat((probability[2], structure[2], prototype[2]), 1)
    agree = pred_g == pred_l
    if not torch.equal(cg[agree], cl[agree]):
        raise ValueError('Agreement candidate evidence must be identical')
    for is_agreement, root in ((False, disagreement_root), (True, agreement_root)):
        population = agree if is_agreement else ~agree
        vi, qi = valid[population[valid]], query[population[query]]
        tab = torch.cat((cg, global_), 1) if is_agreement else torch.cat((cg, cl, global_), 1)
        def bank(row_ids, labeled=False):
            out = dict(dataset=dataset, ids=row_ids, tab=tab[row_ids].float().contiguous(),
                g=g[row_ids].float().contiguous(), l=hidden[row_ids].half().contiguous(),
                gnn_pred=pred_g[row_ids], llm_pred=pred_l[row_ids])
            if labeled:
                if is_agreement:
                    halves = deterministic_calibration_threshold_split(row_ids, dataset, 'agreement')
                    out.update(target=pred_g[row_ids] == labels[row_ids],
                               cal_mask=halves.calibration, policy_mask=halves.threshold)
                else:
                    cal, pol = _historical_masks(row_ids, dataset)
                    state = torch.where(pred_g[row_ids] == labels[row_ids], 0,
                                        torch.where(pred_l[row_ids] == labels[row_ids], 1, 2))
                    out.update(state=state.long(), cal_mask=cal, policy_mask=pol)
            if is_agreement:
                validate_agreement(out, split='validation' if labeled else 'query', dataset=dataset)
            else:
                validate_disagreement(out, labeled=labeled)
            return out
        banks = dict(validation=bank(vi, True), query=bank(qi))
        if not is_agreement:
            banks['unlabeled'] = take(banks['query'], _deterministic_subset_indices(qi, dataset))
        destination = root / 'banks' / dataset
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = destination.parent / f'.{dataset}.{uuid.uuid4().hex}'
        stage.mkdir()
        try:
            for name, payload in banks.items():
                torch.save(payload, stage / f'{name}.pt')
            manifest = dict(schema='agreement_increment_bank_v1' if is_agreement else 'router_v8_learned_bank_v1',
                dataset=dataset, split_seed=split_seed, model_seed=42,
                bank_hashes={f'{name}.pt': sha256(stage / f'{name}.pt') for name in banks},
                input_hashes=inputs, prototype_fit_ids=support.tolist(),
                counts={name: len(payload['ids']) for name, payload in banks.items()},
                split_ids=dict(gold=support.tolist(), validation=valid.tolist(), query=query.tolist()),
                label_access=dict(support_prototype_only=True, validation_supervised=True,
                                  query_labels_read=False, whole_graph_truth_loaded=False),
                features='original probability_evidence + structural_evidence + prototype_evidence',
                slurm_job_id=os.environ['SLURM_JOB_ID'])
            write(stage / 'manifest.json', manifest)
            os.rename(stage, destination)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    return dict(dataset=dataset, split_seed=split_seed, prepared=True, test_truth_read=False)
