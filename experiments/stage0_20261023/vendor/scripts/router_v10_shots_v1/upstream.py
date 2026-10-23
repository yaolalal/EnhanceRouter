"""Exact frozen 3/10-shot inputs and fresh supervised/structural OOF experts."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'extra_experiments')]
from scripts.multiseed_reproduction.audit_splits import sha, save

DATASETS = ('cora', 'citeseer', 'pubmed', 'arxiv', 'ogbn-products')
SEEDS = (42, 43, 44)
SHOTS = (3, 10)
CELLS = tuple((shot, seed, dataset) for shot in SHOTS for seed in SEEDS for dataset in DATASETS)
DEFAULT_RUN = ROOT / 'runs/router_v10_shots_v1_20260914'
REFERENCE = ROOT.parent / 'GNN-as-Judge'
FROZEN = REFERENCE / 'results/ablations/no_gnn_retrain_3_10shot/frozen_splits'
HISTORICAL = ROOT / 'runs/router_v5_structural_oof_fixed_20260729_v1'
SSL = ROOT / 'runs/gcn_ssl_v2_selected_20260623_v1'
RAW = ROOT / 'runs/joint_direct_numbered_router_export_20260813_v1'


def normalize_frozen_split(source_path, record, manifest_path):
    """Validate exported identity; copy exact arrays without sampling or sorting."""
    source_path, manifest_path = Path(source_path), Path(manifest_path)
    if source_path.resolve() != Path(record['frozen_path']).resolve() or sha(source_path) != record['frozen_sha256']:
        raise ValueError('Frozen source path or SHA-256 mismatch')
    source = json.loads(source_path.read_text())
    for key in ('dataset', 'shots', 'seed', 'hash'):
        if source[key] != record[key]:
            raise ValueError(f'Manifest metadata mismatch: {key}')
    n = source['node_count']
    if type(n) is not int or n < 1:
        raise ValueError('Invalid node count')
    keys = ('train_ids', 'val_ids', 'test_ids', 'evaluation_test_ids')
    for key in keys:
        ids = source[key]
        if not isinstance(ids, list) or any(type(i) is not int or i < 0 or i >= n for i in ids):
            raise ValueError(f'Invalid {key}')
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate {key}')
    train, valid, query, evaluation = (source[k] for k in keys)
    sets = list(map(set, (train, valid, query)))
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Forbidden support/validation/query overlap')
    if set.union(*sets) != set(range(n)):
        raise ValueError('Incomplete graph partition')
    if not set(evaluation).issubset(sets[2]):
        raise ValueError('Evaluation IDs are outside query')
    legacy = hashlib.md5(str(train + valid + query).encode()).hexdigest()[:8]
    if legacy != source['hash']:
        raise ValueError('Frozen split hash mismatch')
    for key, ids in zip(('train_count', 'val_count', 'test_pool_count', 'evaluation_count'), (train, valid, query, evaluation)):
        if record[key] != len(ids):
            raise ValueError(f'Manifest count mismatch: {key}')
    return {'dataset': source['dataset'], 'shots': source['shots'], 'split_seed': source['seed'],
            'num_nodes': n, 'reference_path': str(source_path.resolve()),
            'reference_sha256': sha(source_path), 'reference_split_hash': legacy,
            'reference_device': source.get('device'),
            'frozen_manifest_path': str(manifest_path.resolve()), 'frozen_manifest_sha256': sha(manifest_path),
            'support_ids': train, 'valid_ids': valid, 'unlabeled_ids': query,
            'evaluation_full_ids': query, 'evaluation_1000_ids': evaluation, 'evaluation_test_ids': evaluation,
            'evaluation_1000_provenance': 'exact_frozen_evaluation_test_ids_in_saved_order',
            'test_truth_exported': False}


def audit_splits(run_root):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('All numerical execution requires Slurm')
    manifest_path = FROZEN / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    records = manifest['rows']
    expected = set(CELLS)
    actual = [(r['shots'], r['seed'], r['dataset']) for r in records]
    if len(actual) != len(expected) or set(actual) != expected or manifest['count'] != len(expected):
        raise ValueError('Frozen manifest does not cover exactly all 30 cells')
    graph_hashes = {d: sha(REFERENCE / 'datasets' / f'{d}.pt') for d in DATASETS}
    audited = []
    for record in records:
        dataset, shot, seed = record['dataset'], record['shots'], record['seed']
        source = FROZEN / f'{dataset}_{shot}shot_seed{seed}.json'
        normalized = normalize_frozen_split(source, record, manifest_path)
        # Bind the actual baseline split and fixed evaluation export as well.
        split_src, eval_src = Path(record['split_path']), Path(record['test_sample_path'])
        if sha(split_src) != record['split_sha256'] or sha(eval_src) != record['test_sample_sha256']:
            raise ValueError('Baseline split/evaluation source SHA-256 mismatch')
        original, evaluation = json.loads(split_src.read_text()), json.loads(eval_src.read_text())
        assert original['train_ids'] == normalized['support_ids']
        assert original['val_ids'] == normalized['valid_ids']
        assert original['hash'] == normalized['reference_split_hash']
        assert original['test_count'] == len(normalized['unlabeled_ids'])
        assert evaluation['selected_node_ids'] == normalized['evaluation_test_ids']
        normalized.update(graph_path=str(REFERENCE / 'datasets' / f'{dataset}.pt'), graph_sha256=graph_hashes[dataset])
        output = run_root / f'shot{shot}/splits/{source.name}'
        save(output, normalized)
        loaded = json.loads(output.read_text())
        for old_key, new_key in [('train_ids', 'support_ids'), ('val_ids', 'valid_ids'), ('test_ids', 'unlabeled_ids'), ('evaluation_test_ids', 'evaluation_1000_ids')]:
            assert loaded[new_key] == json.loads(source.read_text())[old_key]
        audited.append({'dataset': dataset, 'shot': shot, 'seed': seed,
                        'export': str(output.relative_to(run_root)), 'export_sha256': sha(output),
                        'reference_sha256': normalized['reference_sha256'], 'reference_hash': normalized['reference_split_hash']})
        print(f'Exact frozen split verified: {dataset}/{shot}shot/seed{seed}', flush=True)
    report_path = run_root / 'SPLIT_AUDIT.json'
    report = {'all_checks_pass': True, 'count': len(audited), 'records': audited,
              'manifest_path': str(manifest_path), 'manifest_sha256': sha(manifest_path),
              'no_splits_generated': True, 'evaluation_order_preserved': True,
              'slurm_job_id': os.environ['SLURM_JOB_ID']}
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        report['slurm_job_id'] = previous['slurm_job_id']
    save(report_path, report)


def poison_query_labels(labels, train, valid, query):
    """Expose only permitted targets and discard all query truth before fitting."""
    clean = labels.view(-1).long().clone()
    supervision = {'support_ids': train.cpu(), 'support_targets': clean[train].cpu().clone(),
                   'valid_ids': valid.cpu(), 'valid_targets': clean[valid].cpu().clone()}
    clean[query] = -1
    assert bool((clean[query] == -1).all())
    return clean, supervision


def validate_support_counts(targets, *, num_classes, shot):
    import torch
    if bool(((targets < 0) | (targets >= num_classes)).any()):
        raise ValueError('Support label outside frozen expert class space')
    counts = torch.bincount(targets, minlength=num_classes)
    if bool((counts > shot).any()):
        raise ValueError('Support class count exceeds shot budget')
    return counts


def run_cell(run_root, dataset, split_seed, shot):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use Slurm for all numerical work')
    import torch
    from eargtc.data import load_graph_dataset, get_texts
    from eargtc.utils import set_seed
    from eargtc.router import _stratified_folds
    from eargtc.router_bc_structural_fix.oof import build_fold_full_graph_oof_rows, combine_fold_oof_rows
    from eargtc.router_bc_structural_fix.audit import audit_fold_oof_result
    from scripts.router_bc_structural_fix.build_corrected_input_root import write_corrected_oof_artifacts
    from tune_gcn_ssl_1000_eval import train_head, HeadTrial
    torch.set_num_threads(4)
    final = run_root / f'shot{shot}/seed{split_seed}/upstream/{dataset}'
    audit = json.loads((run_root / 'SPLIT_AUDIT.json').read_text())
    assert audit['all_checks_pass'] and sha(FROZEN / 'manifest.json') == audit['manifest_sha256']
    record = next(r for r in audit['records'] if (r['dataset'], r['seed'], r['shot']) == (dataset, split_seed, shot))
    split_path = run_root / record['export']
    assert sha(split_path) == record['export_sha256']
    split = json.loads(split_path.read_text())
    source_split = Path(split['reference_path'])
    assert sha(source_split) == split['reference_sha256']
    if (final / 'COMPLETE.json').exists():
        completed = json.loads((final / 'COMPLETE.json').read_text())
        assert (completed['dataset'], completed['split_seed'], completed['shot']) == (dataset, split_seed, shot)
        assert completed['split_sha256'] == sha(split_path)
        for path, digest in completed['files'].items():
            assert sha(final / path) == digest, path
        print(f'Validated complete upstream {dataset}/{shot}shot/seed{split_seed}', flush=True)
        return
    if final.exists():
        raise FileExistsError(f'Refusing incomplete final upstream directory: {final}')
    dest = final.with_name(f'.{dataset}.stage-{os.environ["SLURM_JOB_ID"]}')
    dest.mkdir(parents=True, exist_ok=False)
    pipeline = dest / 'pipeline'
    pipeline.mkdir()
    if not torch.cuda.is_available():
        raise RuntimeError('All production supervised heads require CUDA, including seed42')
    device = torch.device('cuda:0')
    assert sha(Path(split['graph_path'])) == split['graph_sha256']
    data = load_graph_dataset(dataset, REFERENCE, device)
    n = int(data.num_nodes)
    assert n == split['num_nodes']
    train = torch.tensor(split['support_ids'], dtype=torch.long, device=device)
    valid = torch.tensor(split['valid_ids'], dtype=torch.long, device=device)
    query = torch.tensor(split['unlabeled_ids'], dtype=torch.long, device=device)
    data.y, supervised = poison_query_labels(data.y, train, valid, query)
    torch.save(supervised, dest / 'supervision.pt')
    torch.save({'edge_index': data.edge_index.cpu(), 'num_nodes': n}, dest / 'graph.pt')
    data.train_mask = torch.zeros(n, dtype=torch.bool, device=device)
    data.val_mask = torch.zeros(n, dtype=torch.bool, device=device)
    data.test_mask = torch.zeros(n, dtype=torch.bool, device=device)
    data.train_mask[train] = True
    data.val_mask[valid] = True
    data.test_mask[query] = True
    old_pipeline = HISTORICAL / dataset / 'pipeline'
    old_predictions_path = old_pipeline / 'predictions.pt'
    old = torch.load(old_predictions_path, map_location='cpu', weights_only=True)
    llm_logits = old['llm_logits'].float().to(device)
    del old
    num_classes = int(llm_logits.shape[1])
    assert llm_logits.shape[0] == n
    support_class_counts = validate_support_counts(data.y[train], num_classes=num_classes, shot=shot)
    ssl_path = SSL / dataset / 'encoder_and_embeddings.pt'
    emb = torch.load(ssl_path, map_location='cpu', weights_only=True)['embeddings'].float()
    raw_path = RAW / dataset / 'router_tensors.pt'
    raw_g = torch.load(raw_path, map_location='cpu', weights_only=True)['gnn_embeddings'].float()
    assert torch.equal(emb, raw_g), 'Historical raw GNN input differs from SSL representation'
    del raw_g
    torch.save({'embeddings': emb, 'node_ids': torch.arange(n)}, dest / 'embeddings.pt')
    embeddings = emb.to(device)
    del emb
    metrics_path = SSL / dataset / 'mlp_metrics.json'
    trial = json.loads(metrics_path.read_text())['trial']
    selected = json.loads((SSL / 'manifest.json').read_text())['datasets'][dataset]
    trial_no = int(Path(selected['source']['selected_trial_dir']).name.split('_')[0])
    model_seed = 42 + trial_no
    old_manifest_path = HISTORICAL / dataset / 'dataset_manifest.json'
    head_config = json.loads(old_manifest_path.read_text())['oof_head_config']
    old_config = json.loads((old_pipeline / 'config.json').read_text())
    config = {**old_config, 'seed': split_seed, 'split_seed': split_seed, 'shots': shot, 'model_seed': model_seed,
              'oof_seed': 42, 'output_dir': str(final / 'pipeline'), 'reference_split_path': str(source_split),
              'pretrained_ssl_encoder_fixed_across_splits': True, 'llm_zero_shot_fixed_across_splits': True}
    save(pipeline / 'config.json', config)
    split_dest = pipeline / 'splits' / source_split.name
    split_dest.parent.mkdir()
    split_dest.write_bytes(source_split.read_bytes())
    set_seed(model_seed)
    gnn_logits, stats, artifact = train_head(embeddings, data.y, train, valid, num_classes, HeadTrial(**trial))
    torch.save({k: v.cpu() for k, v in artifact['state_dict'].items()}, dest / 'gnn_head.pt')
    save(dest / 'head_training.json', {'trial': trial, 'stats': stats, 'history': artifact['history']})
    folds = _stratified_folds(train, data.y, int(old_config['router_num_folds']), 42)
    results = build_fold_full_graph_oof_rows(
        data=data, texts=get_texts(data), embeddings=embeddings, llm_logits=llm_logits,
        train_idx=train, val_idx=valid, num_classes=num_classes, fold_ids=folds, head_config=head_config, seed=42)
    rows = combine_fold_oof_rows(results, train_idx=train, labels=data.y)
    fold_audits = [audit_fold_oof_result(result=r, data=data, num_classes=num_classes) for r in results]
    assert all(r['all_checks_pass'] for r in fold_audits)
    write_corrected_oof_artifacts(pipeline_dir=pipeline, results=results, combined_rows=rows,
                                 audit_report={'all_checks_pass': True, 'folds': fold_audits})
    payload = {'gnn_logits': gnn_logits.cpu().float(), 'llm_logits': llm_logits.cpu().float()}
    assert all(bool(torch.isfinite(x).all()) for x in payload.values())
    torch.save(payload, pipeline / 'predictions.pt')
    rows = [json.loads(line) for line in (pipeline / 'router/oof/router_train.jsonl').read_text().splitlines() if line.strip()]
    assert [row['node_id'] for row in rows] == split['support_ids']
    assert all(row['y_true'] == int(data.y[row['node_id']].item()) for row in rows)
    # Reload the saved supervised head and verify predictions, before publishing.
    from eargtc.models.gnn import MLPHead
    reloaded = MLPHead(input_dim=embeddings.shape[1], hidden_dim=trial['head_hidden_dim'],
                       output_dim=num_classes, dropout=trial['dropout']).to(device)
    reloaded.load_state_dict(torch.load(dest / 'gnn_head.pt', map_location=device, weights_only=True))
    reloaded.eval()
    with torch.no_grad():
        assert torch.equal(reloaded(embeddings), gnn_logits), 'Reloaded head differs from saved predictions'
    provenance = {'dataset': dataset, 'shot': shot, 'split_seed': split_seed, 'model_seed': model_seed, 'oof_seed': 42,
                  'training': {'historical_seed42_replayed': False, 'fresh_supervised_head': True, 'fresh_structural_oof': True, 'trial': trial, 'stats': stats},
                  'query_labels_poisoned_before_training': True, 'reloaded_head_logits_exact': True,
                  'raw_gnn_embedding_exactly_equals_ssl_cache': True, 'same_fixed_frontend_across_seeds': True,
                  'actual_oof_fold_count': len(folds), 'support_class_counts': support_class_counts.cpu().tolist(), 'sources': {str(p): sha(p) for p in (source_split, split_path, ssl_path, raw_path, old_predictions_path, metrics_path, old_manifest_path, Path(__file__))},
                  'execution_device': str(device), 'torch_version': str(torch.__version__), 'slurm_job_id': os.environ['SLURM_JOB_ID']}
    save(dest / 'PROVENANCE.json', provenance)
    save(dest / 'COMPLETE.json', {'dataset': dataset, 'shot': shot, 'split_seed': split_seed, 'all_checks_pass': True,
        'split_sha256': sha(split_path), 'test_truth_exported': False,
        'files': {str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}})
    dest.rename(final)
    print(f'Completed fresh upstream {dataset}/{shot}shot/seed{split_seed}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--cell-index', type=int)
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--split-seed', type=int, choices=SEEDS)
    parser.add_argument('--shot', type=int, choices=SHOTS)
    args = parser.parse_args()
    if args.audit_only:
        audit_splits(args.run_root.resolve())
    else:
        if args.cell_index is not None:
            if not 0 <= args.cell_index < len(CELLS):
                parser.error('Invalid cell index')
            args.shot, args.split_seed, args.dataset = CELLS[args.cell_index]
        if args.dataset is None or args.split_seed is None or args.shot is None:
            parser.error('Provide cell-index or dataset, split-seed and shot')
        run_cell(args.run_root.resolve(), args.dataset, args.split_seed, args.shot)
