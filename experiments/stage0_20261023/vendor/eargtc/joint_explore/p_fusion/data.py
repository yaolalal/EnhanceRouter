"""Authenticate historical arrays and split validation without query labels."""
import hashlib
import torch
from .protocol import ROOT, SOURCE, DEPLOYMENT, RUN, read, sha, verify, load, slurm


def load_data(seed, ds):
    slurm()
    from eargtc.multiseed_joint.experiment import prepare_arrays, input_paths, DEFAULT_RUN
    from scripts.v9new_multiseed.final_audit import verify_deployed_router
    from eargtc.joint_cross_repair.artifacts import validate_hidden
    reg = read(RUN / 'REGISTRATION.json')
    paths = input_paths(DEFAULT_RUN, seed, 'r9_new', ds)
    paths['router'] = SOURCE / f'seed{seed}/router/r9_new/{ds}/predictions.pt'
    source_cell = SOURCE / f'seed{seed}/joint/r9_new/{ds}'
    paths.update(saved_supervision=source_cell/'supervision.pt', saved_pools=source_cell/'pools.pt',
                 deployed=DEPLOYMENT/f'seed{seed}/pipeline/{ds}.pt',
                 disagreement=DEPLOYMENT/f'seed{seed}/models/{ds}/disagreement/query.pt')
    hashes = {}
    for path in paths.values():
        # Authenticate the actual resolved input even when a historical path is a symlink.
        key = str(path.resolve())
        if key not in reg['input_hashes']:
            raise ValueError('Input not sealed: ' + key)
        hashes[key] = reg['input_hashes'][key]
    verify(hashes)
    split = read(paths['split'])
    if split['dataset'] != ds or split['split_seed'] != seed or split.get('test_truth_exported') is not False:
        raise ValueError('Historical split identity differs')
    upstream, emb, safe, router = [load(paths[k]) for k in ('upstream','embeddings','supervision','router')]
    features, supervision, pools = prepare_arrays(split, upstream, emb, safe, router, ds)
    hidden = load(paths['hidden'])
    validate_hidden(hidden, split['num_nodes'])
    if hidden['metadata']['dataset'] != ds:
        raise ValueError('LLM hidden dataset differs')
    verify_deployed_router(load(paths['deployed']), load(paths['disagreement']), router, split)
    for current, key in ((supervision, 'saved_supervision'), (pools, 'saved_pools')):
        saved = load(paths[key])
        if set(current) != set(saved) or any(not torch.equal(v, saved[k]) for k, v in current.items()):
            raise ValueError('Historical supervision/pools differ')
    accepted = torch.cat([pools[k] for k in ('teach_g', 'teach_l', 'agreement')]).sort().values
    if len(accepted.unique()) != len(accepted) or not torch.equal(accepted, router['node_ids'][router['accepted']]):
        raise ValueError('Accepted pseudo pool differs from deployment')
    ranked = sorted(supervision['selection_ids'].tolist(), key=lambda i:
        (hashlib.sha256(f'p-fusion-validation-v1\0{ds}\0{seed}\0{i}'.encode()).hexdigest(), i))
    cut = (len(ranked) + 1)//2
    by_id = dict(zip(supervision['selection_ids'].tolist(), supervision['selection_targets'].tolist()))
    def partition(ids):
        ids = sorted(ids)
        return dict(ids=torch.tensor(ids, dtype=torch.long), targets=torch.tensor([by_id[i] for i in ids]))
    checkpoint, selection = partition(ranked[:cut]), partition(ranked[cut:])
    data = dict(graph=features['gnn_embeddings'], text=hidden['hidden'], classes=features['logp_g'].shape[1],
                node_ids=features['node_ids'], gold_ids=supervision['gold_ids'], gold_targets=supervision['gold_targets'],
                accepted_ids=accepted, pseudo_labels=features['selected_pred'], agreement=features['agreement'], seed=seed)
    if torch.isin(accepted, torch.cat((supervision['gold_ids'], supervision['selection_ids'], supervision['threshold_ids']))).any():
        raise ValueError('Pseudo supervision overlaps protected IDs')
    metadata = dict(seed=seed, dataset=ds, input_hashes=hashes, historical_split_sha256=sha(paths['split']),
                    counts=dict(gold=len(supervision['gold_ids']), checkpoint=len(checkpoint['ids']),
                                selection=len(selection['ids']), accepted=len(accepted), query=len(supervision['test_ids'])),
                    exact_deployment_parity=True, exact_historical_supervision=True, test_truth_read=False)
    return data, checkpoint, selection, features, supervision, pools, split, metadata
