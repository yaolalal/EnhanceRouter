"""Explicit, read-only historical sources; output supervision comes from safe gold only."""
import hashlib
import json
import os
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/router_augmentation_v8_20260911_v1'
SAFE=ROOT/'runs/joint_direct_numbered_router_export_20260813_v1'
GOLD=ROOT/'runs/router_v8_v6_exp02_superseded_20260909/banks'
GRAPH=ROOT/'runs/router_v7_graph_environments_20260815_v2'
TEXT=ROOT/'runs/router_v7_text_environments_20260815_v2'
TEXT_INPUT=ROOT/'runs/router_v7_text_inputs_20260815_v3'
TEXT_LOGITS=ROOT/'runs/router_v7_text_logits_20260815_v2'
CLEAN_LOGITS=ROOT/'runs/router_v7_text_clean_logits_20260815_v1'
HIDDEN=ROOT/'runs/router_v7_decision_llm_hidden_20260827_v1'
SSL=ROOT/'runs/gcn_ssl_v2_selected_20260623_v1'
DATASETS=('cora','citeseer','pubmed','arxiv','ogbn-products')


def require_slurm():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Numerical work requires Slurm')


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def load(path):
    require_slurm()
    return torch.load(path,map_location='cpu',weights_only=True,mmap=True)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:torch.save(value,f)


def gold_source(dataset):
    meta=read(SAFE/dataset/'safe_input_metadata.json')
    splits=read(SAFE/dataset/'split_ids.json')
    supervision=load(SAFE/dataset/'training_supervision.pt')
    ids=supervision['gold_support_ids'].long(); y=supervision['gold_support_targets'].long()
    assert ids.tolist()==splits['train']
    assert not set(ids.tolist()) & (set(splits['validation'])|set(splits['query']))
    raw=load(SAFE/dataset/'router_tensors.pt')
    return dict(ids=ids,y=y,meta=meta,splits=splits,raw=raw)


def cached_environment(dataset, name):
    """Discard historical support OOF and all correctness/label tensors immediately.

    Legacy containers physically include whole-graph labels. None of those fields
    are returned or used: only the full-head query probabilities/predictions and
    IDs are whitelisted. New target construction uses safe gold_source.y only.
    """
    graph=name=='clean' or name.startswith('graph')
    root=GRAPH if graph else TEXT
    container=load(root/dataset/('graph_environments.pt' if graph else 'text_environments.pt'))
    query=container['environments'][name]['query']
    return {k:query[k] for k in ('node_ids','gnn_prob','llm_prob','gnn_pred','llm_pred')}


def positions(all_ids, wanted):
    lookup={int(n):i for i,n in enumerate(all_ids)}
    if len(lookup)!=len(all_ids):raise ValueError('duplicate source node IDs')
    try:return torch.tensor([lookup[int(n)] for n in wanted],dtype=torch.long)
    except KeyError as e:raise ValueError('source misses gold node') from e
