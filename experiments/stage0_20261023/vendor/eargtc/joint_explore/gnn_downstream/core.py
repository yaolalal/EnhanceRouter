"""Numerical primitives; called only by Slurm entrypoints."""
import os,random
import numpy as np
import torch
import torch.nn.functional as F

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def assert_open(root):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    if any((root/p).exists() for p in ('FROZEN.json','EVALUATION_STARTED.json')):
        raise RuntimeError('Training is permanently closed')

def validate_pools(n,support,valid,query,pseudo):
    groups=[list(map(int,g)) for g in (support,valid,query,pseudo)]
    for g in groups:
        if len(g)!=len(set(g)) or any(i<0 or i>=n for i in g):raise ValueError('Invalid node IDs')
    s,v,q,p=map(set,groups)
    if s&v or s&q or v&q or s|v|q!=set(range(n)) or not p<=q:raise ValueError('Unsafe supervision pools')
    if not s or not v:raise ValueError('Support and validation required')

def adjacency(edge,n):
    # Identical normalization and duplicate/self-loop convention to _gcn_aggregate.
    loops=torch.arange(n,device=edge.device)
    row=torch.cat((edge[0],loops));col=torch.cat((edge[1],loops))
    deg=torch.bincount(row,minlength=n).float().clamp_min(1)
    weights=deg[row].pow(-.5)*deg[col].pow(-.5)
    return torch.sparse_coo_tensor(torch.stack((col,row)),weights,(n,n)).coalesce()

def encode(encoder,x,adj):
    for i,conv in enumerate(encoder.convs):
        if conv.gnn_type!='GCN':raise ValueError('Only historical GCN supported')
        x=conv.linear(torch.sparse.mm(adj,x))
        if i<len(encoder.convs)-1:
            x=F.dropout(F.relu(encoder.norms[i](x)),p=encoder.dropout,training=encoder.training)
    return x

def loss(logits,support,targets,pseudo,pseudo_targets):
    out=F.cross_entropy(logits[support],targets)
    if pseudo.numel():out=out+F.cross_entropy(logits[pseudo],pseudo_targets)
    return out

def optimizer(encoder,head,tune,head_lr,encoder_lr,wd):
    encoder.requires_grad_(tune)
    groups=[{'params':head.parameters(),'lr':head_lr}]
    if tune:groups.append({'params':encoder.parameters(),'lr':encoder_lr})
    return torch.optim.AdamW(groups,weight_decay=wd)
