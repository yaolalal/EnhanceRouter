"""Immutable manifests and checkpoint construction."""
import hashlib,json,os
from pathlib import Path
import torch
from eargtc.models.gnn import GNNEncoder,MLPHead
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/router_v9new_gnn_downstream_20260913_v1'
UP=ROOT/'runs/multiseed_reproduction_20260909_v1'
SSL=ROOT/'runs/gcn_ssl_v2_selected_20260623_v1'
ROUTER=(ROOT/'runs/router_v9_new_final').resolve()
JOINT=ROOT/'runs/router_v9_new_joint_multiseed_20260912_v2'
DATASETS=('cora','citeseer','pubmed','arxiv','ogbn-products')
VARIANTS=('router_frozen','router_finetune','gold_frozen','gold_finetune')

def require():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if p.exists():
        if p.read_text()!=content:raise RuntimeError('Refusing overwrite: '+str(p))
    else:
        with p.open('x') as f:f.write(content)
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise RuntimeError('Refusing overwrite: '+str(p))
    tmp=p.with_suffix(p.suffix+'.tmp');torch.save(obj,tmp);tmp.replace(p)
def verify(mapping):
    for p,h in mapping.items():
        if sha(p)!=h:raise ValueError('Hash mismatch: '+str(p))
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def models(bundle,device):
    c=bundle['encoder_config'];t=bundle['head_trial']
    e=GNNEncoder(bundle['input_dim'],c['gnn_hidden_dim'],c['embedding_dim'],c['gnn_layers'],'GCN',c.get('gnn_dropout',.3))
    h=MLPHead(c['embedding_dim'],t['head_hidden_dim'],bundle['classes'],t['dropout'])
    e.load_state_dict(bundle['encoder_state']);h.load_state_dict(bundle['head_state'])
    return e.to(device),h.to(device)
def code_hashes():
    paths=list((ROOT/'eargtc/joint_explore/gnn_downstream').glob('*.py'))+list((ROOT/'scripts/joint_explore/gnn_downstream').glob('*.py'))
    paths += [ROOT/'eargtc/models/gnn.py',ROOT/'eargtc/data.py',ROOT/'eargtc/utils.py']
    return {str(p):sha(p) for p in sorted(paths)}

def same_content(a,b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(same_content(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,type(a)):
        return len(a)==len(b) and all(same_content(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b

def save_or_verify(p,obj):
    p=Path(p)
    if p.exists():
        if not same_content(load(p),obj):raise ValueError('Existing bundle differs from reconstructed source: '+str(p))
    else:save(p,obj)
