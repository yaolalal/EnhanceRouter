"""Isolated orchestration around the exact Router-v10 v2 frozen engines."""
import sys,os,json,hashlib,copy
from pathlib import Path
REPO=Path('/home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter')
BASE=REPO/'runs/router_v10_final'
ROOT=REPO/'runs/router_v10_shots_v1_20260914'
SHOT=int(os.environ.get('ROUTER_SHOTS','3'))
if SHOT not in (3,10):raise ValueError('Only authorized new shots 3 and 10')
DATA=ROOT/f'shot{SHOT}'
RUN=DATA/'router'
PREVIOUS=RUN/'incumbents'
CODE=REPO/'scripts/router_v10_shots_v1'
CHOICES=REPO/'runs/router_v10_observed_best_five_seeds_20260913_v1/CHOICES.json'
sys.path.insert(0,str(BASE/'code'))
import numpy as np
import torch,joblib
from eargtc.router_aug_search.common import guard_bank,engine as old_engine,fit_affine_grouped
from eargtc.router_v10_tune.runtime import scores,predict
DATASETS=('cora','citeseer','pubmed','arxiv','ogbn-products')
SEEDS=(42,43,44)
BRANCHES=('agreement','disagreement')
QUERY_KEYS={'dataset','ids','tab','g','l','gnn_pred','llm_pred'}
def require():
 if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p)
 if not p.resolve().is_relative_to(RUN.resolve()):raise ValueError('New run only')
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def save(p,v):
 require();p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not p.resolve().is_relative_to(RUN.resolve()):raise ValueError('New run only')
 with p.open('xb') as f:torch.save(v,f)
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not p.resolve().is_relative_to(RUN.resolve()):raise ValueError('New run only')
 with p.open('xb') as f:joblib.dump(v,f,compress=3)
def tensor_tree(x):
 if isinstance(x,np.ndarray):return torch.as_tensor(x)
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:tensor_tree(v) for k,v in x.items()}
 if isinstance(x,list):return [tensor_tree(v) for v in x]
 if isinstance(x,tuple):return tuple(tensor_tree(v) for v in x)
 return x

def load(p):
 require();return torch.load(p,weights_only=True,map_location='cpu')
def checked(p):
 p=Path(p);reg=read(RUN/'REGISTRATION.json');assert sha(p)==reg['inputs'][str(p)],str(p)
 return p
def check_files(files):
 for p,h in files.items():assert sha(p)==h,p

def check_training_identity(parents,s):
 p=set(parents)
 if not p<=set(s['support_ids']) or p.intersection(s['valid_ids']) or p.intersection(s['unlabeled_ids']):raise ValueError('Non-gold training parent')
def check_query_keys(b):
 if set(b)!=QUERY_KEYS:raise ValueError('Unsafe query keys')
def split_path(seed,d):return DATA/'splits'/f'{d}_{SHOT}shot_seed{seed}.json'
def split(seed,d):
 s=read(checked(split_path(seed,d)))
 assert s['split_seed']==seed and s['dataset']==d and s['shots']==SHOT
 return s

def training(seed,branch):
 require();assert_open(seed);out={}
 for d in DATASETS:
  s=split(seed,d);p=DATA/f'seed{seed}/banks'/d/(branch+'_train.pt');b=load(checked(p))
  check_training_identity(b['node_ids'].tolist(),s)
  guard_bank(b,dict(gold_ids=s['support_ids'],validation_ids=s['valid_ids'],query_ids=s['unlabeled_ids']))
  assert b['dataset']==d;out[d]=b
 return out

def eval_bank(seed,d,branch,role):
 require();assert role in ('validation','query')
 if role=='validation':assert_open(seed)
 if role=='query':assert (RUN/f'seed{seed}/MODELS_FROZEN.json').exists(),'Query features before freeze'
 p=DATA/f'seed{seed}/banks'/d/(branch+'_'+role+'.pt')
 b=load(checked(p));s=split(seed,d)
 assert b['dataset']==d and set(b['ids'].tolist())<=set(s['valid_ids'] if role=='validation' else s['unlabeled_ids'])
 assert len(b['ids'])==len(set(b['ids'].tolist()))
 assert bool(((b['gnn_pred']==b['llm_pred'])==(branch=='agreement')).all())
 if role=='query':
  # Historical banks include role metadata; enumerate and reject every unknown field before stripping it.
  assert set(b)<=QUERY_KEYS|{'split'} and ('split' not in b or b['split']=='query')
  b={k:b[k] for k in QUERY_KEYS};check_query_keys(b)
 return b

def source_key(cfg,head,sources,banks):
 # Byte-equivalent to frozen _source_key for nonempty arrays; NumPy memoryview.cast
 # rejects zero-length multidimensional buffers. tobytes preserves the zero-byte payload.
 c={k:v for k,v in cfg.items() if k not in ('id','target_steps')}
 h=hashlib.sha256(json.dumps(dict(head=head,config=c),sort_keys=True).encode())
 for d in sources:
  h.update(d.encode())
  for k in ('ids','node_ids','tab','g','l','state'):
   v=np.ascontiguousarray(np.asarray(banks[d][k]));h.update(json.dumps([k,str(v.dtype),list(v.shape)]).encode());h.update(v.tobytes(order='C'))
 return h.hexdigest()
def engine(name):
 if name=='tune':
  import tune_engine as e
 else:e=old_engine(name)
 if name=='v6':e._source_key=source_key
 return e

def fit(w,h,banks,d):return dict(engine=w['engine'],model=engine(w['engine']).fit_candidate(copy.deepcopy(w['model']['config']),h,banks,d))
def norm(w,h,banks,d):
 if w['engine']=='v6':
  r=fit_affine_grouped(banks,d,h,lambda bs:fit(w,h,bs,d),predict)
  return {k:float(r['affine'][k]) for k in ('slope','bias')},r
 m=w['model'];return dict(slope=1/m['scale'],bias=-m['center']/m['scale']),dict(mode='train_center_scale',center=m['center'],scale=m['scale'])
def exact(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(exact(v,b[k]) for k,v in a.items())
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return a==b

def eligible(w,h,banks,d):
 c=w['model']['config'];scope=c.get('scope','transfer' if c.get('kind')=='neural' else 'target')
 use=[d] if scope=='target' else [x for x in DATASETS if x!=d] if scope=='source' else DATASETS
 return sum(int((b['state']!=2).sum()) if h=='preference' else len(b['ids']) for x in use for b in [banks[x]])
def freeze_files(folder):return {str(p):sha(p) for p in Path(folder).rglob('*') if p.is_file()}


def assert_open(seed):
 require()
 if (ROOT/'ROUTER_FROZEN.json').exists() or any((RUN/f'seed{seed}'/p).exists() for p in ('MODELS_FROZEN.json','FROZEN.json','EVALUATION_STARTED.json')):raise RuntimeError('Training/calibration permanently closed')

def check_code():
 reg=read(RUN/'REGISTRATION.json');check_files(reg['code'])

def raw(w,b):
 z=predict(w,b)
 if 'gold_normalization' in w:
  a=w['gold_normalization'];return a['slope']*z+a['bias']
 if w['engine']=='v6' and 'target' in b and w['model']['head']=='agreement':
  from scipy.special import logit
  return logit(np.clip(z,1e-6,1-1e-6))
 if w['engine']=='v6' and w['model']['head']=='agreement':
  from scipy.special import logit
  return logit(np.clip(z,1e-6,1-1e-6))
 m=w['model']
 if 'center' in m:return (z-m['center'])/m['scale']
 return z

def predict_bundle(bundle,b):
 if bundle.get('unavailable') or not len(b['ids']):
  n=len(b['ids']);return dict(score=np.zeros(n),route=np.zeros(n,dtype=np.int64),q=np.tile([0.,0.,1.],(n,1)))
 if bundle.get('tuning_schema')!='crossfit_v1':return scores(bundle,b)
 from scipy.special import expit
 if bundle['branch']=='agreement':
  z=raw(bundle['models']['agreement'],b);a=bundle['calibration']['agreement']
  return dict(score=expit(a['slope']*z+a['bias']))
 from policy import apply_pair
 z={h:raw(w,b) for h,w in bundle['models'].items()}
 return apply_pair(z,bundle['calibration'],bundle['route_recipe'])


def source_choice(seed,d,branch):
 rows=read(checked(CHOICES))['rows']
 return next(r for r in rows if (r['seed'],r['dataset'],r['branch'])==(seed,d,branch))

def validation_features(seed,d,branch):
 p=DATA/f'seed{seed}/banks'/d/(branch+'_validation.pt')
 b=load(checked(p));ids=set(b['ids'].tolist())
 assert ids<=set(split(seed,d)['valid_ids'])
 return {k:b[k] for k in QUERY_KEYS}
