"""Stage-0 preregistered experiments; every numerical entrypoint is Slurm-only."""
from pathlib import Path
import os,sys,json,hashlib,itertools,subprocess,time,copy
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT.parent/'EALR-GTC'
VENDOR=HERE/'vendor'
sys.path[:0]=[str(VENDOR),str(VENDOR/'extra_experiments'),str(ROOT/'src'),str(HERE)]
DS=('cora','citeseer','pubmed','arxiv','ogbn-products')
SHOTS=(3,5,10)
DEV=(42,43,44,45,46)
HELDOUT=()
CELLS=list(itertools.product(SHOTS,DEV,DS))
_run=Path(os.environ.get('ENHANCEROUTER_STAGE0_RUN','runs/stage0-20261026-v2'))
RUN=(_run if _run.is_absolute() else ROOT/_run).resolve()
assert RUN.is_relative_to(ROOT/'runs')
ENVIRONMENTS=('clean','text_truncate_050','text_mask_mixed_030','text_span_delete_030','graph_drop_030','graph_ego_mixed_050')

def require():
 if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('All numerical work must use Slurm')

def read(p):return json.loads(Path(p).read_text())
def jsonable(x):
 if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
 if hasattr(x,'tolist'):return x.tolist()
 return x
def write(p,x):
 p=Path(p);assert p.resolve().is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(jsonable(x),f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def digest(p,kind='sha256'):
 h=hashlib.new(kind)
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 import torch
 p=Path(p);assert p.resolve().is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:torch.save(x,f)
def load(p):
 require();import torch
 return torch.load(p,map_location='cpu',weights_only=True,mmap=True)
def key(shot,seed,ds):return f'{shot}/{seed}/{ds}'
def folder(shot,seed,ds):return RUN/f'cells/shot{shot}/seed{seed}/{ds}'
def variant_folder(shot,seed,ds,variant):
 if variant.startswith('d1/'):
  registration=read(RUN/'EFFECTIVE_GAMMA_REGISTRATION.json')
  affected={r['key'] for r in registration['affected_historical_cells']}
  if key(shot,seed,ds) in affected or (seed>44 and shot!=5):
   variant=variant.replace('d1/','d1_frozen_v2/',1)
 return folder(shot,seed,ds)/variant

def split(shot,seed,ds):return read(RUN/f'splits/{shot}/{seed}/{ds}.json')
def setup():
 require()
 freeze=RUN/'CODE_FREEZE.json'
 if freeze.exists():
  for rel,expected in read(freeze)['files'].items():
   assert digest(ROOT/rel)==expected,'Registered code/config changed before execution: '+rel
 from enhancerouter.io import guard_writes
 guard_writes()
 import torch
 torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',os.environ.get('OMP_NUM_THREADS',2))))
def graph(ds):
 from eargtc.data import _install_pyg_optional_extension_stubs
 _install_pyg_optional_extension_stubs()
 import torch
 idx=read(ROOT/'configs/inputs.json');p=idx['cells'][f'5/42/{ds}']['graph']
 return torch.load(p,map_location='cpu',weights_only=False,mmap=True)
def seal(p,meta):
 p=Path(p);write(p/'COMPLETE.json',dict(meta,files={str(q.relative_to(p)):digest(q) for q in sorted(p.rglob('*')) if q.is_file()}))
def subset(b,mask):
 import torch
 n=len(b['ids']);mask=torch.as_tensor(mask,dtype=torch.bool)
 return {k:(v[mask] if torch.is_tensor(v) and v.ndim and len(v)==n else v) for k,v in b.items()}
def merge_banks(a,b):
 import torch
 assert a['dataset']==b['dataset'] and a.keys()==b.keys()
 return {k:torch.cat((v,b[k])) if torch.is_tensor(v) else v for k,v in a.items()}
def metadata():
 from enhancerouter.io import environment,assert_imports_isolated
 assert_imports_isolated()
 import platform
 env=(dict(base_environment_sha256=digest(RUN/'environment.json'),python=sys.version,host=platform.node())
      if (RUN/'environment.json').exists() else environment())
 return dict(environment=env,code_freeze_sha256=digest(RUN/'CODE_FREEZE.json') if (RUN/'CODE_FREEZE.json').exists() else None,source=read(ROOT/'provenance/source.json'),
  git_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
  code={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(HERE.rglob('*.py'))|set((ROOT/'src').rglob('*.py')))},slurm_job=os.environ['SLURM_JOB_ID'])
def all_sources():
 result=read(RUN/'SOURCE_FILES.json')
 for name in ('SOURCE_EXTENSION_FILES_V2.json','HEAD_SOURCE_FILES.json'):
  if (RUN/name).exists():result.update(read(RUN/name))
 return result
def canonical(ds):
 idx=read(ROOT/'configs/inputs.json');c=idx['cells'][f'5/42/{ds}']
 raw=load(c['inputs']['upstream'])
 return dict(g=load(c['inputs']['embeddings'])['embeddings'].float(),l=load(c['inputs']['hidden'])['hidden'],llm_logits=raw['llm_logits'].float())
def textcache(ds):return RUN/f'shared/text/{ds}.pt'
def bank(shot,seed,ds,branch,role,small=False):
 if not small and (seed<=44 or shot==5):
  idx=read(RUN/'HISTORICAL_INPUTS_V2.json')
  return load(idx['cells'][key(shot,seed,ds)]['branches'][branch][role])
 directory=folder(shot,seed,ds)/('small_native' if small else standard_mode(shot,seed))
 p=directory/f'{branch}_{role}.pt';entry=read(directory/'BANK_COMPLETE.json')['files'][p.name]
 assert digest(p,'md5')==entry['md5'],str(p)
 return load(p)
def config(shot,seed,ds):return read(folder(shot,seed,ds)/'b0/config.json')


def evaluation_ids(seed,ds):
 return read(RUN/'EVALUATION_IDS.json')[f'{seed}/{ds}']

def registered_protocol():
 return read(RUN/'PROTOCOL.json')

def upstream_bank(shot,seed,ds,small=False):
 if not small and (seed<=44 or shot==5):
  c=read(RUN/'HISTORICAL_INPUTS_V2.json')['cells'][key(shot,seed,ds)]
  return load(c['inputs']['upstream'])
 return load(folder(shot,seed,ds)/('small_native' if small else standard_mode(shot,seed))/'upstream.pt')

def supervision_bank(shot,seed,ds,small=False):
 if not small and (seed<=44 or shot==5):
  c=read(RUN/'HISTORICAL_INPUTS_V2.json')['cells'][key(shot,seed,ds)]
  return load(c['inputs']['supervision'])
 return load(folder(shot,seed,ds)/('small_native' if small else standard_mode(shot,seed))/'supervision.pt')

def head_state(shot,seed,ds):
 if seed<=44 or shot==5:
  r=read(RUN/'HEAD_REFERENCES.json')[key(shot,seed,ds)];p=Path(r['path']);assert digest(p)==r['sha256']
  obj=load(p);return obj[r['container']] if r['container'] else obj
 return load(folder(shot,seed,ds)/'standard_native/head.pt')

def standard_mode(shot,seed):
 return 'standard' if seed<=44 or shot==5 else 'standard_native'

def gnn_training_seed(ds):
 c=read(ROOT/'configs/inputs.json')['cells'][f'5/42/{ds}']
 p=Path(c['inputs']['embeddings']).parent/'PROVENANCE.json'
 return int(read(p)['model_seed'])

def canonical_edge(ds):
 c=read(ROOT/'configs/inputs.json')['cells'][f'5/42/{ds}']
 return load(Path(c['inputs']['embeddings']).parent/'graph.pt')['edge_index']

def level1_budget(scores,eval_ids,fraction,mode):
 import torch
 from fractions import Fraction
 mask=torch.isin(scores['ids'],eval_ids);n=int(mask.sum());actual=int(scores['accepted'][mask].sum())
 if mode=='deployed_intersection':primary=actual
 elif mode=='rho_eval_ceil':
  r=Fraction(str(fraction));primary=(n*r.numerator+r.denominator-1)//r.denominator
 else:raise ValueError('Unregistered Level1 budget mode')
 return dict(primary=primary,deployed_intersection=actual,eval_population=n,fraction=float(fraction))

def metric_budget_registration():
 p=RUN/'METRIC_BUDGET_REGISTRATION.json'
 if not p.exists():raise RuntimeError('Awaiting user decision on undefined Level1 acceptance budget; evaluation labels remain closed')
 r=read(p);assert r['mode'] in ('rho_eval_ceil','deployed_intersection');return r
