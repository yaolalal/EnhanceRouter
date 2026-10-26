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
def split(shot,seed,ds):return read(RUN/f'splits/{shot}/{seed}/{ds}.json')
def setup():
 require()
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
 from enhancerouter.io import environment
 import platform
 env=(dict(base_environment_sha256=digest(RUN/'environment.json'),python=sys.version,host=platform.node())
      if (RUN/'environment.json').exists() else environment())
 return dict(environment=env,source=read(ROOT/'provenance/source.json'),
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
 return load(folder(shot,seed,ds)/('small_stratified' if small else 'standard')/f'{branch}_{role}.pt')
def config(shot,seed,ds):return read(folder(shot,seed,ds)/'b0/config.json')


def evaluation_ids(seed,ds):
 return read(RUN/'EVALUATION_IDS.json')[f'{seed}/{ds}']

def registered_protocol():
 return read(RUN/'PROTOCOL.json')

def upstream_bank(shot,seed,ds,small=False):
 if not small and (seed<=44 or shot==5):
  c=read(RUN/'HISTORICAL_INPUTS_V2.json')['cells'][key(shot,seed,ds)]
  return load(c['inputs']['upstream'])
 return load(folder(shot,seed,ds)/('small_stratified' if small else 'standard')/'upstream.pt')

def supervision_bank(shot,seed,ds,small=False):
 if not small and (seed<=44 or shot==5):
  c=read(RUN/'HISTORICAL_INPUTS_V2.json')['cells'][key(shot,seed,ds)]
  return load(c['inputs']['supervision'])
 return load(folder(shot,seed,ds)/('small_stratified' if small else 'standard')/'supervision.pt')

def head_state(shot,seed,ds):
 if seed<=44 or shot==5:
  c=read(RUN/'HISTORICAL_INPUTS_V2.json')['cells'][key(shot,seed,ds)]
  p=Path(c['inputs']['embeddings']).parent/'gnn_head.pt'
  expected=read(RUN/'HEAD_SOURCE_FILES.json')[str(p)]['sha256'];assert digest(p)==expected
  return load(p)
 return load(folder(shot,seed,ds)/'standard/head.pt')
