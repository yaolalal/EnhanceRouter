"""Register immutable inputs, tune a seed, export, or seal all new-shot routers."""
from common import *
import argparse, concurrent.futures, subprocess

ROUTER_FILES=('common.py','policy.py','catalog.py','tune_engine.py','search.py','export.py','router_run.py','canonical.py','test_router.py')
def register():
 require();inputs={};code={}
 for name in ROUTER_FILES:
  p=CODE/name;code[str(p)]=sha(p)
 inputs[str(CHOICES)]=sha(CHOICES)
 for r in read(CHOICES)['rows']:
  if r['seed'] in SEEDS:
   p=Path(r['model_source']);assert sha(p)==r['model_sha256'];inputs[str(p)]=sha(p)
 for seed in SEEDS:
  for d in DATASETS:
   sp=split_path(seed,d);s=read(sp);assert s['shots']==SHOT and s['split_seed']==seed
   inputs[str(sp)]=sha(sp)
   folder=DATA/f'seed{seed}/banks'/d;m=read(folder/'manifest.json')
   assert m['dataset']==d and m['shots']==SHOT and m['split_seed']==seed
   assert m['support_ids']==s['support_ids'] and m['validation_ids']==s['valid_ids'] and m['query_ids']==s['unlabeled_ids']
   check_files(m['input_hashes'])
   inputs.update(m['input_hashes'])
   inputs[str(folder/'manifest.json')]=sha(folder/'manifest.json')
   for branch in BRANCHES:
    for role in ('train','validation','query'):
     p=folder/(branch+'_'+role+'.pt');assert sha(p)==m['bank_hashes'][p.name];inputs[str(p)]=sha(p)
   up=DATA/f'seed{seed}/upstream'/d
   for name in ('COMPLETE.json','pipeline/predictions.pt','supervision.pt'):
    p=up/name;inputs[str(p)]=sha(p)
 # Bind imported frozen engines. These are never modified by the new namespace.
 for p in (BASE/'code').rglob('*.py'):code[str(p)]=sha(p)
 from catalog import initial
 write(RUN/'REGISTRATION.json',dict(shots=SHOT,seeds=list(SEEDS),datasets=list(DATASETS),inputs=inputs,code=code,
  catalogs={h:initial(h) for h in ('agreement','trust','preference')},max_initial=30,max_refine=20,
  source='Router-v10-MS-Original selected per-seed branch recipes, refit on new support',
  model_selection='gold-only fitting; historical threefold validation calibration and routing, fixed-K Wilson improvement >=.002 without precision decline',
  baseline='freshly refit source recipe with own-validation calibration and OOF min20 acceptance',
  scope='same shot, same split seed only; preserve configured dataset scope',
  gold_folds='historical parent node grouping, six views equal unit weight',
  query_truth_access='separate evaluator only after ROOT/ROUTER_FROZEN.json all30',
  historical_query_seen=True,new_query_selection=False,slurm_job_id=os.environ['SLURM_JOB_ID']))

def run_seed(seed):
 require();assert seed in SEEDS;check_code();assert_open(seed)
 root=RUN/f'seed{seed}';root.mkdir(parents=True,exist_ok=True)
 def worker(d):
  done=root/'search'/d/'COMPLETE.json'
  if done.exists():return
  log=root/'logs'/(d+'.log');log.parent.mkdir(exist_ok=True)
  with log.open('a') as f:
   subprocess.run([sys.executable,'-B',str(CODE/'search.py'),str(seed),d],stdout=f,stderr=subprocess.STDOUT,check=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(worker,DATASETS))
 for d in DATASETS:assert read(root/'search'/d/'COMPLETE.json')['all_checks_pass']
 check_code();write(root/'MODELS_FROZEN.json',dict(files=freeze_files(root/'models'),search_files=freeze_files(root/'search'),
  incumbent_files=freeze_files(root/'incumbents'),registration_sha256=sha(RUN/'REGISTRATION.json'),seed=seed,shots=SHOT,query_truth_used=False))
 from export import main
 main(seed)

def seal_all():
 require();files={}
 for shot in (3,10):
  r=ROOT/f'shot{shot}/router'
  for seed in SEEDS:
   folder=r/f'seed{seed}';cert=read(folder/'COMPLETE.json');assert cert['all_checks_pass']
   assert sha(folder/'FROZEN.json')==cert['frozen_sha256']
   freeze=read(folder/'FROZEN.json');check_files(freeze['files']);files.update(freeze['files'])
   files[str(folder/'FROZEN.json')]=sha(folder/'FROZEN.json')
 p=ROOT/'ROUTER_FROZEN.json'
 if p.exists():check_files(read(p)['files']);return
 # global artifact intentionally above shot-specific write guard
 with p.open('x') as f:json.dump(dict(files=files,all_checks_pass=True,shots=[3,10],seeds=list(SEEDS),cells=30,cell_count=30,query_truth_used=False,slurm_job_id=os.environ['SLURM_JOB_ID']),f,indent=2)
 print('ALL 30 ROUTERS FROZEN',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=('register','seed','seal'));p.add_argument('--seed',type=int);a=p.parse_args()
 if a.command=='register':register()
 elif a.command=='seed':run_seed(a.seed)
 else:seal_all()
