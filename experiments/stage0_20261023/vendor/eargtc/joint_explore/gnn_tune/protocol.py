"""Validation-only search specification, identities and supervision contracts."""
import hashlib,json
import torch
from eargtc.joint_explore.gnn_downstream.io import ROOT,RUN as SOURCE,DATASETS,VARIANTS,read,write,sha,verify,require
from eargtc.joint_explore.gnn_downstream.core import validate_pools
RUN=ROOT/'runs/router_v9new_gnn_tune_20260914_v1'
PLAN=ROOT/'docs/superpowers/plans/2026-09-14-router-v9new-gnn-tune.md'
SPEC={'head_lrs':[1e-4,3e-4,1e-3],'pseudo_weights':[.1,.3,1.,3.],
      'encoder_lrs':[1e-6,1e-5,3e-5,1e-4],'modes':['all','warm20','last'],
      'refine_wd':[1e-5,1e-3],'refine_dropout':[.1,.3,.5],'refine_anchor':1.,
      'ramp_epochs':20,'max_epochs':500,'extended_epochs':1000,
      'early_dense_epochs':20,'validation_interval':5,'patience_epochs':100,
      'datasets':list(DATASETS),'seeds':list(range(42,47)),
      'training_seed':'split_seed','device':'cuda','precision':'float32, TF32 disabled',
      'selection':'per split validation correct DESC, NLL ASC, epoch ASC, candidate ID ASC',
      'initial_checkpoint_eligible':True,'legacy_validation_candidate':True,
      'training_head_forward':'supervised subset; historical full-node dropout RNG differs',
      'test_truth_for_selection':False,'inner_ensemble':False}

def config_id(c):return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]
def rank(r):return (-r['validation']['correct'],r['validation']['nll'],r['selected_epoch'],r['id'])
def choose(rows):return min(rows,key=rank)
def open_training(root=RUN):
    require()
    if any((root/p).exists() for p in ('FROZEN.json','EVALUATION_STARTED.json')):raise RuntimeError('Training permanently closed')
def check_item(item,n,seed):
    forbidden={'y','labels','targets','query_targets','query_labels','test_targets','test_labels','eval1000_targets','eval1000_labels'}
    if forbidden&item.keys():raise ValueError('Query truth is forbidden in training input')
    validate_pools(n,item['support_ids'],item['valid_ids'],item['query_ids'],item['pseudo_ids'])
    if item['model_seed']!=seed or item['split_seed']!=seed:raise ValueError('Seed mismatch')
    if not set(item['eval1000_ids'].tolist())<=set(item['query_ids'].tolist()):raise ValueError('Invalid eval subset')
    for prefix in ('support','valid','pseudo'):
        y=item[prefix+'_targets'];ids=item[prefix+'_ids']
        if len(y)!=len(ids) or bool(((y<0)|(y>=item['classes'])).any()):raise ValueError('Invalid labels')
def base(item,arm):
    return {'head_lr':.001,'encoder_lr':.0001,'wd':.0001,'dropout':item['head_trial']['dropout'],
            'lambda':1. if arm=='router' else 0.,'mode':'frozen','ramp':0,'anchor':0.,'max_epochs':500}
def primary_frozen(item,arm):
    b=base(item,arm)
    return [{**b,'head_lr':lr,'lambda':w} for lr in SPEC['head_lrs'] for w in (SPEC['pseudo_weights'] if arm=='router' else [0.])]
def primary_fine(winner):
    b=winner['config']
    return [{**b,'mode':mode,'encoder_lr':lr,'anchor':0.,'ramp':0,'max_epochs':500} for lr in SPEC['encoder_lrs'] for mode in SPEC['modes']]
def refinement(winner,arm,fine):
    b=winner['config'];changes=[{'wd':v} for v in SPEC['refine_wd']]+[{'dropout':v} for v in SPEC['refine_dropout']]
    if fine:changes.append({'anchor':SPEC['refine_anchor']})
    if arm=='router':changes.append({'ramp':SPEC['ramp_epochs']})
    changes.append({'max_epochs':SPEC['extended_epochs']})
    return [{**b,**v} for v in changes]
def code_hashes():
    paths=list((ROOT/'eargtc/joint_explore/gnn_tune').glob('*.py'))+list((ROOT/'scripts/joint_explore/gnn_tune').glob('*.py'))
    paths += [ROOT/'eargtc/joint_explore/gnn_downstream'/f for f in ('core.py','io.py')]+[ROOT/'eargtc/models/gnn.py']
    return {str(p):sha(p) for p in sorted(paths)}
def gate():
    require();p=read(RUN/'PREREGISTERED.json');verify(p['code']);verify(p['manifests'])
    if read(RUN/'CONFIG.json')!=SPEC:raise ValueError('Configuration changed')
    return p

def bind_legacy(path,sealed_files):
    """Bind the local completion record to the previously frozen global manifest."""
    completion=path/'COMPLETE.json'
    verify({str(completion):sealed_files[str(completion)]})
    done=read(completion)
    for f,h in done['files'].items():
        if sealed_files.get(f)!=h:raise ValueError('Historical seal disagreement: '+f)
    return done
