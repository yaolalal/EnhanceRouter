"""D1: same B0 heads, inputs and native selection; numerical tuning only.

Keep the earlier all-frozen D1 in its original directories. This protocol owns
separate outputs and never consults evaluation labels for model selection.
"""
from pathlib import Path
import sys, os, copy, json, hashlib, gc
from contextlib import ExitStack
from unittest.mock import patch
OLD = Path(__file__).resolve().parent.parent / 'stage0_20261026'
sys.path.insert(0, str(OLD))
from common import *

# Explicit allowlist: seeds, rank, feature modes, scopes, source profiles, loss
# definitions, ensemble membership and domain weights are structural and fixed.
NUMERIC = frozenset(('C', 'gamma', 'width', 'dropout', 'lr', 'weight_decay',
 'steps', 'adapt_steps', 'pretrain_steps', 'source_steps', 'target_steps',
 'depth', 'leaf', 'n_estimators', 'max_depth', 'min_samples_leaf', 'max_iter',
 'learning_rate', 'l2_regularization', 'alpha'))
IDENTITY = frozenset(('id', 'stage'))
JOINT_NUMERIC = frozenset(('dim','cross','lr','gate_lr','warmup','aux','keep'))

def joint_locked(cfg):
 from eargtc.joint_explore.joint_purf_v10_v1.training import DEFAULTS
 full=dict(DEFAULTS,**cfg)
 return {k:v for k,v in full.items() if k not in JOINT_NUMERIC}

def native_joint(shot,seed,ds,router,out,base_cfg):
 """Run the original adaptive search; constrain non-numerical B0 choices."""
 import compute,native_joint_search as search
 from eargtc.joint_explore.joint_purf_v10_v1.training import DEFAULTS
 anchor=dict(DEFAULTS,**base_cfg)
 def constrained(rows):
  candidates=[]
  for row in rows:
   cfg=copy.deepcopy(anchor)
   for name in JOINT_NUMERIC:cfg[name]=row[name]
   assert joint_locked(cfg)==joint_locked(anchor)
   candidates.append(cfg)
  return search.dedup(candidates)
 with ExitStack() as stack:
  for name in ('core','schemes','refinement','removals'):
   original=getattr(search,name)
   def generator(*args,_fn=original):return constrained(_fn(*args))
   stack.enter_context(patch.object(search,name,new=generator))
  return compute.joint(shot,seed,ds,router,out)

def locked(cfg):
 return {k: locked(v) if isinstance(v, dict) else copy.deepcopy(v)
         for k, v in cfg.items() if k not in NUMERIC | IDENTITY}

def fingerprint(cfg):
 return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

def native_catalog(w, ds, head):
 """Filter the original engine catalog without converting its model family."""
 from enhancerouter.router import engine
 e = engine(w['engine']); cfg = w['model']['config']; rows = [copy.deepcopy(cfg)]
 if cfg.get('scope') == 'source': return rows
 if w['engine'] == 'v6':
  with ExitStack() as stack:
   for name in ('FINAL', 'EXP04', 'EXP03', 'GOLD'):
    stack.enter_context(patch.object(e, name, SOURCE/'runs'/getattr(e, name).name))
   rows += e.candidate_grid(ds, head)
 elif w['engine'] == 'v7': rows += e.candidate_grid(ds, head)
 elif w['engine'] == 'expanded':
  for stage in ('original', 'capacity', 'nonlinear', 'training'):
   rows += e.candidate_grid(ds, head, stage)
 else:
  for stage in ('regularization', 'transfer', 'training'):
   rows += e.candidate_grid(ds, head, stage, w)
 result = []; seen = set()
 for row in rows:
  merged=copy.deepcopy(cfg)
  def merge(dst,src):
   for k,v in src.items():
    if k in IDENTITY:continue
    if isinstance(v,dict) and isinstance(dst.get(k),dict):merge(dst[k],v)
    else:dst[k]=copy.deepcopy(v)
  merge(merged,row)
  # Fitted configs contain derived fields (e.g. target_domain) absent from
  # catalog recipes. Inherit these fields; never mistake absence for a change.
  if locked(merged) != locked(cfg): continue
  # Keep the B0 configuration shape and all structural fields exactly.
  candidate = copy.deepcopy(cfg)
  def overlay(dst, src):
   for k, v in src.items():
    if k in NUMERIC: dst[k] = copy.deepcopy(v)
    elif isinstance(v, dict) and isinstance(dst.get(k), dict): overlay(dst[k], v)
  overlay(candidate, merged)
  payload = copy.deepcopy(candidate); payload.pop('id', None); payload.pop('stage', None)
  ident = fingerprint(payload)
  if ident in seen: continue
  seen.add(ident); result.append(candidate)
 assert result and locked(result[0]) == locked(cfg)
 return result

def train_candidate(template, cfg, head, banks, ds):
 from frozen_fit import refit, input_hash
 w = copy.deepcopy(template); w['model']['config'] = copy.deepcopy(cfg)
 # A numerical gamma can be selected explicitly; the symbolic B0 scale keeps
 # its actual fitted value through the same controlled refit helper.
 fitted, audit = refit(w, head, banks, ds)
 assert locked(fitted['model']['config']) == locked(template['model']['config'])
 assert input_hash(fitted) == input_hash(template)
 return fitted, audit

def selected_heads(original, shot, seed, ds, getter, out):
 import numpy as np, torch, joblib
 from enhancerouter.router import raw
 import native_router_search as native
 from enhancerouter.policy import folds
 models = copy.deepcopy(original); audits = {}
 for branch,m in models.items():
  banks={d:getter(shot,seed,d,branch,'train') for d in DS}
  for h,w in m.get('models',{}).items():
   m['models'][h],audits[h+'/fresh_incumbent']=train_candidate(w,w['model']['config'],h,banks,ds)
 # Reuse the native branch shortlist, OOF pair evaluation, adoption threshold
 # and tie breaking. Only the candidate generator/fit is constrained by B0.
 def fit_head(s, d, h, banks, vb, fold, fraction, old):
  branch = 'agreement' if h == 'agreement' else 'disagreement'
  template = original[branch]['models'][h]; dest = out/'numeric-search'/h
  rows = []; catalog = native_catalog(template, ds, h)
  registration=read(RUN/'D1_NUMERIC_REGISTRATION_20261027.json')
  registered=next(r for r in registration['rows'] if r['key']==key(shot,seed,ds))
  assert registered['heads'][branch+'/'+h]['catalog_sha256']==fingerprint(catalog)
  write(dest/'CATALOG.json', dict(configs=catalog, locked=locked(template['model']['config']), engine=template['engine']))
  for i, cfg in enumerate(catalog):
   cid = 'incumbent' if i == 0 else 'numeric_'+fingerprint(cfg)[:16]
   w, audit = train_candidate(template, cfg, h, banks, ds)
   z = raw(w, vb); metric = native.head_metric(z, vb, h, fold, fraction)
   mp = dest/(cid+'.joblib'); rp = dest/(cid+'.pt')
   with mp.open('xb') as f: joblib.dump(w, f, compress=3)
   save(rp, dict(raw=torch.as_tensor(z), validation_ids=vb['ids']))
   row = dict(id=cid, stage='numeric_only', config=w['model']['config'],
    engine=w['engine'], model_path=str(mp), model_sha256=digest(mp),
    raw_path=str(rp), raw_sha256=digest(rp), metric=metric)
   rows.append(row); audits[h+'/'+cid] = audit
   print('D1_NUMERIC_CANDIDATE', shot, seed, ds, h, i+1, len(catalog), flush=True)
   del w; gc.collect()
  shortlist = sorted(rows, key=native.ranking)[:3]
  if not any(r['id']=='incumbent' for r in shortlist): shortlist.append(rows[0])
  write(dest/'RESULT.json', dict(rows=rows, shortlist=[r['id'] for r in shortlist], evaluation_labels_used=False))
  return shortlist, rows
 with patch.object(native, 'fit_head', new=fit_head), patch.object(native, 'bank', new=getter):
  result = native.select(models, shot, seed, ds, False, out)
 # Historical 5-shot non-crossfit branches retain their own calibration family.
 # A selected numerical head must not implicitly replace that workflow.
 for branch, bundle in result.items():
  source = original[branch]
  if source.get('tuning_schema') != 'crossfit_v1':
   bundle.pop('tuning_schema', None)
   for k in ('affine', 'boundary'):
    if k in source: bundle[k] = copy.deepcopy(source[k])
  else: bundle['tuning_schema'] = source['tuning_schema']
  bundle['policy'] = copy.deepcopy(source['policy'])
  for h, w in bundle.get('models', {}).items():
   assert w['engine'] == original[branch]['models'][h]['engine']
   assert locked(w['model']['config']) == locked(original[branch]['models'][h]['model']['config'])
 return result, audits

def main(index, fractions=(.25, .5, .75, 1.)):
 setup()
 registration=read(RUN/'D1_NUMERIC_REGISTRATION_20261027.json')
 assert registration['all_checks_pass'] and registration['files']['d1_numeric.py']==digest(__file__)
 if tuple(fractions)==(.25,.5,.75,1.):assert read(RUN/'D1_NUMERIC_PILOT_PASS.json')['all_checks_pass']
 import torch, joblib
 import compute
 from enhancerouter.router import engine
 from frozen_fit import input_hash
 shot, seed, ds = CELLS[index]; base = folder(shot, seed, ds)/'b0'
 for rel, h in read(base/'COMPLETE.json')['files'].items(): assert digest(base/rel)==h
 original = joblib.load(base/'models.joblib'); native_bank = compute.bank
 for fraction in fractions:
  out = folder(shot, seed, ds)/'d1_numeric'/str(fraction)
  if (out/'COMPLETE.json').exists():
   for rel, h in read(out/'COMPLETE.json')['files'].items(): assert digest(out/rel)==h
   continue
  if out.exists():
   archive = out.with_name(out.name+'.interrupted-'+str(os.environ['SLURM_JOB_ID']))
   out.rename(archive)
  parents = torch.tensor(split(shot, seed, ds)['support_ids'], dtype=torch.long)
  gen = torch.Generator().manual_seed(seed+913)
  parents = parents[torch.randperm(len(parents), generator=gen)[:max(1,int(len(parents)*fraction+.5))]]
  cache = {}
  def getter(sh, se, d, branch, role, small=False):
   k = (sh,se,d,branch,role,small)
   if k not in cache:
    b = native_bank(sh,se,d,branch,role,small)
    if d==ds and role=='train': b = subset(b, torch.isin(b['node_ids'],parents))
    cache[k] = b
   return cache[k]
  # Verify the six-view parent unit jointly across both branches.
  node_env = {}
  for branch in ('agreement','disagreement'):
   b = getter(shot,seed,ds,branch,'train')
   for node, env in zip(b['node_ids'].tolist(), b['environment_index'].tolist()):
    node_env.setdefault(node,[]).append(env)
  assert set(node_env)==set(parents.tolist())
  assert all(sorted(env)==list(range(6)) for env in node_env.values())
  bank_hashes={}
  for branch in ('agreement','disagreement'):
   for d in DS:
    bank_hashes[branch+'/'+d]=joblib.hash(getter(shot,seed,d,branch,'train'),hash_name='sha1')
  models, audit = selected_heads(original,shot,seed,ds,getter,out)
  # Gold parent OOF calibration must also fit with the B0 input transform.
  with ExitStack() as stack:
   stack.enter_context(patch.object(compute,'bank',new=getter))
   for ename in {w['engine'] for m in models.values() for w in m.get('models',{}).values()}:
    e = engine(ename); native_fit = e.fit_candidate
    def controlled_fit(cfg,h,banks,target,_e=e,_native=native_fit,_ename=ename):
     branch='agreement' if h=='agreement' else 'disagreement'
     template=original[branch]['models'][h]
     assert template['engine']==_ename
     with patch.object(_e,'fit_candidate',new=_native):
      return train_candidate(template,cfg,h,banks,target)[0]['model']
    stack.enter_context(patch.object(e,'fit_candidate',new=controlled_fit))
   models = compute.calibrated(models,shot,seed,ds)
  for branch,m in models.items():
   for h,w in m.get('models',{}).items():
    assert input_hash(w)==input_hash(original[branch]['models'][h])
    training_audit=w['model'].get('fit_audit',w['model'].get('training_audit',{}))
    tb=getter(shot,seed,ds,branch,'train')
    eligible=int((tb['state']!=2).sum()) if h=='preference' else len(tb['ids'])
    scope=w['model']['config'].get('scope','transfer' if w['model']['config'].get('kind')=='neural' else 'target')
    if scope=='target' and not eligible:
     fitted=w['model'].get('fitted',w['model'].get('estimator',{}))
     assert fitted['kind']=='constant'
     continue # refit receipt explicitly records the zero-row prior; old audit is retained as provenance.
    for field in ('supervised_parent_ids','training_parent_ids','fit_parent_ids','shared_fit_parent_ids'):
     domains=training_audit.get(field) or {}
     if ds in domains:assert set(domains[ds])<=set(parents.tolist()),(h,field)
   assert m.get('tuning_schema')==original[branch].get('tuning_schema')
  for branch in ('agreement','disagreement'):
   for d in DS:assert joblib.hash(getter(shot,seed,d,branch,'train'),hash_name='sha1')==bank_hashes[branch+'/'+d]
  with patch.object(compute,'bank',new=getter): router,scores=compute.route(models,shot,seed,ds)
  save(out/'router.pt',router)
  for b,r in scores.items(): save(out/f'{b}-scores.pt',r)
  with (out/'models.joblib').open('xb') as f: joblib.dump(models,f,compress=3)
  # Keep the same four-stage model selection code as B0, constraining its
  # non-numerical training choices (notably gate_gold) to the B0 values.
  anchor=read(base/'config.json')['joint']; chosen=native_joint(shot,seed,ds,router,out,anchor)
  assert joint_locked(chosen)==joint_locked(anchor)
  write(out/'config.json',dict(dataset=ds,shot=shot,seed=seed,fraction=fraction,
   parent_ids=parents.tolist(),parent_selection_seed=seed+913,
   frozen=['experts','augmentation','source banks','prototypes','input transforms','head engines/families','feature modes','training scopes','Joint non-numerical choices and training schedule'],
   retuned=['numerical head hyperparameters','native validation calibration/routing policy','Joint original numeric search fields'],
   joint=chosen,joint_B0_locked=joint_locked(anchor),
   native_selection='same OOF shortlist/pair/adoption kernels, B0-constrained engine catalog',
   B0_seal_sha256=digest(base/'COMPLETE.json'),new_code_sha256=digest(__file__),
   metadata=metadata()))
  write(out/'controlled-audit.json',dict(all_checks_pass=True,heads=audit,
   six_views_per_parent=True,source_banks_unchanged=True,input_transforms_frozen=True,
   training_bank_sha1_before_and_after=bank_hashes,
   native_calibration_schema_retained=True,query_truth_used=False))
  seal(out,dict(models_frozen=True,query_truth_used=False,protocol='D1 numerical tuning and native calibration, user-confirmed 2026-10-27'))
  print('D1_NUMERIC_COMPLETE',index,shot,seed,ds,fraction,flush=True)
  del cache,models; gc.collect()

if __name__=='__main__':
 main(int(os.environ.get('SLURM_ARRAY_TASK_ID',sys.argv[1] if len(sys.argv)>1 else -1)),
      tuple(float(x) for x in sys.argv[2:]) if len(sys.argv)>2 else (.25,.5,.75,1.))
