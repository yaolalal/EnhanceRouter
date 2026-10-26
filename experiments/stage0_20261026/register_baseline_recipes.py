from common import *
setup()
import joblib
idx=read(ROOT/'configs/inputs.json');rows=[]
for shot,seed,ds in CELLS:
 if seed<=44 or shot==5:continue
 c=idx['cells'][f'{shot}/42/{ds}'];routers={}
 for b,v in c['branches'].items():
  m=joblib.load(v['bundle']);routers[b]=dict(source=v['bundle'],source_sha256=digest(v['bundle']),schema=m.get('tuning_schema'),reference_fraction=m['policy']['fraction'],models={h:dict(engine=w['engine'],config=w['model']['config']) for h,w in m['models'].items()})
 candidates=[]
 for s in (42,43,44):
  cfg=idx['cells'][f'{shot}/{s}/{ds}']['config']
  if cfg not in candidates:candidates.append(cfg)
 rows.append(dict(shot=shot,seed=seed,dataset=ds,router=routers,joint_candidates=candidates,joint_selection='Own validation accuracy then NLL',gnn_model_seed=gnn_training_seed(ds)))
write(RUN/'BASELINE_RECIPE_REGISTRATION.json',dict(rows=rows,new_cells=20,full_family_catalog_researched=False,rule='Refit matching-shot seed42 historical family/mode/scope recipe; native schema-specific gold normalization, validation calibration/route/rho and frozen finite Joint candidate selection. Historical55 retain exact per-cell bundles. D3 reruns its own baseline recipe with reduced validation, without a full family/GNN-hyperparameter catalog search.',query_labels_used=False,slurm_job=os.environ['SLURM_JOB_ID']))
print('BASELINE_RECIPES_REGISTERED',len(rows),flush=True)
