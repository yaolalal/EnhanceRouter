from common import *
setup()
import torch,joblib,math
rows=[]
for shot,seed,ds in CELLS:
 base=folder(shot,seed,ds)/'b0'
 if not (base/'COMPLETE.json').exists():continue
 ids=torch.tensor(evaluation_ids(seed,ds));scores=load(base/'disagreement-scores.pt');mask=torch.isin(scores['ids'],ids)
 k=int(scores['accepted'][mask].sum());n=int(mask.sum());models=joblib.load(base/'models.joblib');rho=float(models['disagreement']['policy']['fraction'])
 row=dict(shot=shot,seed=seed,dataset=ds,eval_disagreement_nodes=n,B0_deployed_intersection_K=k,B0_rho=rho,ceil_rho_eval_K=math.ceil(rho*n));rows.append(row)
 if k==0:print('ZERO_EVAL_BUDGET',row,flush=True)
write(RUN/'EVAL_BUDGET_AUDIT.json',dict(cells=len(rows),rows=rows,zero_cells=[r for r in rows if r['B0_deployed_intersection_K']==0],query_truth_read=False,slurm_job=os.environ['SLURM_JOB_ID']))
print('BUDGET_AUDIT_COMPLETE',len(rows),flush=True)
