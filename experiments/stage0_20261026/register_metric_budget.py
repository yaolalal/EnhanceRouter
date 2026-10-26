"""Register the human's choice before any held-out metric outcomes are exposed."""
from common import *
import argparse
p=argparse.ArgumentParser();p.add_argument('mode',choices=('rho_eval_ceil','deployed_intersection'));p.add_argument('--user-confirmation',required=True);args=p.parse_args()
setup()
assert args.user_confirmation.strip()
assert not (RUN/'d1-report').exists() and not (RUN/'report').exists(),'Primary definition cannot change after evaluation release'
write(RUN/'METRIC_BUDGET_REGISTRATION.json',dict(mode=args.mode,user_confirmation=args.user_confirmation,reason='One historical Arxiv10/42 B0 has deployed evaluation disagreement K=0; primary definition must be fixed before outcomes.',evaluation_ids_changed=False,deployment_fractions_changed=False,models_changed=False,query_evaluation_outcomes_used=False,slurm_job=os.environ['SLURM_JOB_ID']))
print('METRIC_BUDGET_REGISTERED',args.mode,flush=True)
