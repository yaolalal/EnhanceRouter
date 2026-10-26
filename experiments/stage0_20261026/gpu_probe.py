from common import *
setup()
import torch
from datetime import datetime,timezone
result=dict(job=os.environ['SLURM_JOB_ID'],gpu_env={k:v for k,v in os.environ.items() if any(t in k for t in ('CUDA','GPU','TRES'))},visible_count=torch.cuda.device_count(),names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
write(RUN/f'GPU_PROBE-{os.environ["SLURM_JOB_ID"]}.json',result)
print(json.dumps(result,indent=2),flush=True)
