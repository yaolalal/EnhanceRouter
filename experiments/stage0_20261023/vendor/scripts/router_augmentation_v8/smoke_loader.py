"""Exercise the public reader against every delivered head and feature view."""
import sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from eargtc.router_augmentation_v8.sources import RUN,DATASETS,require_slurm,write
from eargtc.router_augmentation_v8.data import load_head
require_slurm();records=[]
for d in DATASETS:
    for head in ('agreement','trust','preference'):
        base=load_head(RUN,d,head,'v6')
        for mode in ('prob','full','compact','context','embed'):
            value=load_head(RUN,d,head,'v7',mode)
            for key in ('sample_id','node_ids','target','weight'):assert torch.equal(base[key],value[key])
            assert value['x'].shape[0]==len(value['target'])
            assert bool(torch.isfinite(value['x']).all())
        records.append(dict(dataset=d,head=head,rows=len(base['target'])))
write(RUN/'LOADER_VERIFIED.json',dict(all_checks_pass=True,records=records,slurm_job_id=os.environ['SLURM_JOB_ID']))
print('Public loader passed for all 15 head datasets and all five v7 modes',flush=True)
