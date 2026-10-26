from common import *
setup()
import torch
rows=[]
for ds in DS:
 ids=set(read(RUN/f'shared/text-request-v2-{ds}.json')['node_ids']);p=ROOT/f'runs/stage0-20261023-v1/shared/text/{ds}.pt';old=load(p);cached=set(old['node_ids'].tolist());missing=sorted(ids-cached)
 row=dict(dataset=ds,requested=len(ids),prior_cache=len(cached),not_in_prior_cache=len(missing),missing_node_ids=missing);rows.append(row);print('COVERAGE',ds,len(ids),len(missing),flush=True)
write(RUN/'CACHE_COVERAGE.json',dict(rows=rows,labels_read=False,slurm_job=os.environ['SLURM_JOB_ID']))
