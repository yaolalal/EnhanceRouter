import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from eargtc.router_augmentation_v8.sources import *
from eargtc.router_augmentation_v8.data import head_view
require_slurm();torch.set_num_threads(2)
for d in DATASETS:
    for branch in ('agreement','disagreement'):
        bank=load(RUN/'banks'/d/(branch+'_train.pt'))
        for h in (('agreement',) if branch=='agreement' else ('trust','preference')):
            saved=load(RUN/'banks'/d/(h+'_v7.pt'))
            for mode,x in saved['features'].items():
                actual=head_view(bank,h,'v7',mode)['x'];diff=(actual-x).abs()
                if not torch.equal(actual,x):print(d,h,mode,'shape',tuple(x.shape),'max',float(diff.max()),'different',int((diff>0).sum()),flush=True)
