#!/usr/bin/env python3
import argparse,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','source','source_pair','target'],required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--target',required=True);p.add_argument('--held');p.add_argument('--device',default='cpu');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Numerical stages require Slurm')
    if a.stage=='prepare':
        from eargtc.router_v8_learned.data import prepare
        prepare(a.root.resolve(),a.target)
    elif a.stage=='source_pair':
        from eargtc.router_v8_learned.experiment import run_source_pair
        run_source_pair(a.root.resolve(),a.target,a.held,a.device)
    elif a.stage=='source':
        from eargtc.router_v8_learned.experiment import run_source
        run_source(a.root.resolve(),a.target,a.held,a.device)
    else:
        from eargtc.router_v8_learned.experiment import run_target
        run_target(a.root.resolve(),a.target,a.device)
if __name__=='__main__':main()
