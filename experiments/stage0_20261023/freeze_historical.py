from common import *
setup()
import csv,shutil
out=ROOT/'reports/stage0-20261023-historical';out.mkdir(parents=True,exist_ok=False)
refs={3:ROOT/'runs/stage0-b0-historical-3shot-20261023',5:ROOT/'runs/acceptance-5shot-20261022-v2',10:ROOT/'runs/stage0-b0-historical-10shot-20261023'}
results={};rows=[]
for shot,run in refs.items():
 r=read(run/'evaluation/ACCEPTANCE.json');assert r['all_checks_pass'] and r['cells']==15 and r['all_predictions_exact'] and r['all_metrics_match_historical'] and r['source_dependencies_unchanged']
 results[shot]=r;rows+=r['per_seed']
 for seed in (42,43,44):
  for ds in DS:
   folder=run/f'cells/shot{shot}/seed{seed}/{ds}'
   dest=out/f'frozen-configs/shot{shot}/seed{seed}/{ds}';dest.mkdir(parents=True)
   for name in ('config.json','FROZEN.json','diagnostics.json','frozen-inputs.json'):
    if (folder/name).exists():shutil.copyfile(folder/name,dest/name)
write(out/'RESULTS.json',dict(historical45_exact=True,replays=results,dev_diagnostics_complete=False,protocol_note='Historical per-seed 10shot Query1000 lists retained. New shared-across-seeds list is a separate development protocol.',metadata=metadata()))
lines=['# Phase0 historical B0 reproduction','', '45/45 exact replay PASS. Metrics: accuracy percentage, mean ± population standard deviation over seeds42/43/44.','', '| Dataset | 3-shot | 5-shot | 10-shot |','|---|---:|---:|---:|']
for ds in DS:
 values=[]
 for shot in SHOTS:
  x=next(v for v in results[shot]['summary'] if v['dataset']==ds);values.append(f"{100*x['accuracy_mean']:.2f} ± {100*x['accuracy_std_ddof0']:.2f}")
 lines.append('| '+ds+' | '+' | '.join(values)+' |')
lines+=['','This completes historical baseline replay only. D1-D4 and the stage1 decision require the new75-cell development run.','All45 configurations, affine/routing/policy parameters, input hashes and run receipts are copied alongside this report.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
seal(out,dict(historical_cells=45,all_predictions_exact=True,all_metrics_match=True,dev_diagnostics_complete=False))
print('\n'.join(lines),flush=True)
