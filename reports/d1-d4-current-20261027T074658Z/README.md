# D2 complete; D1 partial current snapshot

Snapshot (Asia/Shanghai): 2026-10-27T15:47:26.671424+08:00. Slurm report job120992.

D2: all75 configurations and all450 variants complete. Independent hash gate120659 passed all450 seals; proof is D2_ALL75_VERIFIED.json. D3 and both D4 methods remain complete across75 cells. New numeric D1 counts by fraction: {'0.25': 23, '0.5': 20, '0.75': 18, '1.0': 20}; common four-fraction cohort: 18 cells. Its complete75-cell conclusion remains pending.

REPORT.md contains the tables. RESULTS.json preserves all individual results, matched B0, actual Oracle N, defined AUROC counts and completed D3/D4 accuracy. per_cell.csv gives exact physical artifact paths and seal hashes. FROZEN_SNAPSHOT.json fixes the included artifacts; later completions are excluded.

D2 adds true labels from either the unlabeled pool (a) or B0 high-confidence agreement pool (b), excludes eval1000, and uses the same six views and controlled B0 refit. Requested N=50/200/1000 is an upper bound: some b pools have fewer nodes. Differences where actual N differs cannot isolate distribution from label count. No extra Joint accuracy was trained for D2.

New D1 fixes B0 feature mode, training combination, model family, fitted input transforms, experts and six views, allowing only numerical parameters and native calibration to retune. Its100% is an internal reference. Old frozen D1 remains separate.

Code: experiments/stage0_20261027/d1_numeric.py; experiments/stage0_20261026/compute.py:oracle_requests/oracle_banks/d2; frontend.py for experts/views; report_current_d1_d4.py for this descriptive report. The reporting-only wording update removes a stale Products-incomplete sentence and changes no training code/config.

P@1/5/20/100% ranks disagreement nodes within fixed eval1000 and evaluates the selected expert prediction. Trust/Preference AUROC are budget-independent diagnostics. D-Prec@B0 and A-Prec@B0 remain deferred; these tables do not choose that budget or decide Stage0/Stage1. Historical B05-shot selection has selection_uses_query_truth=true; see docs/FAIRROUTER_BASELINE.md.

Remote root: /home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter. Immutable result source: runs/stage0-20261026-v2/current-results/20261027T074658Z.
