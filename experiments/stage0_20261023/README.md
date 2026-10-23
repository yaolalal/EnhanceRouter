# EnhanceRouter Phase 0

From this project root, one command launches a fresh independent development run:

```bash
bash experiments/stage0_20261023/run.sh stage0-reproduction-$(date +%Y%m%d-%H%M%S)
```

Uses the existing `lf` Conda environment and Slurm. Every numerical operation, including inspection and reporting, runs on Slurm. `ENHANCEROUTER_GPU_PARTITIONS` for heads defaults to `RTX2080,RTX3090,RTX4090`; CPU work uses `cpu`. Text work defaults to a single24GB GPU (`ENHANCEROUTER_TEXT_GPU_PARTITIONS=RTX3090,RTX4090`, `ENHANCEROUTER_TEXT_GRES=gpu:1`). For11GB cards set `ENHANCEROUTER_TEXT_GPU_PARTITIONS=RTX2080 ENHANCEROUTER_TEXT_GRES=gpu:2`; the same fp16 model is split over2 GPUs with no quantization or CPU model layers. Text batches checkpoint in the new run directory and request bounded requeue after90 minutes to fit2-hour scheduler windows. Do not run two writers against the same run directory.

## Historical B0

All45 original cells (5 datasets x3 shots x seeds42/43/44) are separately replayed using the original per-seed historical10shot Query1000 list. These are historical reproducibility checks, distinct from the development protocol below. Their directories are:

- `runs/stage0-b0-historical-3shot-20261023`
- `runs/acceptance-5shot-20261022-v2`
- `runs/stage0-b0-historical-10shot-20261023`

To repeat any shot, use `ENHANCEROUTER_SHOT=3 bash reproduce.sh UNIQUE_RUN_NAME` (replace3 with5 or10). Historical configuration snapshots and results are collected in `reports/stage0-20261023-historical/` after its Slurm reporting job completes.

## Registered development protocol

`PROTOCOL.json` is written before fitting. `PROTOCOL_AMENDMENT_01.json` records label-blind cache/sampling corrections made before any dev router fit or evaluation. Dev seeds142-146; heldout1042-1046 reserved and never trained/evaluated here. Shared evaluation IDs are sampled label-blind from the intersection of historical10shot Query sets; one ordered1000-node list per dataset is reserved before ANY new support or validation sampling. This implements the PDF page2 annotation that the evaluation nodes must also match across seeds. Old historical artifacts are never overwritten.

For new seeds, B0 keeps the selected historical seed42 scorer recipe at each dataset/shot, refits it on new support banks, then recalibrates affine/routing/acceptance on its own validation. Joint candidates are the unique historical winners at the same dataset/shot from42/43/44; selection uses only new validation. This is an explicitly fixed candidate protocol, not a claim of rerunning every historical hyperparameter search. Each cell saves its exact chosen configuration and normalization.

D1 retains25/50/75/100% of original target support PARENTS, using the same parent subset for both branches and retaining all their views. Experts and prototypes are frozen: this measures router supervision amount, not the total expert-label budget. Source-only heads keep their original scope.

D2a samples random unlabeled parents; D2b samples B0 high-confidence agreement parents (top20%, cap2000). Both exclude the fixed1000 evaluation nodes. Random parent order is shared across shots, reserving the largest10shot support. N=50/200/1000; actual N is capped and reported. Oracle truth is restricted to selected parents and isolated as analysis. Only T/P observations are added; the agreement scorer stays fixed. Source-only heads remain source-only. The6 views are the exact historical clean+3text+2graph views. The page6 mention of3 graph perturbations is not used to invent another view. Missing perturbed probabilities and4096-wide hL are extracted with the original prompt, category permutation and paired-clean `align_text_replay`; unchanged text reuses exact frozen clean outputs. Sparse historical caches are never treated as full-node caches.

D3 samples stratified k nodes/class from the nontraining pool excluding all fixed evaluation nodes and reruns GNN checkpoint selection, calibration/rho and Joint validation selection. D4 reproduces both confidence references; Confidence router uses geometric-mean confidence for agreement ranking. B0 accepted counts are matched per branch; Joint hyperparameters stay fixed. D3 actual class counts are capped only where the complete eligible class pool itself has fewer than k nodes (including absent classes); exact shortages are reported. PROTOCOL_AMENDMENT_02 documents this physical data limitation. Original random500-validation subset pilots under `small/` are excluded; final D3 uses `small_stratified/`.

## Evaluation and gate

Models for all75 cells and variants must be sealed before the final reporting job can load evaluation labels. Primary D-Prec@B0 is computed on the common1000 nodes at the ACTUAL count of B0's deployed accepted nodes intersecting that fixed evaluation set (not a separately rounded rho times1000 count). Report includes trust/preference AUROC,1/5/20/100% precision, agreement precision where changed, and end-to-end accuracy/macro-F1 (macro-F1 over all class columns). Each paired unit is dataset/shot/devseed;10000 bootstrap replicates, fixed seed731901,95% percentile CI. Stage1 uses the preregistered D2b N200 gate: positive overall lower CI, positive mean on at least4/5 datasets, no dataset loss over1pp. Other oracle sizes are descriptive. Heldout results are not generated.

Final artifacts: `RUN/report/RESULTS.json`, per-cell and dataset CSVs, `stage0_diagnostics.png`, `REPORT.txt`, source MD5 recheck and completion seal. Mere Slurm submission is not completion: require `RUN/report/COMPLETE.json` with `all_checks_pass=true`.

## Isolation and provenance

All vendor code is copied under this project; `source-code.json` records original EALR-GTC paths and MD5/SHA256. Original files are read-only. Every new output/cache is protected by the EnhanceRouter write guard. `SOURCE_FILES.json` records frozen data/model paths, MD5/SHA256/size/mtime; extra prompt/input paths are in text receipts and rechecked at final reporting. Full environment is captured once per run, referenced by hash from every cell. EALR-GTC has no Git repository: its commit is recorded as null with hashes; no invented commit. Record this project's commit and code hashes for every run.

The existing `submit_remaining.sh` and `submit_text_v2.sh` document the initial run's submission history; use `run.sh` for a fresh replay. Initial sparse-cache failures and their logs are retained for audit.
