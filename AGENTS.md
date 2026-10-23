# EnhanceRouter boundaries

- EALR-GTC is read-only. Do not modify its code, artifacts, Git metadata, logs or caches.
- Copy runtime code here; never import or execute Python modules from EALR-GTC.
- Keep all new code, experiment outputs, checkpoints, logs and caches inside this project.
- All numerical work, including tensor inspection and tests, must run through Slurm.
- Seeds are 42, 43, 44. Keep the selected Original Router recipes and Joint/PURF V1 configs as the baseline.
- For each dataset and seed, reuse the exact saved historical 10-shot Query 1,000-node list in its original order for every shot. Never resample.
- Record frozen input paths and MD5, full configs, source provenance, environment and EnhanceRouter commit for every run.
- EALR-GTC has no Git repository at migration time. Record its unavailable commit ID as null with source hashes; never invent a commit or initialize Git there.
- New experiments use a new run directory. Do not overwrite completed runs.

## Stage 0 protocol update, explicitly requested 2026-10-23

- Follow the user supplied EnhanceRouter improvement plan, including its page2 annotation.
- Historical replay keeps seeds42/43/44 and original per-seed lists above.
- New development experiments reserve one label-blind list of1000 nodes per dataset before support/validation generation, shared in exact order across ALL shots and seeds.
- Dev seeds142-146; held-out1042-1046 reserved for final confirmation only.
- Stage0 preregistration lives in runs/stage0-20261023-v1/PROTOCOL.json, written before fitting.
- Oracle labels are allowed ONLY in isolated D2 analyses on selected parents excluding all fixed evaluation nodes.
- No stage1 work until phase0 diagnoses and gate are complete.
