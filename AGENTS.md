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
