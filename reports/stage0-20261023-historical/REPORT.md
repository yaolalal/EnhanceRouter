# Phase0 historical B0 reproduction

45/45 exact replay PASS. Metrics: accuracy percentage, mean ± population standard deviation over seeds42/43/44.

| Dataset | 3-shot | 5-shot | 10-shot |
|---|---:|---:|---:|
| cora | 79.37 ± 0.17 | 80.70 ± 1.91 | 81.27 ± 1.36 |
| citeseer | 73.37 ± 0.12 | 72.60 ± 0.79 | 73.47 ± 1.25 |
| pubmed | 90.80 ± 1.04 | 91.47 ± 1.30 | 91.63 ± 0.83 |
| arxiv | 58.13 ± 0.12 | 60.40 ± 1.56 | 61.60 ± 0.86 |
| ogbn-products | 72.30 ± 0.85 | 72.43 ± 0.41 | 72.80 ± 0.82 |

This completes historical baseline replay only. D1-D4 and the stage1 decision require the new75-cell development run.
All45 configurations, affine/routing/policy parameters, input hashes and run receipts are copied alongside this report.
