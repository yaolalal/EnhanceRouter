# Completed B0 / D4 final-accuracy results

Five split seeds42–46; each cell evaluates the fixed1000. L1 and Stage0 gates are not assessed.

| Dataset | Shot | FairRouter B0 | Confidence router | Confidence preference |
|---|---:|---:|---:|---:|
| cora | 3 | 79.14 ± 0.82 | 73.26 ± 1.26 | 72.74 ± 2.12 |
| citeseer | 3 | 73.36 ± 0.48 | 72.64 ± 0.59 | 73.12 ± 0.91 |
| pubmed | 3 | 90.16 ± 1.15 | 89.82 ± 0.91 | 88.64 ± 2.32 |
| arxiv | 3 | 57.56 ± 1.04 | 55.28 ± 2.15 | 56.44 ± 1.99 |
| ogbn-products | 3 | 72.24 ± 0.70 | 70.98 ± 1.23 | 71.10 ± 2.38 |
| cora | 5 | 80.56 ± 1.49 | 75.78 ± 2.11 | 76.40 ± 3.10 |
| citeseer | 5 | 72.76 ± 0.67 | 72.28 ± 0.59 | 71.46 ± 1.27 |
| pubmed | 5 | 90.92 ± 1.21 | 89.26 ± 2.12 | 90.06 ± 2.22 |
| arxiv | 5 | 59.52 ± 1.64 | 59.02 ± 1.72 | 58.08 ± 1.40 |
| ogbn-products | 5 | 72.18 ± 0.77 | 70.40 ± 1.38 | 70.88 ± 1.22 |
| cora | 10 | 81.30 ± 1.08 | 77.12 ± 1.58 | 75.70 ± 2.22 |
| citeseer | 10 | 74.28 ± 1.39 | 72.48 ± 0.47 | 72.68 ± 0.79 |
| pubmed | 10 | 91.00 ± 1.09 | 90.26 ± 1.39 | 90.42 ± 1.48 |
| arxiv | 10 | 61.76 ± 0.70 | 60.80 ± 1.07 | 60.32 ± 1.21 |
| ogbn-products | 10 | 73.56 ± 1.56 | 72.14 ± 0.80 | 73.26 ± 0.69 |

d4/confidence_router: paired accuracy delta=-1.9187pp,95%CI=[-2.3827,-1.4707],n=75

d4/confidence_preference: paired accuracy delta=-1.9333pp,95%CI=[-2.4813,-1.4227],n=75
