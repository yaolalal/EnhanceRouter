# D3 small-validation final-accuracy results

All75 D3 models passed the completion/hash gate before evaluation. Split seeds42–46, own-seed fixed eval1000 shared across shots.

D3 caps each class at k labels from the original validation pool. Classes with fewer than k available labels retain their actual counts; all per-class counts are recorded in RESULTS.json. L1 and Stage0 acceptance are not assessed.

| Dataset | Shot | B0 accuracy (%) | D3 accuracy (%) | Paired delta (pp) |
|---|---:|---:|---:|---:|
| cora | 3 | 79.14 ± 0.82 | 75.30 ± 2.77 | -3.84 |
| citeseer | 3 | 73.36 ± 0.48 | 70.78 ± 1.92 | -2.58 |
| pubmed | 3 | 90.16 ± 1.15 | 88.24 ± 2.65 | -1.92 |
| arxiv | 3 | 57.56 ± 1.04 | 52.30 ± 3.99 | -5.26 |
| ogbn-products | 3 | 72.24 ± 0.70 | 69.18 ± 1.18 | -3.06 |
| cora | 5 | 80.56 ± 1.49 | 75.82 ± 4.46 | -4.74 |
| citeseer | 5 | 72.76 ± 0.67 | 70.42 ± 2.71 | -2.34 |
| pubmed | 5 | 90.92 ± 1.21 | 88.44 ± 1.74 | -2.48 |
| arxiv | 5 | 59.52 ± 1.64 | 58.86 ± 0.93 | -0.66 |
| ogbn-products | 5 | 72.18 ± 0.77 | 65.38 ± 3.52 | -6.80 |
| cora | 10 | 81.30 ± 1.08 | 80.68 ± 1.50 | -0.62 |
| citeseer | 10 | 74.28 ± 1.39 | 71.66 ± 1.48 | -2.62 |
| pubmed | 10 | 91.00 ± 1.09 | 88.92 ± 2.18 | -2.08 |
| arxiv | 10 | 61.76 ± 0.70 | 60.44 ± 1.16 | -1.32 |
| ogbn-products | 10 | 73.56 ± 1.56 | 72.28 ± 2.46 | -1.28 |

Overall paired accuracy delta=-2.7733pp,95%CI=[-3.4480,-2.1533],n=75.
