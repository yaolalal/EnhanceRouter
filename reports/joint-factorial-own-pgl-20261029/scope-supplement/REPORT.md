# 纠正性指标：分歧区、定义分母及调参效应补充

原论文Fig10是分歧区指标，因此同时展示“实际接受分歧伪标签精度”和“A+D全部接受精度”；原报告和原相关性保持封存。纠正性精度的主参照为各模型自身pGL。
空纠正集合不能定义精度，均值只统计有定义配置；n列是有定义配置数/全部配置数。小样本均值不能等同节点池化精度。

| Cohort | Dataset | Setting | Accuracy% | D accepted precision% | All accepted precision% | Corrective precision% | Defined n | Corrective nodes | Net gain | Right-minus-wrong |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all75 | ALL | B0 | 75.353 | 83.495 | 90.980 | 57.425 | 54/75 | 69.640 | 8.360 | -0.813 |
| all75 | ALL | D1_25__fixed | 75.037 | 80.775 | 89.943 | 51.539 | 59/75 | 105.160 | 19.640 | 3.747 |
| all75 | ALL | D1_25__retuned | 75.253 | 80.775 | 89.943 | 49.201 | 56/75 | 84.307 | 13.213 | -0.013 |
| all75 | ALL | D2a_1000__fixed | 74.999 | 83.115 | 90.889 | 62.068 | 48/75 | 61.947 | 12.800 | 5.147 |
| all75 | ALL | D2a_1000__retuned | 75.265 | 83.115 | 90.889 | 58.150 | 49/75 | 70.747 | 18.120 | 9.893 |
| all75 | ALL | D4_confidence__fixed | 73.435 | 75.945 | 88.288 | 48.767 | 35/75 | 19.707 | 7.600 | 6.347 |
| all75 | ALL | D4_confidence__retuned | 74.393 | 75.945 | 88.288 | 59.757 | 38/75 | 59.907 | 25.573 | 21.693 |
| all75 | cora | B0 | 80.333 | 90.215 | 96.040 | 82.728 | 8/15 | 6.600 | 4.933 | 4.467 |
| all75 | cora | D1_25__fixed | 79.940 | 89.308 | 95.760 | 81.066 | 9/15 | 10.467 | 6.733 | 5.533 |
| all75 | cora | D1_25__retuned | 80.533 | 89.308 | 95.760 | 59.317 | 7/15 | 5.067 | 3.800 | 3.333 |
| all75 | cora | D2a_1000__fixed | 79.807 | 88.133 | 95.653 | 82.866 | 7/15 | 4.867 | 3.000 | 2.600 |
| all75 | cora | D2a_1000__retuned | 80.333 | 88.133 | 95.653 | 48.450 | 7/15 | 5.733 | 1.600 | 1.067 |
| all75 | cora | D4_confidence__fixed | 75.387 | 45.566 | 84.638 | 23.844 | 12/15 | 22.867 | -7.533 | -10.200 |
| all75 | cora | D4_confidence__retuned | 78.280 | 45.566 | 84.638 | 29.286 | 14/15 | 57.400 | -9.667 | -17.533 |
| all75 | citeseer | B0 | 73.467 | 70.374 | 87.963 | 57.736 | 11/15 | 19.067 | 5.733 | 2.933 |
| all75 | citeseer | D1_25__fixed | 73.787 | 69.124 | 87.499 | 46.345 | 11/15 | 9.467 | 2.800 | 1.333 |
| all75 | citeseer | D1_25__retuned | 73.573 | 69.124 | 87.499 | 53.799 | 11/15 | 11.267 | 4.067 | 2.467 |
| all75 | citeseer | D2a_1000__fixed | 73.520 | 66.612 | 86.227 | 56.654 | 11/15 | 11.600 | 5.200 | 4.267 |
| all75 | citeseer | D2a_1000__retuned | 73.780 | 66.612 | 86.227 | 51.283 | 10/15 | 9.333 | 3.600 | 2.667 |
| all75 | citeseer | D4_confidence__fixed | 72.467 | 70.013 | 85.015 | 100.000 | 1/15 | 1.067 | 1.067 | 1.067 |
| all75 | citeseer | D4_confidence__retuned | 73.087 | 70.013 | 85.015 | 100.000 | 1/15 | 0.067 | 0.067 | 0.067 |
| all75 | pubmed | B0 | 90.693 | 90.070 | 95.631 | 64.169 | 14/15 | 84.800 | 50.067 | 47.467 |
| all75 | pubmed | D1_25__fixed | 90.393 | 87.322 | 94.828 | 65.520 | 14/15 | 232.933 | 32.933 | 28.933 |
| all75 | pubmed | D1_25__retuned | 90.360 | 87.322 | 94.828 | 64.020 | 13/15 | 212.600 | 8.267 | 4.733 |
| all75 | pubmed | D2a_1000__fixed | 90.433 | 92.290 | 96.279 | 76.327 | 9/15 | 85.333 | 42.000 | 40.533 |
| all75 | pubmed | D2a_1000__retuned | 90.467 | 92.290 | 96.279 | 82.661 | 11/15 | 108.067 | 60.000 | 58.467 |
| all75 | pubmed | D4_confidence__fixed | 89.780 | 92.892 | 96.154 | 44.205 | 11/15 | 27.533 | 20.933 | 20.867 |
| all75 | pubmed | D4_confidence__retuned | 90.553 | 92.892 | 96.154 | 71.628 | 11/15 | 146.133 | 129.600 | 129.333 |
| all75 | arxiv | B0 | 59.613 | 82.278 | 86.243 | 50.868 | 6/15 | 164.000 | 0.200 | -28.667 |
| all75 | arxiv | D1_25__fixed | 59.273 | 74.878 | 83.093 | 39.108 | 10/15 | 194.733 | 52.200 | -8.200 |
| all75 | arxiv | D1_25__retuned | 59.613 | 74.878 | 83.093 | 42.878 | 10/15 | 125.800 | 53.133 | 2.600 |
| all75 | arxiv | D2a_1000__fixed | 59.153 | 83.857 | 87.494 | 72.395 | 7/15 | 29.333 | -6.600 | -10.267 |
| all75 | arxiv | D2a_1000__retuned | 59.327 | 83.857 | 87.494 | 68.494 | 7/15 | 35.600 | -3.867 | -8.267 |
| all75 | arxiv | D4_confidence__fixed | 58.367 | 83.382 | 85.307 | 77.193 | 5/15 | 5.133 | 0.133 | -0.067 |
| all75 | arxiv | D4_confidence__retuned | 58.967 | 83.382 | 85.307 | 75.607 | 5/15 | 78.200 | -3.200 | -13.533 |
| all75 | ogbn-products | B0 | 72.660 | 84.538 | 89.023 | 40.029 | 15/15 | 73.733 | -19.133 | -30.267 |
| all75 | ogbn-products | D1_25__fixed | 71.793 | 83.245 | 88.535 | 32.871 | 15/15 | 78.200 | 3.533 | -8.867 |
| all75 | ogbn-products | D1_25__retuned | 72.187 | 83.245 | 88.535 | 32.480 | 15/15 | 66.800 | -3.200 | -13.200 |
| all75 | ogbn-products | D2a_1000__fixed | 72.080 | 84.683 | 88.792 | 41.593 | 14/15 | 178.600 | 20.400 | -11.400 |
| all75 | ogbn-products | D2a_1000__retuned | 72.420 | 84.683 | 88.792 | 43.475 | 14/15 | 195.000 | 29.267 | -4.467 |
| all75 | ogbn-products | D4_confidence__fixed | 71.173 | 87.875 | 90.323 | 74.749 | 6/15 | 41.933 | 23.400 | 20.067 |
| all75 | ogbn-products | D4_confidence__retuned | 71.080 | 87.875 | 90.323 | 84.977 | 7/15 | 17.733 | 11.067 | 10.133 |
| fig10_5shot_seed42_44 | pubmed | B0 | 91.467 | 89.637 | 96.674 | 65.142 | 3/3 | 14.333 | 4.000 | 3.667 |
| fig10_5shot_seed42_44 | pubmed | D1_25__fixed | 91.733 | 89.067 | 96.247 | 75.621 | 3/3 | 45.333 | 27.333 | 27.333 |
| fig10_5shot_seed42_44 | pubmed | D1_25__retuned | 91.400 | 89.067 | 96.247 | 74.297 | 3/3 | 116.333 | 76.667 | 76.333 |
| fig10_5shot_seed42_44 | pubmed | D2a_1000__fixed | 91.433 | 95.542 | 97.919 | 100.000 | 1/3 | 0.667 | 0.667 | 0.667 |
| fig10_5shot_seed42_44 | pubmed | D2a_1000__retuned | 91.100 | 95.542 | 97.919 | 100.000 | 2/3 | 2.333 | 2.333 | 2.333 |
| fig10_5shot_seed42_44 | pubmed | D4_confidence__fixed | 90.633 | 95.612 | 96.878 | 43.750 | 2/3 | 3.000 | 1.667 | 1.667 |
| fig10_5shot_seed42_44 | pubmed | D4_confidence__retuned | 91.100 | 95.612 | 96.878 | 48.980 | 2/3 | 16.667 | 15.333 | 15.333 |
| fig10_5shot_seed42_44 | ogbn-products | B0 | 72.433 | 87.090 | 91.992 | 49.274 | 3/3 | 29.000 | -9.000 | -10.333 |
| fig10_5shot_seed42_44 | ogbn-products | D1_25__fixed | 72.067 | 84.801 | 90.215 | 53.633 | 3/3 | 73.333 | 21.333 | 13.333 |
| fig10_5shot_seed42_44 | ogbn-products | D1_25__retuned | 70.967 | 84.801 | 90.215 | 51.141 | 3/3 | 45.333 | 3.000 | -2.667 |
| fig10_5shot_seed42_44 | ogbn-products | D2a_1000__fixed | 72.267 | 91.053 | 93.356 | 63.519 | 3/3 | 29.333 | 22.000 | 20.667 |
| fig10_5shot_seed42_44 | ogbn-products | D2a_1000__retuned | 71.900 | 91.053 | 93.356 | 62.585 | 3/3 | 27.667 | 19.667 | 18.333 |
| fig10_5shot_seed42_44 | ogbn-products | D4_confidence__fixed | 70.667 | 91.038 | 92.642 | 85.714 | 1/3 | 2.333 | 1.667 | 1.667 |
| fig10_5shot_seed42_44 | ogbn-products | D4_confidence__retuned | 69.867 | 91.038 | 92.642 | 100.000 | 1/3 | 0.333 | 0.333 | 0.333 |

## 同样本相关比较

比较使用相同有定义配置的相对B0差值，D2的B0已匹配剩余Query。按dataset×seed簇重采样4000次，保持shot依赖；数据集内仅5簇，区间是探索性的。不显著不代表不相关。

| Dataset | Treatment | Joint | Precision comparison | Method | n | Corrective r | Precision r | Difference 95% CI |
|---|---|---|---|---|---:|---:|---:|---|
| ALL | D1_25 | fixed | D_accepted_precision_delta_pp | Pearson | 51 | 0.162 | -0.003 | [-0.372, 0.582] |
| ALL | D1_25 | fixed | D_accepted_precision_delta_pp | Spearman | 51 | 0.082 | 0.028 | [-0.309, 0.441] |
| ALL | D1_25 | fixed | all_accepted_precision_delta_pp | Pearson | 51 | 0.162 | -0.037 | [-0.396, 0.626] |
| ALL | D1_25 | fixed | all_accepted_precision_delta_pp | Spearman | 51 | 0.082 | 0.040 | [-0.348, 0.432] |
| ALL | D1_25 | retuned | D_accepted_precision_delta_pp | Pearson | 48 | -0.238 | 0.031 | [-0.572, 0.031] |
| ALL | D1_25 | retuned | D_accepted_precision_delta_pp | Spearman | 48 | -0.063 | -0.005 | [-0.421, 0.323] |
| ALL | D1_25 | retuned | all_accepted_precision_delta_pp | Pearson | 48 | -0.238 | 0.079 | [-0.612, -0.015] |
| ALL | D1_25 | retuned | all_accepted_precision_delta_pp | Spearman | 48 | -0.063 | 0.082 | [-0.490, 0.193] |
| ALL | D2a_1000 | fixed | D_accepted_precision_delta_pp | Pearson | 42 | 0.283 | -0.108 | [0.063, 0.649] |
| ALL | D2a_1000 | fixed | D_accepted_precision_delta_pp | Spearman | 42 | 0.345 | -0.024 | [0.084, 0.624] |
| ALL | D2a_1000 | fixed | all_accepted_precision_delta_pp | Pearson | 42 | 0.283 | -0.224 | [0.074, 0.795] |
| ALL | D2a_1000 | fixed | all_accepted_precision_delta_pp | Spearman | 42 | 0.345 | -0.063 | [0.088, 0.714] |
| ALL | D2a_1000 | retuned | D_accepted_precision_delta_pp | Pearson | 42 | -0.197 | -0.247 | [-0.256, 0.328] |
| ALL | D2a_1000 | retuned | D_accepted_precision_delta_pp | Spearman | 42 | -0.211 | -0.257 | [-0.254, 0.308] |
| ALL | D2a_1000 | retuned | all_accepted_precision_delta_pp | Pearson | 42 | -0.197 | -0.373 | [-0.240, 0.474] |
| ALL | D2a_1000 | retuned | all_accepted_precision_delta_pp | Spearman | 42 | -0.211 | -0.319 | [-0.239, 0.413] |
| ALL | D4_confidence | fixed | D_accepted_precision_delta_pp | Pearson | 25 | 0.410 | 0.621 | [-0.445, 0.040] |
| ALL | D4_confidence | fixed | D_accepted_precision_delta_pp | Spearman | 25 | 0.352 | 0.430 | [-0.469, 0.319] |
| ALL | D4_confidence | fixed | all_accepted_precision_delta_pp | Pearson | 25 | 0.410 | 0.353 | [-0.233, 0.300] |
| ALL | D4_confidence | fixed | all_accepted_precision_delta_pp | Spearman | 25 | 0.352 | 0.260 | [-0.236, 0.446] |
| ALL | D4_confidence | retuned | D_accepted_precision_delta_pp | Pearson | 28 | 0.212 | 0.456 | [-0.554, 0.104] |
| ALL | D4_confidence | retuned | D_accepted_precision_delta_pp | Spearman | 28 | 0.185 | 0.439 | [-0.637, 0.079] |
| ALL | D4_confidence | retuned | all_accepted_precision_delta_pp | Pearson | 28 | 0.212 | 0.348 | [-0.430, 0.270] |
| ALL | D4_confidence | retuned | all_accepted_precision_delta_pp | Spearman | 28 | 0.185 | 0.329 | [-0.530, 0.198] |
| cora | D1_25 | fixed | D_accepted_precision_delta_pp | Pearson | 6 | -0.716 | -0.635 | [-0.341, 0.865] |
| cora | D1_25 | fixed | D_accepted_precision_delta_pp | Spearman | 6 | -0.406 | -0.667 | [0.000, 1.010] |
| cora | D1_25 | fixed | all_accepted_precision_delta_pp | Pearson | 6 | -0.716 | -0.355 | [-0.697, 0.822] |
| cora | D1_25 | fixed | all_accepted_precision_delta_pp | Spearman | 6 | -0.406 | -0.754 | [0.000, 1.481] |
| cora | D1_25 | retuned | D_accepted_precision_delta_pp | Pearson | 4 | -0.804 | -0.972 | [-0.057, 0.250] |
| cora | D1_25 | retuned | D_accepted_precision_delta_pp | Spearman | 4 | -0.800 | -0.800 | [0.000, 0.000] |
| cora | D1_25 | retuned | all_accepted_precision_delta_pp | Pearson | 4 | -0.804 | -0.839 | [-0.000, 0.104] |
| cora | D1_25 | retuned | all_accepted_precision_delta_pp | Spearman | 4 | -0.800 | -0.800 | [0.000, 0.000] |
| cora | D2a_1000 | fixed | D_accepted_precision_delta_pp | Pearson | 6 | -0.671 | -0.426 | [-0.449, 0.446] |
| cora | D2a_1000 | fixed | D_accepted_precision_delta_pp | Spearman | 6 | -0.348 | -0.314 | [-0.255, 0.000] |
| cora | D2a_1000 | fixed | all_accepted_precision_delta_pp | Pearson | 6 | -0.671 | -0.143 | [-0.800, 0.477] |
| cora | D2a_1000 | fixed | all_accepted_precision_delta_pp | Spearman | 6 | -0.348 | 0.029 | [-0.839, 0.000] |
| cora | D2a_1000 | retuned | D_accepted_precision_delta_pp | Pearson | 5 | -0.495 | -0.021 | [-2.000, 0.026] |
| cora | D2a_1000 | retuned | D_accepted_precision_delta_pp | Spearman | 5 | -0.600 | 0.300 | [-2.000, 0.000] |
| cora | D2a_1000 | retuned | all_accepted_precision_delta_pp | Pearson | 5 | -0.495 | 0.128 | [-2.000, 0.000] |
| cora | D2a_1000 | retuned | all_accepted_precision_delta_pp | Spearman | 5 | -0.600 | 0.400 | [-2.000, 0.000] |
| cora | D4_confidence | fixed | D_accepted_precision_delta_pp | Pearson | 6 | 0.272 | -0.024 | [-1.107, 0.848] |
| cora | D4_confidence | fixed | D_accepted_precision_delta_pp | Spearman | 6 | 0.143 | -0.143 | [-1.125, 1.000] |
| cora | D4_confidence | fixed | all_accepted_precision_delta_pp | Pearson | 6 | 0.272 | -0.275 | [-0.325, 1.435] |
| cora | D4_confidence | fixed | all_accepted_precision_delta_pp | Spearman | 6 | 0.143 | -0.886 | [0.000, 1.529] |
| cora | D4_confidence | retuned | D_accepted_precision_delta_pp | Pearson | 8 | 0.804 | 0.400 | [-0.399, 0.891] |
| cora | D4_confidence | retuned | D_accepted_precision_delta_pp | Spearman | 8 | 0.619 | 0.381 | [-0.800, 1.080] |
| cora | D4_confidence | retuned | all_accepted_precision_delta_pp | Pearson | 8 | 0.804 | 0.429 | [-0.645, 1.632] |
| cora | D4_confidence | retuned | all_accepted_precision_delta_pp | Spearman | 8 | 0.619 | 0.238 | [-0.943, 1.585] |
| citeseer | D1_25 | fixed | D_accepted_precision_delta_pp | Pearson | 10 | -0.262 | 0.051 | [-1.287, 0.633] |
| citeseer | D1_25 | fixed | D_accepted_precision_delta_pp | Spearman | 10 | -0.562 | -0.096 | [-1.345, 0.435] |
| citeseer | D1_25 | fixed | all_accepted_precision_delta_pp | Pearson | 10 | -0.262 | 0.225 | [-1.360, 0.636] |
| citeseer | D1_25 | fixed | all_accepted_precision_delta_pp | Spearman | 10 | -0.562 | -0.084 | [-1.348, 0.720] |
| citeseer | D1_25 | retuned | D_accepted_precision_delta_pp | Pearson | 10 | -0.031 | 0.183 | [-0.984, 0.282] |
| citeseer | D1_25 | retuned | D_accepted_precision_delta_pp | Spearman | 10 | -0.012 | 0.263 | [-1.195, 0.176] |
| citeseer | D1_25 | retuned | all_accepted_precision_delta_pp | Pearson | 10 | -0.031 | 0.166 | [-0.951, 0.449] |
| citeseer | D1_25 | retuned | all_accepted_precision_delta_pp | Spearman | 10 | -0.012 | 0.330 | [-1.327, 0.559] |
| citeseer | D2a_1000 | fixed | D_accepted_precision_delta_pp | Pearson | 9 | 0.352 | 0.218 | [-0.350, 0.539] |
| citeseer | D2a_1000 | fixed | D_accepted_precision_delta_pp | Spearman | 9 | 0.435 | 0.301 | [-0.177, 0.600] |
| citeseer | D2a_1000 | fixed | all_accepted_precision_delta_pp | Pearson | 9 | 0.352 | 0.236 | [-0.321, 0.560] |
| citeseer | D2a_1000 | fixed | all_accepted_precision_delta_pp | Spearman | 9 | 0.435 | 0.259 | [-0.154, 0.800] |
| citeseer | D2a_1000 | retuned | D_accepted_precision_delta_pp | Pearson | 8 | 0.056 | 0.148 | [-0.557, 0.580] |
| citeseer | D2a_1000 | retuned | D_accepted_precision_delta_pp | Spearman | 8 | 0.193 | 0.072 | [-1.144, 0.943] |
| citeseer | D2a_1000 | retuned | all_accepted_precision_delta_pp | Pearson | 8 | 0.056 | 0.204 | [-0.786, 0.524] |
| citeseer | D2a_1000 | retuned | all_accepted_precision_delta_pp | Spearman | 8 | 0.193 | -0.024 | [-1.144, 0.943] |
| pubmed | D1_25 | fixed | D_accepted_precision_delta_pp | Pearson | 14 | 0.081 | 0.296 | [-1.300, 0.059] |
| pubmed | D1_25 | fixed | D_accepted_precision_delta_pp | Spearman | 14 | 0.112 | 0.529 | [-1.169, 0.349] |
| pubmed | D1_25 | fixed | all_accepted_precision_delta_pp | Pearson | 14 | 0.081 | 0.136 | [-0.881, 1.045] |
| pubmed | D1_25 | fixed | all_accepted_precision_delta_pp | Spearman | 14 | 0.112 | -0.123 | [-0.693, 1.200] |
| pubmed | D1_25 | retuned | D_accepted_precision_delta_pp | Pearson | 13 | -0.156 | -0.131 | [-0.709, 0.336] |
| pubmed | D1_25 | retuned | D_accepted_precision_delta_pp | Spearman | 13 | 0.050 | -0.033 | [-0.718, 0.641] |
| pubmed | D1_25 | retuned | all_accepted_precision_delta_pp | Pearson | 13 | -0.156 | -0.299 | [-0.276, 0.824] |
| pubmed | D1_25 | retuned | all_accepted_precision_delta_pp | Spearman | 13 | 0.050 | -0.548 | [-0.417, 1.287] |
| pubmed | D2a_1000 | fixed | D_accepted_precision_delta_pp | Pearson | 8 | -0.093 | 0.175 | [-0.482, 0.001] |
| pubmed | D2a_1000 | fixed | D_accepted_precision_delta_pp | Spearman | 8 | -0.241 | 0.024 | [-0.483, 0.000] |
| pubmed | D2a_1000 | fixed | all_accepted_precision_delta_pp | Pearson | 8 | -0.093 | -0.073 | [-0.264, 0.092] |
| pubmed | D2a_1000 | fixed | all_accepted_precision_delta_pp | Spearman | 8 | -0.241 | -0.241 | [0.000, 0.000] |
| pubmed | D2a_1000 | retuned | D_accepted_precision_delta_pp | Pearson | 10 | -0.544 | -0.112 | [-0.841, -0.057] |
| pubmed | D2a_1000 | retuned | D_accepted_precision_delta_pp | Spearman | 10 | -0.717 | -0.116 | [-0.845, 0.000] |
| pubmed | D2a_1000 | retuned | all_accepted_precision_delta_pp | Pearson | 10 | -0.544 | -0.149 | [-0.867, -0.131] |
| pubmed | D2a_1000 | retuned | all_accepted_precision_delta_pp | Spearman | 10 | -0.717 | -0.164 | [-0.793, -0.043] |
| pubmed | D4_confidence | fixed | D_accepted_precision_delta_pp | Pearson | 10 | 0.468 | -0.069 | [-0.882, 1.339] |
| pubmed | D4_confidence | fixed | D_accepted_precision_delta_pp | Spearman | 10 | 0.322 | 0.061 | [-1.038, 1.531] |
| pubmed | D4_confidence | fixed | all_accepted_precision_delta_pp | Pearson | 10 | 0.468 | -0.079 | [-0.864, 1.259] |
| pubmed | D4_confidence | fixed | all_accepted_precision_delta_pp | Spearman | 10 | 0.322 | -0.091 | [-0.828, 1.500] |
| pubmed | D4_confidence | retuned | D_accepted_precision_delta_pp | Pearson | 10 | 0.194 | 0.199 | [-0.766, 0.719] |
| pubmed | D4_confidence | retuned | D_accepted_precision_delta_pp | Spearman | 10 | 0.127 | 0.503 | [-1.278, 0.494] |
| pubmed | D4_confidence | retuned | all_accepted_precision_delta_pp | Pearson | 10 | 0.194 | 0.179 | [-0.732, 0.576] |
| pubmed | D4_confidence | retuned | all_accepted_precision_delta_pp | Spearman | 10 | 0.127 | 0.285 | [-1.333, 0.700] |
| arxiv | D1_25 | fixed | D_accepted_precision_delta_pp | Pearson | 6 | -0.716 | -0.782 | [-0.446, 0.669] |
| arxiv | D1_25 | fixed | D_accepted_precision_delta_pp | Spearman | 6 | -0.754 | -0.899 | [-0.935, 0.750] |
| arxiv | D1_25 | fixed | all_accepted_precision_delta_pp | Pearson | 6 | -0.716 | -0.768 | [-0.427, 1.290] |
| arxiv | D1_25 | fixed | all_accepted_precision_delta_pp | Spearman | 6 | -0.754 | -0.986 | [0.000, 0.866] |
| arxiv | D1_25 | retuned | D_accepted_precision_delta_pp | Pearson | 6 | -0.089 | 0.224 | [-0.728, 0.300] |
| arxiv | D1_25 | retuned | D_accepted_precision_delta_pp | Spearman | 6 | -0.429 | 0.029 | [-1.500, 0.000] |
| arxiv | D1_25 | retuned | all_accepted_precision_delta_pp | Pearson | 6 | -0.089 | 0.125 | [-0.760, 1.188] |
| arxiv | D1_25 | retuned | all_accepted_precision_delta_pp | Spearman | 6 | -0.429 | -0.143 | [-1.500, 0.500] |
| arxiv | D2a_1000 | fixed | D_accepted_precision_delta_pp | Pearson | 5 | 0.423 | 0.183 | [-0.000, 0.719] |
| arxiv | D2a_1000 | fixed | D_accepted_precision_delta_pp | Spearman | 5 | 0.400 | 0.200 | [-0.121, 1.111] |
| arxiv | D2a_1000 | fixed | all_accepted_precision_delta_pp | Pearson | 5 | 0.423 | -0.582 | [0.000, 2.000] |
| arxiv | D2a_1000 | fixed | all_accepted_precision_delta_pp | Spearman | 5 | 0.400 | -0.600 | [0.000, 2.000] |
| arxiv | D2a_1000 | retuned | D_accepted_precision_delta_pp | Pearson | 5 | 0.137 | 0.386 | [-0.325, 0.324] |
| arxiv | D2a_1000 | retuned | D_accepted_precision_delta_pp | Spearman | 5 | -0.051 | 0.051 | [-0.364, 0.667] |
| arxiv | D2a_1000 | retuned | all_accepted_precision_delta_pp | Pearson | 5 | 0.137 | -0.661 | [0.000, 2.000] |
| arxiv | D2a_1000 | retuned | all_accepted_precision_delta_pp | Spearman | 5 | -0.051 | -0.821 | [0.000, 2.000] |
| ogbn-products | D1_25 | fixed | D_accepted_precision_delta_pp | Pearson | 15 | 0.759 | 0.214 | [-0.316, 0.799] |
| ogbn-products | D1_25 | fixed | D_accepted_precision_delta_pp | Spearman | 15 | 0.691 | 0.248 | [-0.293, 0.852] |
| ogbn-products | D1_25 | fixed | all_accepted_precision_delta_pp | Pearson | 15 | 0.759 | 0.237 | [-0.412, 0.834] |
| ogbn-products | D1_25 | fixed | all_accepted_precision_delta_pp | Spearman | 15 | 0.691 | 0.308 | [-0.441, 0.826] |
| ogbn-products | D1_25 | retuned | D_accepted_precision_delta_pp | Pearson | 15 | -0.385 | 0.023 | [-1.014, 0.630] |
| ogbn-products | D1_25 | retuned | D_accepted_precision_delta_pp | Spearman | 15 | -0.125 | -0.027 | [-0.801, 0.861] |
| ogbn-products | D1_25 | retuned | all_accepted_precision_delta_pp | Pearson | 15 | -0.385 | 0.236 | [-1.109, 0.307] |
| ogbn-products | D1_25 | retuned | all_accepted_precision_delta_pp | Spearman | 15 | -0.125 | 0.261 | [-0.953, 0.465] |
| ogbn-products | D2a_1000 | fixed | D_accepted_precision_delta_pp | Pearson | 14 | 0.717 | -0.048 | [0.248, 1.095] |
| ogbn-products | D2a_1000 | fixed | D_accepted_precision_delta_pp | Spearman | 14 | 0.852 | 0.177 | [0.340, 1.074] |
| ogbn-products | D2a_1000 | fixed | all_accepted_precision_delta_pp | Pearson | 14 | 0.717 | 0.002 | [0.252, 1.000] |
| ogbn-products | D2a_1000 | fixed | all_accepted_precision_delta_pp | Spearman | 14 | 0.852 | 0.150 | [0.337, 1.157] |
| ogbn-products | D2a_1000 | retuned | D_accepted_precision_delta_pp | Pearson | 14 | -0.096 | -0.467 | [-0.774, 0.849] |
| ogbn-products | D2a_1000 | retuned | D_accepted_precision_delta_pp | Spearman | 14 | -0.035 | -0.270 | [-0.906, 0.731] |
| ogbn-products | D2a_1000 | retuned | all_accepted_precision_delta_pp | Pearson | 14 | -0.096 | -0.505 | [-0.825, 0.821] |
| ogbn-products | D2a_1000 | retuned | all_accepted_precision_delta_pp | Spearman | 14 | -0.035 | -0.283 | [-0.868, 0.851] |
| ogbn-products | D4_confidence | fixed | D_accepted_precision_delta_pp | Pearson | 6 | 0.081 | -0.001 | [-0.226, 0.893] |
| ogbn-products | D4_confidence | fixed | D_accepted_precision_delta_pp | Spearman | 6 | 0.087 | -0.232 | [-0.200, 1.125] |
| ogbn-products | D4_confidence | fixed | all_accepted_precision_delta_pp | Pearson | 6 | 0.081 | -0.108 | [-0.066, 0.953] |
| ogbn-products | D4_confidence | fixed | all_accepted_precision_delta_pp | Spearman | 6 | 0.087 | -0.406 | [0.000, 1.750] |
| ogbn-products | D4_confidence | retuned | D_accepted_precision_delta_pp | Pearson | 7 | -0.681 | 0.164 | [-1.363, 0.648] |
| ogbn-products | D4_confidence | retuned | D_accepted_precision_delta_pp | Spearman | 7 | -0.357 | 0.286 | [-1.500, 0.947] |
| ogbn-products | D4_confidence | retuned | all_accepted_precision_delta_pp | Pearson | 7 | -0.681 | 0.190 | [-1.547, 0.708] |
| ogbn-products | D4_confidence | retuned | all_accepted_precision_delta_pp | Spearman | 7 | -0.357 | 0.357 | [-1.579, 0.947] |

## 调参效果（同Router，retuned−fixed）

| Dataset | Treatment | n | Accuracy gain pp | Cluster 95% CI pp |
|---|---|---:|---:|---|
| ALL | D1_25 | 75 | 0.216 | [-0.157, 0.633] |
| ALL | D2a_1000 | 75 | 0.267 | [0.020, 0.543] |
| ALL | D4_confidence | 75 | 0.959 | [0.461, 1.495] |
| cora | D1_25 | 15 | 0.593 | [0.233, 1.067] |
| cora | D2a_1000 | 15 | 0.527 | [0.193, 0.787] |
| cora | D4_confidence | 15 | 2.893 | [2.020, 3.587] |
| citeseer | D1_25 | 15 | -0.213 | [-0.633, 0.193] |
| citeseer | D2a_1000 | 15 | 0.260 | [0.053, 0.513] |
| citeseer | D4_confidence | 15 | 0.620 | [0.193, 0.973] |
| pubmed | D1_25 | 15 | -0.033 | [-0.427, 0.360] |
| pubmed | D2a_1000 | 15 | 0.033 | [-0.413, 0.327] |
| pubmed | D4_confidence | 15 | 0.773 | [0.240, 1.420] |
| arxiv | D1_25 | 15 | 0.340 | [-0.507, 1.307] |
| arxiv | D2a_1000 | 15 | 0.173 | [-0.327, 0.721] |
| arxiv | D4_confidence | 15 | 0.600 | [-0.040, 1.160] |
| ogbn-products | D1_25 | 15 | 0.393 | [-1.008, 1.993] |
| ogbn-products | D2a_1000 | 15 | 0.340 | [-0.514, 1.447] |
| ogbn-products | D4_confidence | 15 | -0.093 | [-0.940, 0.927] |

历史45组B0 cross计数：{'0.0': 2, '0.25': 13, '1.0': 30}

重调同时开放多个Joint数值项，因此不能把全部调参收益归因cross。描述性相关也不能证明因果机制。
