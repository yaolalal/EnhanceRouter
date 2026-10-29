# 纠正性指标：分歧区、定义分母及调参效应补充

原论文Fig10是分歧区指标，因此同时展示“实际接受分歧伪标签精度”和“A+D全部接受精度”；原报告和原相关性保持封存。纠正性精度的主参照固定为B0 pGL，自身pGL另存。
空纠正集合不能定义精度，均值只统计有定义配置；n列是有定义配置数/全部配置数。小样本均值不能等同节点池化精度。

| Cohort | Dataset | Setting | Accuracy% | D accepted precision% | All accepted precision% | Corrective precision% | Defined n | Corrective nodes | Net gain | Right-minus-wrong |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all75 | ALL | B0 | 75.353 | 83.495 | 90.980 | 57.425 | 54/75 | 69.640 | 8.360 | -0.813 |
| all75 | ALL | D1_25__fixed | 75.037 | 80.775 | 89.943 | 51.820 | 57/75 | 97.133 | 11.067 | -4.280 |
| all75 | ALL | D1_25__retuned | 75.253 | 80.775 | 89.943 | 51.820 | 57/75 | 97.133 | 11.067 | -4.280 |
| all75 | ALL | D2a_1000__fixed | 74.999 | 83.115 | 90.889 | 59.210 | 46/75 | 78.000 | 21.267 | 11.600 |
| all75 | ALL | D2a_1000__retuned | 75.265 | 83.115 | 90.889 | 59.210 | 46/75 | 78.000 | 21.267 | 11.600 |
| all75 | ALL | D4_confidence__fixed | 73.435 | 75.945 | 88.288 | 47.789 | 40/75 | 35.107 | 4.787 | 2.867 |
| all75 | ALL | D4_confidence__retuned | 74.393 | 75.945 | 88.288 | 47.789 | 40/75 | 35.107 | 4.787 | 2.867 |
| all75 | cora | B0 | 80.333 | 90.215 | 96.040 | 82.728 | 8/15 | 6.600 | 4.933 | 4.467 |
| all75 | cora | D1_25__fixed | 79.940 | 89.308 | 95.760 | 72.894 | 7/15 | 8.400 | 5.333 | 4.267 |
| all75 | cora | D1_25__retuned | 80.533 | 89.308 | 95.760 | 72.894 | 7/15 | 8.400 | 5.333 | 4.267 |
| all75 | cora | D2a_1000__fixed | 79.807 | 88.133 | 95.653 | 66.384 | 5/15 | 3.467 | 1.733 | 1.467 |
| all75 | cora | D2a_1000__retuned | 80.333 | 88.133 | 95.653 | 66.384 | 5/15 | 3.467 | 1.733 | 1.467 |
| all75 | cora | D4_confidence__fixed | 75.387 | 45.566 | 84.638 | 22.304 | 15/15 | 36.933 | -12.067 | -16.800 |
| all75 | cora | D4_confidence__retuned | 78.280 | 45.566 | 84.638 | 22.304 | 15/15 | 36.933 | -12.067 | -16.800 |
| all75 | citeseer | B0 | 73.467 | 70.374 | 87.963 | 57.736 | 11/15 | 19.067 | 5.733 | 2.933 |
| all75 | citeseer | D1_25__fixed | 73.787 | 69.124 | 87.499 | 54.462 | 11/15 | 10.000 | 3.333 | 1.733 |
| all75 | citeseer | D1_25__retuned | 73.573 | 69.124 | 87.499 | 54.462 | 11/15 | 10.000 | 3.333 | 1.733 |
| all75 | citeseer | D2a_1000__fixed | 73.520 | 66.612 | 86.227 | 56.201 | 11/15 | 11.133 | 5.067 | 4.067 |
| all75 | citeseer | D2a_1000__retuned | 73.780 | 66.612 | 86.227 | 56.201 | 11/15 | 11.133 | 5.067 | 4.067 |
| all75 | citeseer | D4_confidence__fixed | 72.467 | 70.013 | 85.015 | 91.026 | 1/15 | 5.200 | 4.467 | 4.267 |
| all75 | citeseer | D4_confidence__retuned | 73.087 | 70.013 | 85.015 | 91.026 | 1/15 | 5.200 | 4.467 | 4.267 |
| all75 | pubmed | B0 | 90.693 | 90.070 | 95.631 | 64.169 | 14/15 | 84.800 | 50.067 | 47.467 |
| all75 | pubmed | D1_25__fixed | 90.393 | 87.322 | 94.828 | 63.641 | 14/15 | 210.533 | -1.400 | -5.200 |
| all75 | pubmed | D1_25__retuned | 90.360 | 87.322 | 94.828 | 63.641 | 14/15 | 210.533 | -1.400 | -5.200 |
| all75 | pubmed | D2a_1000__fixed | 90.433 | 92.290 | 96.279 | 73.368 | 9/15 | 90.733 | 43.467 | 41.933 |
| all75 | pubmed | D2a_1000__retuned | 90.467 | 92.290 | 96.279 | 73.368 | 9/15 | 90.733 | 43.467 | 41.933 |
| all75 | pubmed | D4_confidence__fixed | 89.780 | 92.892 | 96.154 | 43.979 | 12/15 | 77.400 | 9.733 | 9.267 |
| all75 | pubmed | D4_confidence__retuned | 90.553 | 92.892 | 96.154 | 43.979 | 12/15 | 77.400 | 9.733 | 9.267 |
| all75 | arxiv | B0 | 59.613 | 82.278 | 86.243 | 50.868 | 6/15 | 164.000 | 0.200 | -28.667 |
| all75 | arxiv | D1_25__fixed | 59.273 | 74.878 | 83.093 | 39.153 | 10/15 | 195.867 | 52.733 | -7.733 |
| all75 | arxiv | D1_25__retuned | 59.613 | 74.878 | 83.093 | 39.153 | 10/15 | 195.867 | 52.733 | -7.733 |
| all75 | arxiv | D2a_1000__fixed | 59.153 | 83.857 | 87.494 | 65.511 | 7/15 | 30.200 | -6.467 | -10.200 |
| all75 | arxiv | D2a_1000__retuned | 59.327 | 83.857 | 87.494 | 65.511 | 7/15 | 30.200 | -6.467 | -10.200 |
| all75 | arxiv | D4_confidence__fixed | 58.367 | 83.382 | 85.307 | 75.834 | 6/15 | 14.333 | -1.000 | -1.933 |
| all75 | arxiv | D4_confidence__retuned | 58.967 | 83.382 | 85.307 | 75.834 | 6/15 | 14.333 | -1.000 | -1.933 |
| all75 | ogbn-products | B0 | 72.660 | 84.538 | 89.023 | 40.029 | 15/15 | 73.733 | -19.133 | -30.267 |
| all75 | ogbn-products | D1_25__fixed | 71.793 | 83.245 | 88.535 | 37.461 | 15/15 | 60.867 | -4.667 | -14.467 |
| all75 | ogbn-products | D1_25__retuned | 72.187 | 83.245 | 88.535 | 37.461 | 15/15 | 60.867 | -4.667 | -14.467 |
| all75 | ogbn-products | D2a_1000__fixed | 72.080 | 84.683 | 88.792 | 46.761 | 14/15 | 254.467 | 62.533 | 20.733 |
| all75 | ogbn-products | D2a_1000__retuned | 72.420 | 84.683 | 88.792 | 46.761 | 14/15 | 254.467 | 62.533 | 20.733 |
| all75 | ogbn-products | D4_confidence__fixed | 71.173 | 87.875 | 90.323 | 83.869 | 6/15 | 41.667 | 22.800 | 19.533 |
| all75 | ogbn-products | D4_confidence__retuned | 71.080 | 87.875 | 90.323 | 83.869 | 6/15 | 41.667 | 22.800 | 19.533 |
| fig10_5shot_seed42_44 | pubmed | B0 | 91.467 | 89.637 | 96.674 | 65.142 | 3/3 | 14.333 | 4.000 | 3.667 |
| fig10_5shot_seed42_44 | pubmed | D1_25__fixed | 91.733 | 89.067 | 96.247 | 74.036 | 3/3 | 53.000 | 31.333 | 31.000 |
| fig10_5shot_seed42_44 | pubmed | D1_25__retuned | 91.400 | 89.067 | 96.247 | 74.036 | 3/3 | 53.000 | 31.333 | 31.000 |
| fig10_5shot_seed42_44 | pubmed | D2a_1000__fixed | 91.433 | 95.542 | 97.919 | 75.000 | 2/3 | 2.667 | 0.667 | 0.667 |
| fig10_5shot_seed42_44 | pubmed | D2a_1000__retuned | 91.100 | 95.542 | 97.919 | 75.000 | 2/3 | 2.667 | 0.667 | 0.667 |
| fig10_5shot_seed42_44 | pubmed | D4_confidence__fixed | 90.633 | 95.612 | 96.878 | 0.000 | 2/3 | 2.000 | -2.000 | -2.000 |
| fig10_5shot_seed42_44 | pubmed | D4_confidence__retuned | 91.100 | 95.612 | 96.878 | 0.000 | 2/3 | 2.000 | -2.000 | -2.000 |
| fig10_5shot_seed42_44 | ogbn-products | B0 | 72.433 | 87.090 | 91.992 | 49.274 | 3/3 | 29.000 | -9.000 | -10.333 |
| fig10_5shot_seed42_44 | ogbn-products | D1_25__fixed | 72.067 | 84.801 | 90.215 | 51.936 | 3/3 | 47.667 | 5.333 | -0.333 |
| fig10_5shot_seed42_44 | ogbn-products | D1_25__retuned | 70.967 | 84.801 | 90.215 | 51.936 | 3/3 | 47.667 | 5.333 | -0.333 |
| fig10_5shot_seed42_44 | ogbn-products | D2a_1000__fixed | 72.267 | 91.053 | 93.356 | 62.436 | 3/3 | 17.667 | 10.667 | 9.667 |
| fig10_5shot_seed42_44 | ogbn-products | D2a_1000__retuned | 71.900 | 91.053 | 93.356 | 62.436 | 3/3 | 17.667 | 10.667 | 9.667 |
| fig10_5shot_seed42_44 | ogbn-products | D4_confidence__fixed | 70.667 | 91.038 | 92.642 | 83.333 | 1/3 | 2.000 | 1.333 | 1.333 |
| fig10_5shot_seed42_44 | ogbn-products | D4_confidence__retuned | 69.867 | 91.038 | 92.642 | 83.333 | 1/3 | 2.000 | 1.333 | 1.333 |

## 同样本相关比较

比较使用相同有定义配置的相对B0差值，D2的B0已匹配剩余Query。按dataset×seed簇重采样4000次，保持shot依赖；数据集内仅5簇，区间是探索性的。不显著不代表不相关。

| Dataset | Treatment | Joint | Precision comparison | Method | n | Corrective r | Precision r | Difference 95% CI |
|---|---|---|---|---|---:|---:|---:|---|
| ALL | D1_25 | fixed | D_accepted_precision_delta_pp | Pearson | 51 | 0.252 | -0.003 | [-0.240, 0.616] |
| ALL | D1_25 | fixed | D_accepted_precision_delta_pp | Spearman | 51 | 0.210 | 0.028 | [-0.154, 0.514] |
| ALL | D1_25 | fixed | all_accepted_precision_delta_pp | Pearson | 51 | 0.252 | -0.037 | [-0.272, 0.649] |
| ALL | D1_25 | fixed | all_accepted_precision_delta_pp | Spearman | 51 | 0.210 | 0.040 | [-0.185, 0.506] |
| ALL | D1_25 | retuned | D_accepted_precision_delta_pp | Pearson | 51 | -0.194 | 0.029 | [-0.476, -0.026] |
| ALL | D1_25 | retuned | D_accepted_precision_delta_pp | Spearman | 51 | -0.069 | 0.007 | [-0.365, 0.178] |
| ALL | D1_25 | retuned | all_accepted_precision_delta_pp | Pearson | 51 | -0.194 | 0.073 | [-0.510, -0.036] |
| ALL | D1_25 | retuned | all_accepted_precision_delta_pp | Spearman | 51 | -0.069 | 0.070 | [-0.430, 0.157] |
| ALL | D2a_1000 | fixed | D_accepted_precision_delta_pp | Pearson | 41 | 0.167 | -0.101 | [0.027, 0.489] |
| ALL | D2a_1000 | fixed | D_accepted_precision_delta_pp | Spearman | 41 | 0.306 | 0.006 | [0.014, 0.594] |
| ALL | D2a_1000 | fixed | all_accepted_precision_delta_pp | Pearson | 41 | 0.167 | -0.220 | [-0.025, 0.710] |
| ALL | D2a_1000 | fixed | all_accepted_precision_delta_pp | Spearman | 41 | 0.306 | -0.030 | [0.014, 0.682] |
| ALL | D2a_1000 | retuned | D_accepted_precision_delta_pp | Pearson | 41 | -0.157 | -0.224 | [-0.237, 0.343] |
| ALL | D2a_1000 | retuned | D_accepted_precision_delta_pp | Spearman | 41 | -0.127 | -0.239 | [-0.211, 0.375] |
| ALL | D2a_1000 | retuned | all_accepted_precision_delta_pp | Pearson | 41 | -0.157 | -0.390 | [-0.211, 0.527] |
| ALL | D2a_1000 | retuned | all_accepted_precision_delta_pp | Spearman | 41 | -0.127 | -0.313 | [-0.188, 0.500] |
| ALL | D4_confidence | fixed | D_accepted_precision_delta_pp | Pearson | 28 | 0.382 | 0.661 | [-0.538, 0.004] |
| ALL | D4_confidence | fixed | D_accepted_precision_delta_pp | Spearman | 28 | 0.335 | 0.550 | [-0.539, 0.109] |
| ALL | D4_confidence | fixed | all_accepted_precision_delta_pp | Pearson | 28 | 0.382 | 0.387 | [-0.342, 0.283] |
| ALL | D4_confidence | fixed | all_accepted_precision_delta_pp | Spearman | 28 | 0.335 | 0.430 | [-0.459, 0.254] |
| ALL | D4_confidence | retuned | D_accepted_precision_delta_pp | Pearson | 28 | 0.019 | 0.446 | [-0.686, -0.077] |
| ALL | D4_confidence | retuned | D_accepted_precision_delta_pp | Spearman | 28 | 0.154 | 0.434 | [-0.613, 0.041] |
| ALL | D4_confidence | retuned | all_accepted_precision_delta_pp | Pearson | 28 | 0.019 | 0.352 | [-0.627, 0.156] |
| ALL | D4_confidence | retuned | all_accepted_precision_delta_pp | Spearman | 28 | 0.154 | 0.378 | [-0.643, 0.150] |
| cora | D1_25 | fixed | D_accepted_precision_delta_pp | Pearson | 6 | -0.887 | -0.635 | [-1.190, 0.060] |
| cora | D1_25 | fixed | D_accepted_precision_delta_pp | Spearman | 6 | -0.841 | -0.667 | [-1.017, 0.889] |
| cora | D1_25 | fixed | all_accepted_precision_delta_pp | Pearson | 6 | -0.887 | -0.355 | [-1.219, 0.087] |
| cora | D1_25 | fixed | all_accepted_precision_delta_pp | Spearman | 6 | -0.841 | -0.754 | [-0.889, 0.889] |
| cora | D1_25 | retuned | D_accepted_precision_delta_pp | Pearson | 6 | -0.405 | -0.365 | [-0.421, 0.173] |
| cora | D1_25 | retuned | D_accepted_precision_delta_pp | Spearman | 6 | -0.657 | -0.486 | [-1.200, 0.889] |
| cora | D1_25 | retuned | all_accepted_precision_delta_pp | Pearson | 6 | -0.405 | -0.549 | [-0.302, 0.398] |
| cora | D1_25 | retuned | all_accepted_precision_delta_pp | Spearman | 6 | -0.657 | -0.714 | [-0.750, 0.889] |
| cora | D2a_1000 | fixed | D_accepted_precision_delta_pp | Pearson | 5 | -0.641 | -0.502 | [-0.246, 0.000] |
| cora | D2a_1000 | fixed | D_accepted_precision_delta_pp | Spearman | 5 | -0.500 | -0.500 | [0.000, 0.000] |
| cora | D2a_1000 | fixed | all_accepted_precision_delta_pp | Pearson | 5 | -0.641 | -0.200 | [-0.541, 0.000] |
| cora | D2a_1000 | fixed | all_accepted_precision_delta_pp | Spearman | 5 | -0.500 | -0.100 | [-0.842, 0.000] |
| cora | D2a_1000 | retuned | D_accepted_precision_delta_pp | Pearson | 5 | 0.023 | 0.082 | [-0.109, 0.071] |
| cora | D2a_1000 | retuned | D_accepted_precision_delta_pp | Spearman | 5 | -0.100 | -0.100 | [0.000, 0.000] |
| cora | D2a_1000 | retuned | all_accepted_precision_delta_pp | Pearson | 5 | 0.023 | 0.164 | [-0.222, 0.175] |
| cora | D2a_1000 | retuned | all_accepted_precision_delta_pp | Spearman | 5 | -0.100 | 0.400 | [-1.000, 0.000] |
| cora | D4_confidence | fixed | D_accepted_precision_delta_pp | Pearson | 8 | 0.262 | -0.103 | [-1.570, 1.049] |
| cora | D4_confidence | fixed | D_accepted_precision_delta_pp | Spearman | 8 | 0.048 | -0.071 | [-1.636, 0.906] |
| cora | D4_confidence | fixed | all_accepted_precision_delta_pp | Pearson | 8 | 0.262 | -0.341 | [-0.278, 1.572] |
| cora | D4_confidence | fixed | all_accepted_precision_delta_pp | Spearman | 8 | 0.048 | -0.762 | [-0.302, 1.585] |
| cora | D4_confidence | retuned | D_accepted_precision_delta_pp | Pearson | 8 | 0.751 | 0.400 | [-0.331, 0.939] |
| cora | D4_confidence | retuned | D_accepted_precision_delta_pp | Spearman | 8 | 0.571 | 0.381 | [-0.774, 1.120] |
| cora | D4_confidence | retuned | all_accepted_precision_delta_pp | Pearson | 8 | 0.751 | 0.429 | [-0.654, 1.677] |
| cora | D4_confidence | retuned | all_accepted_precision_delta_pp | Spearman | 8 | 0.571 | 0.238 | [-0.755, 1.585] |
| citeseer | D1_25 | fixed | D_accepted_precision_delta_pp | Pearson | 10 | -0.379 | 0.051 | [-1.247, 0.487] |
| citeseer | D1_25 | fixed | D_accepted_precision_delta_pp | Spearman | 10 | -0.554 | -0.096 | [-1.225, 0.217] |
| citeseer | D1_25 | fixed | all_accepted_precision_delta_pp | Pearson | 10 | -0.379 | 0.225 | [-1.303, 0.541] |
| citeseer | D1_25 | fixed | all_accepted_precision_delta_pp | Spearman | 10 | -0.554 | -0.084 | [-1.228, 0.633] |
| citeseer | D1_25 | retuned | D_accepted_precision_delta_pp | Pearson | 10 | -0.083 | 0.183 | [-0.996, 0.238] |
| citeseer | D1_25 | retuned | D_accepted_precision_delta_pp | Spearman | 10 | -0.018 | 0.263 | [-1.141, 0.156] |
| citeseer | D1_25 | retuned | all_accepted_precision_delta_pp | Pearson | 10 | -0.083 | 0.166 | [-0.930, 0.412] |
| citeseer | D1_25 | retuned | all_accepted_precision_delta_pp | Spearman | 10 | -0.018 | 0.330 | [-1.215, 0.546] |
| citeseer | D2a_1000 | fixed | D_accepted_precision_delta_pp | Pearson | 9 | 0.349 | 0.218 | [-0.380, 0.611] |
| citeseer | D2a_1000 | fixed | D_accepted_precision_delta_pp | Spearman | 9 | 0.418 | 0.301 | [-0.189, 0.526] |
| citeseer | D2a_1000 | fixed | all_accepted_precision_delta_pp | Pearson | 9 | 0.349 | 0.236 | [-0.301, 0.687] |
| citeseer | D2a_1000 | fixed | all_accepted_precision_delta_pp | Spearman | 9 | 0.418 | 0.259 | [-0.103, 0.526] |
| citeseer | D2a_1000 | retuned | D_accepted_precision_delta_pp | Pearson | 9 | 0.069 | 0.268 | [-0.829, 0.047] |
| citeseer | D2a_1000 | retuned | D_accepted_precision_delta_pp | Spearman | 9 | 0.050 | 0.202 | [-1.144, 0.312] |
| citeseer | D2a_1000 | retuned | all_accepted_precision_delta_pp | Pearson | 9 | 0.069 | 0.312 | [-0.789, 0.004] |
| citeseer | D2a_1000 | retuned | all_accepted_precision_delta_pp | Spearman | 9 | 0.050 | 0.134 | [-1.144, 0.357] |
| pubmed | D1_25 | fixed | D_accepted_precision_delta_pp | Pearson | 14 | 0.153 | 0.296 | [-1.168, 0.043] |
| pubmed | D1_25 | fixed | D_accepted_precision_delta_pp | Spearman | 14 | 0.172 | 0.529 | [-1.109, 0.440] |
| pubmed | D1_25 | fixed | all_accepted_precision_delta_pp | Pearson | 14 | 0.153 | 0.136 | [-0.715, 1.065] |
| pubmed | D1_25 | fixed | all_accepted_precision_delta_pp | Spearman | 14 | 0.172 | -0.123 | [-0.650, 1.185] |
| pubmed | D1_25 | retuned | D_accepted_precision_delta_pp | Pearson | 14 | -0.179 | -0.102 | [-1.021, 0.344] |
| pubmed | D1_25 | retuned | D_accepted_precision_delta_pp | Spearman | 14 | 0.044 | 0.095 | [-0.953, 0.547] |
| pubmed | D1_25 | retuned | all_accepted_precision_delta_pp | Pearson | 14 | -0.179 | -0.262 | [-0.358, 0.611] |
| pubmed | D1_25 | retuned | all_accepted_precision_delta_pp | Spearman | 14 | 0.044 | -0.416 | [-0.507, 1.195] |
| pubmed | D2a_1000 | fixed | D_accepted_precision_delta_pp | Pearson | 8 | -0.135 | 0.294 | [-0.899, 0.234] |
| pubmed | D2a_1000 | fixed | D_accepted_precision_delta_pp | Spearman | 8 | -0.282 | 0.160 | [-0.806, 0.171] |
| pubmed | D2a_1000 | fixed | all_accepted_precision_delta_pp | Pearson | 8 | -0.135 | -0.001 | [-0.883, 0.235] |
| pubmed | D2a_1000 | fixed | all_accepted_precision_delta_pp | Spearman | 8 | -0.282 | -0.086 | [-0.641, 0.309] |
| pubmed | D2a_1000 | retuned | D_accepted_precision_delta_pp | Pearson | 8 | -0.335 | 0.150 | [-1.123, 0.026] |
| pubmed | D2a_1000 | retuned | D_accepted_precision_delta_pp | Spearman | 8 | -0.446 | 0.145 | [-1.049, 0.033] |
| pubmed | D2a_1000 | retuned | all_accepted_precision_delta_pp | Pearson | 8 | -0.335 | -0.018 | [-1.157, 0.055] |
| pubmed | D2a_1000 | retuned | all_accepted_precision_delta_pp | Spearman | 8 | -0.446 | 0.060 | [-1.273, 0.214] |
| pubmed | D4_confidence | fixed | D_accepted_precision_delta_pp | Pearson | 11 | -0.035 | 0.294 | [-1.213, 0.195] |
| pubmed | D4_confidence | fixed | D_accepted_precision_delta_pp | Spearman | 11 | -0.036 | 0.228 | [-1.080, 0.464] |
| pubmed | D4_confidence | fixed | all_accepted_precision_delta_pp | Pearson | 11 | -0.035 | 0.095 | [-1.020, 0.346] |
| pubmed | D4_confidence | fixed | all_accepted_precision_delta_pp | Spearman | 11 | -0.036 | 0.050 | [-0.619, 0.484] |
| pubmed | D4_confidence | retuned | D_accepted_precision_delta_pp | Pearson | 11 | -0.167 | 0.320 | [-1.245, 0.235] |
| pubmed | D4_confidence | retuned | D_accepted_precision_delta_pp | Spearman | 11 | 0.018 | 0.555 | [-1.185, 0.227] |
| pubmed | D4_confidence | retuned | all_accepted_precision_delta_pp | Pearson | 11 | -0.167 | 0.228 | [-1.241, 0.339] |
| pubmed | D4_confidence | retuned | all_accepted_precision_delta_pp | Spearman | 11 | 0.018 | 0.300 | [-1.111, 0.352] |
| arxiv | D1_25 | fixed | D_accepted_precision_delta_pp | Pearson | 6 | -0.721 | -0.782 | [-0.446, 0.669] |
| arxiv | D1_25 | fixed | D_accepted_precision_delta_pp | Spearman | 6 | -0.754 | -0.899 | [-0.935, 0.750] |
| arxiv | D1_25 | fixed | all_accepted_precision_delta_pp | Pearson | 6 | -0.721 | -0.768 | [-0.427, 1.290] |
| arxiv | D1_25 | fixed | all_accepted_precision_delta_pp | Spearman | 6 | -0.754 | -0.986 | [0.000, 0.866] |
| arxiv | D1_25 | retuned | D_accepted_precision_delta_pp | Pearson | 6 | 0.100 | 0.224 | [-0.221, 0.749] |
| arxiv | D1_25 | retuned | D_accepted_precision_delta_pp | Spearman | 6 | 0.029 | 0.029 | [-1.125, 0.500] |
| arxiv | D1_25 | retuned | all_accepted_precision_delta_pp | Pearson | 6 | 0.100 | 0.125 | [-0.127, 1.190] |
| arxiv | D1_25 | retuned | all_accepted_precision_delta_pp | Spearman | 6 | 0.029 | -0.143 | [0.000, 0.750] |
| arxiv | D2a_1000 | fixed | D_accepted_precision_delta_pp | Pearson | 5 | 0.403 | 0.183 | [-0.000, 0.703] |
| arxiv | D2a_1000 | fixed | D_accepted_precision_delta_pp | Spearman | 5 | 0.400 | 0.200 | [-0.121, 1.111] |
| arxiv | D2a_1000 | fixed | all_accepted_precision_delta_pp | Pearson | 5 | 0.403 | -0.582 | [0.000, 2.000] |
| arxiv | D2a_1000 | fixed | all_accepted_precision_delta_pp | Spearman | 5 | 0.400 | -0.600 | [0.000, 2.000] |
| arxiv | D2a_1000 | retuned | D_accepted_precision_delta_pp | Pearson | 5 | 0.148 | 0.386 | [-0.327, 0.397] |
| arxiv | D2a_1000 | retuned | D_accepted_precision_delta_pp | Spearman | 5 | -0.051 | 0.051 | [-0.364, 0.667] |
| arxiv | D2a_1000 | retuned | all_accepted_precision_delta_pp | Pearson | 5 | 0.148 | -0.661 | [-0.000, 2.000] |
| arxiv | D2a_1000 | retuned | all_accepted_precision_delta_pp | Spearman | 5 | -0.051 | -0.821 | [0.000, 2.000] |
| ogbn-products | D1_25 | fixed | D_accepted_precision_delta_pp | Pearson | 15 | 0.845 | 0.214 | [0.046, 0.830] |
| ogbn-products | D1_25 | fixed | D_accepted_precision_delta_pp | Spearman | 15 | 0.878 | 0.248 | [0.121, 0.908] |
| ogbn-products | D1_25 | fixed | all_accepted_precision_delta_pp | Pearson | 15 | 0.845 | 0.237 | [0.001, 0.860] |
| ogbn-products | D1_25 | fixed | all_accepted_precision_delta_pp | Spearman | 15 | 0.878 | 0.308 | [0.057, 0.950] |
| ogbn-products | D1_25 | retuned | D_accepted_precision_delta_pp | Pearson | 15 | -0.400 | 0.023 | [-0.848, 0.222] |
| ogbn-products | D1_25 | retuned | D_accepted_precision_delta_pp | Spearman | 15 | -0.272 | -0.027 | [-0.742, 0.468] |
| ogbn-products | D1_25 | retuned | all_accepted_precision_delta_pp | Pearson | 15 | -0.400 | 0.236 | [-1.045, 0.061] |
| ogbn-products | D1_25 | retuned | all_accepted_precision_delta_pp | Spearman | 15 | -0.272 | 0.261 | [-0.892, 0.072] |
| ogbn-products | D2a_1000 | fixed | D_accepted_precision_delta_pp | Pearson | 14 | 0.475 | -0.048 | [0.219, 0.970] |
| ogbn-products | D2a_1000 | fixed | D_accepted_precision_delta_pp | Spearman | 14 | 0.695 | 0.177 | [0.194, 1.070] |
| ogbn-products | D2a_1000 | fixed | all_accepted_precision_delta_pp | Pearson | 14 | 0.475 | 0.002 | [0.021, 1.013] |
| ogbn-products | D2a_1000 | fixed | all_accepted_precision_delta_pp | Spearman | 14 | 0.695 | 0.150 | [0.194, 1.203] |
| ogbn-products | D2a_1000 | retuned | D_accepted_precision_delta_pp | Pearson | 14 | -0.174 | -0.467 | [-0.748, 0.694] |
| ogbn-products | D2a_1000 | retuned | D_accepted_precision_delta_pp | Spearman | 14 | -0.066 | -0.270 | [-0.901, 0.709] |
| ogbn-products | D2a_1000 | retuned | all_accepted_precision_delta_pp | Pearson | 14 | -0.174 | -0.505 | [-0.767, 0.710] |
| ogbn-products | D2a_1000 | retuned | all_accepted_precision_delta_pp | Spearman | 14 | -0.066 | -0.283 | [-0.901, 0.884] |
| ogbn-products | D4_confidence | fixed | D_accepted_precision_delta_pp | Pearson | 6 | 0.032 | -0.352 | [-0.234, 2.000] |
| ogbn-products | D4_confidence | fixed | D_accepted_precision_delta_pp | Spearman | 6 | -0.143 | -0.600 | [-0.235, 2.000] |
| ogbn-products | D4_confidence | fixed | all_accepted_precision_delta_pp | Pearson | 6 | 0.032 | -0.400 | [-1.305, 2.000] |
| ogbn-products | D4_confidence | fixed | all_accepted_precision_delta_pp | Spearman | 6 | -0.143 | -0.371 | [-0.902, 2.000] |
| ogbn-products | D4_confidence | retuned | D_accepted_precision_delta_pp | Pearson | 6 | -0.918 | -0.213 | [-2.000, 0.000] |
| ogbn-products | D4_confidence | retuned | D_accepted_precision_delta_pp | Spearman | 6 | -0.714 | 0.029 | [-2.000, 0.000] |
| ogbn-products | D4_confidence | retuned | all_accepted_precision_delta_pp | Pearson | 6 | -0.918 | 0.169 | [-2.000, 0.000] |
| ogbn-products | D4_confidence | retuned | all_accepted_precision_delta_pp | Spearman | 6 | -0.714 | 0.543 | [-2.000, 0.000] |

## 调参效果（同Router，retuned−fixed）

| Dataset | Treatment | n | Accuracy gain pp | Cluster 95% CI pp |
|---|---|---:|---:|---|
| ALL | D1_25 | 75 | 0.216 | [-0.156, 0.644] |
| ALL | D2a_1000 | 75 | 0.267 | [0.021, 0.541] |
| ALL | D4_confidence | 75 | 0.959 | [0.461, 1.491] |
| cora | D1_25 | 15 | 0.593 | [0.233, 1.067] |
| cora | D2a_1000 | 15 | 0.527 | [0.173, 0.793] |
| cora | D4_confidence | 15 | 2.893 | [2.020, 3.567] |
| citeseer | D1_25 | 15 | -0.213 | [-0.633, 0.193] |
| citeseer | D2a_1000 | 15 | 0.260 | [0.053, 0.480] |
| citeseer | D4_confidence | 15 | 0.620 | [0.240, 0.973] |
| pubmed | D1_25 | 15 | -0.033 | [-0.427, 0.360] |
| pubmed | D2a_1000 | 15 | 0.033 | [-0.413, 0.327] |
| pubmed | D4_confidence | 15 | 0.773 | [0.240, 1.420] |
| arxiv | D1_25 | 15 | 0.340 | [-0.507, 1.307] |
| arxiv | D2a_1000 | 15 | 0.173 | [-0.347, 0.740] |
| arxiv | D4_confidence | 15 | 0.600 | [-0.040, 1.160] |
| ogbn-products | D1_25 | 15 | 0.393 | [-1.007, 1.993] |
| ogbn-products | D2a_1000 | 15 | 0.340 | [-0.540, 1.447] |
| ogbn-products | D4_confidence | 15 | -0.093 | [-0.940, 0.927] |

历史45组B0 cross计数：{'0.0': 2, '0.25': 13, '1.0': 30}

重调同时开放多个Joint数值项，因此不能把全部调参收益归因cross。描述性相关也不能证明因果机制。
