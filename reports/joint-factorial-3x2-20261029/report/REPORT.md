# 3×2：Router处理 × Joint数值超参选择

D1使用numeric25% Router，D2使用随机a1000，D4使用confidence_router；每格5datasets×3shots×5seeds=75。B0另作参照。
固定指固定B0超参数并重新训练权重；重调沿用原生四阶段验证集数值搜索，非搜索设置固定B0。每行两设置的Router输出、已接受伪标签池逐项相同。
纠正性精度：实际已接受伪标签与B0固定pGL预测不同的节点中，伪标签正确比例。净纠正=原错变对−原对变错；另报纠正标签正误差=正确伪标签数−错误伪标签数（该错误数包含错转错）。两口径均保留，避免把错转错算作准确率损失。空集合为null；各模型自身pGL另存。
Router/伪标签主指标用完整Query；D2排除新增训练节点，B0同名单配对。最终分类只用原顺序eval1000。D2额外真标签只进入原Router训练，Joint gold仍为原support。
完整Query节点量不同，净纠正原始数与每1000 Query归一化值同时给出。数值均为配置均值，非节点池化。

| Dataset | Setting | n | Accuracy% | Trust AUROC | Pref AUROC | D P@1% | Accepted prec% | Corrective prec% | Corrective n | Net | Net/1000Q | Corrective right-minus-wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | B0 | 75 | 75.353 | 0.636 | 0.778 | 86.547 | 90.980 | 57.425 | 69.640 | 8.360 | 1.340 | -0.813 |
| ALL | D1_25__fixed | 75 | 75.037 | 0.632 | 0.754 | 85.445 | 89.943 | 51.820 | 97.133 | 11.067 | 0.789 | -4.280 |
| ALL | D1_25__retuned | 75 | 75.253 | 0.632 | 0.754 | 85.445 | 89.943 | 51.820 | 97.133 | 11.067 | 0.789 | -4.280 |
| ALL | D2a_1000__fixed | 75 | 74.999 | 0.748 | 0.810 | 89.698 | 90.889 | 59.210 | 78.000 | 21.267 | 1.624 | 11.600 |
| ALL | D2a_1000__retuned | 75 | 75.265 | 0.748 | 0.810 | 89.698 | 90.889 | 59.210 | 78.000 | 21.267 | 1.624 | 11.600 |
| ALL | D4_confidence__fixed | 75 | 73.435 | 0.677 | 0.767 | 81.372 | 88.288 | 47.789 | 35.107 | 4.787 | -0.598 | 2.867 |
| ALL | D4_confidence__retuned | 75 | 74.393 | 0.677 | 0.767 | 81.372 | 88.288 | 47.789 | 35.107 | 4.787 | -0.598 | 2.867 |
| cora | B0 | 15 | 80.333 | 0.636 | 0.793 | 96.759 | 96.040 | 82.728 | 6.600 | 4.933 | 2.276 | 4.467 |
| cora | D1_25__fixed | 15 | 79.940 | 0.622 | 0.770 | 95.926 | 95.760 | 72.894 | 8.400 | 5.333 | 2.469 | 4.267 |
| cora | D1_25__retuned | 15 | 80.533 | 0.622 | 0.770 | 95.926 | 95.760 | 72.894 | 8.400 | 5.333 | 2.469 | 4.267 |
| cora | D2a_1000__fixed | 15 | 79.807 | 0.786 | 0.804 | 95.667 | 95.653 | 66.384 | 3.467 | 1.733 | 1.494 | 1.467 |
| cora | D2a_1000__retuned | 15 | 80.333 | 0.786 | 0.804 | 95.667 | 95.653 | 66.384 | 3.467 | 1.733 | 1.494 | 1.467 |
| cora | D4_confidence__fixed | 15 | 75.387 | 0.571 | 0.775 | 49.074 | 84.638 | 22.304 | 36.933 | -12.067 | -5.594 | -16.800 |
| cora | D4_confidence__retuned | 15 | 78.280 | 0.571 | 0.775 | 49.074 | 84.638 | 22.304 | 36.933 | -12.067 | -5.594 | -16.800 |
| citeseer | B0 | 15 | 73.467 | 0.577 | 0.691 | 77.685 | 87.963 | 57.736 | 19.067 | 5.733 | 2.174 | 2.933 |
| citeseer | D1_25__fixed | 15 | 73.787 | 0.580 | 0.692 | 79.352 | 87.499 | 54.462 | 10.000 | 3.333 | 1.262 | 1.733 |
| citeseer | D1_25__retuned | 15 | 73.573 | 0.580 | 0.692 | 79.352 | 87.499 | 54.462 | 10.000 | 3.333 | 1.262 | 1.733 |
| citeseer | D2a_1000__fixed | 15 | 73.520 | 0.686 | 0.703 | 81.333 | 86.227 | 56.201 | 11.133 | 5.067 | 3.088 | 4.067 |
| citeseer | D2a_1000__retuned | 15 | 73.780 | 0.686 | 0.703 | 81.333 | 86.227 | 56.201 | 11.133 | 5.067 | 3.088 | 4.067 |
| citeseer | D4_confidence__fixed | 15 | 72.467 | 0.629 | 0.656 | 91.521 | 85.015 | 91.026 | 5.200 | 4.467 | 1.674 | 4.267 |
| citeseer | D4_confidence__retuned | 15 | 73.087 | 0.629 | 0.656 | 91.521 | 85.015 | 91.026 | 5.200 | 4.467 | 1.674 | 4.267 |
| pubmed | B0 | 15 | 90.693 | 0.632 | 0.846 | 91.818 | 95.631 | 64.169 | 84.800 | 50.067 | 2.607 | 47.467 |
| pubmed | D1_25__fixed | 15 | 90.393 | 0.652 | 0.810 | 92.364 | 94.828 | 63.641 | 210.533 | -1.400 | -0.076 | -5.200 |
| pubmed | D1_25__retuned | 15 | 90.360 | 0.652 | 0.810 | 92.364 | 94.828 | 63.641 | 210.533 | -1.400 | -0.076 | -5.200 |
| pubmed | D2a_1000__fixed | 15 | 90.433 | 0.847 | 0.892 | 97.397 | 96.279 | 73.368 | 90.733 | 43.467 | 2.388 | 41.933 |
| pubmed | D2a_1000__retuned | 15 | 90.467 | 0.847 | 0.892 | 97.397 | 96.279 | 73.368 | 90.733 | 43.467 | 2.388 | 41.933 |
| pubmed | D4_confidence__fixed | 15 | 89.780 | 0.796 | 0.829 | 85.988 | 96.154 | 43.979 | 77.400 | 9.733 | 0.507 | 9.267 |
| pubmed | D4_confidence__retuned | 15 | 90.553 | 0.796 | 0.829 | 85.988 | 96.154 | 43.979 | 77.400 | 9.733 | 0.507 | 9.267 |
| arxiv | B0 | 15 | 59.613 | 0.674 | 0.788 | 80.710 | 86.243 | 50.868 | 164.000 | 0.200 | 0.001 | -28.667 |
| arxiv | D1_25__fixed | 15 | 59.273 | 0.642 | 0.757 | 73.255 | 83.093 | 39.153 | 195.867 | 52.733 | 0.379 | -7.733 |
| arxiv | D1_25__retuned | 15 | 59.613 | 0.642 | 0.757 | 73.255 | 83.093 | 39.153 | 195.867 | 52.733 | 0.379 | -7.733 |
| arxiv | D2a_1000__fixed | 15 | 59.153 | 0.733 | 0.817 | 84.439 | 87.494 | 65.511 | 30.200 | -6.467 | -0.047 | -10.200 |
| arxiv | D2a_1000__retuned | 15 | 59.327 | 0.733 | 0.817 | 84.439 | 87.494 | 65.511 | 30.200 | -6.467 | -0.047 | -10.200 |
| arxiv | D4_confidence__fixed | 15 | 58.367 | 0.678 | 0.762 | 84.741 | 85.307 | 75.834 | 14.333 | -1.000 | -0.007 | -1.933 |
| arxiv | D4_confidence__retuned | 15 | 58.967 | 0.678 | 0.762 | 84.741 | 85.307 | 75.834 | 14.333 | -1.000 | -0.007 | -1.933 |
| ogbn-products | B0 | 15 | 72.660 | 0.661 | 0.770 | 85.765 | 89.023 | 40.029 | 73.733 | -19.133 | -0.358 | -30.267 |
| ogbn-products | D1_25__fixed | 15 | 71.793 | 0.665 | 0.741 | 86.326 | 88.535 | 37.461 | 60.867 | -4.667 | -0.087 | -14.467 |
| ogbn-products | D1_25__retuned | 15 | 72.187 | 0.665 | 0.741 | 86.326 | 88.535 | 37.461 | 60.867 | -4.667 | -0.087 | -14.467 |
| ogbn-products | D2a_1000__fixed | 15 | 72.080 | 0.688 | 0.834 | 89.656 | 88.792 | 46.761 | 254.467 | 62.533 | 1.197 | 20.733 |
| ogbn-products | D2a_1000__retuned | 15 | 72.420 | 0.688 | 0.834 | 89.656 | 88.792 | 46.761 | 254.467 | 62.533 | 1.197 | 20.733 |
| ogbn-products | D4_confidence__fixed | 15 | 71.173 | 0.712 | 0.813 | 95.535 | 90.323 | 83.869 | 41.667 | 22.800 | 0.428 | 19.533 |
| ogbn-products | D4_confidence__retuned | 15 | 71.080 | 0.712 | 0.813 | 95.535 | 90.323 | 83.869 | 41.667 | 22.800 | 0.428 | 19.533 |

## Fig.10相关范围：5-shot、seeds42–44、PubMed/Products

| Dataset | Setting | n | Accuracy% | Accepted precision% | Corrective precision% | Net | Corrective right-minus-wrong |
|---|---|---:|---:|---:|---:|---:|---:|
| pubmed | B0 | 3 | 91.467 | 96.674 | 65.142 | 4.000 | 3.667 |
| pubmed | D1_25__fixed | 3 | 91.733 | 96.247 | 74.036 | 31.333 | 31.000 |
| pubmed | D1_25__retuned | 3 | 91.400 | 96.247 | 74.036 | 31.333 | 31.000 |
| pubmed | D2a_1000__fixed | 3 | 91.433 | 97.919 | 75.000 | 0.667 | 0.667 |
| pubmed | D2a_1000__retuned | 3 | 91.100 | 97.919 | 75.000 | 0.667 | 0.667 |
| pubmed | D4_confidence__fixed | 3 | 90.633 | 96.878 | 0.000 | -2.000 | -2.000 |
| pubmed | D4_confidence__retuned | 3 | 91.100 | 96.878 | 0.000 | -2.000 | -2.000 |
| ogbn-products | B0 | 3 | 72.433 | 91.992 | 49.274 | -9.000 | -10.333 |
| ogbn-products | D1_25__fixed | 3 | 72.067 | 90.215 | 51.936 | 5.333 | -0.333 |
| ogbn-products | D1_25__retuned | 3 | 70.967 | 90.215 | 51.936 | 5.333 | -0.333 |
| ogbn-products | D2a_1000__fixed | 3 | 72.267 | 93.356 | 62.436 | 10.667 | 9.667 |
| ogbn-products | D2a_1000__retuned | 3 | 71.900 | 93.356 | 62.436 | 10.667 | 9.667 |
| ogbn-products | D4_confidence__fixed | 3 | 70.667 | 92.642 | 83.333 | 1.333 | 1.333 |
| ogbn-products | D4_confidence__retuned | 3 | 69.867 | 92.642 | 83.333 | 1.333 | 1.333 |

## 重调带来的分类变化与cross选择

| Dataset | Treatment | Tuning gain pp | Fixed cross 0/.25/1 counts | Retuned cross 0/.25/1 counts |
|---|---|---:|---|---|
| ALL | D1_25 | +0.216 | 4/20/51 | 5/25/45 |
| ALL | D2a_1000 | +0.267 | 4/20/51 | 5/20/50 |
| ALL | D4_confidence | +0.959 | 4/20/51 | 27/20/28 |
| cora | D1_25 | +0.593 | 0/6/9 | 0/4/11 |
| cora | D2a_1000 | +0.527 | 0/6/9 | 1/7/7 |
| cora | D4_confidence | +2.893 | 0/6/9 | 13/2/0 |
| citeseer | D1_25 | -0.213 | 1/6/8 | 0/5/10 |
| citeseer | D2a_1000 | +0.260 | 1/6/8 | 2/6/7 |
| citeseer | D4_confidence | +0.620 | 1/6/8 | 6/6/3 |
| pubmed | D1_25 | -0.033 | 2/4/9 | 3/6/6 |
| pubmed | D2a_1000 | +0.033 | 2/4/9 | 2/3/10 |
| pubmed | D4_confidence | +0.773 | 2/4/9 | 0/4/11 |
| arxiv | D1_25 | +0.340 | 1/3/11 | 0/5/10 |
| arxiv | D2a_1000 | +0.173 | 1/3/11 | 0/2/13 |
| arxiv | D4_confidence | +0.600 | 1/3/11 | 3/5/7 |
| ogbn-products | D1_25 | +0.393 | 0/1/14 | 2/5/8 |
| ogbn-products | D2a_1000 | +0.340 | 0/1/14 | 0/2/13 |
| ogbn-products | D4_confidence | -0.093 | 0/1/14 | 5/3/7 |

## 离线相关性

下表用相对B0的配对差值；D2的伪标签参照已匹配剩余Query。完整数据集/shot/seed明细在CSV/JSON。相关性不能替代干预证据；不以测试结果选择方法。

| Dataset | Treatment | Joint | Predictor delta | n | Pearson | Spearman |
|---|---|---|---|---:|---:|---:|
| ALL | D1_25 | fixed | corrective_precision_delta_pp | 51 | 0.252 | 0.210 |
| ALL | D1_25 | fixed | pseudo_precision_delta_pp | 75 | -0.016 | 0.018 |
| ALL | D1_25 | fixed | net_per_1000_query_delta | 75 | 0.051 | 0.016 |
| ALL | D1_25 | retuned | corrective_precision_delta_pp | 51 | -0.194 | -0.069 |
| ALL | D1_25 | retuned | pseudo_precision_delta_pp | 75 | 0.043 | -0.007 |
| ALL | D1_25 | retuned | net_per_1000_query_delta | 75 | 0.047 | 0.124 |
| ALL | D2a_1000 | fixed | corrective_precision_delta_pp | 41 | 0.167 | 0.306 |
| ALL | D2a_1000 | fixed | pseudo_precision_delta_pp | 75 | 0.019 | 0.032 |
| ALL | D2a_1000 | fixed | net_per_1000_query_delta | 75 | 0.070 | -0.002 |
| ALL | D2a_1000 | retuned | corrective_precision_delta_pp | 41 | -0.157 | -0.127 |
| ALL | D2a_1000 | retuned | pseudo_precision_delta_pp | 75 | -0.065 | -0.127 |
| ALL | D2a_1000 | retuned | net_per_1000_query_delta | 75 | 0.201 | -0.005 |
| ALL | D4_confidence | fixed | corrective_precision_delta_pp | 28 | 0.382 | 0.335 |
| ALL | D4_confidence | fixed | pseudo_precision_delta_pp | 75 | 0.440 | 0.331 |
| ALL | D4_confidence | fixed | net_per_1000_query_delta | 75 | 0.401 | 0.342 |
| ALL | D4_confidence | retuned | corrective_precision_delta_pp | 28 | 0.019 | 0.154 |
| ALL | D4_confidence | retuned | pseudo_precision_delta_pp | 75 | 0.251 | 0.219 |
| ALL | D4_confidence | retuned | net_per_1000_query_delta | 75 | 0.212 | 0.086 |
| cora | D1_25 | fixed | corrective_precision_delta_pp | 6 | -0.887 | -0.841 |
| cora | D1_25 | fixed | pseudo_precision_delta_pp | 15 | 0.095 | -0.382 |
| cora | D1_25 | fixed | net_per_1000_query_delta | 15 | 0.065 | 0.177 |
| cora | D1_25 | retuned | corrective_precision_delta_pp | 6 | -0.405 | -0.657 |
| cora | D1_25 | retuned | pseudo_precision_delta_pp | 15 | -0.467 | -0.456 |
| cora | D1_25 | retuned | net_per_1000_query_delta | 15 | 0.461 | 0.567 |
| cora | D2a_1000 | fixed | corrective_precision_delta_pp | 5 | -0.641 | -0.500 |
| cora | D2a_1000 | fixed | pseudo_precision_delta_pp | 15 | 0.239 | 0.050 |
| cora | D2a_1000 | fixed | net_per_1000_query_delta | 15 | -0.389 | -0.404 |
| cora | D2a_1000 | retuned | corrective_precision_delta_pp | 5 | 0.023 | -0.100 |
| cora | D2a_1000 | retuned | pseudo_precision_delta_pp | 15 | 0.021 | 0.089 |
| cora | D2a_1000 | retuned | net_per_1000_query_delta | 15 | -0.217 | -0.089 |
| cora | D4_confidence | fixed | corrective_precision_delta_pp | 8 | 0.262 | 0.048 |
| cora | D4_confidence | fixed | pseudo_precision_delta_pp | 15 | -0.281 | -0.357 |
| cora | D4_confidence | fixed | net_per_1000_query_delta | 15 | -0.340 | -0.493 |
| cora | D4_confidence | retuned | corrective_precision_delta_pp | 8 | 0.751 | 0.571 |
| cora | D4_confidence | retuned | pseudo_precision_delta_pp | 15 | 0.027 | 0.002 |
| cora | D4_confidence | retuned | net_per_1000_query_delta | 15 | 0.070 | -0.265 |
| citeseer | D1_25 | fixed | corrective_precision_delta_pp | 10 | -0.379 | -0.554 |
| citeseer | D1_25 | fixed | pseudo_precision_delta_pp | 15 | -0.139 | -0.204 |
| citeseer | D1_25 | fixed | net_per_1000_query_delta | 15 | 0.481 | 0.099 |
| citeseer | D1_25 | retuned | corrective_precision_delta_pp | 10 | -0.083 | -0.018 |
| citeseer | D1_25 | retuned | pseudo_precision_delta_pp | 15 | -0.023 | 0.054 |
| citeseer | D1_25 | retuned | net_per_1000_query_delta | 15 | 0.741 | 0.500 |
| citeseer | D2a_1000 | fixed | corrective_precision_delta_pp | 9 | 0.349 | 0.418 |
| citeseer | D2a_1000 | fixed | pseudo_precision_delta_pp | 15 | 0.396 | 0.106 |
| citeseer | D2a_1000 | fixed | net_per_1000_query_delta | 15 | 0.198 | 0.106 |
| citeseer | D2a_1000 | retuned | corrective_precision_delta_pp | 9 | 0.069 | 0.050 |
| citeseer | D2a_1000 | retuned | pseudo_precision_delta_pp | 15 | 0.544 | 0.272 |
| citeseer | D2a_1000 | retuned | net_per_1000_query_delta | 15 | 0.142 | -0.154 |
| citeseer | D4_confidence | fixed | corrective_precision_delta_pp | 1 | NA | NA |
| citeseer | D4_confidence | fixed | pseudo_precision_delta_pp | 15 | -0.003 | -0.107 |
| citeseer | D4_confidence | fixed | net_per_1000_query_delta | 15 | 0.322 | 0.510 |
| citeseer | D4_confidence | retuned | corrective_precision_delta_pp | 1 | NA | NA |
| citeseer | D4_confidence | retuned | pseudo_precision_delta_pp | 15 | 0.257 | 0.273 |
| citeseer | D4_confidence | retuned | net_per_1000_query_delta | 15 | 0.647 | 0.390 |
| pubmed | D1_25 | fixed | corrective_precision_delta_pp | 14 | 0.153 | 0.172 |
| pubmed | D1_25 | fixed | pseudo_precision_delta_pp | 15 | 0.143 | -0.140 |
| pubmed | D1_25 | fixed | net_per_1000_query_delta | 15 | 0.180 | 0.068 |
| pubmed | D1_25 | retuned | corrective_precision_delta_pp | 14 | -0.179 | 0.044 |
| pubmed | D1_25 | retuned | pseudo_precision_delta_pp | 15 | -0.253 | -0.407 |
| pubmed | D1_25 | retuned | net_per_1000_query_delta | 15 | -0.148 | -0.014 |
| pubmed | D2a_1000 | fixed | corrective_precision_delta_pp | 8 | -0.135 | -0.282 |
| pubmed | D2a_1000 | fixed | pseudo_precision_delta_pp | 15 | -0.017 | -0.108 |
| pubmed | D2a_1000 | fixed | net_per_1000_query_delta | 15 | -0.223 | -0.200 |
| pubmed | D2a_1000 | retuned | corrective_precision_delta_pp | 8 | -0.335 | -0.446 |
| pubmed | D2a_1000 | retuned | pseudo_precision_delta_pp | 15 | -0.121 | -0.127 |
| pubmed | D2a_1000 | retuned | net_per_1000_query_delta | 15 | -0.316 | -0.181 |
| pubmed | D4_confidence | fixed | corrective_precision_delta_pp | 11 | -0.035 | -0.036 |
| pubmed | D4_confidence | fixed | pseudo_precision_delta_pp | 15 | 0.124 | 0.102 |
| pubmed | D4_confidence | fixed | net_per_1000_query_delta | 15 | 0.276 | -0.070 |
| pubmed | D4_confidence | retuned | corrective_precision_delta_pp | 11 | -0.167 | 0.018 |
| pubmed | D4_confidence | retuned | pseudo_precision_delta_pp | 15 | 0.442 | 0.463 |
| pubmed | D4_confidence | retuned | net_per_1000_query_delta | 15 | -0.370 | -0.229 |
| arxiv | D1_25 | fixed | corrective_precision_delta_pp | 6 | -0.721 | -0.754 |
| arxiv | D1_25 | fixed | pseudo_precision_delta_pp | 15 | -0.168 | 0.029 |
| arxiv | D1_25 | fixed | net_per_1000_query_delta | 15 | -0.291 | -0.480 |
| arxiv | D1_25 | retuned | corrective_precision_delta_pp | 6 | 0.100 | 0.029 |
| arxiv | D1_25 | retuned | pseudo_precision_delta_pp | 15 | 0.129 | 0.041 |
| arxiv | D1_25 | retuned | net_per_1000_query_delta | 15 | 0.045 | -0.214 |
| arxiv | D2a_1000 | fixed | corrective_precision_delta_pp | 5 | 0.403 | 0.400 |
| arxiv | D2a_1000 | fixed | pseudo_precision_delta_pp | 15 | -0.139 | -0.050 |
| arxiv | D2a_1000 | fixed | net_per_1000_query_delta | 15 | -0.193 | -0.360 |
| arxiv | D2a_1000 | retuned | corrective_precision_delta_pp | 5 | 0.148 | -0.051 |
| arxiv | D2a_1000 | retuned | pseudo_precision_delta_pp | 15 | -0.038 | -0.078 |
| arxiv | D2a_1000 | retuned | net_per_1000_query_delta | 15 | -0.205 | -0.456 |
| arxiv | D4_confidence | fixed | corrective_precision_delta_pp | 2 | NA | NA |
| arxiv | D4_confidence | fixed | pseudo_precision_delta_pp | 15 | -0.445 | -0.486 |
| arxiv | D4_confidence | fixed | net_per_1000_query_delta | 15 | -0.155 | -0.127 |
| arxiv | D4_confidence | retuned | corrective_precision_delta_pp | 2 | NA | NA |
| arxiv | D4_confidence | retuned | pseudo_precision_delta_pp | 15 | -0.160 | 0.032 |
| arxiv | D4_confidence | retuned | net_per_1000_query_delta | 15 | 0.200 | 0.213 |
| ogbn-products | D1_25 | fixed | corrective_precision_delta_pp | 15 | 0.845 | 0.878 |
| ogbn-products | D1_25 | fixed | pseudo_precision_delta_pp | 15 | 0.237 | 0.308 |
| ogbn-products | D1_25 | fixed | net_per_1000_query_delta | 15 | 0.247 | 0.382 |
| ogbn-products | D1_25 | retuned | corrective_precision_delta_pp | 15 | -0.400 | -0.272 |
| ogbn-products | D1_25 | retuned | pseudo_precision_delta_pp | 15 | 0.236 | 0.261 |
| ogbn-products | D1_25 | retuned | net_per_1000_query_delta | 15 | 0.141 | -0.161 |
| ogbn-products | D2a_1000 | fixed | corrective_precision_delta_pp | 14 | 0.475 | 0.695 |
| ogbn-products | D2a_1000 | fixed | pseudo_precision_delta_pp | 15 | -0.003 | 0.118 |
| ogbn-products | D2a_1000 | fixed | net_per_1000_query_delta | 15 | 0.322 | 0.647 |
| ogbn-products | D2a_1000 | retuned | corrective_precision_delta_pp | 14 | -0.174 | -0.066 |
| ogbn-products | D2a_1000 | retuned | pseudo_precision_delta_pp | 15 | -0.519 | -0.305 |
| ogbn-products | D2a_1000 | retuned | net_per_1000_query_delta | 15 | 0.535 | 0.570 |
| ogbn-products | D4_confidence | fixed | corrective_precision_delta_pp | 6 | 0.032 | -0.143 |
| ogbn-products | D4_confidence | fixed | pseudo_precision_delta_pp | 15 | 0.030 | 0.034 |
| ogbn-products | D4_confidence | fixed | net_per_1000_query_delta | 15 | 0.280 | 0.104 |
| ogbn-products | D4_confidence | retuned | corrective_precision_delta_pp | 6 | -0.918 | -0.714 |
| ogbn-products | D4_confidence | retuned | pseudo_precision_delta_pp | 15 | 0.087 | 0.025 |
| ogbn-products | D4_confidence | retuned | net_per_1000_query_delta | 15 | 0.248 | 0.089 |

这次重调同时开放原Joint数值搜索项，包括cross/aux/keep/学习率等；调参收益不能单独归因cross。原L1预算争议和阶段1未推进。

文件：per_cell.csv逐模型；paired_deltas.csv相对B0；tuning_effects.csv同Router冻结/重调差；correlations.csv离线相关；RESULTS.json包含完整配置/来源/分母。
