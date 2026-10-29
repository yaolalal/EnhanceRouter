# 3×2完成结果与解读

全部525份分类结果完成，300份新训练及225份复用模型通过输入/超参/伪标签池核验；独立从checkpoint重新推理eval1000，预测标签逐节点完全一致，概率容差2e-6。
主要结论：Joint调参混杂确实存在；D4重调恢复了部分分类差距。纠正性指标目前没有稳定优于整体伪标签精度，不能据此宣称已普遍解释Fig10的PubMed/Products现象。

## 设置及定义

D1使用已完成numeric25% Router；D2使用用户确认的随机a1000；D4是confidence_router。每个处理的两组Joint使用完全相同Router/接受名单/伪标签/专家/划分/完整gold support，只改变是否重新搜索Joint数值超参。固定超参仍重训权重。
主参照pGL=(alpha_B0*pG+beta_B0*pL)/(alpha_B0+beta_B0)，采用B0最终checkpoint固定系数，避免重调同时移动参照。每个模型自身alpha/beta的结果另列。
纠正集合C={已接受且pseudo != argmax(pGL)}；纠正性精度=其中伪标签正确数/|C|。空集合为NA。净准确率纠正=原错变对−原对变错；标签正误差=正确纠正标签−错误纠正标签（含错转错）。原始计数和归一化值都保留，不能混称。
Router/伪标签在完整Query；D2剔除新增训练父节点，B0同名单配对。分类仍是原顺序eval1000。D2新增gold仅进入Router，Joint继续用原support。

## 七设置：分类与纠正性指标

每行75组配置均值；纠正性精度只平均有定义配置，n明确列出。因此跨方法原始均值不一定基于相同有效配置，共同有效配置对照见paired-reference。

| Setting | Final accuracy% | Corrective precision% | Defined n/75 | Own-pGL corrective precision% | Own-pGL defined n/75 | Corrective nodes | Net accuracy corrections | Correct-labels minus wrong-labels | Net/1000 Query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 75.353 | 57.425 | 54/75 | 57.425 | 54/75 | 69.640 | 8.360 | -0.813 | 1.340 |
| D1_25__fixed | 75.037 | 51.820 | 57/75 | 51.539 | 59/75 | 97.133 | 11.067 | -4.280 | 0.789 |
| D1_25__retuned | 75.253 | 51.820 | 57/75 | 49.201 | 56/75 | 97.133 | 11.067 | -4.280 | 0.789 |
| D2a_1000__fixed | 74.999 | 59.210 | 46/75 | 62.068 | 48/75 | 78.000 | 21.267 | 11.600 | 1.624 |
| D2a_1000__retuned | 75.265 | 59.210 | 46/75 | 58.150 | 49/75 | 78.000 | 21.267 | 11.600 | 1.624 |
| D4_confidence__fixed | 73.435 | 47.789 | 40/75 | 48.767 | 35/75 | 35.107 | 4.787 | 2.867 | -0.598 |
| D4_confidence__retuned | 74.393 | 47.789 | 40/75 | 59.757 | 38/75 | 35.107 | 4.787 | 2.867 | -0.598 |

## Router与伪标签

同一处理两种Joint设置的这些指标完全相同，下表只列一次。D2配对B0单列，不能用原Query的B0指标代替。精度单位%。

| Router | Trust AUROC | Pref AUROC | D P@1% | D P@5% | D P@20% | D accepted precision% | A+D accepted precision% |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.636 | 0.778 | 86.547 | 85.726 | 79.776 | 83.495 | 90.980 |
| D1_25__fixed | 0.632 | 0.754 | 85.445 | 84.204 | 78.416 | 80.775 | 89.943 |
| D2a_1000__fixed | 0.748 | 0.810 | 89.698 | 86.679 | 80.093 | 83.115 | 90.889 |
| D4_confidence__fixed | 0.677 | 0.767 | 81.372 | 79.084 | 72.135 | 75.945 | 88.288 |
| B0 matched D2 Query | 0.635 | 0.780 | 87.429 | 86.713 | 80.439 | 84.158 | 91.073 |

## Joint调参的影响

| Treatment | Retuned−fixed accuracy pp | Cluster 95% CI pp | B0 cross 0/.25/1 | Retuned cross 0/.25/1 |
|---|---:|---|---|---|
| D1_25 | 0.216 | [-0.156, 0.644] | 4/20/51 | 5/25/45 |
| D2a_1000 | 0.267 | [0.021, 0.541] | 4/20/51 | 5/20/50 |
| D4_confidence | 0.959 | [0.461, 1.491] | 4/20/51 | 27/20/28 |

历史45格cross=0有2格、cross=.25有13格的说法已核实。扩到75格B0为4/20/51，D4重调后为27/20/28；其中Cora15格有13格选cross=0。验证集确实会选择关小或关闭分歧伪标签。重调还允许dim、学习率、aux、keep等改变，不能把全部收益单独归因cross。

## 纠正性指标是否解释最终分类

使用相对B0的配对差值；下面两种相关在相同有定义配置上计算。整体精度取Fig10对应的实际接受分歧伪标签精度。按dataset×seed簇重采样，保持shot相依；原报告也保留A+D整体精度、Spearman、净纠正及分数据集结果。

| Treatment | Joint | Common n | Corrective-precision Pearson r | D-precision Pearson r | Difference 95% CI |
|---|---|---:|---:|---:|---|
| D1_25 | fixed | 51 | 0.252 | -0.003 | [-0.240, 0.616] |
| D1_25 | retuned | 51 | -0.194 | 0.029 | [-0.476, -0.026] |
| D2a_1000 | fixed | 41 | 0.167 | -0.101 | [0.027, 0.489] |
| D2a_1000 | retuned | 41 | -0.157 | -0.224 | [-0.237, 0.343] |
| D4_confidence | fixed | 28 | 0.382 | 0.661 | [-0.538, 0.004] |
| D4_confidence | retuned | 28 | 0.019 | 0.446 | [-0.686, -0.077] |

D4重调后，纠正性精度与分类变化的相关接近0，而分歧伪标签整体精度仍呈正相关；D1/D2的方向也不稳定。因此尚不支持“分类跟纠正性精度相关、跟整体精度不相关”的普遍判断。不显著不能证明不相关；这是探索性离线分析。

## Fig10范围及反例

5-shot、seeds42–44的PubMed/Products全部七设置均已单列在scope-supplement。PubMed的confidence纠正集合很小，并有空集合，不能忽略分母。
Products/5-shot/seed44是同一配置内的反例：B0纠正性精度44.44%（9节点）、净纠正0、最终准确率71.90%；D4纠正性精度83.33%（6节点）、净纠正+4，但最终准确率固定Joint69.30%、重调Joint69.10%。两种纠正指标提高仍未保证分类提高，不能把所提指标写成已成立的统一机制。

## 定位

- [完整主报告](report/REPORT.md)：五数据集、三shot、五seed全部指标。
- [分歧区/有效分母/同样本相关区间](scope-supplement/REPORT.md)。
- [D2匹配B0与共同有效配置对照](paired-reference/REPORT.md)。
- [逐模型CSV](report/per_cell.csv)、[逐节点证据](cells/)、[525份配置及训练history](training-records/)、[新训练全部候选选择](training-selection/)。
- [独立复核通过证据](proofs/INDEPENDENT_AUDIT_PASS.json)。历史5-shot配方来源的已知限制继续见[基线说明](../../docs/FAIRROUTER_BASELINE.md#原始历史选择的限制)。

所有核验与补充数值计算都通过Slurm；旧模型/报告保持封存，未推进阶段1或选择旧L1争议口径。
