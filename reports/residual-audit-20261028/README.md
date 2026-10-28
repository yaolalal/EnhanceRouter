# B0 / FairRouter residual核查与训练曲线

[完整分析报告（图例复核版）](REPORT_REVIEWED.md) · [逐模型消融CSV](per_cell.csv) · [完整JSON](RESULTS.json) · [训练系数原始记录](coefficient_history.csv) · [Arxiv配对差](arxiv_paired.csv)

[训练监督与损失配置补充](training-context/REPORT.md) 单列每组Router父节点数、Joint gold数、cross/aux/keep、实际接受伪标签数量与精度，并核对Arxiv各变体的专家输入哈希一致。

- B0训练曲线：3/5/10-shot × seeds42–44 × 5数据集，共45组，每组同时画alpha和beta。[45组矢量PDF](figures/B0_historical45_alpha_beta.pdf)。
- 分类分解：B0全部75组，以及Arxiv重调参D1的25%与100%各15组，共105份冻结checkpoint。完整pJ、pR单独、仅移除pR及标签改变率均在原固定、原顺序eval1000计算。
- Arxiv五seed曲线：[25% vs100%内部对照](figures-reviewed/Arxiv_D1_25_vs_100.pdf)、[25% vs B0](figures-reviewed/Arxiv_D1_25_vs_B0.pdf)，每图覆盖3种shot × seeds42–46。
- 训练曲线直接来自最终入选超参数候选的原checkpoint history，默认每50steps的真实验证记录；不串接不同候选的搜索过程，没有重训或补造逐step数据。选中的validation checkpoint与停止训练的末步分别标记。
- 真正公式为`pJ=(alpha*pG+beta*pL+pR)/(alpha+beta+1)`。去pR只在推理时改为`(alpha*pG+beta*pL)/(alpha+beta)`，保持模型和已训练alpha/beta不变。pR单独准确率不表示单独重训一个分类器。
- “可争夺区”按用户定义为加入pR后改变argmax标签的eval1000节点比例；同时报告改对、改错及换成另一错误类别。各组cells目录的逐节点CSV可复核。
- D1的25%作用于Router训练bank目标support父节点；Joint residual的gold support仍是原shot的完整数量。逐组同时列出这两个数量，不把它解释为整个系统金标签减少到25%。
- Router伪标签精度复用已封存的完整Query指标，区别于最终分类eval1000。25%对照历史B0与对照重调参100%的差值分别列出，避免混用参照或早期部分完成快照。

代码定位：[模型与融合](../../src/eargtc/joint_explore/joint_purf_v10_v1/model.py#L8-L37)、[训练及系数history](../../src/eargtc/joint_explore/joint_purf_v10_v1/training.py#L34-L112)、[B0复现入口](../../src/enhancerouter/cli.py#L84-L142)、[25%父节点抽样](../../experiments/stage0_20261027/d1_numeric.py#L182-L192)、[本次分析与作图](../../experiments/stage0_20261027/analyze_residual.py)。

本次只读取冻结模型、输入和历史训练记录，不修改原实现或任何旧实验产物。完整预测逐节点核对原模型输出，分类准确率必须与此前发布值一致；组件分解公式及“净收益＝改对−改错”也逐组检查。原模型/数据文件的源路径与哈希保存在JSON的source字段。

本次“未用Query真值作选择”仅描述此次冻结模型拆分分析；历史B0的既有配方来源限制继续保留，尤其5-shot的历史选择登记，见[基线说明](../../docs/FAIRROUTER_BASELINE.md#原始历史选择的限制)。

视觉复核仅调整Arxiv合集共享图例的位置，避免遮挡子图标题；数值序列未变。[原始封存报告](REPORT.md)与原图保留，复核登记和逐条作图序列见[figures-reviewed](figures-reviewed/)。[完整结果压缩包](https://github.com/yaolalal/EnhanceRouter/releases/tag/residual-audit-20261028)包含原始数据、CSV、原图及复核图。
