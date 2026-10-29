# 3×2 Joint调参混杂核查与纠正性指标

[完整七设置主报告](report/REPORT.md) · [Fig10分歧区及有效分母补充](scope-supplement/REPORT.md) · [原范围相关及置信区间](associations/REPORT.md)

全部525份分类结果完成：B0及{D1 numeric25%, D2随机a1000, D4 confidence_router}×{Joint超参固定B0, 原生数值重调}；五数据集、3/5/10-shot、seeds42–46。新增300份训练，复用225份已冻结模型。16个试运行、全量训练、指标与独立checkpoint预测复核均通过。固定超参仍重训权重。每个Router处理内的两组Joint拥有完全相同的Router输出和训练伪标签池。

主纠正性精度只看已接受伪标签 != argmax(B0固定pGL) 的节点。α/β采用B0最终选中checkpoint；各模型自身pGL的结果另列，不能混为同一参照。空纠正集合的精度是null，务必结合scope-supplement中的有定义配置数与纠正节点数。净准确率纠正=原错变对−原对变错；纠正标签正误差=正确标签−错误标签（错误含错转错），两者分列。

Router和伪标签在完整Query评估，D2剔除新增训练节点并匹配B0同一名单；最终分类在原顺序eval1000。Fig10讨论分歧区，因此分歧区实际接受精度和A+D全部接受精度同时给出。原范围报告保留，不能混用两种精度。Joint仍接收完整原support gold；D2新增真标签仅用于Router。

每种Joint重调沿用原生四阶段验证集搜索，开放dim/cross/lr/gate_lr/warmup/aux/keep，其余设置固定B0。重调的收益不能单独归因cross；离线相关不能证明因果机制，不显著也不能证明不相关。旧预算争议与阶段1仍暂缓，定时监控保持暂停。

- report/per_cell.csv、RESULTS.json：525份模型的指标、完整配置、实际分母与来源。
- scope-supplement：分歧区对照、Fig10范围、同样本相关比较及簇重采样区间。
- associations：三张配对差值散点图、两种相关系数及不确定性。
- cells：每个划分的七设置指标和eval1000逐节点证据。
- training-records：525份最终入选模型的配置、checkpoint选择及原训练history。
- training-selection：300份新训练的输入登记、全部候选验证指标、四阶段计划。SERVER_COMPLETE.json是服务器原封存记录，里面的二进制哈希不表示二进制已上传。
- independent-audit：从525份checkpoint重新推理eval1000，标签与原保存预测逐节点完全相同；概率容差2e-6；检查训练池一致、固定超参、指标恒等式及汇总值。
- proofs、logs：登记、作业依赖、失败路径修复、试运行和完成核验证据。

模型权重、候选checkpoint、完整图/特征缓存和全Query二进制预测仍在服务器；全部可报告的文本指标、候选选择、训练历史、逐节点评估结果、图片和核验证据包含在本目录及Release压缩包中。来源路径和哈希随记录保存。

代码：[实验入口](../../experiments/stage0_20261029/factorial_joint.py)、[独立复核](../../experiments/stage0_20261029/audit_factorial.py)、[分歧区补充](../../experiments/stage0_20261029/scope_analysis.py)、[协议](../../docs/JOINT_FACTORIAL_3X2.md)。

[本次结果解读与七设置总表](INTERPRETATION.md) · [同名单B0与共同有效配置对照](paired-reference/REPORT.md)
