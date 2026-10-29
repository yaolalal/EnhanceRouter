# 用户要求：B0及3×2完整指标报告

仅评估已经冻结的B0及{D1 numeric25%, D2随机a1000, D4 confidence_router}×{Joint超参冻结, 数值重调}，五数据集、3/5/10-shot、seeds42–46，共525模型。不得重训、重新选择模型、改变Router/接受节点/伪标签或输入变换。原模型、624文件冻结、既有报告保持原样。

全量Query：AUROC使用评分器predict_candidate的原始输出，明确在外层gold_normalization/center-scale/affine/pair calibration之前；D4没有可学习校准，使用原置信度规则。P@1/5/20/100%使用既有校准后部署评分和路由结果。D2排除新增真标签训练节点，配对B0使用相同剩余Query。

伪标签指标使用实际接受集合，分agreement/disagreement/全部分别报告精度与计数；纠正性指标仅在disagreement实际接受集合，以各模型自身pGL为参照，报告C/R/H/W及净数。校验与上次own-pGL封存统计一致。

分类在原eval1000上：pR单独、去pR的自身pGL、完整pJ。新增子集S={argmax pR != argmax pGL}，报告占比、人数和S内三项准确率，不得用pJ!=pGL旧集合代替。复用封存逐节点预测并核对节点顺序和真值，不重训。

所有数值计算通过Slurm。报告总体与每个dataset×shot；同组七配置放在一起，另列D2同名单B0。精度为配置等权均值±总体标准差，有效n单列或注明；空集合NA。计数同时保留总数/每配置均值，跨配置总数是节点—配置次数。为用户生成完整Markdown并复制到本机Downloads；同步云端且核验哈希。阶段1、旧L1预算、定时监控保持暂缓/暂停。
