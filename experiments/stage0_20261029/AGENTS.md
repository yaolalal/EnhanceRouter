# 2026-10-28/29 用户授权的 3×2 Joint 混杂核查

本目录仅用于 {D1 numeric 25%, D2a random1000, D4 confidence_router} × {Joint超参数固定B0, 原生数值重调}，加B0。继承根AGENTS.md及stage0_20261027约束。所有数值计算必须用Slurm；历史624文件冻结和全部既有产物不改。

每个Router处理的两个Joint设置必须使用相同专家、划分、Router分数、已接受节点及伪标签、完整Joint gold support。只改变Joint的数值超参数是否按原四阶段验证集流程选择；冻结超参数不冻结Joint权重。D1使用已完成d1_numeric/0.25 Router，不能以旧d1_frozen代替。D2使用用户确认的随机a/1000，新增真标签仅用于原D2 Router训练，不新增Joint gold标签。D4使用完整confidence_router。

新代码和新运行单独登记。可复用通过哈希/协议检查的D1-retuned和D4-fixed及B0；D1-fixed、D2-fixed、D2-retuned、D4-retuned新训练。不用Query真值搜索超参或接受方法。

纠正性指标以实际已接受伪标签中 pseudo != argmax(pGL) 为主，报告精度、改对、改错、错转错、净纠正及分母。主pGL固定B0 checkpoint alpha/beta；另报各模型自身alpha/beta，防止移动参照混杂。主范围完整Query；D2剔除新增训练父节点，配对B0也用相同剩余Query。原Query/实际训练池（含D2已知节点）另列不能混用。Joint最终分类仍为原顺序eval1000。

相关性只作离线解释，保留无定义值、节点量和数据集/shot分层；配对分类差值与伪标签/纠正性差值比较，避免把跨数据集原始难度相关当作机制证据。两Joint设置间Router指标应相同，检查不一致必须停止对应分析。原L1预算和阶段1仍暂缓；不恢复定时监控。

“净纠正”明确区分实际准确率净增（原错变对−原对变错）与纠正标签正误差（正确伪标签−错误伪标签，含错转错）；两者都报告。原论文Fig.10涉及的5-shot、seeds42–44、PubMed/Products另列，避免与全75组混用。离线相关性按dataset×seed簇重采样并比较两种相关系数的差；不得把不显著等同于不相关。
