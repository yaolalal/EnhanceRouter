# Joint 数值调参与纠正性伪标签：3×2受控实验

用户2026-10-28授权：{D1 numeric25%, D2随机a1000, D4 confidence_router} × {Joint超参固定B0, Joint数值重调}，另加B0。范围五数据集、3/5/10-shot、seeds42–46，共75组划分、7种设置。D2随机a已由用户明确确认。

实验入口：`experiments/stage0_20261029/factorial_joint.py`。运行：`runs/joint-factorial-3x2-20261029-v1`。旧代码/产物不变。

每个Router处理的两组Joint必须共用完全相同的Router、接受名单、伪标签、专家、划分、全部Joint gold support。固定表示超参数固定后重新训练权重；重调复用`d1_numeric.native_joint`，原生四阶段搜索仅开放dim/cross/lr/gate_lr/warmup/aux/keep，其他设置固定B0。B0、现有D1重调及D4固定可复用；D1固定、D2固定/重调、D4重调新训练。新增D2真标签只进入Router，Joint仍使用原support真标签。

纠正性指标主参照：B0最终checkpoint的alpha/beta定义固定pGL=(alpha*pG+beta*pL)/(alpha+beta)。只在实际已接受伪标签 != argmax(pGL) 的节点计算精度；改对为pGL错误而伪标签正确，改错为pGL正确而伪标签错误；净纠正=改对−改错。错转另一错误类单列；同时报告“正确纠正标签数−错误纠正标签数”（错误数含错转错），与实际准确率意义的净纠正区分。空纠正集合为null，不能填0。另报各模型自身alpha/beta的pGL参照。

主指标在完整Query计算。D2剔除其新增真标签训练节点；B0同一剩余Query配对。原Query上的实际训练池另列，不能混入主指标。最终分类仅用原顺序eval1000。原始净纠正计数、纠正节点数及每1000Query的归一化值都报告，避免节点规模混杂。cross=0时仍可评估Router质量，但不能把未启用的伪标签称为实际损失贡献。

报告7种设置的Trust/Pref AUROC、分歧区P@1/5/20%、接受伪标签精度、纠正性精度/数量/净纠正和分类准确率；按dataset、shot、seed保留明细。检验两Joint设置的Router指标及固定pGL纠正性指标逐项相同；配对分类差值对比伪标签/纠正性指标差值，另分数据集，避免仅用跨数据集难度产生的相关。相关性是离线描述，不能预先宣布解释假设成立；Joint重调同时改变多个数值项，收益也不能自动归因cross。

计算通过Slurm。先源文件/输入哈希登记，再覆盖Cora、Arxiv、Products及seed45原生输入的16个试运行，经完成检查后批量训练。训练和报告各自封存哈希，失败保留中断目录，不覆盖旧结果。原定时监控继续暂停；L1预算争议与阶段1不在本次范围内。
