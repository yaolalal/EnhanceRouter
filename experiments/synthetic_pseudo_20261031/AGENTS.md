# 真值合成分歧伪标签：用户授权诊断实验

继承根AGENTS.md；这是用户明确授权的真值构造诊断，不是方法，不作方法选择或接纳。只使用5-shot、split seeds42–46，保持B0专家、gold/valid/Query和有序eval1000。所有计算经Slurm，旧624冻结文件、历史来源及结果只读。

候选为Query去掉eval1000/support/valid，且两专家分歧、恰好一个正确。完整无向简单图去自环度数。选点seed0均匀抽N=min(1000,candidates)，不足1000正常记capped。十规则同点同N，K=round(.8N)，正确标签/错误标签分别选正确/错误专家argmax；R6整类错误及最后部分类写明。margin固定B0最终alpha/beta。R7/R8按LLM最大概率降/升序，R9/R10按GNN最大概率降/升序，沿用噪声seed打破并列。

R1–R10噪声0/1/2×训练0/1；A0/A100训练0/1且不复制噪声试验。主版本25格×64=1600模型；last500版本仅R1/A0/A100，25格×10=250模型。B0直接引用，另在训练前用新trainer重放原池/原配置/训练id0，25格eval1000标签必须完全一致。任何门槛失败停止后续作业，保留中断文件，不换方法掩盖。

用户本次指定synthetic teach_g表示被选中GNN，teach_l表示被选中LLM。历史adapter命名恰好相反（被修复对象），因此仅B0重放保留历史原池顺序；synthetic按用户定义分池，不调用相反命名的旧pool validator。loss循环的两个独立RNG流、各batch同大小、有放回、各cross/2完全保留，不按池人数重加权。

training.py从旧trainer复制，来源SHA及修改登记。train(...split_seed,training_seed=0/1)的有效RNG基数=split_seed+1000003*training_seed；id0与旧初始化及四个batch RNG偏移完全一致；不改变split。固定500版保留best历史，但实际取last500状态，不早停。

所有合成运行keep=0；非A0若B0 cross=0强制1，A0始终cross=0。分别记录actual cross_forced与cell_B0_cross_zero；剔除强制格时按后者排除整个配对单元，避免A0比较不对称。其他B0超参固定。

指标固定B0 pGL作为行为参照，另记录自身pGL；head/tail用完整图真实类别频率中位数，head严格大于中位数、tail不大于，分类集合仅用于离线评估。预测熵用eval1000硬预测类别直方图的自然对数熵，另记录归一化熵、类别数、最大类别比例。

主汇总按split×noise×train，锚点仅唯一两次训练。配对时锚点与相同split/train的每份R1噪声结果比较，但先在每split内平均，CI按5个split整簇重采样（固定bootstrap RNG、10000次）。R1主方差分解按split→noise→train的条件总方差公式，三项精确加总；另外给出平衡有限ANOVA的三个主效应和全部交互项，明确前者按顺序分摊交互，不能冒充独立随机效应估计。选点seed只有0，不能估计选点方差。

报告先观察后解读，包含全部结果、第0步/强制cross/候选capped、剔除第0步（配对剔除双方任一step0）及剔除强制cross。last500与best独立汇总。禁止Query真值用于选择训练配置或checkpoint；validation按旧accuracy/NLL规则选择。所有truth合成池与eval1000严格隔离；预先封存池列表、逐节点标签/专家归属及全部源哈希。
