# Stage 0：用户确认的五 seed 协议（2026-10-26）

有效运行：`runs/stage0-20261026-v2`。仅使用 split seeds 42、43、44、45、46；不设 dev/held-out 两组。模型内部历史 ensemble seeds 保持原配置，不新增实验 split seeds。

每个 dataset/seed 使用自己的有序 eval1000，三个 shot 共用。只要求与同一 seed 的 train/valid 无交集；跨 seed 交集已披露并经用户确认允许。42–44 沿用历史10-shot名单。45/46 的缺失3/10-shot经用户授权使用复制的原 CUDA 采样语句补建，先逐项核对历史5-shot，再检验嵌套。全部75格已通过，见运行目录 SPLIT_AUDIT.json。

有效入口：prepare_native.py、hydrate_historical.py、register_exact_five_shot.py、reuse_baseline.py、cache_requests.py、frontend.py、small_stratified.py、compute.py、d1_report.py、report.py。job.sh 是统一 Slurm 包装。其余从 stage0_20261023 留下的辅助入口仅作历史参考，不用于当前协议；尤其禁止运行旧 prepare.py / prepare_amend.py / preflight.py / recover 脚本。

55个历史B0格子保留原专家、scorer、完整输入变换、仿射、路由规则、接受比例和Joint选择；先核对原始路由张量再封存。只有45/46的3/10-shot共20格需要补算。HISTORICAL_INPUTS_V2.json 记录逐格实际来源；V1衍生猜测未用于模型拟合。

5-shot 根据已选 bundle 的实际 schema 保留分支流程：原非 crossfit 分支使用原生 agreement/dense-K 或 disagreement 校准选择；原来已经采用 crossfit 的分支保持该流程。3/10-shot 继承对应基线接受比例，不写死15%。D1/D2的训练使用 frozen_fit.py，在训练开始前固定已拟合输入处理；输出归一化与完整路由 bundle 也保持B0原值。不得只训练完再替换 scaler。代表性完整数据重放、减少数据后输入哈希一致性和原生5-shot政策回归均已通过 Slurm 测试，记录在 reports/audit-20261026。

D1按support原始节点取25/50/75/100%，同节点六视图整体保留。D2排除同seed固定评估节点，oracle标签仅用于隔离诊断。D3从原validation按类缩减至最多k，确实不足的类别按实际数量报告，不另从Query补标签。D4保持B0实际接受数和Joint超参。所有75格对应模型封存后才发布评估结果；D1可以独立发布，但不能据此宣称阶段0完成。

历史来源注意：EXACT_FIVE_SHOT_REGISTRATION.json 保留原 CHOICES.json 中 selection_uses_query_truth=true 的既有来源事实；这里只复现历史B0，不将该文件的选择机制用于任何新增方法或超参选择。新比较全部遵守固定协议与验证集选择。

文本缓存按节点ID复用冻结专家输出，不使用标签；仅缺失节点需要原fp16 LLM前向。资源不足时增加Slurm主存/GPU配额，不改模型或精度。所有输出、缓存及运行代码留在EnhanceRouter；EALR-GTC只读，绝不从其目录导入代码。

任务提交不代表完成。阶段0只有在report/COMPLETE.json存在且all_checks_pass=true时才算完成。D2b N=200的L1门槛决定是否推进阶段1；没有结果不得臆造接受结论。
