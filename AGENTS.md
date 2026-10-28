# EnhanceRouter 工作约束

## 当前用户要求（2026-10-26，优先于方案中的冲突内容）

1. 对每个数据集和 seed，从历史 **10-shot Query** 中固定抽取 1,000 个节点，3／5／10-shot 共用这些节点及顺序。已存在的历史 1,000 节点名单必须逐项、逐序沿用，不得从交集重新抽样，也不得强制不同 seed 共用同一名单。若 seed 45/46 没有历史名单，必须先明确来源并登记；不得冒充历史名单。
   - 用户再次确认：跨 seed 的评估名单可以不同；只要求每份名单与同一 seed 各 shot 的 train、valid 均无重叠。跨 seed 的交集单独披露，不据此重抽名单。
2. 实验仅使用 seeds **42、43、44、45、46**，用于开发、诊断、决策和最终比较。暂不预留额外种子，不使用 142–146 或 1042–1046。
3. 严格遵循用户提供的《EnhanceRouter 改进实验方案》；冲突以本次用户指令为准。阶段 0 的 D1–D4 和判定记录完成前不得推进阶段 1；各阶段按单因素、Level 1 → Level 2 门槛逐步接纳。

## 工程与实验边界

- EALR-GTC 只读：不修改其代码、产物、Git 元数据、日志或缓存；仅使用本工程内复制的运行代码，不从 EALR-GTC 导入或执行模块。
- 所有新增代码、输出、检查点、日志和缓存均保存在本工程内。
- 所有数值工作（包括 tensor 检查、测试和报告）必须通过 Slurm。
- 历史大表 45 格复现保持原样、单独归档；不把历史 seeds 42–44 的范围误当成新实验仅可用 3 个 seed。
- 保留 Original Router 与 Joint/PURF V1 的历史逐 split 选择流程。不得把原 5-shot agreement/disagreement 流程替换为统一三折校准；3/10-shot 的参考接受比例必须继承对应基线值，不能写死为 15%。
- 受控对照冻结 B0 的 scorer 族、特征模式、训练 scope、完整已拟合输入变换（标准化、筛列、投影、rank 等）、输出归一化、仿射、路由规则、接受比例和下游超参；只重新拟合被测变量允许变动的参数。不得用恢复输出 center/scale 冒充冻结输入变换。
- D1 按 support 原始节点子采样，保留/丢弃节点全部六视图；固定专家、原型、源数据及其他已选配置。
- Oracle 真标签仅在隔离的 D2 分析训练和冻结后的评估中使用；D2 排除固定评估节点，不能用于方法超参选择。
- 每次运行记录输入路径及 MD5/SHA256、完整配置、来源、环境、工程 commit 和未提交代码哈希；EALR-GTC 无 Git commit 时记 null，不编造、不初始化。
- 新协议使用新运行目录。不得覆盖已完成运行，不把未通过审查的旧结果作为有效新基线。提交任务不等于完成；需实际完成日志、检查与封存结果。

## 当前有效记录

- 活跃实验代码：experiments/stage0_20261026；当前运行：runs/stage0-20261026-v2。
- 用户已授权按原始 CUDA 划分流程补建 seeds45/46 的3/10-shot，并先验证与其历史5-shot嵌套。SPLIT_AUDIT.json 已通过全部75格检查：42–44原划分逐项重放一致，45/46历史5-shot逐项一致。
- EVALUATION_IDS.json 以 seed/dataset 为键，共25份；不得回退到仅dataset为键的新交集名单。
- stage0_20261023 和新目录中保留的未列入 README 有效入口的辅助脚本，仅作旧协议留档；不得使用旧142–146入口提交新实验。

- 五个实验 seeds 指 support/split 的42–46；不要将它们替换到原基线内部所有 RNG。历史 GNN 头的固定 model_seed 为 Cora47、Citeseer46、PubMed43、Arxiv46、Products46，必须原样保留；Joint训练seed继续使用该split seed。参见 GNN_TRAINING_SEEDS.json 和 HEAD_REFERENCES.json。
- 新生成的标准前端只使用 standard_native（45/46的3/10-shot）；D3只使用 small_native。standard 中这20格旧生成输出和 small_stratified 全部作为被排除的历史记录保留，禁止纳入结果。
- 批量训练前必须通过 verify_prerequisites.py，校验55个历史B0、55个历史D1和95套原生前端并冻结其哈希；不能仅移除或替换已过期的Slurm作业ID来跳过前置条件。

- 受控重训同时固定 B0 SVC 的实际 `_gamma`，不能让 `gamma=scale` 随新样本方差重算；D1通过 variant_folder 选择已审定结果，十个历史受影响组合与新增20格使用 d1_frozen_v2，旧结果保留。
- 新增20格 B0 使用同 seed 的历史5-shot作为原生 incumbent 配方来源，重拟合全部权重/输入统计，再执行原生30初始候选、最多20细化及 OOF 分支选择；不得仅转移seed42选中配方。D3也重跑完整scorer候选选择和Joint/PURF V1四阶段自适应搜索。55个历史B0保持逐格原样。新登记 NATIVE_BASELINE_SEARCH_REGISTRATION.json 优先于旧有限候选登记。


## 2026-10-27 用户确认：Router 与最终分类的评估范围

Router 的结果应该在全量 Query 节点上计算，不管是 P@k 伪标签准确率还是 Trust/Pref AUROC。最终分类的结果才在 eval1000 上算。

D2：从该配置的完整 Query 剔除实际新增的真标签训练节点；B0 必须使用完全相同的剩余节点名单作配对比较。不得将新增训练节点算入 Router 测试结果。

D3：报告其小验证集划分下的完整 Query；与 B0 的配对比较另在两者共同、均未用于训练或验证的 Query 上进行，明确区分两种评估范围。

本次仅重算已冻结模型的描述性指标，不重新训练或用 Query 标签选择模型。P@1/5/20/100% 的 k 为各分支完整评估池大小乘相应比例向上取整；每组记录实际节点数和 k。旧报告中在 eval1000 计算的 Router 指标已被本次全 Query 报告取代；最终分类仍使用原固定 eval1000。此前暂缓的 D-Prec@B0/A-Prec@B0 预算决策及阶段1保持暂缓，定时监控保持暂停。

用户本次明确要求更新本文件。修改前的冻结指令原文和 SHA256 保存在 provenance/full-query-amendment-20261027；旧 CODE_FREEZE.json 及所有训练代码与模型不变。新的指标报告执行入口校验归档指令与原冻结哈希一致，当前指令与本次追加登记一致，其余冻结文件逐一按原登记校验。

## 2026-10-28 指令入口合并与存储迁移

- 根目录 `AGENTS.md` 是唯一维护完整项目指令的文件。此前 `Agent.md` 与它内容完全相同，没有独立职责；现将 `Agent.md` 改为指向 `AGENTS.md` 的相对软链接，兼容用户既有链接和历史文件引用。用户此前写的 `agent.md` 是笔误，不另建第三份。实验子目录的 `AGENTS.md` 继续作为该目录的补充约束。
- Codex 默认发现的是 `AGENTS.md`；其他文件名只有显式配置为 fallback 时才参与自动发现。参考官方文档：https://learn.chatgpt.com/docs/agent-configuration/agents-md 。不要再复制维护两个根指令副本。
- GNN-as-judge 的当前真实位置是 `/data1/wujingyao-20354/wujingyao/codes/GNN-as-judge`。Linux 区分大小写：新路径末段为 `GNN-as-judge`，旧路径为 `/home/wujingyao/codes/TAG_Fewshot_Learning/GNN-as-Judge`。
- 已核实旧路径保留为指向新目录的软链接。`configs/inputs.json` 的45个历史 graph 引用和冻结来源记录仍使用旧路径；当前依靠该兼容链接读取同一文件。迁移不会改变已完成实验的数值，也不构成重新训练的理由。
- 冻结配置和历史记录不批量改写为新路径；保留旧链接，运行前验证其解析目标、图数据及OGB划分文件的MD5/SHA256与原冻结记录一致。如果兼容链接失效，先停止依赖它的执行并登记新运行的路径映射，不能绕过哈希检查或悄悄换数据。GNN-as-judge 与 EALR-GTC 来源均只读。
- 本次仅合并指令、记录路径并校验迁移/上传完整性，不修改模型训练、推理、划分、指标或已封存结果。旧指令和上一版指令修订登记保存在 `provenance/instructions-and-storage-20261028/`；`CODE_FREEZE.json` 的624项原始哈希不变，文档修订仍通过归档原文与当前指令哈希显式验证。
- 当前新D1仍固定B0的特征模式、训练组合、模型族、完整已拟合输入变换、专家和六视图；仅允许原流程的数值超参数及校准重调。旧全冻结D1单独保留。Router完整Query、D2剔除新增训练节点并匹配B0、最终分类eval1000的规则不变；监控保持暂停，原预算争议和阶段1继续暂缓。
- 结果上传范围必须说清楚：GitHub仓库保存全部实验报告、逐组指标和哈希定位；Release的 `experiment-results.tar.gz` 保存1350份配置/诊断/选择记录及已导出的eval1000逐节点结果，`full-query-metrics-20261028.tar.gz` 保存完整Query更正指标和75组核验证据。模型权重、完整图缓存和原始全Query二进制评分文件仍在服务器，不能声称这些大型二进制已上传。它们的路径和哈希由结果索引记录。
