# D1–D4 当前结果完整汇总

指标快照：北京时间2026-10-27 14:59；汇总作业120949以0:0退出。全部实验使用split seeds42–46、3/5/10-shot，每dataset/seed固定eval1000，跨shot共用节点及顺序。

## 文件

- REPORT.md：完整汇总表，包括1%/5%/20%/100%覆盖率精度、Trust/Preference AUROC，以及D3/D4最终准确率。
- per_cell.csv：927条已完成配置记录，包括75份B0、300份旧冻结D1、42份新D1、360份D2、150份D4；记录逐格指标、物理输出路径及封存哈希。
- RESULTS.json：全部原始汇总数、均值/标准差/有效样本数、匹配B0、D2实际N、共同完成的新D1配置，以及D3/D4逐格最终准确率。
- FROZEN_SNAPSHOT.json：本次实际纳入的完整有序产物名单及封存哈希，统计过程不纳入之后新完成的产物。
- completion_by_variant.csv：逐实验、逐数据集完成数量。
- RUNTIME_QUEUE.txt：发布时队列，时间晚于指标快照；D1_RUNTIME_CONCURRENCY.json登记并发2→6的运行调整，每组仍为2CPU/24G，数值代码/配置未改。

D-Prec@B0和A-Prec@B0仍按用户要求留空，本次没有替用户确定预算，也没有作阶段0或阶段1接纳判断。P@20%是在eval1000中的分歧节点上，按该方法排序取前20%后实际路由标签的精度；不等同于D-Prec@B0。Trust AUROC评估至少一位专家是否正确，Preference AUROC在至少一位正确的分歧节点上评估应选GNN还是LLM。

## 实验与代码定位

D1旧版：仅改变目标support原始节点比例25/50/75/100%，保留节点全部六视图；超参、完整输入处理、校准和路由策略固定。入口 experiments/stage0_20261026/compute.py:d1，冻结重拟合在frozen_fit.py。300/300份完成，仅做路由诊断，没有另训Joint。该版100%本次与B0六项路由指标逐项核对一致。

D1当前版：逐split固定B0特征模式、训练组合、模型族、完整已拟合输入处理、专家和增强，允许数值超参/原校准重调；入口experiments/stage0_20261027/d1_numeric.py，数值过滤native_catalog、选择selected_heads、原四阶段Joint约束native_joint。42/300份完成；共同完成四档的7组仅含3-shot/seed42五数据集及3-shot/seed43的Cora、Citeseer，不能视为完整75格结论。新100%是本版内部对照，不能与旧版75格均值直接归因比较。

D2：两种Oracle扩充，a从全部未标注池抽样，b从B0高置信一致候选池抽样。请求N=50/200/1000，实际数量受候选池限制；排除eval1000，使用真标签和同样六视图，仅作分析。入口experiments/stage0_20261026/compute.py:oracle_requests/oracle_banks/d2。完整冻结B0非数据配置，仅读入新增目标训练观测重拟合；360/450份完成，Products等待LLM补算。没有新增Joint最终准确率。

D3：验证标签每类最多取k个，原池不足k时保留实际数量，重跑专家头选择、router选择及Joint。入口compute.py:baseline(...,d3)，frontend.py:heads；空分支恢复扩展在experiments/stage0_20261027/d3_recovery.py。75/75格通过独立哈希检查，最终准确率已完成。专门报告见../completed-d3-20261027/。

D4：Confidence router用专家置信度替代路由打分与偏好，Confidence preference保留学习到的trust并用专家置信度选择模态；整个Query每分支接受数量匹配B0，Joint配置固定B0。入口compute.py:d4及route。两种各75/75份完成，最终准确率在各自固定eval1000上计算；专门报告见../completed-diagnostics-20261027/。

共同专家/六视图构造位于experiments/stage0_20261026/frontend.py，训练银行在common.py，原Joint选择在native_joint_search.py。服务器逻辑工程根为/home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter，当前产物根为runs/stage0-20261026-v2。逐格路由分数、模型、预测和配置的完整路径均在CSV/JSON中。

历史B0的5-shot配方选择登记标注selection_uses_query_truth=true；该来源限制随基线保留，新的数值搜索只读validation。相关披露见docs/FAIRROUTER_BASELINE.md。
