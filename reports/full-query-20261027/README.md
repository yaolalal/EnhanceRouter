# Router 全量 Query 指标更正

[完整报告](REPORT.md) · [逐组CSV](per_cell.csv) · [汇总CSV](summary.csv) · [完整JSON](RESULTS.json)

本报告取代此前 all-results-20261027 等报告在 eval1000 上计算的 Router 指标。Router 的 P@k、Trust/Preference AUROC 在完整 Query 上计算；最终分类继续使用原固定、原顺序的 eval1000，结果不变。

- D2：排除该配置实际新增真标签训练父节点，B0 使用完全相同的剩余 Query 名单。
- D3：报告小验证集协议下的完整 Query；B0 配对比较另用双方共同的原 Query，避免把 B0 验证节点纳入其测试。
- P@1/5/20/100% 分别基于分歧、一致分支的完整评估池，k 向上取整。Trust/Pref 的条件与原定义不变。
- 1350份冻结模型输出，共75组、seeds42–46；无重新训练、无模型选择；旧冻结D1和新数值D1分开。
- 暂缓的 D-Prec@B0/A-Prec@B0 预算决策、阶段0接纳及阶段1均未执行；定时监控保持暂停。

代码：[report_full_query.py](../../experiments/stage0_20261027/report_full_query.py)；手算语义验证：[check_full_query_metrics.py](../../experiments/stage0_20261027/check_full_query_metrics.py)。指令已更新至[Agent.md](../../Agent.md)、[AGENTS.md](../../AGENTS.md)和活动实验目录；[指令修改登记及原文](../../provenance/full-query-amendment-20261027/AMENDMENT.json)保留旧冻结哈希。

RESULTS.json保留完整Query节点数、分支计数、实际k、节点名单SHA256、D2排除父节点、匹配B0指标、输入评分文件的路径与哈希，以及最终分类的eval1000名单哈希。per_cell.csv为1350条扁平记录；完整匹配细节见JSON。cells/内保留75份Slurm逐组结果与完成封存。

原二进制评分文件和模型保存在服务器，路径见JSON的source。此前[Release](https://github.com/yaolalal/EnhanceRouter/releases/tag/stage0-results-20261027)仍保留全部模型配置、诊断及eval1000逐节点CSV；这些抽样Router CSV不代表完整Query计算全集。本次完整Query指标和输入定位已单独发布，旧封存报告未覆盖。

服务器运行目录：`runs/stage0-20261026-v2/full-query-20261027`。
