# 结果上传与来源迁移核验

全部75组实验、1350份模型输出的报告及已导出结果均已上传。逐项核验见 [RESULTS.json](RESULTS.json)，脚本为 [audit_storage_uploads.py](../../experiments/stage0_20261027/audit_storage_uploads.py)。

| 内容 | 云端位置 | 范围 |
|---|---|---|
| 完整Query Router指标 | [更正报告](../full-query-20261027/REPORT.md)、[逐组CSV](../full-query-20261027/per_cell.csv) | 1350条，含D2相同名单B0比较、D3完整及共同Query；取代旧eval1000 Router指标 |
| 最终分类 | [完整表](../full-query-20261027/REPORT.md#最终分类固定eval1000结果未变) | 固定eval1000，600份已有Joint输出；旧冻结D1和D2未额外训练Joint |
| 配置、诊断、选择记录及逐节点输出 | [experiment-results.tar.gz](https://github.com/yaolalal/EnhanceRouter/releases/download/stage0-results-20261027/experiment-results.tar.gz) | 1350份模型的文本产物、1350份eval1000 Router CSV、600份最终分类CSV |
| 完整Query更正结果包 | [full-query-metrics-20261028.tar.gz](https://github.com/yaolalal/EnhanceRouter/releases/download/stage0-results-20261027/full-query-metrics-20261028.tar.gz) | 完整更正指标、75组完成证明、计算代码；包内指令是发布时的历史快照 |
| 二进制模型权重、完整图缓存、原始全Query二进制评分 | 服务器原路径 | 未上传大型二进制；公开[原始索引](../all-results-20261027/raw-outputs/INDEX.json)、压缩包MANIFEST及更正JSON记录源路径和哈希 |

两份Release资产的云端SHA256分别为 `5bc29714a3c1baefa0bbd0767436396ff303e8369a9889eb99b178440e7592ff`、`a4a27964234da42423e7af350dfd2e4147f6039e74a327bb113ed676ef677c1e`，与服务器、已下载本地文件一致。本次还逐项检查1350份模型完成登记、归档中配置/诊断/逐节点CSV的哈希，以及两套发布清单。

GNN-as-judge 当前路径为 `/data1/wujingyao-20354/wujingyao/codes/GNN-as-judge`。旧 `/home/wujingyao/codes/TAG_Fewshot_Learning/GNN-as-Judge` 保留兼容软链接，因此45个历史graph引用仍有效。图数据与OGB划分的来源哈希按冻结记录核验；不改冻结配置或重跑实验。

根指令统一维护在 [AGENTS.md](../../AGENTS.md)，[Agent.md](../../Agent.md) 仅为相对软链接。此前两个文件内容完全相同；用户写的`agent.md`是笔误，不另建第三份。旧文档和上一版修订登记已[归档](../../provenance/instructions-and-storage-20261028/)，数值代码、624项旧冻结登记和全部封存指标不变。实验子目录的AGENTS.md继续提供局部补充约束。
