# FairRouter / B0 与 D1–D4：完整实验结果

**Router指标评估范围已更正：请使用[全量Query报告](../full-query-20261027/REPORT.md)。本目录旧REPORT.md、RESULTS.json、per_cell.csv中的Router指标是在eval1000抽样上计算的，已由更正报告取代，原文件不覆盖以保留审计记录。最终分类的eval1000结果保持有效且数值未改。Release中的逐节点Router CSV是eval1000抽样视图，不是全量Query指标的计算全集。**

全部75组配置：5个数据集 × 3/5/10-shot × split seeds42–46。每个dataset/seed固定eval1000，跨shot共用节点及顺序。

## 审阅入口

- [完整指标表](REPORT.md)：B0、旧冻结D1、新数值重调D1、D2、D3、D4。
- [逐组结果CSV](per_cell.csv)：1275条路由结果；D3的75组最终准确率在RESULTS.json的completed_L2中，也见[独立D3逐组CSV](../completed-d3-20261027/per_cell.csv)。
- [完整结果JSON](RESULTS.json)：逐组指标、标准差、有效样本数、匹配B0、D2实际标签数、D3/D4最终准确率。
- [原始文本结果与逐节点输出压缩包](https://github.com/yaolalal/EnhanceRouter/releases/download/stage0-results-20261027/experiment-results.tar.gz)：1350份已封存模型配置对应的JSON/CSV/诊断/选择记录，及各自eval1000的路由分数；已有Joint的配置同时含最终预测类别与置信度。
- [产物索引](raw-outputs/INDEX.json)：逐模型路径、完成登记哈希和逐节点输出哈希。压缩包内MANIFEST.json还保留所有源文件的路径与哈希。
- [校验证据](proofs/)：新D1的300份、D2的450份、D3的75份及总计1350份封存结果的检查记录。
- [历史大表FairRouter复现](../fairrouter-baseline/REPORT.md)、[独立D3报告](../completed-d3-20261027/REPORT.md)、[独立D4报告](../completed-diagnostics-20261027/REPORT.md)。

## 代码定位

[D1数值重调](../../experiments/stage0_20261027/d1_numeric.py)、[B0/D2/D4入口](../../experiments/stage0_20261026/compute.py)、[D3边界恢复](../../experiments/stage0_20261027/d3_recovery.py)、[专家与六视图](../../experiments/stage0_20261026/frontend.py)、[完整指标汇总](../../experiments/stage0_20261027/report_current_d1_d4.py)、[逐节点结果导出](../../experiments/stage0_20261027/pack_review_results.py)。

## 完成范围与实验边界

B0为75组；旧全冻结D1为300份，当前数值重调D1另有300份；D2为450份；D3为75份；D4两种各75份。总计1350份均通过封存文件哈希检查。新D1所有四档均覆盖相同75组，其100%为内部对照。

D1固定B0的特征模式、训练组合、模型族、完整已拟合输入变换、专家和六视图；仅数值超参数与原校准可重调。旧冻结D1独立保存。D2请求N为上限，部分高置信池实际数量不足；a/b之间存在实际标签数差异的配置，不能仅归因于分布差异。D2及旧冻结D1没有另训Joint最终准确率。D3每类最多使用k个validation标签，原池不足时记录实际数量。

P@1/5/20/100%是固定eval1000内分歧节点的排序精度。D-Prec@B0、A-Prec@B0的预算口径仍按用户要求暂缓，本结果不作阶段0接纳或阶段1推进判断。历史B0的5-shot选择登记含selection_uses_query_truth=true，来源限制见[基线说明](../../docs/FAIRROUTER_BASELINE.md)。

完整压缩包保存在本仓库Release下载区，文件名为experiment-results.tar.gz；SHA256在raw-outputs/INDEX.json和PUBLICATION_SHA256.json中。逐节点CSV保持原始eval1000名单及顺序，eval_order从0开始。压缩包包含可审阅的实验结果、配置、诊断和eval1000输出。二进制模型权重及全图缓存的源路径和哈希保存在清单中，原文件保留在服务器。

指标快照时间（Asia/Shanghai）：2026-10-27T21:40:16.364574+08:00。
服务器原始汇总：runs/stage0-20261026-v2/current-results/20261027T134000Z。
逐节点导出：runs/stage0-20261026-v2/cloud-review/20261027T134000Z。
