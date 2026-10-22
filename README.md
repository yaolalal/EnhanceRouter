# EnhanceRouter

独立的 **Router-v10-MS-Original + Joint/PURF V1** 工程。代码和所有新结果归本工程管理；EALR-GTC 仅作为只读冻结输入来源。

## 一条命令复现

在服务器上运行：

```bash
cd /home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter && ./reproduce.sh
```

命令提交 Slurm 依赖链：冻结文件身份校验 → 15 组独立训练（5 数据集 × seed 42/43/44，5-shot）→ 全部模型冻结 → 统一评估和验收。默认使用 `/home/wujingyao/miniconda3/envs/lf/bin/python`，CPU、每任务 2 核/24 GiB、并发 3。可用 `ENHANCEROUTER_PYTHON` 或 `ENHANCEROUTER_CONCURRENCY` 调整环境/并发。

终端打印 run 路径和 job ID。通过 `squeue -u "$USER"` 查看；最终结果在 `runs/<run>/evaluation/REPORT.txt`、`per_seed.csv`、`summary.csv` 和 `ACCEPTANCE.json`。只有 `ACCEPTANCE.json` 中 `all_checks_pass: true` 才表示验收通过。每次使用新 run 名；失败保留日志，不覆盖旧运行。也可指定 `./reproduce.sh my_run_name`。

代码同时保留 3/10-shot 的最终配置及只读输入索引，可用 `ENHANCEROUTER_SHOT=3 ./reproduce.sh` 或 `ENHANCEROUTER_SHOT=10 ./reproduce.sh` 执行相应 15 组；首次验收范围为 5-shot。

## 实际重跑范围

重新拟合已选定 Router 评分器，并以相同划分 seed 从随机初始化训练 Joint/PURF V1。复用已冻结的专家 logits、GNN embeddings、LLM hidden、路由训练特征、校准和接受策略；不重新训练专家，也不重复历史超参数搜索。保存新 Router 模型和 Joint checkpoint，并逐元素比对历史 Joint 权重、validation 和全节点预测；评分器数值容差 `1e-10`，路由决策、Joint 权重及预测要求精确相等。

每个数据集和 seed 使用历史 **10-shot Query 的固定 1,000 个节点及其顺序**。3/5/10-shot 共用同一列表，完全不重新抽样。Accuracy、Macro-F1 逐 seed 对照 EALR-GTC 已确认的 `standard_10shot_eval1000` Full V1 表格；汇总标准差为总体标准差 `ddof=0`。

当前任务包含历史 query-observed Router 配方来源，迁移和重跑不消除该来源历史。

## 文件和来源

- `src/enhancerouter/`：独立的输入认证、Router refit/推理、训练/评估调度。数学内核从 EALR-GTC 复制；来源与变换见 `provenance/copied-sources.json`。
- `src/eargtc/`：复制的原始数值模块及其依赖闭包，保留原模块名以兼容冻结模型。执行时不从 EALR-GTC 导入代码。
- `configs/inputs.json`：45 组最终配置、只读输入路径、源 SHA256 以及固定评估节点。数据不复制到本工程。
- `provenance/source.json`：EALR-GTC 的 commit 信息。迁移时源目录没有 Git 仓库，真实 commit ID 不可取得，明确记录 `null` 和原因，以源码内容指纹追踪来源；不在 EALR-GTC 初始化 Git。
- 每次运行保存 `config.json`、`environment.json`（Python、依赖、Conda、线程、Slurm、代码 commit）、`frozen-inputs.json` 和 `frozen-files.md5`。每组还保存完整 Router 配方/Joint 配置、模型、预测及诊断。
- 每个冻结依赖记录原路径、解析路径、MD5、SHA256、大小和修改时间。运行前认证、结束后再次核查；大型 checkpoint/预测保留在本工程的 `runs/` 中，不纳入 Git。
- 新实验代码用本工程 Git 分支/commit 管理，结果写入新的 `runs/` 目录。禁止写入 EALR-GTC。

训练进程设置独立缓存路径、禁止生成 pyc，并对 Python 文件写入施加本工程路径边界。评估必须等待全部 15 组模型完成并封存后，才读取图真值。

## 已完成验收

2026-10-22 的 5-shot 验收已通过，覆盖五数据集及 seeds 42/43/44 共 15 组。验收任务为 prepare `118449`、array `118450`、evaluate `118451`；evaluate 以 ExitCode 0:0 完成。

| Dataset | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| Cora | 80.70 ± 1.91 | 78.70 ± 2.26 |
| CiteSeer | 72.60 ± 0.79 | 68.74 ± 0.68 |
| PubMed | 91.47 ± 1.30 | 90.48 ± 1.60 |
| ogbn-arxiv | 60.40 ± 1.56 | 42.50 ± 2.36 |
| ogbn-products subset | 72.43 ± 0.41 | 28.86 ± 1.19 |

以上为三个 seed 的均值 ± 总体标准差。15 组逐 seed Accuracy 和 Macro-F1 与 EALR-GTC 差异均为零；新训练 Joint 权重、validation 和全节点概率逐元素精确相等，Router 决策一致。269 个冻结依赖的运行前后 MD5 一致，迁移源码的原文件 MD5 也保持一致。

完整输出：`runs/acceptance-5shot-20261022-v2/`。Git 管理的轻量验收材料：`reports/acceptance-5shot-20261022/`，包含报告、逐 seed 表、完整配置、环境、冻结输入路径与 MD5、每组诊断和完整输出路径。训练所用代码 commit 记录在报告中；后续仅增加验收材料的 commit 不改变数值代码。
