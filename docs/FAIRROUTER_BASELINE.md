# FairRouter 基线与可审阅证据

FairRouter 是本工程对 `Router-v10-MS-Original + Joint/PURF V1` 的统一名称，实验中的 B0 指这套基线。`src/fairrouter` 直接导出已核验的实现，不复制新数学模块。

分类residual实现位于 `src/eargtc/joint_explore/joint_purf_v10_v1/model.py` 的 `UnifiedRepair`，由同目录 `training.py` 训练，历史复现调用见 `src/enhancerouter/cli.py:119`。实际概率为 `(alpha*pG+beta*pL+pR)/(alpha+beta+1)`；alpha/beta是经softplus的两个全局可训练标量，pR是双侧embedding进入MLP得到的概率分类头。实现定位、原checkpoint系数曲线与固定权重推理拆分见[residual核查](../reports/residual-audit-20261028/README.md)。

## 已复现的大表

历史三种 shot、seeds42/43/44，共45格已经逐节点预测、Joint权重及指标核对一致。结果见 `reports/fairrouter-baseline/REPORT.md`；`RESULTS.json` 保存逐格数据，`COMPLETE.json` 保存原报告及配置哈希。

原运行目录：

- `runs/stage0-b0-historical-3shot-20261023`
- `runs/acceptance-5shot-20261022-v2`
- `runs/stage0-b0-historical-10shot-20261023`

入口 `python -m fairrouter` 与 `python -m enhancerouter.cli` 指向完全同一个 `main`。prepare/cell/evaluate 均须通过Slurm，保存历史逐格配置。完整复现是从已选配方重训并核对，不是重新发明一遍历史搜索；尤其不得把5-shot不同分支统一替换为3/10-shot的三折校准。

## 新的五seed诊断

只使用42–46；每个dataset/seed自己的有序eval1000跨3/5/10-shot共用，与同seed所有support/valid无交集。历史55格B0（含5-shot45/46）精确重放；缺少的20格另行执行同seed配方重拟合及原生验证集搜索，二者来源分开登记。

输入特征、冻结专家、图/文本增强、训练矩阵、标签边界、候选族、超参数选择、校准和下游模型应逐项审查，不能用同名函数或提交成功代替数值等价性证据。source-only等头在D1可能根本不使用目标样本，结果不随D1比例变化属于训练组合本身。

2026-10-27用户确认：D1特征模式、训练组合、模型族固定为B0，仅允许按验证集重新选择数值超参数和拟合校准。既有四档D1是“输入变换、超参和校准全部冻结”的独立版本；不得改名为“已重新调参版”。

新增入口 `experiments/stage0_20261027/d1_numeric.py` 将四档输出保存在独立 `d1_numeric` 目录。各档复用同一套训练、原生OOF候选比较、校准和推理代码，完整B0输入变换继续固定。只从对应B0引擎的原生候选表中过滤出结构完全一致的数值候选，不把v6等旧模型转换成tune引擎。保留每格原来的校准分支；5-shot非crossfit分支不会因换数值超参而被替换成统一三折校准。

Joint也执行原来的四阶段自适应验证集搜索，复用`native_joint_search.run`及同一训练、checkpoint选择和推理函数。仅开放原候选中的`dim/cross/lr/gate_lr/warmup/aux/keep`数值项；`gate_gold`及其他非搜索项固定B0。没有用“固定一个Joint配置只重训一次”替代原来的模型选择流程。

四个比例均重新运行这套受控流程，100%为D1内部对照。由于允许重新调参和校准，100%的输出可能与原历史大表不同；不得把这种变化当作数据减少造成的变化，也不得宣称重新选择后的模型仍逐预测复现历史大表。新的候选选择只读validation。注册、试运行和完成检查分别保存，代码完成不等于试运行已通过。

## 原始历史选择的限制

5-shot历史 `CHOICES.json` 标记 `selection_uses_query_truth=true`，这一既有来源事实已保留在 `EXACT_FIVE_SHOT_REGISTRATION.json`。准确复现这些历史结果不等于证明历史选择没有偏差。新的搜索只允许使用各自validation；结论须披露继承历史配方的事实。

## 已发现的D3边界

小验证集 Citeseer/3-shot/seed43 共18个一致节点、0个分歧节点。原Joint验证集二分函数不接受空分支。`experiments/stage0_20261027/d3_recovery.py` 对空分支返回空划分，对单节点分支只放入selection；不重复标签，不从Query补节点。其他74格的划分与原函数逐项完全相同，测试证据在运行目录 `D3_SMALL_VALIDATION_EXTENSION_TEST.json`。此扩展仅处理D3边界，不修改冻结的旧代码。

## 审阅入口

- 历史数学内核：`src/enhancerouter/`、`src/eargtc/`，来源登记 `provenance/copied-sources.json`。
- 原专家/增强与新五seed入口：`experiments/stage0_20261026/frontend.py`、`small_stratified.py`、`vendor/`。
- 原生候选选择：`native_router_search.py`、`native_router_catalog.py`、`native_joint_search.py`。
- 既有冻结D1与D2：`compute.py`、`frozen_fit.py`；不代表2026-10-27新增的可重调参口径已执行。
- D3失败恢复：`experiments/stage0_20261027/`。

二进制权重、数据集、缓存与完整运行日志仍保存在服务器；GitHub审阅包含代码、文本结果、配置和哈希登记，不包含登录凭据。

## 2026-10-27 试运行与恢复状态

新D1五格的25%及100%共10份结果全部完成，作业120662统一检查通过；完整75组作业120663已启动。核验包含完成文件哈希、控制变量审计、源版本、四阶段Joint搜索及其非数值字段固定。证据位于 reports/runtime-recovery-20261027/snapshot-20261027T052120Z。

该快照的封存数量为B0 75、D3 65、D2每个arm/N各45、D4每种75。D2余下Arxiv/Products等待各自LLM补算前置条件。快照中的封存计数与最终全模型完成检查分开表述；完整D1曲线尚未完成。

## D3 已完成最终准确率汇总

D3全部75格于后续恢复中完成，120658全量哈希核验通过，120914最终准确率汇总通过。相对B0整体配对差−2.7733pp，95%CI=[−3.4480,−2.1533]。完整表及来源见 reports/completed-d3-20261027；原validation池每类不足k时保留实际数量，逐类计数已保存。L1及阶段0接纳仍未评估，阶段1未开始。

## 2026-10-27 完整结果已归档

B0、新旧D1、D2、D3、D4全部完成，总计1350份封存结果通过检查。完整逐组指标、配置、诊断、eval1000路由输出与可用最终预测见[完整结果入口](../reports/all-results-20261027/README.md)。这代表模型及可报告结果完成，L1预算和阶段0接纳判断继续暂缓。
