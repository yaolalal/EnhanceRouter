# 合成伪标签发布结果独立核验

继承根 AGENTS.md。只对 synthetic_pseudo_20261031 的已完成发布结果做核验，不修改训练、池、模型、历史输出或冻结代码。数值核验通过 Slurm。核验脚本与训练代码分目录，不改变在运行代码的登记哈希。结果写入新的 reports/synthetic_pseudo_audit_20261031，失败保留现场。
