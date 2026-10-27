# 2026-10-27 追加工作约束

用户本轮明确要求代码同步GitHub，将已完成大表的流程命名FairRouter并作为B0，审查D1控制变量，并处理D1–D4的运行失败。旧L1预算口径争议暂时搁置，不得替用户选口径。

所有数值工作仍使用Slurm；EALR-GTC只读。旧stage0_20261026的624文件代码冻结保持原样；新恢复/审查代码留在本目录，记录自己的哈希。

既有D1是全冻结版，不能当作重新搜索和校准后的版本。用户已确认：新D1特征模式、source/target/pooled训练组合、模型族固定为B0，仅重新选择数值超参数及拟合校准。完整B0输入变换继续固定；原生内部随机种子和ensemble成员不变；Joint执行原四阶段验证集搜索，仅开放原生数值搜索项，gate_gold及其他非搜索项固定B0；原生训练、验证checkpoint选择与推理不变。新结果保存为d1_numeric，旧D1单独保留。评估真标签不得参与搜索或选择。

用户已给出GitHub目标https://github.com/yaolalal/EnhanceRouter（用户创建的public仓库）。不得上传到无关仓库，不自行授予连接器账号aubphlex权限。用户密码和其他凭据不得加入代码、日志或上传内容。

用户已明确授权：配额耗尽时将整个工程迁至/data1/wujingyao-20354/wujingyao/EnhanceRouter，逐文件校验后用软链接保留原/home/.../EnhanceRouter路径。EALR-GTC原目录只读，不迁移或修改。
