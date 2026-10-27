# 评估名单及划分核查

五个split seeds为42–46，共75格。每个dataset/seed使用自己的有序eval1000，跨3/5/10-shot逐项逐序一致，与该seed三个shot的support和valid均无重叠。跨seed重叠允许并单独记录。

SPLIT_AUDIT.json记录全部75格及历史回放/嵌套检查。EVALUATION_IDS.json保留25份有序节点ID及来源，包含按用户授权补建的45/46名单；不能把新补建的名单标作原有历史名单。

GNN_TRAINING_SEEDS.json区分split seeds与原专家模型内部RNG，后者原样保留。以上均为既有已通过记录的副本，不重新抽取节点，也不读取新实验测试结果。
