"""Export sealed numeric results and textual training evidence; no model edits."""
from factorial_joint import *
import shutil,tarfile

def main():
    registered();audit=read(OUT/'INDEPENDENT_AUDIT_PASS.json');assert audit['all_checks_pass'] and audit['models']==525
    check_seal(OUT/'report');check_seal(OUT/'associations');check_seal(OUT/'scope-supplement')
    assert audit['report_seal_sha256']==digest(OUT/'report/COMPLETE.json')
    assert read(OUT/'scope-supplement/INPUTS.json')['report_seal_sha256']==audit['report_seal_sha256']
    dest=ROOT/'reports/joint-factorial-3x2-20261029';assert not dest.exists();dest.mkdir()
    for source,target in [('report','report'),('associations','associations'),('scope-supplement','scope-supplement'),('metrics','cells'),('independent-audit','independent-audit'),('training-records','training-records'),('logs','logs')]:
        shutil.copytree(OUT/source,dest/target)
    proof=dest/'proofs';proof.mkdir()
    for name in ('REGISTRATION.json','REGISTRATION_FAILURE_RECOVERY_V2.json','JOB_GRAPH_V2.json','AUDIT_JOB_GRAPH.json','SCOPE_JOB.json','PILOT_PASS.json','ALL_TRAINING_PASS.json','INDEPENDENT_AUDIT_PASS.json'):
        shutil.copy2(OUT/name,proof/name)
    for path in sorted((OUT/'models').glob('*/*/*/COMPLETE.json')):
        base=path.parent;target=dest/'training-selection'/base.relative_to(OUT/'models');target.mkdir(parents=True)
        shutil.copy2(path,target/'SERVER_COMPLETE.json')
        for name in ('INPUTS_BEFORE_TRAINING.json','config.json','joint-selection.json'):
            shutil.copy2(base/name,target/name)
        if (base/'joint-search').exists():
            for p in sorted((base/'joint-search').glob('stage*.json')):shutil.copy2(p,target/p.name)
    files={str(p.relative_to(dest)):digest(p) for p in dest.rglob('*') if p.is_file()}
    write(dest/'NUMERIC_EXPORT_SHA256.json',files)
    readme='''# 3×2 Joint调参混杂核查与纠正性指标

[完整七设置主报告](report/REPORT.md) · [Fig10分歧区及有效分母补充](scope-supplement/REPORT.md) · [原范围相关及置信区间](associations/REPORT.md)

全部525份分类结果完成：B0及{D1 numeric25%, D2随机a1000, D4 confidence_router}×{Joint超参固定B0, 原生数值重调}；五数据集、3/5/10-shot、seeds42–46。新增300份训练，复用225份已冻结模型。16个试运行、全量训练、指标与独立checkpoint预测复核均通过。固定超参仍重训权重。每个Router处理内的两组Joint拥有完全相同的Router输出和训练伪标签池。

主纠正性精度只看已接受伪标签 != argmax(B0固定pGL) 的节点。α/β采用B0最终选中checkpoint；各模型自身pGL的结果另列，不能混为同一参照。空纠正集合的精度是null，务必结合scope-supplement中的有定义配置数与纠正节点数。净准确率纠正=原错变对−原对变错；纠正标签正误差=正确标签−错误标签（错误含错转错），两者分列。

Router和伪标签在完整Query评估，D2剔除新增训练节点并匹配B0同一名单；最终分类在原顺序eval1000。Fig10讨论分歧区，因此分歧区实际接受精度和A+D全部接受精度同时给出。原范围报告保留，不能混用两种精度。Joint仍接收完整原support gold；D2新增真标签仅用于Router。

每种Joint重调沿用原生四阶段验证集搜索，开放dim/cross/lr/gate_lr/warmup/aux/keep，其余设置固定B0。重调的收益不能单独归因cross；离线相关不能证明因果机制，不显著也不能证明不相关。旧预算争议与阶段1仍暂缓，定时监控保持暂停。

- report/per_cell.csv、RESULTS.json：525份模型的指标、完整配置、实际分母与来源。
- scope-supplement：分歧区对照、Fig10范围、同样本相关比较及簇重采样区间。
- associations：三张配对差值散点图、两种相关系数及不确定性。
- cells：每个划分的七设置指标和eval1000逐节点证据。
- training-records：525份最终入选模型的配置、checkpoint选择及原训练history。
- training-selection：300份新训练的输入登记、全部候选验证指标、四阶段计划。SERVER_COMPLETE.json是服务器原封存记录，里面的二进制哈希不表示二进制已上传。
- independent-audit：从525份checkpoint重新推理eval1000，标签与原保存预测逐节点完全相同；概率容差2e-6；检查训练池一致、固定超参、指标恒等式及汇总值。
- proofs、logs：登记、作业依赖、失败路径修复、试运行和完成核验证据。

模型权重、候选checkpoint、完整图/特征缓存和全Query二进制预测仍在服务器；全部可报告的文本指标、候选选择、训练历史、逐节点评估结果、图片和核验证据包含在本目录及Release压缩包中。来源路径和哈希随记录保存。

代码：[实验入口](../../experiments/stage0_20261029/factorial_joint.py)、[独立复核](../../experiments/stage0_20261029/audit_factorial.py)、[分歧区补充](../../experiments/stage0_20261029/scope_analysis.py)、[协议](../../docs/JOINT_FACTORIAL_3X2.md)。
'''
    (dest/'README.md').write_text(readme)
    write(dest/'PUBLICATION_SHA256.json',{str(p.relative_to(dest)):digest(p) for p in sorted(dest.rglob('*')) if p.is_file()})
    asset=ROOT/'runs/joint-factorial-3x2-20261029-results.tar.gz';assert not asset.exists()
    with tarfile.open(asset,'w:gz') as t:
        t.add(dest,arcname=str(dest.relative_to(ROOT)))
        t.add(THIS,arcname=str(THIS.relative_to(ROOT)))
        t.add(ROOT/'docs/JOINT_FACTORIAL_3X2.md',arcname='docs/JOINT_FACTORIAL_3X2.md')
    receipt=dict(all_checks_pass=True,publication=str(dest),files=len(list(dest.rglob('*'))),archive=str(asset),archive_sha256=digest(asset),archive_bytes=asset.stat().st_size,publisher_sha256=digest(__file__),audit_pass_sha256=digest(OUT/'INDEPENDENT_AUDIT_PASS.json'))
    write(OUT/'PUBLICATION.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
