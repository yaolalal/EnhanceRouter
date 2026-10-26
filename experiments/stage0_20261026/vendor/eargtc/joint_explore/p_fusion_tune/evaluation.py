"""Frozen-only paired metrics and Chinese five-seed reporting."""
import csv
import itertools
import os
import numpy as np
from .protocol import *


def metrics(probability, targets):
    slurm()
    p, y = np.asarray(probability), np.asarray(targets, dtype=np.int64)
    if p.ndim != 2 or len(p) != len(y) or not len(y) or not np.isfinite(p).all():
        raise ValueError('Invalid metric inputs')
    pred = p.argmax(1)
    c = p.shape[1]
    matrix = np.bincount(y*c+pred,minlength=c*c).reshape(c,c)
    denom = matrix.sum(0)+matrix.sum(1)
    f1 = np.divide(2*matrix.diagonal(),denom,out=np.zeros(c,dtype=float),where=denom>0)
    return dict(accuracy=float((pred==y).mean()),macro_f1=float(f1.mean()),
                nll=float(-np.log(np.maximum(p[np.arange(len(y)),y].astype(float),1e-12)).mean()))


def moments(values):
    slurm()
    values = np.asarray([x for x in values if x is not None],dtype=float)
    return dict(mean=float(values.mean()) if len(values) else None,
                std_ddof0=float(values.std(ddof=0)) if len(values) else None,
                std_ddof1=float(values.std(ddof=1)) if len(values)>1 else None,
                valid_seeds=len(values))


def csv_write(path, rows):
    path = Path(path)
    with path.open('x',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def csv_read(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def exploration_row(ds, seed, trial, final_id):
    if trial['config']['seed'] != seed:
        raise ValueError('Exploration seed differs from outer')
    return dict(dataset=ds,candidate_id=trial['id'],selected=trial['id']==final_id,
                search_accuracy=trial['accuracy'],search_nll=trial['nll'],
                checkpoint_step=trial['checkpoint']['step'],elapsed_seconds=trial['elapsed_seconds'],
                **trial['config'])


def evaluate():
    from .experiment import verify_frozen
    import torch
    torch.set_num_threads(1)
    verify_frozen()  # All identities and all 25 cell completions precede graph access.
    if (RUN/'evaluation').exists():
        return audit()
    write(RUN/'EVALUATION_STARTED.json',dict(frozen_sha256=sha(RUN/'FROZEN.json')))
    out = RUN/'evaluation.staging'
    if out.exists():
        raise FileExistsError('Inspect interrupted evaluation staging before retrying')
    out.mkdir()
    from eargtc.data import _install_pyg_optional_extension_stubs
    _install_pyg_optional_extension_stubs()
    rows, exploration, choices, label_hashes = [], [], {}, {}
    old_rows = {(int(r['split_seed']),r['dataset'],r['population']):r
                for r in csv_read(SOURCE/'joint/evaluation/per_split.csv')}
    for ds in DATASETS:
        first = read(SOURCE/'splits'/f'{ds}_5shot_seed42.json')
        graph_path = first['graph_path']
        verify({graph_path:first['graph_sha256']})
        label_hashes[graph_path] = first['graph_sha256']
        graph = torch.load(graph_path,weights_only=False,map_location='cpu')
        truth = graph.y.reshape(-1).long().numpy(); del graph
        for seed in SEEDS:
            dest = cell(seed,ds)
            split = read(SOURCE/'splits'/f'{ds}_5shot_seed{seed}.json')
            assert split['graph_path'] == graph_path and split['graph_sha256'] == first['graph_sha256']
            saved = load(dest/'predictions.pt')
            old = load(SOURCE/f'seed{seed}/joint/r9_new/{ds}/predictions.pt')
            router = load(SOURCE/f'seed{seed}/router/r9_new/{ds}/predictions.pt')
            assert torch.equal(saved['node_ids'],old['node_ids'])
            selection = read(dest/'selection.json')
            choices[f'{seed}/{ds}'] = {k:selection[k] for k in ('final_id','final_config','stages','matched_gold_id','no_kd_id')}
            for trial in selection['trials']:
                exploration.append(exploration_row(ds, seed, trial, selection['final_id']))
            for population,key in (('full_query','evaluation_full_ids'),('eval1000','evaluation_1000_ids')):
                ids = np.array(split[key],dtype=np.int64)
                accepted = router['accepted'][ids].numpy()
                pseudo_ids = ids[accepted]
                pseudo_accuracy = float((router['selected_pred'][pseudo_ids].numpy()==truth[pseudo_ids]).mean()) if len(pseudo_ids) else None
                row = dict(dataset=ds,seed=seed,population=population,query_count=len(ids),
                    accepted_count=len(pseudo_ids),pseudo_accuracy=pseudo_accuracy,pseudo_coverage=len(pseudo_ids)/len(ids))
                models = {k:saved[k] for k in ('fixed','p_fusion','gold_only','initial_fusion','previous_fusion','no_kd')}
                models['historical_ensemble'] = old['families']['optimized']['ensemble']
                models.update({'stage_'+k:v for k,v in saved['stages'].items()})
                for name,probs in models.items():
                    row.update({name+'_'+k:v for k,v in metrics(probs[ids].numpy(),truth[ids]).items()})
                for metric in ('accuracy','macro_f1'):
                    row['gain_'+metric+'_pp'] = 100*(row['p_fusion_'+metric]-row['fixed_'+metric])
                    row['gain_vs_historical_'+metric+'_pp'] = 100*(row['p_fusion_'+metric]-row['historical_ensemble_'+metric])
                    row['gain_vs_gold_'+metric+'_pp'] = 100*(row['p_fusion_'+metric]-row['gold_only_'+metric])
                    row['gain_vs_previous_'+metric+'_pp'] = 100*(row['p_fusion_'+metric]-row['previous_fusion_'+metric])
                    row['gain_vs_no_kd_'+metric+'_pp'] = 100*(row['p_fusion_'+metric]-row['no_kd_'+metric])
                reference = old_rows[(seed,ds,population)]
                for metric in ('accuracy','macro_f1'):
                    assert abs(row['historical_ensemble_'+metric]-float(reference[metric])) < 1e-10
                assert row['accepted_count']==int(reference['accepted_count'])
                assert abs(row['pseudo_coverage']-float(reference['pseudo_coverage']))<1e-12
                if pseudo_accuracy is not None:
                    assert abs(pseudo_accuracy-float(reference['pseudo_accuracy']))<1e-12
                rows.append(row)
    summaries = []
    metric_keys = [k for k in rows[0] if k not in ('dataset','seed','population','query_count','accepted_count')]
    for ds,population in itertools.product(DATASETS,('full_query','eval1000')):
        group = [r for r in rows if r['dataset']==ds and r['population']==population]
        assert len(group)==5
        for key in metric_keys:
            summaries.append(dict(dataset=ds,population=population,metric=key,**moments([r[key] for r in group])))
    csv_write(out/'per_seed.csv',rows); csv_write(out/'mean_std.csv',summaries)
    csv_write(out/'exploration.csv',exploration)
    for population in ('full_query','eval1000'):
        csv_write(out/f'{population}_per_seed.csv',[r for r in rows if r['population']==population])
        csv_write(out/f'{population}_mean_std.csv',[r for r in summaries if r['population']==population])
    write(out/'selected_configs.json',choices)
    write(out/'metrics.json',dict(rows=rows,summaries=summaries,label_hashes=label_hashes,
                                 frozen_sha256=sha(RUN/'FROZEN.json')))
    report(out,rows,summaries,exploration)
    out.rename(RUN/'evaluation')
    return audit()


def report(out,rows,summaries,exploration):
    lookup = {(r['dataset'],r['population'],r['metric']):r for r in summaries}
    def summary(ds,pop,k,scale=100):
        r=lookup[ds,pop,k]
        return '未定义' if r['mean'] is None else f"{scale*r['mean']:.2f} ± {scale*r['std_ddof0']:.2f}"
    lines=['# Router-v9-new：Joint-Fixed 与蒸馏/非对称降维 p_fusion 五种子调优','',
        '新模型分别将冻结 GNN/LLM embedding 降维，拼接后用单分类头 softmax 输出 p_fusion。最终概率不加入专家或 Router 概率。', '',
        '主表两种模型均为训练 seed = 历史划分 seed（42–46），无内层 seeds、无概率集成。Joint-Fixed 使用原结构、原九配方及原 validation 选择；42–46 全部复用上一轮匹配 seed 的已认证成员，并按原 validation 规则重放，逐节点验证与旧 Fixed 一致。历史内层42/43/44集成结果单列参考。', '',
        '所有数值工作通过 Slurm。每个 outer 独立搜索；gold/support、valid、full-query、eval1000 IDs 与历史完全一致。原 Joint 模型选择 validation 再按不读取标签的稳定哈希分为 checkpoint/search 两半；threshold 半不参与。checkpoint 按 Accuracy、NLL、最早步数选取；方案按 search Accuracy、NLL、稳定 ID 排序。新模型和旧 Fixed 的搜索预算、validation 用法不同，差值不能解释为单一结构改动的因果贡献。','',
        '有序搜索：软教师/蒸馏权重/温度 → GNN32/64与LLM64/128/256非对称宽度 → dropout/weight decay/标签平滑/A-D采样 → 学习率/投影学习率倍率/batch/步数/监督及蒸馏权重复查；每阶段保留 incumbent。全部25单元的方案、checkpoint和输出冻结并重放验证后才打开 query 真值。最优指本次有限搜索的 validation 最优；没有按测试成绩换配置。', '',
        'Router-v9-new 在历史上由 posthoc query metrics 选择；本轮固定使用该部署，不改变 Router。已有测试结果也曾被观察，因此这是既有基准复评，不能声称未观察测试集。full-query/eval1000 共享冻结模型；eval1000 为 query 报告子集，未从伪标签池剔除，属于传导式伪监督实验。', '',
        '均值 ± 总体标准差（ddof=0）；CSV同时保留样本标准差（ddof=1）。逐 seed 增益单位为百分点。五种子共用图且节点重叠，std 描述划分和初始化波动，并非独立样本置信区间。Macro-F1 使用完整类别空间，缺失类别按0计。伪标签准确率只在各评价集合的接受子集计算，覆盖率分母为对应完整评价集合。','']
    lines += ['蒸馏损失为 gold CE + scheduled λ_pseudo CE_pseudo + scheduled λ_KD T² KL(q_T || p_fusion,T)。仅 accepted query 节点参与后两项；teacher 先分别温度缩放，再按 Router 选择（agreement 取专家均值），或全部取专家均值。教师分布完全 detach，不进入最终输出。匹配 no-KD 对照只关闭蒸馏，gold-only 关闭全部伪监督；两项对照在选优后训练，不参与最终配置选择。', '', '本轮从默认配置及上一轮 validation 胜者重新训练，无权重续训。最多46个搜索成员与2个匹配对照/单元，去重后不超过1200个训练。旧测试成绩已观察，因此本轮属于后续适应性基准探索。', '']
    for pop,label in (('full_query','完整 query'),('eval1000','eval1000')):
        lines += [f'## {label}：五种子均值与标准差','',
            '| 数据集 | Fixed Acc | p_fusion Acc | ΔAcc pp | Fixed Macro-F1 | p_fusion Macro-F1 | ΔF1 pp | 伪标签Acc | 覆盖率 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for ds in DATASETS:
            keys=('fixed_accuracy','p_fusion_accuracy','gain_accuracy_pp','fixed_macro_f1','p_fusion_macro_f1','gain_macro_f1_pp','pseudo_accuracy','pseudo_coverage')
            lines.append('| '+ds+' | '+' | '.join(summary(ds,pop,k,1 if 'gain_' in k else 100) for k in keys)+' |')
        lines += ['',f'## {label}：逐 seed 配对结果','',
            '| 数据集 | seed | Fixed Acc | p_fusion Acc | ΔAcc pp | Fixed F1 | p_fusion F1 | ΔF1 pp | 伪标签Acc | 覆盖率 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in [r for r in rows if r['population']==pop]:
            keys=('fixed_accuracy','p_fusion_accuracy','gain_accuracy_pp','fixed_macro_f1','p_fusion_macro_f1','gain_macro_f1_pp','pseudo_accuracy','pseudo_coverage')
            values=['未定义' if r[k] is None else f"{r[k]*(1 if 'gain_' in k else 100):.2f}" for k in keys]
            lines.append(f"| {r['dataset']} | {r['seed']} | "+' | '.join(values)+' |')
        lines += ['',f'## {label}：历史集成与 gold-only 参考','',
            '| 数据集 | 历史集成Acc | 历史集成F1 | 新模型Δ历史Acc pp | 匹配gold-only Acc | 匹配gold-only F1 | 新模型Δgold Acc pp |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for ds in DATASETS:
            keys=('historical_ensemble_accuracy','historical_ensemble_macro_f1','gain_vs_historical_accuracy_pp','gold_only_accuracy','gold_only_macro_f1','gain_vs_gold_accuracy_pp')
            lines.append('| '+ds+' | '+' | '.join(summary(ds,pop,k,1 if 'gain_' in k else 100) for k in keys)+' |')
        lines += ['',f'## {label}：相对上轮与匹配关闭蒸馏对照','',
            '| 数据集 | 上轮Acc | 本轮Acc | Δ上轮Acc pp | 上轮F1 | 本轮F1 | Δ上轮F1 pp | no-KD Acc | no-KD F1 | Δno-KD Acc pp | Δno-KD F1 pp |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for ds in DATASETS:
            keys=('previous_fusion_accuracy','p_fusion_accuracy','gain_vs_previous_accuracy_pp',
                  'previous_fusion_macro_f1','p_fusion_macro_f1','gain_vs_previous_macro_f1_pp',
                  'no_kd_accuracy','no_kd_macro_f1','gain_vs_no_kd_accuracy_pp','gain_vs_no_kd_macro_f1_pp')
            lines.append('| '+ds+' | '+' | '.join(summary(ds,pop,k,1 if 'gain_' in k else 100) for k in keys)+' |')
    lines += ['', '## 搜索记录与解释', '', f'共保存 {len(exploration)} 个去重训练成员（含最终匹配 no-KD、gold-only 对照），每成员仅一个与 outer 相同的训练 seed。', '',
        '| 数据集 | seed | 基础融合search Acc | 训练阶段 | 表示阶段 | 正则阶段 | 优化阶段 | 最终配置ID |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for ds,seed in itertools.product(DATASETS,SEEDS):
        s=read(cell(seed,ds)/'selection.json'); trials={r['id']:r for r in s['trials']}
        values=[trials[s['base_id']]['accuracy']]+[s['stages'][stage]['accuracy'] for stage in ('training','representation','regularization','optimization')]
        lines.append(f'| {ds} | {seed} | '+' | '.join(f'{100*v:.2f}' for v in values)+f" | {s['final_id']} |")
    lines += ['', '阶段提升是给定前序 incumbent 的条件比较；相互作用未穷举，validation 改善不保证 query 改善。逐候选完整配置、步数、耗时和 validation 分数见 exploration.csv，最终配置见 selected_configs.json。阶段冻结预测的两种评价口径指标也保留在 per_seed.csv。', '',
        '代码：eargtc/joint_explore/p_fusion_tune/；运行：scripts/joint_explore/p_fusion_tune/run.py。REGISTRATION.json、INPUTS.json、COMPLETE.json、FROZEN.json 与 FINAL_RECEIPT.json 记录来源和完整性。旧 Joint/Router、历史划分及保留归档未覆盖。上游原始数据、专家和 hidden caches 仍是外部哈希绑定依赖，本目录不是独立完整数据包。','']
    (out/'REPORT.md').write_text('\n'.join(lines))


def audit():
    """Independently recompute final paired rows with sklearn after frozen evaluation."""
    from .experiment import verify_frozen
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from eargtc.data import _install_pyg_optional_extension_stubs
    verify_frozen(); _install_pyg_optional_extension_stubs()
    out=RUN/'evaluation'; payload=read(out/'metrics.json')
    verify(payload['label_hashes'])
    rows=payload['rows']
    assert len(rows)==50 and len({(r['seed'],r['dataset'],r['population']) for r in rows})==50
    for ds in DATASETS:
        split0=read(SOURCE/'splits'/f'{ds}_5shot_seed42.json')
        graph=torch.load(split0['graph_path'],weights_only=False,map_location='cpu')
        truth=graph.y.reshape(-1).numpy();del graph
        for seed in SEEDS:
            split=read(SOURCE/'splits'/f'{ds}_5shot_seed{seed}.json')
            pred=load(cell(seed,ds)/'predictions.pt')
            for r in [r for r in rows if r['dataset']==ds and r['seed']==seed]:
                ids=np.array(split['evaluation_full_ids' if r['population']=='full_query' else 'evaluation_1000_ids'])
                for name in ('fixed','p_fusion','gold_only','initial_fusion','previous_fusion','no_kd'):
                    p=pred[name][ids].numpy(); labels=p.argmax(1)
                    a=accuracy_score(truth[ids],labels)
                    f=f1_score(truth[ids],labels,labels=np.arange(p.shape[1]),average='macro',zero_division=0)
                    assert abs(a-r[name+'_accuracy'])<1e-12 and abs(f-r[name+'_macro_f1'])<1e-12
                for stage,p in pred['stages'].items():
                    a=accuracy_score(truth[ids],p[ids].argmax(1).numpy())
                    f=f1_score(truth[ids],p[ids].argmax(1).numpy(),labels=np.arange(p.shape[1]),average='macro',zero_division=0)
                    assert abs(a-r['stage_'+stage+'_accuracy'])<1e-12 and abs(f-r['stage_'+stage+'_macro_f1'])<1e-12
                router=load(SOURCE/f'seed{seed}/router/r9_new/{ds}/predictions.pt')
                accepted_ids=ids[router['accepted'][ids].numpy()]
                assert len(accepted_ids)==r['accepted_count'] and len(ids)==r['query_count']
                assert abs(len(accepted_ids)/len(ids)-r['pseudo_coverage'])<1e-12
                if len(accepted_ids):
                    assert abs(accuracy_score(truth[accepted_ids],router['selected_pred'][accepted_ids].numpy())-r['pseudo_accuracy'])<1e-12
                for metric in ('accuracy','macro_f1'):
                    for reference in ('gold','previous','no_kd'):
                        model={'gold':'gold_only','previous':'previous_fusion','no_kd':'no_kd'}[reference]
                        assert abs(r['gain_vs_'+reference+'_'+metric+'_pp']-100*(r['p_fusion_'+metric]-r[model+'_'+metric]))<1e-12
                    assert abs(r['gain_'+metric+'_pp']-100*(r['p_fusion_'+metric]-r['fixed_'+metric]))<1e-12
    for r in payload['summaries']:
        vals=[x[r['metric']] for x in rows if x['dataset']==r['dataset'] and x['population']==r['population']]
        expected=moments(vals)
        assert all(r[k]==v for k,v in expected.items())
    def check_csv(path,expected):
        actual=csv_read(path)
        serialized=[{k:('' if v is None else str(v)) for k,v in row.items()} for row in expected]
        assert actual==serialized, 'CSV differs from independently audited payload: '+str(path)
    check_csv(out/'per_seed.csv',rows);check_csv(out/'mean_std.csv',payload['summaries'])
    for population in ('full_query','eval1000'):
        check_csv(out/f'{population}_per_seed.csv',[r for r in rows if r['population']==population])
        check_csv(out/f'{population}_mean_std.csv',[r for r in payload['summaries'] if r['population']==population])
    expected_exploration=[]
    for ds in DATASETS:
        for seed in SEEDS:
            pick=read(cell(seed,ds)/'selection.json')
            expected_exploration += [exploration_row(ds,seed,t,pick['final_id']) for t in pick['trials']]
    check_csv(out/'exploration.csv',expected_exploration)
    reg=read(RUN/'REGISTRATION.json'); verify(reg['preserved_hashes']);verify(reg['code_hashes']);verify(reg['previous_hashes'])
    files={str(p):sha(p) for p in out.iterdir() if p.is_file()}
    receipt=dict(all_checks_pass=True,cells=25,metric_rows=50,inner_seeds=[],training_seed_equals_split_seed=True,
                 full_query_and_eval1000=True,independent_sklearn_recomputation=True,
                 preserved_files_verified=len(reg['preserved_hashes']),files=files,
                 frozen_sha256=sha(RUN/'FROZEN.json'))
    write(RUN/'FINAL_RECEIPT.json',receipt)
    target=RUN/'REPORT.md'
    if target.exists():
        assert target.read_bytes()==(out/'REPORT.md').read_bytes()
    else:
        target.write_bytes((out/'REPORT.md').read_bytes())
    print('FINAL AUDIT PASSED',25,'cells',50,'rows',flush=True)
    return receipt
