"""Replay audit and locked evaluation. Only evaluate() loads query truth."""
import csv
import gc
import os
from pathlib import Path
import joblib
import numpy as np
import torch
from .artifacts import DATASETS, REPO, read_json, write_json, sha256, verify_record, require_slurm, verify_preservation
from .policy import FRACTIONS, top_mask, summarize, selection_metrics, select_policy

SAFE = REPO/'runs/joint_direct_numbered_router_export_20260813_v1'


def audit_pipeline(query, scores, pipeline, selection):
    qallowed = {'dataset','ids','tab','g','l','gnn_pred','llm_pred'}
    sallowed = {'node_ids','gnn_pred','llm_pred','selected_score','baselines'}
    pallowed = {'schema','dataset','node_ids','gnn_pred','llm_pred','selected_score','accepted_mask','pseudo_label'}
    if not set(query).issubset(qallowed) or set(scores) != sallowed or set(pipeline) != pallowed:
        raise ValueError('unexpected delivery/query fields; truth-derived fields are forbidden')
    if pipeline['schema'] != 'agreement_increment_pipeline_v1' or pipeline['dataset'] != selection['dataset']:
        raise ValueError('delivery schema/dataset mismatch')
    ids = query['ids']
    if ids.dtype != torch.int64 or len(torch.unique(ids)) != len(ids):
        raise ValueError('IDs are not unique int64')
    if not torch.equal(query['gnn_pred'],query['llm_pred']):
        raise ValueError('non-agreement rows')
    for key, qkey in [('node_ids','ids'),('gnn_pred','gnn_pred'),('llm_pred','llm_pred')]:
        if not torch.equal(scores[key],query[qkey]) or not torch.equal(pipeline[key],query[qkey]):
            raise ValueError('node/prediction identity mismatch')
    forbidden = set(selection['calibration_ids']) | set(selection['policy_ids'])
    if set(ids.tolist()) & forbidden:
        raise ValueError('query overlaps model/policy supervision')
    score = scores['selected_score']
    if score.shape != ids.shape or not torch.isfinite(score).all() or ((score<0)|(score>1)).any():
        raise ValueError('invalid selected score')
    if not torch.equal(pipeline['selected_score'],score):
        raise ValueError('delivery scores changed')
    expected = torch.as_tensor(top_mask(score.numpy(),ids.numpy(),selection['policy']['fraction']))
    if pipeline['accepted_mask'].dtype != torch.bool or not torch.equal(expected,pipeline['accepted_mask']):
        raise ValueError('acceptance differs from locked policy')
    if not torch.equal(torch.where(expected,query['gnn_pred'],-1),pipeline['pseudo_label']):
        raise ValueError('pseudo-label is not accepted consensus prediction')
    if selection['accepted_count'] != int(expected.sum()):
        raise ValueError('accepted count mismatch')
    for score in scores['baselines'].values():
        if score.shape != ids.shape or not torch.isfinite(score).all():
            raise ValueError('invalid baseline scores')
    return dict(population=len(ids), accepted=int(expected.sum()), label_free=True)


def audit(root):
    require_slurm(); root = Path(root).resolve()
    frozen = read_json(root/'FROZEN.json'); verify_record(frozen)
    from .data import load_bank
    from .models import predict_model, fit_predict, candidate_configs
    records = {}
    for dataset in DATASETS:
        dest = root/'targets'/dataset
        query = load_bank(root,dataset,'query')
        pipeline = torch.load(root/'pipeline'/f'{dataset}.pt',map_location='cpu',weights_only=True)
        scores = torch.load(dest/'scores.pt',map_location='cpu',weights_only=True)
        selection = read_json(dest/'selection.json')
        result = audit_pipeline(query,scores,pipeline,selection)
        model = joblib.load(dest/'model.joblib')
        reproduced = predict_model(model,{k:query[k].numpy() for k in ('tab','g','l')})
        maximum = replay_error(reproduced,scores['selected_score'].numpy())
        vs = torch.load(dest/'validation_scores.pt',map_location='cpu',weights_only=True)
        validation = load_bank(root,dataset,'validation')
        for saved,bank_key in [('node_ids','ids'),('target','target'),('cal_mask','cal_mask'),('policy_mask','policy_mask')]:
            if not torch.equal(vs[saved],validation[bank_key]):
                raise ValueError('saved validation identities/labels/masks differ from bank')
        if selection['calibration_ids'] != validation['ids'][validation['cal_mask']].tolist() or selection['policy_ids'] != validation['ids'][validation['policy_mask']].tolist():
            raise ValueError('selection split identities differ from bank')
        val_replay = predict_model(model,{k:validation[k].numpy() for k in ('tab','g','l')})
        result['validation_replay_error'] = replay_error(val_replay,vs['score'].numpy())
        configs = [c for c in candidate_configs() if c['family']=='baseline']
        if set(scores['baselines']) != {c['id'] for c in configs}:
            raise ValueError('baseline identities differ from preregistration')
        for config in configs:
            baseline,_ = fit_predict(config,{k:validation[k].numpy() for k in ('tab','g','l')},
                validation['target'].numpy(),{k:query[k].numpy() for k in ('tab','g','l')})
            replay_error(baseline,scores['baselines'][config['id']].numpy())
        mask = vs['policy_mask'].numpy()
        reproduced_policy = select_policy(vs['score'].numpy()[mask], vs['target'].numpy()[mask], vs['node_ids'].numpy()[mask])
        if reproduced_policy != selection['policy']: raise ValueError('policy replay differs')
        result['max_replay_error'] = maximum; records[dataset] = result
        print(f'AUDIT {dataset}: {result}', flush=True)
        del model, query, scores, pipeline, reproduced; gc.collect()
    result = dict(all_checks_pass=True, frozen_sha256=sha256(root/'FROZEN.json'),
        datasets=records, preservation=verify_preservation(root), query_truth_read=False,
        slurm_job_id=os.environ['SLURM_JOB_ID'])
    write_json(root/'PRETEST_AUDIT.json',result)
    return result


def replay_error(reproduced, expected):
    reproduced,expected=np.asarray(reproduced),np.asarray(expected)
    if reproduced.shape != expected.shape or not np.isfinite(reproduced).all() or not np.isfinite(expected).all():
        raise ValueError('replayed score shape or finite-value check failed')
    maximum=float(np.max(np.abs(reproduced-expected))) if expected.size else 0.
    if maximum>2e-6: raise ValueError(f'model replay differs: {maximum}')
    return maximum


def validate_evaluation_ready(root):
    root=Path(root)
    frozen=read_json(root/'FROZEN.json');verify_record(frozen)
    audit_record=read_json(root/'PRETEST_AUDIT.json')
    if set(frozen['datasets']) != set(DATASETS) or set(audit_record['datasets']) != set(DATASETS):
        raise ValueError('all five target endpoints must be frozen and audited')
    if not audit_record['all_checks_pass'] or audit_record['query_truth_read'] or audit_record['frozen_sha256'] != sha256(root/'FROZEN.json'):
        raise ValueError('all-target pretest audit not passing')
    return frozen


def evaluate(root):
    require_slurm(); root = Path(root).resolve()
    frozen = validate_evaluation_ready(root)
    out = root/'evaluation'; out.mkdir(exist_ok=True)
    started = dict(frozen_sha256=sha256(root/'FROZEN.json'), slurm_job_id=os.environ['SLURM_JOB_ID'])
    if (out/'COMPLETE.json').exists():
        complete = read_json(out/'COMPLETE.json')
        if complete['frozen_sha256'] != started['frozen_sha256'] or complete['results_sha256'] != sha256(out/'results.json'):
            raise ValueError('completed evaluation identity changed')
        return read_json(out/'results.json')
    if (out/'STARTED.json').exists():
        if read_json(out/'STARTED.json')['frozen_sha256'] != started['frozen_sha256']:
            raise ValueError('cannot resume evaluation with different frozen models')
        write_json(out/f"RESUMED_{os.environ['SLURM_JOB_ID']}.json",started)
    else:
        with (out/'STARTED.json').open('x') as handle:
            import json
            json.dump(started,handle)
    from eargtc.data import _install_pyg_optional_extension_stubs
    _install_pyg_optional_extension_stubs()
    results = {}; flat = []; curves = []
    for dataset in DATASETS:
        selection = read_json(root/'targets'/dataset/'selection.json')
        scores = torch.load(root/'targets'/dataset/'scores.pt',map_location='cpu',weights_only=True)
        metadata = read_json(SAFE/dataset/'safe_input_metadata.json')
        graph_path = Path(metadata['graph_path'])
        if sha256(graph_path) != metadata['graph_path_sha256']:
            raise ValueError('evaluation graph differs from historical safe-source identity')
        graph = torch.load(graph_path,map_location='cpu',weights_only=False)
        ids = scores['node_ids'].numpy(); pred = scores['gnn_pred'].numpy()
        truth = graph.y.detach().cpu().reshape(-1)[scores['node_ids']].numpy()
        correct = pred == truth
        score = scores['selected_score'].numpy()
        keep = top_mask(score,ids,selection['policy']['fraction'])
        selected = summarize(correct,keep)
        base = summarize(correct,np.ones(len(ids),bool))
        baselines = {name:summarize(correct,top_mask(value.numpy(),ids,selection['policy']['fraction']))
                     for name,value in scores['baselines'].items()}
        points = [dict(fraction=f,**summarize(correct,top_mask(score,ids,f))) for f in FRACTIONS]
        metrics = selection_metrics(score,correct,ids)
        result = dict(dataset=dataset, selected=selected, unfiltered=base, same_budget_baselines=baselines,
            fixed_fraction_curve=points, ranking_metrics=metrics,
            model_id=selection['chosen']['id'], model_config=selection['chosen']['model'],
            scope=selection['chosen']['scope'], calibration_metrics=selection['chosen']['metrics'],
            policy=selection['policy'], selected_on='calibration model + policy fraction; no test selection')
        results[dataset] = result
        flat.append(dict(dataset=dataset,model=result['model_id'], fraction=selection['policy']['fraction'],
                         policy_precision=selection['policy']['precision'],
                         unfiltered_precision=base['precision'],**selected))
        curves.extend(dict(dataset=dataset,**r) for r in points)
        # Truth-derived decisions are audit-only and are never put under pipeline/.
        torch.save(dict(node_ids=scores['node_ids'],correct=torch.as_tensor(correct),accepted_mask=torch.as_tensor(keep)),
                   out/f'{dataset}_audit_only.pt')
        write_json(out/f'{dataset}.json',result)
        print(f"RESULT {dataset}: accuracy={selected['precision']} accepted={selected['accepted']} coverage={selected['coverage']}",flush=True)
        del graph,truth,scores; gc.collect()
    write_json(out/'results.json',results)
    for filename,rows in [('summary.csv',flat),('fixed_fraction_curves.csv',curves)]:
        with (out/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    preservation = verify_preservation(root)
    write_report(root,results)
    write_json(out/'COMPLETE.json',dict(all_checks_pass=True, preservation=preservation,
        frozen_sha256=sha256(root/'FROZEN.json'),results_sha256=sha256(out/'results.json'),
        slurm_job_id=os.environ['SLURM_JOB_ID'],five_endpoints_frozen_before_test=True))
    return results


def write_report(root,results):
    def pct(x): return '—' if x is None else f'{100*x:.2f}%'
    lines=['# Independent agreement increment — final results', '',
        'This module independently selects agreement pseudo-labels. It does not modify, consume scores from, or merge results into Router-v8 v4.', '',
        'Best means the winner of the preregistered calibration search, with acceptance fraction selected on policy. Final test results did not select models, hyperparameters, fractions or seeds.', '',
        '| Dataset | Selected accuracy | Accepted | Coverage | All agreement accuracy | Policy accuracy | Model |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for d,r in results.items():
        s=r['selected']
        lines.append(f"| {d} | {pct(s['precision'])} | {s['accepted']} | {pct(s['coverage'])} | {pct(r['unfiltered']['precision'])} | {pct(r['policy']['precision'])} | {r['model_id']} |")
    lines += ['', '## Selection and training', '',
        'The exact candidate grid is in CONFIG.json. Each graph uses five calibration folds to compare fixed confidence baselines, regularized linear models, trees and neural networks. Learned candidates compare target-only and source+target training. Pooled training assigns half the total weight to target calibration and half equally to the other four validation domains.', '',
        'Model selection maximizes OOF mean precision over eligible 5%, 10%, 20% fractions (at least 10 rows); BCE and config ID break ties; if no calibration fraction has ten rows, selection falls back to BCE then ID. Policy selection maximizes empirical precision over the fixed fraction grid, requiring 20 accepted policy rows; Wilson lower bound and coverage break ties. The selected fraction transfers to query with exact decimal ceiling and node-ID tie ordering.', '',
        'Scores serve ranking; they are not claimed to be calibrated correctness probabilities. No query consistency or self-training is used.', '',
        '## Same-budget baselines', '',
        '| Dataset | Baseline | Accuracy | Accepted |', '| --- | --- | ---: | ---: |']
    for d,r in results.items():
        for name,b in r['same_budget_baselines'].items():
            lines.append(f"| {d} | {name} | {pct(b['precision'])} | {b['accepted']} |")
    lines += ['', '## Reproduction and limitations', '',
        '- Agreement-only, label-free delivery: pipeline/<dataset>.pt. Align by node_ids and use accepted_mask/pseudo_label.',
        '- Search evidence: targets/<dataset>/trials.json, chosen.json, cv_manifest.json, SEARCH_SEALED.json, selection.json and model.joblib.',
        '- Frozen evaluation boundary: FROZEN.json and PRETEST_AUDIT.json. All numerical stages ran through Slurm.',
        '- evaluation/*_audit_only.pt contains correctness and is never a training input.',
        '- Independent test precision can be below policy precision; extensive development selection and correlated graph nodes limit validation estimates. Wilson intervals are descriptive, not guarantees.',
        '- The benchmark and upstream validation-selected experts have historical development exposure. This run isolates query truth from new training and selection; it does not create a new untouched benchmark.',
        '- Additional agreement validation supervision is used; this is not a five-shot-only router result. Precision and coverage must be read together. No downstream Joint gain is claimed.',
        '- Protected-source audit hashes small files and checks size/mtime of large historical artifacts, as detailed in PRESERVATION_BEFORE.json. Raw data and caches remain external dependencies.', '']
    (root/'REPORT.md').write_text('\n'.join(lines))

    # Chinese delivery summary is generated from the same locked numerical results.
    chinese=['# Agreement 独立增量模块：最终结果', '',
        '本模块只筛选 agreement 节点，与 Router-v8（v4）独立；没有修改 v4，也没有使用其分数作为特征或教师。', '',
        '“最优”指预注册候选空间中由 calibration 折外结果选出的模型，接收比例由独立 policy 集选择。测试真值只在五图冻结且重放审计通过后读取，没有用于选择模型、参数、比例或随机种子。', '',
        '| 数据集 | 接收节点准确率 | 接收数 | Agreement 覆盖率 | 不筛选准确率 | Policy 准确率 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for dataset,row in results.items():
        selected=row['selected']
        chinese.append(f"| {dataset} | {pct(selected['precision'])} | {selected['accepted']} | {pct(selected['coverage'])} | {pct(row['unfiltered']['precision'])} | {pct(row['policy']['precision'])} |")
    chinese += ['', '## 最终模型及超参数', '']
    for dataset,row in results.items():
        import json
        chinese += [f"### {dataset}", '', f"选择：`{row['model_id']}`；训练范围：`{row['scope']}`；接收比例：{pct(row['policy']['fraction'])}。", '',
                    '```json', json.dumps(row['model_config'],indent=2,ensure_ascii=False), '```', '']
    chinese += ['## 搜索与评价口径', '',
        '- 45 个基础配置，学习模型比较 target-only 与 pooled，合计每图 83 个候选、五折 OOF。固定 seed42；没有按测试选择随机种子。',
        '- 架构包含7个固定指标、20个逻辑回归配置、6个 ExtraTrees、4个梯度提升树、8个 MLP 配方。线性模型的 embedding 输入采用固定随机坐标子集；MLP 对完整原始 embedding 学习投影。',
        '- 用 calibration 的低覆盖率 OOF 准确率选模型，policy 选择精度最高的接收比例，要求 policy 至少接收20行；同精度时优先证据更多的方案。完整固定比例曲线为诊断，不用于测试后重选。',
        '- Pooled 模型只使用目标 calibration 和其他四图 validation，目标总权重0.5，源域平分剩余0.5。MLP 按样本权重抽样。',
        '- 使用新增 agreement validation 监督，不能描述为仅由5-shot训练的 Router。历史 benchmark 和上游 validation 选模的既有限制继续保留。',
        '- 高伪标签准确率不等于已经验证下游 Joint 增益；本轮没有训练新的 Joint。', '',
        '## 产物', '',
        f'- [完整结果与同预算基线](../runs/{root.name}/REPORT.md)',
        f'- [结果 CSV](../runs/{root.name}/evaluation/summary.csv)',
        f'- [固定比例曲线 CSV](../runs/{root.name}/evaluation/fixed_fraction_curves.csv)',
        f'- [冻结清单](../runs/{root.name}/FROZEN.json)',
        f'- [测试前审计](../runs/{root.name}/PRETEST_AUDIT.json)',
        f'- 独立伪标签：`runs/{root.name}/pipeline/<dataset>.pt`，按 `node_ids` 对齐，读取 `accepted_mask` 和 `pseudo_label`。文件不含真值。', '']
    (REPO/'docs/agreement_increment_results_2026-09-08.md').write_text('\n'.join(chinese))
