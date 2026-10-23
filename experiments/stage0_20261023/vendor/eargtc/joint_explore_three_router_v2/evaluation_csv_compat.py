"""Full-population pretest reconstruction and one-shot numerical reporting."""
import csv
import itertools
from pathlib import Path
import statistics
import torch
from eargtc.joint_cross_repair.artifacts import read_json,write_json,sha256
from eargtc.joint_cross_repair.training import _score
from .adapter import load_cell,ROUTERS,DATASETS
from eargtc.joint_v8.evaluation import verify_full_family
from eargtc.joint_five_router.evaluation import _label_rows,labels_for_ids,classification_report,_seed_summary,correction_damage,_verify_historical_expert_identities
from .experiment import verify_frozen,legacy_inputs,close
from eargtc.joint_explore.selection import choose_final,shortlist,rank
from .retrieval import anchors


def _load(path):return torch.load(path,map_location='cpu',weights_only=True)


def verify_seed_slices(observed,expected,message):
    if observed.shape!=expected.shape or observed.shape[0]!=3:raise ValueError(message)
    for a,b in zip(observed,expected):close(a,b,message)


def audit_all(root):
    root=Path(root).resolve();config=verify_frozen(root)
    if config['smoke']:raise ValueError('smoke cannot evaluate test')
    if config['routers']!=list(ROUTERS) or config['datasets']!=list(DATASETS) or config['seeds']!=[42,43,44]:raise ValueError('wrong formal population')
    torch.set_num_threads(next(iter(config['settings'].values()))['threads']);records={}
    for rid,ds in itertools.product(ROUTERS,DATASETS):
        cell=root/rid/ds;f,s,p,h,meta=load_cell(config['export_root'],rid,ds)
        pred=_load(cell/'predictions.pt');val=_load(cell/'validation.pt');sel=read_json(cell/'selection.json')
        if not torch.equal(pred['node_ids'],f['node_ids']) or not torch.equal(val['node_ids'],s['selection_ids']):raise ValueError('full/validation IDs differ')
        old,base=legacy_inputs(config,rid,ds,f,s)
        families=pred['families']
        expected={'legacy','router','gnn','llm','expert_mean','prototype','knn','balanced_direct','balanced_residual','gated_experts','attention_direct','optimized'}
        if set(families)!=expected or set(val['families'])!=expected-{'optimized'}:raise ValueError('missing output family')
        for name,value in families.items():
            verify_full_family(value,name,f['logp_g'].shape)
            if name!='optimized':close(value['ensemble'][s['selection_ids']],val['families'][name],name+' validation mismatch')
        close(families['legacy']['ensemble'],old['ensemble'],'legacy full baseline differs')
        verify_seed_slices(families['legacy']['seeds'],old['seeds'],'legacy seed predictions differ')
        for name,value in base.items():
            close(families[name]['ensemble'],value,'base '+name+' differs')
            verify_seed_slices(families[name]['seeds'],value.unsqueeze(0).repeat(3,1,1),'base seeds differ')
        screening=[];bank={};gold=s['gold_ids'];pseudo=torch.cat([p['teach_g'],p['teach_l'],p['agreement']]);keep=p['unselected']
        for path in sorted((cell/'bank').glob('*/*/seed*.pt')):
            m=_load(path);key=path.parent.parent.name+'/'+path.parent.name;seed=int(path.stem[4:])
            from .migration import verify_import_receipt
            source=Path(config['checkpoint_import']['root'])/rid/ds/'bank'/key/path.name
            verify_import_receipt(path,source,config['checkpoint_import']['config_sha256'],sha256(root/'CONFIG.json'))
            expected_settings={**config['settings'][key],'seed':seed}
            if m['settings']!=expected_settings or m.get('test_labels_read') is not False or not torch.equal(m['validation_ids'],s['selection_ids']):raise ValueError('member provenance mismatch')
            for name,ids in [('gold',gold),('pseudo',pseudo),('keep',keep)]:
                if not torch.equal(m['training_ids'][name].sort().values,ids.sort().values):raise ValueError('training population changed')
                if torch.isin(ids,torch.cat([s['selection_ids'],s['threshold_ids']])).any():raise ValueError('validation in supervised training population')
            bank[key,seed]=m['validation_probability']
            if seed==42:
                choices=[{'id':key,'architecture':key.split('/')[0],'criterion':c,**_score(prob,s['selection_targets'])} for c,prob in m['validation_probability'].items()]
                screening.append(min(choices,key=rank))
        if len(bank)!=40 or len(screening)!=24:raise ValueError('unequal neural search budget')
        retained=shortlist(screening);saved=read_json(cell/'shortlist.json')
        if retained!=saved['retained'] or sorted(screening,key=lambda r:r['id'])!=sorted(saved['rows'],key=lambda r:r['id']):raise ValueError('screening failed reconstruction')
        expected_bank={(key,42) for key in config['settings']}|{(key,seed) for key,seed in itertools.product(retained,[43,44])}
        if set(bank)!=expected_bank:raise ValueError('wrong confirmation candidates')
        neural={}
        for key in retained:
            for criterion in ['accuracy','nll']:
                q=torch.stack([bank[key,seed][criterion] for seed in config['seeds']]).mean(0)
                row={'id':key+'/'+criterion,'key':key,'criterion':criterion,**_score(q,s['selection_targets'])};family=key.split('/')[0]
                if family not in neural or rank(row)<rank(neural[family]):neural[family]=row
        if neural!=sel['neural']:raise ValueError('neural family selection differs')
        for family,pick in neural.items():
            seeds=torch.stack([bank[pick['key'],seed][pick['criterion']] for seed in config['seeds']])
            close(seeds.mean(0),val['families'][family],'neural ensemble changed')
            verify_seed_slices(families[family]['seeds'][:,s['selection_ids']],seeds,'neural per-seed validation differs')
        retrieval=_load(cell/'retrieval.pt');anchor_ids,labels,weights,isgold=anchors(f,s,p)
        for name,value in retrieval['families'].items():
            for key,expected_value in [('anchor_ids',anchor_ids),('anchor_labels',labels),('anchor_weights',weights),('anchor_gold',isgold)]:
                if not torch.equal(value[key],expected_value):raise ValueError('retrieval anchor population/targets changed')
            close(value['validation_probability'],val['families'][name],'retrieval validation changed')
            verify_seed_slices(families[name]['seeds'],families[name]['ensemble'].unsqueeze(0).repeat(3,1,1),'retrieval seeds differ')
            rows=[r for r in retrieval['history'] if r['family']==name]
            best=min(rows,key=lambda r:(-r['accuracy'],r['nll']))
            if best['config']!=value['config'] or _score(value['validation_probability'],s['selection_targets'])!={k:best[k] for k in ['accuracy','nll']}:raise ValueError('retrieval selection failed reconstruction')
        rebuilt,catalog=choose_final(val['families'],s['selection_targets'])
        if rebuilt!=sel['optimized'] or set(catalog)!=set(val['catalog']):raise ValueError('fusion choice changed')
        for name,value in catalog.items():close(value,val['catalog'][name],'fusion catalog changed')
        q=sum(w*families[name]['ensemble'] for name,w in rebuilt['components'].items())
        close(families['optimized']['ensemble'],q,'full final fusion differs')
        expected_seeds=sum(w*families[name]['seeds'] for name,w in rebuilt['components'].items())
        verify_seed_slices(families['optimized']['seeds'],expected_seeds,'final seed-wise fusion differs')
        close(q[s['selection_ids']],catalog[rebuilt['id']],'final validation reproduction differs')
        records[f'{rid}/{ds}']={'test_count':len(s['test_ids']),'accepted_count':meta['accepted_count'],'pseudo_teaching_count':len(pseudo),'accepted_agreement_count':len(p['agreement']),'accepted_conflict_count':len(p['teach_g'])+len(p['teach_l']),'candidate_count':len(s['test_ids']),'neural_members':40,'imported_neural_members':40,'new_neural_training_members':0,'validation_reproduced':True}
        print('Pretest validated',rid,ds,flush=True)
        del f,s,p,h,pred,val,old,base,families,bank,retrieval
    audit={'all_checks_pass':True,'cell_count':15,'test_labels_read':False,'cells':records,'frozen_sha256':sha256(root/'FROZEN.json')}
    path=root/'PRETEST_AUDIT.json'
    if path.exists():
        if read_json(path)!=audit:raise ValueError('pretest audit changed')
    else:write_json(path,audit)
    return config,audit


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def evaluate_all(root):
    root=Path(root).resolve();output=root/'evaluation';stage=root/'evaluation.staging'
    if output.exists() or stage.exists():raise FileExistsError('immutable evaluation already exists')
    config,audit=audit_all(root);stage.mkdir()
    labelroot=Path(config['label_root']);truth={ds:_label_rows(labelroot/ds/'decisions.jsonl') for ds in DATASETS}
    old_rows=list(csv.DictReader((Path(config['legacy_root'])/'evaluation/performance.csv').open()))
    old_by={(r['router'],r['dataset']):r for r in old_rows};rows=[];details={};ablations=[]
    for rid,ds in itertools.product(ROUTERS,DATASETS):
        cell=root/rid/ds;f,s,p,h,meta=load_cell(config['export_root'],rid,ds)
        pred=_load(cell/'predictions.pt');families=pred['families'];ids=s['test_ids'];y=labels_for_ids(truth[ds],ids)
        _verify_historical_expert_identities(truth[ds],ids,families['gnn']['ensemble'][ids],families['llm']['ensemble'][ids])
        metrics={name:classification_report(value['ensemble'][ids],y) for name,value in families.items()}
        if abs(metrics['legacy']['accuracy']-float(old_by[rid,ds]['accuracy']))>1e-12 or len(ids)!=int(old_by[rid,ds]['test_count']) or meta['accepted_count']!=int(old_by[rid,ds]['accepted_count']) or abs(metrics['router']['accuracy']-(float(old_by[rid,ds]['accuracy'])-float(old_by[rid,ds]['router_delta'])))>1e-12:raise ValueError('historical fulltest identity differs')
        seed=_seed_summary(families['optimized']['seeds'],ids,y);selection=read_json(cell/'selection.json')
        delta=metrics['optimized']['accuracy']-metrics['legacy']['accuracy']
        cd=correction_damage(families['legacy']['ensemble'][ids].argmax(1),families['optimized']['ensemble'][ids].argmax(1),y)
        if abs(cd['gain']-delta)>1e-12:raise ValueError('correction/damage mismatch')
        row={'router':rid,'dataset':ds,**{k:metrics['optimized'][k] for k in ['accuracy','macro_f1','nll']},
             'old_joint_accuracy':metrics['legacy']['accuracy'],'delta_old_joint':delta,'router_accuracy':metrics['router']['accuracy'],
             'delta_router':metrics['optimized']['accuracy']-metrics['router']['accuracy'],'seed_accuracy_mean':seed['accuracy']['mean'],
             'seed_accuracy_std':seed['accuracy']['std'],'accepted_count':meta['accepted_count'],'pseudo_teaching_count':meta['pseudo_teaching_count'],'accepted_agreement_count':meta['accepted_agreement_count'],'accepted_conflict_count':meta['accepted_conflict_count'],'candidate_count':meta['candidate_count'],'test_count':len(ids),'selected':selection['optimized']['id']}
        rows.append(row);details[rid+'/'+ds]={'metrics':metrics,'selection':selection,'seed_metrics':seed,'correction_damage_vs_old':cd,'test_label_sha256':sha256(labelroot/ds/'decisions.jsonl')}
        for name,metric in metrics.items():ablations.append({'router':rid,'dataset':ds,'family':name,**{k:metric[k] for k in ['accuracy','macro_f1','nll']}})
        print('Evaluated',rid,ds,flush=True)
        del f,s,p,h,pred,families
    write_json(stage/'results.json',{'cells':details,'config':{k:config[k] for k in ['settings','screening','seeds']}})
    write_csv(stage/'performance.csv',rows);write_csv(stage/'family_performance.csv',ablations)
    matrix=[]
    for rid in ROUTERS:
        for field,title in [('old_joint_accuracy','preserved_joint'),('accuracy','explored_joint'),('delta_old_joint','delta_pp')]:
            row={'router':rid,'variant':title,**{ds:next(r[field] for r in rows if r['router']==rid and r['dataset']==ds) for ds in DATASETS}}
            row['mean']=statistics.fmean(row[ds] for ds in DATASETS);matrix.append(row)
    write_csv(stage/'accuracy_comparison.csv',matrix)
    write_report(stage/'REPORT.md',rows,ablations,matrix,details)
    write_json(stage/'complete.json',{'all_checks_pass':True,'evaluated_cells':15,'neural_members':600,'imported_neural_members':600,'new_neural_training_members':0,
        'frozen_sha256':sha256(root/'FROZEN.json'),'pretest_audit_sha256':sha256(root/'PRETEST_AUDIT.json'),
        'artifacts':{p.name:sha256(p) for p in stage.iterdir() if p.is_file()}})
    stage.rename(output)
    return rows


def write_report(path,rows,ablations,matrix,details):
    lines=['# Joint 架构探索：R6-Legacy-C / R6-Split-C / A20-MS-Split-C 最终结果','',
        '旧 Joint 已独立归档，全部旧代码、配置、checkpoint 和结果保持原样。当前使用相同的冻结 R6-Legacy-C / R6-Split-C / A20-MS-Split-C 接受集合与五数据集划分；golden support 保留。以下是验证集选出的有限搜索结果，不是 test 上挑出的最优值或理论分类上限。','',
        '本轮 v1 在测试前复核发现 kNN 近似并列相似度的排序受矩阵批次形状影响。v2 统一使用固定全局节点块计算近邻相似度，对全部15格重新完成原检索网格和最终验证集选模；600个神经 checkpoint 逐字节复用，架构、损失、神经训练和搜索预算不变。v1 记录保留，未进行测试评分。','',
        '评分兼容说明：全部15格评分前审计已通过；首次评分因历史CSV缺少 router_accuracy 列而中止。历史Router准确率按 accuracy − router_delta 还原后重试。此时测试标签已被读取，但冻结模型、预测、检索参数与最终选择均未改变，也未追加搜索。原失败日志和兼容层哈希单独保留。','',
        '## 主表：完整 test 准确率（%）','',
        '| Router | Joint | Cora | CiteSeer | PubMed | Arxiv | Products | 平均 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for row in matrix:
        title={'preserved_joint':'保留旧版','explored_joint':'本轮探索','delta_pp':'变化（pp）'}[row['variant']]
        fmt=[f'{100*row[d]:+.2f}' if row['variant']=='delta_pp' else f'{100*row[d]:.2f}' for d in [*DATASETS,'mean']]
        lines.append('| '+ROUTERS[row['router']]+' | '+title+' | '+' | '.join(fmt)+' |')
    lines+=['','准确率采用三个种子的概率平均后 argmax；平均列为五数据集等权平均。确定性检索/无修复候选的种子预测相同；混合候选按对应种子组合。','',
        '## 逐格指标与最终选择','',
        '| Router | 数据集 | Acc % | Macro-F1 % | Δ旧Joint pp | ΔRouter-B pp | 单种子Acc均值±标准差 % | 伪标签数 | 最终候选 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['router']} | {r['dataset']} | {r['accuracy']*100:.2f} | {r['macro_f1']*100:.2f} | {r['delta_old_joint']*100:+.2f} | {r['delta_router']*100:+.2f} | {r['seed_accuracy_mean']*100:.2f}±{r['seed_accuracy_std']*100:.2f} | {r['accepted_count']} | {r['selected']} |")
    lines+=['','## 全量候选与教学范围','',
        '未使用候选筛选模块：每个 Router 的候选为原始完整 test。保持 Joint-Explore 架构、损失和搜索规则；监督接口兼容扩展将已被 Router 接受的一致节点也纳入同一伪标签 CE 与检索锚点。原 v8 一致接受池为空，其行为保持不变。golden support 始终保留。表中教学数量表示可用池；gold-only 配方的伪标签损失系数为零。',
        '| Router | 数据集 | 全量候选 | Router接受总数 | 接受冲突节点 | 接受一致节点 | 伪标签教学候选 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {ROUTERS[r['router']]} | {r['dataset']} | {r['candidate_count']} | {r['accepted_count']} | {r['accepted_conflict_count']} | {r['accepted_agreement_count']} | {r['pseudo_teaching_count']} |")
    lines+=['','## 架构和训练','',
        '- balanced_residual：两视图分别归一化并投影到相同维度，加入交互特征和原专家概率，学习 Router-B 的残差。',
        '- balanced_direct：使用同样的平衡表示直接分类。',
        '- gated_experts：按节点学习 GNN、LLM、新联合分类器的混合权重。',
        '- attention_direct：GNN、LLM、先验概率和 CLS token 经两层注意力编码后直接分类。',
        '- prototype / knn：原始全维表示的类原型和精确近邻对照。训练锚点只有 gold 和 Router 接受伪标签；近邻排除节点自身。',
        '- 神经网络以 gold CE + 置信度加权伪标签 CE + 原 Router-B 保持 KL 训练；对齐变体加入无标签成对 InfoNCE。最终联合输出决定 checkpoint，非分别选两个头后再拼装。',
        '- 每格24配置单种子初筛，每架构前2配置补齐3种子，共40个神经训练成员；十五格共600个。六配方涵盖 gold-only、教学强度、保持约束、表示对齐和更宽网络。原型72配置，近邻81配置，搜索范围事先固定。',
        '- 每个家族在原 model-selection validation half 选优，再比较旧 Joint 保底、原专家及有限凸组合。旧 Router阈值、伪标签集合均未重选。','',
        '## 家族消融：完整 test 准确率（%）','',
        '| Router | 家族 | Cora | CiteSeer | PubMed | Arxiv | Products | 平均 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for rid in ROUTERS:
        for family in ['legacy','balanced_residual','balanced_direct','gated_experts','attention_direct','prototype','knn','optimized']:
            values=[next(r['accuracy'] for r in ablations if r['router']==rid and r['dataset']==ds and r['family']==family) for ds in DATASETS]
            lines.append('| '+rid+' | '+family+' | '+' | '.join(f'{v*100:.2f}' for v in [*values,statistics.fmean(values)])+' |')
    lines+=['','## 协议与解释边界','',
        '所有有标签梯度只使用原始 golden support 与冻结伪标签。无标签表示可用于对齐/检索，这是传导式设定。validation 真值只用于选模，threshold half 未用于新 Joint 选模。全部十五格先冻结并通过选模复算、完整节点/种子概率与来源检查，再统一评分原始完整 test；本轮没有根据 test 追加搜索。',
        '历史上游 Router 已使用目标 validation，历史 test 也已观察。本轮沿用既有开发协议，不宣称全流程独立验证或统计显著性。种子标准差不是多划分置信区间。',
        '最终候选允许回退旧 Joint 或无修复专家，也允许神经/检索组合；不能把每格成绩都归因于新神经网络。详细最终网络配置、checkpoint指标和检索参数见 results.json 的 selection 与冻结 CONFIG.json。','']
    Path(path).write_text('\n'.join(lines))
