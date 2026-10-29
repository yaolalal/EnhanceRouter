"""Self-contained Chinese Markdown, with all seven models grouped by dataset/shot."""
from comprehensive import *
import datetime

NAMES={'B0':'B0 / FairRouter','D1_25__fixed':'D1-25% / Joint冻结','D1_25__retuned':'D1-25% / Joint重调','D2a_1000__fixed':'D2-1000 / Joint冻结','D2a_1000__retuned':'D2-1000 / Joint重调','D4_confidence__fixed':'D4 / Joint冻结','D4_confidence__retuned':'D4 / Joint重调',CONTROL:'B0 / D2同名单'}
DATASETS={'cora':'Cora','citeseer':'Citeseer','pubmed':'PubMed','arxiv':'Arxiv','ogbn-products':'Products'}
METRIC_PATHS={
    'trust_raw':('router_raw','trust_auroc'),'preference_raw':('router_raw','preference_auroc'),
    **{f'D_P{k:03d}':('router_calibrated',f'D_Prec_{k:02d}') for k in (1,5,20,100)},
    **{f'pseudo_{b}_precision':('pseudo',f'{b}_precision') for b in ('A','D','all','full_A','full_D','full')},
    'corrective_precision':('corrective_D','precision'),
    **{k:('classification',k) for k in ('pR_accuracy','without_pR_accuracy','joint_accuracy','subset_fraction','subset_pR_accuracy','subset_without_pR_accuracy','subset_joint_accuracy')}
}
COUNT_PATHS={
    'Q':('query_count',),'A_population':('pseudo','full_A_count'),'D_population':('pseudo','full_D_count'),
    **{f'accepted_{b}':('pseudo',f'{b}_count') for b in ('A','D','all')},
    **{k:('corrective_D',k) for k in ('C','R','H','W','net','right_minus_wrong')},
    **{k:('classification',k) for k in ('eval_count','eval_pR_correct','eval_without_pR_correct','eval_joint_correct','subset_count','subset_pR_correct','subset_without_pR_correct','subset_joint_correct')}
}

def get(r,path):
    for p in path:r=r[p]
    return r

def summarize(rows,ds,shot,label):
    metrics={name:stat([get(r,path) for r in rows]) for name,path in METRIC_PATHS.items()}
    counts={name:dict(total=sum(get(r,path) for r in rows),**stat([get(r,path) for r in rows])) for name,path in COUNT_PATHS.items()}
    assert counts['C']['total']==sum(counts[k]['total'] for k in ('R','H','W'))
    assert counts['net']['total']==counts['R']['total']-counts['H']['total']
    assert counts['right_minus_wrong']['total']==counts['R']['total']-counts['H']['total']-counts['W']['total']
    assert counts['accepted_all']['total']==counts['accepted_A']['total']+counts['accepted_D']['total']
    pooled={k:counts[v]['total']/counts['subset_count']['total'] if counts['subset_count']['total'] else None for k,v in (('subset_pR_accuracy','subset_pR_correct'),('subset_without_pR_accuracy','subset_without_pR_correct'),('subset_joint_accuracy','subset_joint_correct'))}
    pooled['corrective_precision']=counts['R']['total']/counts['C']['total'] if counts['C']['total'] else None
    return dict(dataset=ds,shot=shot,variant=label,cells=len(rows),metrics=metrics,counts=counts,pooled=pooled)

def code(file,token=None,label=None):
    p=ROOT/file;lines=p.read_text().splitlines();line=None
    if token is not None:
        found=[i for i,s in enumerate(lines,1) if token in s];assert found,(file,token);line=found[0]
    return '['+(label or file)+'](https://github.com/yaolalal/EnhanceRouter/blob/main/'+file+('#L'+str(line) if line else '')+')'

def table(headers,rows):
    assert all(len(row)==len(headers) for row in rows)
    assert not any('|' in str(x) or '\n' in str(x) for row in [headers,*rows] for x in row)
    return ['| '+' | '.join(map(str,headers))+' |','|'+'|'.join('---' for _ in headers)+'|',*['| '+' | '.join(map(str,row))+' |' for row in rows]]

def render(summary,allrows):
    labels=(*LABELS,CONTROL)
    def pct(x):return 'NA' if x is None else f'{100*x:.2f}'
    def mean(s,k,raw=False):
        v=s['metrics'][k]
        if v['mean'] is None:return f'NA [0/{s["cells"]}]'
        scale,dec=(1,4) if raw else (100,2)
        out=f'{scale*v["mean"]:.{dec}f} ± {scale*v["std"]:.{dec}f}'
        return out+(f' [{v["n"]}/{s["cells"]}]' if v['n']!=s['cells'] else '')
    def count(s,k):
        v=s['counts'][k];return f'{v["total"]:,} / {v["mean"]:.2f}'
    def integer(s,k):return f'{s["counts"][k]["total"]:,}'
    rawlink=code('src/enhancerouter/router.py','def predict(','评分器原始输出 predict()')
    calink=code('src/enhancerouter/router.py','def predict_bundle','原生部署校准 predict_bundle()')
    newlink=code('experiments/comprehensive_report_20261029/comprehensive.py','def raw_router','本次原始输出恢复与部署复核')
    subsetlink=code('experiments/comprehensive_report_20261029/comprehensive.py','def component_predictions','本次eval1000子集计算')
    lines=['# B0与3×2：完整指标、实现对照及代码溯源','',
        '生成时间（Asia/Shanghai）：'+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        '', '范围：Cora、Citeseer、PubMed、Arxiv、Products；3/5/10-shot；split seeds42–46。七种主配置共525份冻结模型。另列75份“B0 / D2同名单”评估参照，不是新增训练配置。',
        '本报告仅进行冻结模型推理/离线统计：不重训、不重选模型，不改变专家、Router、接受名单、伪标签、alpha/beta或数据划分。D1使用numeric25% Router；“Joint冻结”仅表示Joint数值超参沿用B0，绝不是旧的全冻结Router D1。',
        '', '## 1. 如何阅读','',
        '- **校准前AUROC是本次重新计算的原始评分器指标。** 之前报告读到的是保存的校准后Trust/Preference，本报告不把它们冒称校准前分数。',
        '- Router与纠正性指标使用完整Query，排除原support/validation；D2另排除新增1000个真标签训练节点，配对B0使用相同剩余名单。Query不是eval1000。',
        '- 最终分类只用各dataset/seed原顺序eval1000，3/5/10-shot共用名单。三个分类分支均来自同一已训练checkpoint，不是分别重训的消融模型。',
        '- 纠正性参照采用**每个模型自身**最终accuracy checkpoint的alpha/beta。C来自实际接受的disagreement伪标签；分类子集S则来自eval1000，两集合定义和评估范围不同。',
        '- 精度单位为%，AUROC为0–1。表内“均值 ± 标准差”为配置等权均值与总体标准差(ddof=0)；总体跨75配置，各dataset×shot跨5个seed。总体标准差也包含数据集/shot难度差异。',
        '- 数量列写成“**总计 / 每配置均值**”。总体按75组累计；逐dataset×shot按5组累计。累计是节点—配置次数，节点跨shot/seed可能重复，未乘六视图，不能解释为去重图节点数。',
        '- 空接受集合/空C/空S精度为NA，不填0。均值只纳入有定义配置，缺失时附[n/N]；纠正性精度列始终列有效n。计数均值包含零计数。配置平均R/C不等于累计R除以累计C，后者另列“次数池化精度”。',
        '- 接受伪标签数是Router实际接受的节点数；cross或keep为0时，接受不等于在Joint损失中启用该部分伪标签。这里不改变任何损失权重。',
        '', '## 2. 七种配置与B0的统一项目对照','']
    headers=['统一对照项目','B0','D1冻结','D1重调','D2冻结','D2重调','D4冻结','D4重调']
    rows=[
        ['专家GNN/LLM输出','原B0','同对应B0','同对应B0','同对应B0','同对应B0','同对应B0','同对应B0'],
        ['原support/valid/Query划分','原B0','不变','不变','原划分不变','原划分不变','不变','不变'],
        ['目标Router监督父节点','全部原support','随机保留25%','同一25%','原support＋随机1000','同一额外1000','不新增监督','不新增监督'],
        ['图/文本增强及六视图','原B0','保留父节点全部六视图','同左','额外节点同六视图','同左','专家与缓存不变','同左'],
        ['源数据/原型','原B0','固定B0','固定B0','固定B0','固定B0','不使用学习评分作决策','同左'],
        ['Router特征模式/模型族/训练组合','原生逐格选择','固定对应B0','固定对应B0','固定对应B0','固定对应B0','改用专家置信度规则','同左'],
        ['Router已拟合输入变换','原B0','固定B0','固定B0','固定B0','固定B0','不重新拟合','同左'],
        ['Router评分器权重','原B0','减少数据后重拟合','同一Router','加入真标签后重拟合分歧头','同一Router','置信度规则不训练','同一Router'],
        ['Router数值超参','原B0','允许原验证流程重选','同一Router','冻结B0','同一Router','无可学习超参','同左'],
        ['Router校准与路由策略','原生逐格流程','允许原验证流程重拟合','同一Router','冻结B0','同一Router','选置信度更高专家','同左'],
        ['接受数量/比例','B0已选策略','验证流程可重新选择','同一接受名单','B0冻结比例','同一接受名单','每分支数量精确匹配B0','同一接受名单'],
        ['新增真标签送入Joint','无','无','无','不送入','不送入','无','无'],
        ['Joint gold监督','完整原support','完整原support','完整原support','完整原support','完整原support','完整原support','完整原support'],
        ['Joint数值超参','原B0','冻结对应B0','原生四阶段验证重选','冻结对应B0','原生四阶段验证重选','冻结对应B0','原生四阶段验证重选'],
        ['Joint权重及alpha/beta','B0最终checkpoint','重新训练所得','重调并训练所得','重新训练所得','重调并训练所得','重新训练所得','重调并训练所得'],
        ['纠正性pGL参照','自身alpha/beta','自身alpha/beta','自身alpha/beta','自身alpha/beta','自身alpha/beta','自身alpha/beta','自身alpha/beta'],
        ['Router测试Query','原完整Query','同B0','同B0','剔除新增训练节点','同左','同B0','同B0'],
        ['分类eval1000','原名单/顺序','不变','不变','不变且无新增训练节点','不变','不变','不变']]
    lines+=table(headers,rows)
    lines+=['','D1抽样按目标support总名单、seed+913随机排列，取max(1,round(N×0.25))个父节点（程序用int(N×0.25+0.5)）；不是每类都固定保留四分之一。来源数据及专家保持对应B0，不能把它解释成GNN专家也只用25% support重训。',
        'D2采用用户指定的随机a/1000，新增真标签只用于Router分歧分支训练；不是高置信池b。评估统一排除这1000节点，即使它们在某一增强视图中属于agreement也不回填。',
        'D4采用confidence_router，不是confidence_preference：cG=max softmax(GNN logits)，cL=max softmax(LLM logits)；分歧Trust/排序分数=max(cG,cL)，Preference分数=cG−cL，选择GNN当cG≥cL；一致分支排序分数=sqrt(cG×cL)。每分支接受人数匹配对应B0。D4没有“学习校准之前”的另一份模型输出，AUROC直接使用此原始置信度规则。',
        'Joint重调只开放原候选中的dim/cross/lr/gate_lr/warmup/aux/keep；gate_gold及其他非搜索设置固定对应B0。冻结超参不等于冻结Joint权重。B0 / D2同名单仅改变评估名单，不改变B0权重或重新训练。',
        '', '### 2.1 实现定位','']
    sources=[
        ('B0原始路由/拟合入口','src/enhancerouter/router.py','def rebuild'),
        ('专家训练、专家缓存和增强入口','experiments/stage0_20261026/frontend.py','def '),
        ('历史/原生Query特征库加载','experiments/stage0_20261026/common.py','def bank'),
        ('D1 numeric父节点抽样、冻结及重调项','experiments/stage0_20261027/d1_numeric.py','def main'),
        ('D2新增真标签、D4受控置信度','experiments/stage0_20261026/compute.py','def d2'),
        ('D4置信度公式和接受人数匹配','experiments/stage0_20261026/compute.py','def route'),
        ('3×2共用Router并训练不同Joint','experiments/stage0_20261029/factorial_joint.py','def train_task'),
        ('Joint残差概率头、alpha/beta和融合','src/eargtc/joint_explore/joint_purf_v10_v1/model.py','class UnifiedRepair'),
        ('Joint训练、验证checkpoint选择及预测','src/eargtc/joint_explore/joint_purf_v10_v1/training.py','def train'),
        ('本次全Query重算、C/R/H/W及冻结复核','experiments/comprehensive_report_20261029/comprehensive.py','def worker')]
    lines+=table(['环节','代码'],[[name,code(file,token)] for name,file,token in sources])
    lines+=['','历史来源说明：55格B0复用历史冻结结果，其余20格运行已登记的原生搜索；历史5-shot选择记录含selection_uses_query_truth=true，这一已知来源限制未被本次离线报告消除。新增分析不使用Query真值选模型。详见'+code('docs/FAIRROUTER_BASELINE.md',None,'B0来源及历史选择限制')+'。',
        '', '## 3. 指标的集合、分数和计算方式','',
        '记Q为当前dataset/shot/seed完整Query（split.unlabeled_ids），A为GNN与LLM预测一致节点，D为两者预测不一致节点。D2测试池Q*=Q去掉新增训练父节点；配对B0同样使用Q*。记K为原Router实际接受标记。伪标签为Router选择的专家标签，不是Joint或pR的预测。',
        '', '### 3.1 校准前AUROC与校准后D-PR@k%','',
        '本报告的校准前输出取'+rawlink+'：engine.predict_candidate直接输出的Trust/Preference分数。**不执行raw()中的gold_normalization或输出中心化/缩放，也不执行外层affine、pair_calibration、route_recipe。** 模型内部已拟合输入标准化/筛列/投影仍保留，这是模型输入变换，不在本次改动。没有根据Query标签翻转分数方向。',
        'Trust AUROC：在D∩Q（D2为D∩Q*）上，真值是“GNN或LLM至少一个正确”，评分是原Trust头输出；Preference AUROC：只在上述至少一个正确的节点上，真值是“GNN正确”，评分是原Preference头输出。只有单一真值类别时记NA。D4使用前述置信度规则。',
        '校准后部署由'+calink+'实现。本次把缓存的原始头输出送入原部署函数，验证校准后score/trust/preference和路由与原封存结果一致，确认没有换模型或换特征；实现见'+newlink+'。',
        'D-PR@k%（本报告等同D-P@k%）：在当前完整Query的分歧池内，用封存的校准后最终score降序排序，同分按node_id升序；取前ceil(k×分歧池大小/100)个节点，统计原Router所选专家标签的准确率。k=1、5、20、100。不是把k当作固定节点数，也不是先按接受集合裁剪后再算P@k。D2在剔除训练节点后重新排序取k；其实际接受名单仅剔除、不补选。',
        '', '### 3.2 伪标签精度、数量与disagreement纠正性精度','',
        '主伪标签表分别在K∩A∩Q、K∩D∩Q、K∩Q计算正确标签数/接受数，并报告数量；D2将Q替换为Q*。另附“未筛选全Query分支”表，给出未施加接受筛选时A、D及合计的准确率/人数，避免将分歧全池精度(P@100%)与接受分歧精度混用。',
        '每模型使用自身已训练alpha、beta：pGL=(alpha×pG+beta×pL)/(alpha+beta)。令b=argmax(pGL)，pseudo为原Router伪标签，y为真标签。在实际接受分歧集合中定义：',
        '- C={节点属于K∩D∩Q，且pseudo≠b}；D2使用Q*。',
        '- R：C中b≠y且pseudo=y（改对）；H：C中b=y且pseudo≠y（改错）；W：C中b≠y且pseudo≠y（换成另一个错标签）。',
        '- C总数=R+H+W；纠正性精度=R/C；净纠正=R−H；正确−错误纠正标签数=R−H−W。C空时精度NA而计数均为0。',
        '本次直接在分歧节点上重算C/R/H/W，并逐项核对上一版corrective_own_pGL统计。一致分支中两专家argmax相同且alpha/beta为正，pGL仍选该标签；代码也逐节点检查接受的一致节点没有被错误计入C。计算见'+code('experiments/comprehensive_report_20261029/comprehensive.py','def worker','全Query分支计数与C/R/H/W')+'。',
        '', '### 3.3 分类分解与eval1000子集','',
        'pR是同一checkpoint的修复分类头概率；w/o pR为该模型自身pGL；完整分类概率pJ=(alpha×pG+beta×pL+pR)/(alpha+beta+1)。三项准确率均为对应argmax标签与真标签比较，不另训练pR-only或无pR模型。',
        '**本次新子集S={节点属于原eval1000，argmax(pR)≠argmax(pGL)}。** 这里比较预测类别，不比较概率向量是否逐元素相等。占比=人数S/1000；三项子集准确率都在同一模型的同一S上计算。S为空则三项精度NA。不同配置的S可不同，不能将子集精度直接当成同名单配对比较。',
        'S与旧的“pR改变完整模型标签”集合不同：旧集合是argmax(pJ)≠argmax(pGL)。本报告使用用户本次要求的pR与pGL分歧集合；另将旧集合大小保留在逐组CSV用于核查。',
        '本次读取已封存逐节点pR/pGL/pJ标签，核对原eval1000顺序、真值、原完整模型预测和原自身pGL证据，再筛选S；三项全eval1000准确率逐组与原报告一致。实现见'+subsetlink+'。所有模型都采用最终accuracy checkpoint，而非训练最后一步或重新挑选checkpoint。',
        '', '### 3.4 平均方式与可复核性','',
        '下表主精度先在每个seed配置计算，再对配置等权取均值；std为ddof=0。纠正性/子集精度只平均非空配置，计数平均纳入零。累计R/C以及累计子集正确数/累计S人数是另一种按节点次数加权的“池化精度”，报告分列或保存在summary.csv，不替代主均值。',
        '原始head分数二进制存服务器并记录路径/SHA256；每个配置的Query/eval1000名单哈希、模型/评分器/数据源哈希、原校准指标、逐节点C/R/H/W和eval1000标签/子集标记在本次JSON/CSV中保留。旧模型及报告不覆盖。所有数值计算均由Slurm完成。',
        '', '## 4. 结果导航','', '[总体](#overall) · '+' · '.join('['+DATASETS[d]+' '+str(s)+'-shot](#'+d+'-shot'+str(s)+')' for d in DS for s in SHOTS),'']
    groups=[('ALL','ALL'),*[(d,s) for d in DS for s in SHOTS]]
    for ds,shot in groups:
        ss=[next(s for s in summary if s['dataset']==ds and s['shot']==shot and s['variant']==v) for v in labels]
        anchor='overall' if ds=='ALL' else ds+'-shot'+str(shot)
        title='总体：五数据集 × 三shot × 五seed' if ds=='ALL' else DATASETS[ds]+' · '+str(shot)+'-shot · seeds42–46'
        lines+=['','<a id="'+anchor+'"></a>','', '## '+title,'', f'每种主配置{ss[0]["cells"]}组。B0 / D2同名单只用于D2配对，不计为第八种训练设置。数值为均值±标准差；数量为总计/每组均值。',
            '', '### Router：校准前AUROC，校准后D-PR@k%','']
        lines+=table(['配置','Trust AUROC（前）','Pref AUROC（前）','D-PR@1%（后）','D-PR@5%（后）','D-PR@20%（后）','D-PR@100%（后）'],[[NAMES[s['variant']],mean(s,'trust_raw',True),mean(s,'preference_raw',True),*[mean(s,'D_P'+f'{k:03d}') for k in (1,5,20,100)]] for s in ss])
        lines+=['','### 伪标签：实际接受的agreement／disagreement／合计','']
        lines+=table(['配置','Agreement精度%','Agreement数量Σ/均值','Disagreement精度%','Disagreement数量Σ/均值','合计精度%','合计数量Σ/均值'],[[NAMES[s['variant']],mean(s,'pseudo_A_precision'),count(s,'accepted_A'),mean(s,'pseudo_D_precision'),count(s,'accepted_D'),mean(s,'pseudo_all_precision'),count(s,'accepted_all')] for s in ss])
        lines+=['','### 实际接受disagreement上的自身pGL纠正性指标','']
        lines+=table(['配置','纠正性精度%（有效n）','累计R/C%','C Σ/均值','R Σ/均值','H Σ/均值','W Σ/均值','净R−H 总计','R−H−W 总计'],[[NAMES[s['variant']],mean(s,'corrective_precision')+(f' [{s["metrics"]["corrective_precision"]["n"]}/{s["cells"]}]' if s['metrics']['corrective_precision']['n']==s['cells'] else ''),pct(s['pooled']['corrective_precision']),count(s,'C'),count(s,'R'),count(s,'H'),count(s,'W'),integer(s,'net'),integer(s,'right_minus_wrong')] for s in ss])
        lines+=['','### 分类：完整eval1000与S={argmax pR≠argmax pGL}子集','']
        lines+=table(['配置','eval1000 pR%','eval1000 w/o pR%','eval1000完整%','S占eval1000%','S人数Σ/均值','S内pR%','S内w/o pR%','S内完整%','S非空配置数'],[[NAMES[s['variant']],*[mean(s,k) for k in ('pR_accuracy','without_pR_accuracy','joint_accuracy','subset_fraction')],count(s,'subset_count'),*[mean(s,k) for k in ('subset_pR_accuracy','subset_without_pR_accuracy','subset_joint_accuracy')],str(s['metrics']['subset_pR_accuracy']['n'])+'/'+str(s['cells'])] for s in ss])
        lines+=['','### 补充：未做接受筛选的完整Query分支','', '这张表是全分支标签准确率，不是上方实际接受伪标签精度；D全分支精度与D-PR@100%一致。','']
        lines+=table(['配置','A全分支精度%','A全分支人数Σ/均值','D全分支精度%','D全分支人数Σ/均值','全Query标签精度%','Query人数Σ/均值'],[[NAMES[s['variant']],mean(s,'pseudo_full_A_precision'),count(s,'A_population'),mean(s,'pseudo_full_D_precision'),count(s,'D_population'),mean(s,'pseudo_full_precision'),count(s,'Q')] for s in ss])
    overall=[s for s in summary if s['dataset']=='ALL']
    lines+=['','## 5. 与旧AUROC口径的核对','', '旧报告AUROC来自校准后保存分数；本表只用于溯源，并不替代上方要求的校准前结果。相同排序的正向仿射或sigmoid不改变AUROC，但负斜率、并列和数值饱和可能导致不同，故不能未经检查直接复用。','']
    old=[]
    for s in overall:
        g=[r for r in allrows if r['variant']==s['variant']]
        t=stat([r['router_calibrated']['trust_auroc'] for r in g]);p=stat([r['router_calibrated']['preference_auroc'] for r in g])
        old.append([NAMES[s['variant']],mean(s,'trust_raw',True),f'{t["mean"]:.4f}' if t['mean'] is not None else 'NA',mean(s,'preference_raw',True),f'{p["mean"]:.4f}' if p['mean'] is not None else 'NA'])
    lines+=table(['配置','本次Trust校准前','旧Trust校准后均值','本次Pref校准前','旧Pref校准后均值'],old)
    lines+=['','## 6. 结果文件与复现入口','',
        '- '+code('experiments/comprehensive_report_20261029/AGENTS.md',None,'本次要求与运行边界')+'；'+code('experiments/comprehensive_report_20261029/comprehensive.py',None,'冻结推理、指标计算与封存代码')+'；'+code('experiments/comprehensive_report_20261029/render_report.py',None,'汇总及本Markdown生成代码')+'。',
        '- [逐配置600行CSV（525模型＋75配对B0）](https://github.com/yaolalal/EnhanceRouter/blob/main/reports/comprehensive-b0-3x2-20261029/per_cell.csv)。',
        '- [总体及各dataset×shot汇总CSV](https://github.com/yaolalal/EnhanceRouter/blob/main/reports/comprehensive-b0-3x2-20261029/summary.csv)，含均值/标准差/有效n/总数量/均数量。',
        '- [完整JSON及来源哈希](https://github.com/yaolalal/EnhanceRouter/blob/main/reports/comprehensive-b0-3x2-20261029/RESULTS.json)；[75组逐节点证据](https://github.com/yaolalal/EnhanceRouter/tree/main/reports/comprehensive-b0-3x2-20261029/cells)。',
        '- [本次完整归档下载](https://github.com/yaolalal/EnhanceRouter/releases/tag/comprehensive-b0-3x2-20261029)。归档包含报告、全部分组统计、C/R/H/W节点及eval1000标签/子集证据、代码与登记。原模型权重、图缓存和原始head分数二进制保留服务器，路径与SHA256在JSON中列出。',
        '- 原始运行：runs/comprehensive-b0-3x2-20261029-v1；历史模型运行：runs/stage0-20261026-v2及runs/joint-factorial-3x2-20261029-v1；根目录为/data1/wujingyao-20354/wujingyao/EnhanceRouter（原/home工程路径为兼容软链接）。',
        '', '旧L1预算争议与阶段1仍暂缓；本报告不依据Query真值调参或接纳新方法。','']
    text='\n'.join(lines)
    assert text.count('### Router：校准前AUROC，校准后D-PR@k%')==16
    assert text.count('### 分类：完整eval1000与S=')==16
    return text
