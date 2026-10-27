"""Export sealed experiment outputs for review; never train or select models."""
from d3_recovery import *
setup()
import csv, io, tarfile
from datetime import datetime, timezone
import torch

gate = read(RUN/'STAGE0_MODELS_READY_20261027.json')
assert gate['all_models_complete'] and not gate['stage0_decision_complete']
assert len(gate['seals']) == 1350
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out = RUN/'cloud-review'/stamp
out.mkdir(parents=True)
archive = out/'experiment-results.tar.gz'
variants = ['b0', *[f'd1/{f}' for f in (.25,.5,.75,1.)],
            *[f'd1_numeric/{f}' for f in (.25,.5,.75,1.)], 'd3',
            *[f'd2{a}/{n}' for a in ('a','b') for n in (50,200,1000)],
            'd4/confidence_router','d4/confidence_preference']
manifest = []

def add_bytes(tar, name, data):
    entry = tarfile.TarInfo(name)
    entry.size = len(data)
    entry.mode = 0o644
    tar.addfile(entry, io.BytesIO(data))

def scalar(value, index):
    if torch.is_tensor(value):
        return value[index].item()
    return value[index]

def check_keys(value):
    if isinstance(value,dict):
        for key,item in value.items():
            assert str(key).lower() not in {'password','access_token','refresh_token','api_key','authorization','client_secret'}, key
            check_keys(item)
    elif isinstance(value,list):
        for item in value: check_keys(item)

with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
    for sh,se,ds in CELLS:
        ids = list(evaluation_ids(se,ds))
        ranks = {int(node):i for i,node in enumerate(ids)}
        assert len(ranks)==1000
        wanted = torch.tensor(ids,dtype=torch.long)
        for variant in variants:
            path = variant_folder(sh,se,ds,variant)
            receipt = path/'COMPLETE.json'
            assert digest(receipt)==gate['seals'][str(receipt)]
            record = read(receipt)
            display = variant.replace('d1/','d1_frozen/',1) if variant.startswith('d1/') else variant
            prefix = f'shot{sh}/seed{se}/{ds}/{display}'
            item = dict(key=key(sh,se,ds),variant=display,source_path=str(path),
                        seal_sha256=digest(receipt),source_files=record['files'],
                        exported_text_files={},exports={})
            for rel,h in record['files'].items():
                source = path/rel
                if source.suffix.lower() not in ('.json','.csv','.tsv','.txt','.md'):continue
                assert digest(source)==h,(source,'changed text artifact')
                data=source.read_bytes()
                if source.suffix.lower()=='.json':check_keys(json.loads(data))
                add_bytes(tar,prefix+'/'+rel,data)
                item['exported_text_files'][rel]=h
            add_bytes(tar,prefix+'/COMPLETE.json',receipt.read_bytes())
            score_csv=io.StringIO(newline='')
            fields=['eval_order','node_id','branch','trust','preference','score','route','accepted']
            writer=csv.DictWriter(score_csv,fieldnames=fields,lineterminator='\n');writer.writeheader()
            coverage=set(); score_rows=[]
            for branch in ('agreement','disagreement'):
                score_path=path/f'{branch}-scores.pt'
                assert digest(score_path)==record['files'][score_path.name]
                scores=load(score_path)
                indices=torch.where(torch.isin(scores['ids'].long(),wanted))[0].tolist()
                for i in indices:
                    node=int(scores['ids'][i]);assert node not in coverage;coverage.add(node)
                    row=dict(eval_order=ranks[node],node_id=node,branch=branch)
                    for name in fields[3:]:
                        if name in scores:row[name]=scalar(scores[name],i)
                    score_rows.append(row)
                del scores
            assert coverage==set(ranks),(path,'eval1000 coverage')
            writer.writerows(sorted(score_rows,key=lambda row:row['eval_order']))
            data=score_csv.getvalue().encode();add_bytes(tar,prefix+'/eval1000-router-scores.csv',data)
            item['exports']['eval1000-router-scores.csv']=hashlib.sha256(data).hexdigest()
            if variant in ('b0','d3','d4/confidence_router','d4/confidence_preference') or variant.startswith('d1_numeric/'):
                pred=path/'predictions.pt'
                if pred.exists():assert digest(pred)==record['files']['predictions.pt']
                else:
                    reference=read(path/'predictions-reference.json');pred=Path(reference['path'])
                    assert digest(pred)==reference['sha256']
                probability=load(pred)['probability'][wanted]
                confidence,label=probability.max(1)
                result=io.StringIO(newline='');writer=csv.writer(result,lineterminator='\n')
                writer.writerow(['eval_order','node_id','predicted_class','confidence'])
                writer.writerows((i,int(node),int(label[i]),float(confidence[i])) for i,node in enumerate(ids))
                data=result.getvalue().encode();add_bytes(tar,prefix+'/eval1000-final-predictions.csv',data)
                item['exports']['eval1000-final-predictions.csv']=hashlib.sha256(data).hexdigest()
                del probability,confidence,label
            manifest.append(item)
        print('REVIEW_EXPORT_CELL',sh,se,ds,flush=True)
    metadata_blob=json.dumps(dict(cells=75,variants=1350,artifacts=manifest,
        scope='Sealed text artifacts, eval1000 router outputs and available final predictions',
        binary_weights_and_full_graph_caches='Source paths and hashes retained; binary payloads remain on server',
        L1_budget='deferred',Stage0_decision_complete=False,test_outcomes_used_for_selection=False),indent=2).encode()
    add_bytes(tar,'MANIFEST.json',metadata_blob)
write(out/'INDEX.json',dict(cells=75,variants=len(manifest),archive=str(archive),
    archive_size=archive.stat().st_size,archive_sha256=digest(archive),
    entries=[dict(key=x['key'],variant=x['variant'],source_path=x['source_path'],seal_sha256=x['seal_sha256'],exports=x['exports']) for x in manifest],
    training_performed=False,model_selection_performed=False,L1_budget='deferred',Stage0_decision_complete=False))
seal(out,dict(all_checks_pass=True,cells=75,variants=1350,source_gate_sha256=digest(RUN/'STAGE0_MODELS_READY_20261027.json'),export_code_sha256=digest(__file__),training_performed=False))
print('REVIEW_EXPORT_COMPLETE',out,archive.stat().st_size,digest(archive),flush=True)
