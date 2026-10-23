"""Audited byte-identical checkpoint imports; missing imports never retrain."""
from pathlib import Path
import shutil
import torch
from eargtc.joint_cross_repair.artifacts import read_json,write_json,sha256,verify_files


def import_member(source,receipt,destination,old_digest,new_digest,settings,s,p):
    source,receipt,destination=map(Path,(source,receipt,destination))
    if not source.is_file() or not receipt.is_file():raise FileNotFoundError('required preserved checkpoint is missing')
    old=read_json(receipt)
    if old['config_sha256']!=old_digest:raise ValueError('source checkpoint CONFIG mismatch')
    if sha256(source)!=old['sha256']:raise ValueError('source checkpoint hash mismatch')
    value=torch.load(source,map_location='cpu',weights_only=True)
    if value['settings']!=settings or value.get('test_labels_read') is not False or not torch.equal(value['validation_ids'],s['selection_ids']):raise ValueError('source member settings or validation identity changed')
    expected={'gold':s['gold_ids'],'pseudo':torch.cat([p['teach_g'],p['teach_l'],p['agreement']]),'keep':p['unselected']}
    if set(value['training_ids'])!=set(expected):raise ValueError('source training keys changed')
    for name,ids in expected.items():
        if not torch.equal(value['training_ids'][name],ids):raise ValueError('source training pool changed')
    marker=destination.with_suffix('.json')
    note={'sha256':old['sha256'],'config_sha256':new_digest,'imported_from':{'path':str(source.resolve()),'receipt':str(receipt.resolve()),'config_sha256':old_digest,'receipt_sha256':sha256(receipt),'checkpoint_sha256':old['sha256']}}
    if marker.exists():
        if read_json(marker)!=note or not destination.is_file() or sha256(destination)!=old['sha256']:raise ValueError('imported checkpoint receipt/content differs')
    else:
        if destination.exists():raise ValueError('orphan imported checkpoint')
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
        if sha256(destination)!=old['sha256']:raise ValueError('checkpoint copy mismatch')
        write_json(marker,note)
    return value


def import_bindings(source):
    from eargtc.joint_explore_three_router.experiment import verify_config,check_complete
    source=Path(source).resolve();config=verify_config(source);digest=sha256(source/'CONFIG.json')
    if config['smoke'] or config['candidate_strategy']!='full':raise ValueError('formal full-candidate source required')
    if (source/'PRETEST_AUDIT.json').exists() or (source/'evaluation').exists() or (source/'evaluation.staging').exists():raise ValueError('checkpoint source must precede test evaluation')
    bindings={**config['source_code'],**config['input_bindings'],str(source/'CONFIG.json'):digest};total=0;completed=0
    for rid in config['routers']:
        for ds in config['datasets']:
            cell=source/rid/ds;short=read_json(cell/'shortlist.json')
            expected={(key,42) for key in config['settings']}|{(key,seed) for key in short['retained'] for seed in [43,44]}
            files=list((cell/'bank').glob('*/*/seed*.pt'))
            observed={(str(p.parent.relative_to(cell/'bank')),int(p.stem[4:])) for p in files}
            if len(expected)!=40 or observed!=expected:raise ValueError('source must contain exact 24+16 members')
            for p in files:
                receipt=p.with_suffix('.json');note=read_json(receipt)
                if note['config_sha256']!=digest or note['sha256']!=sha256(p):raise ValueError('source member provenance failed')
                bindings[str(p)]=note['sha256'];bindings[str(receipt)]=sha256(receipt);total+=1
            for name in ['shortlist.json','selection.json']:
                bindings[str(cell/name)]=sha256(cell/name)
            if (cell/'complete.json').exists():
                check_complete(source,rid,ds);completed+=1
                bindings[str(cell/'complete.json')]=sha256(cell/'complete.json')
                bindings.update({str(cell/k):v for k,v in read_json(cell/'complete.json')['artifacts'].items()})
    if total!=600:raise ValueError('all600 imports required')
    return {'root':str(source),'config_sha256':digest,'expected_neural_members':600,'completed_source_cells':completed},bindings


def verify_import_receipt(destination,source,old_digest,new_digest):
    destination,source=map(Path,(destination,source))
    original_receipt=source.with_suffix('.json');old=read_json(original_receipt);note=read_json(destination.with_suffix('.json'))
    source_hash=sha256(source);destination_hash=sha256(destination)
    expected={'sha256':source_hash,'config_sha256':new_digest,'imported_from':{'path':str(source.resolve()),'receipt':str(original_receipt.resolve()),'config_sha256':old_digest,'receipt_sha256':sha256(original_receipt),'checkpoint_sha256':source_hash}}
    if old['config_sha256']!=old_digest or old['sha256']!=source_hash or destination_hash!=source_hash or note!=expected:raise ValueError('independent import audit failed')
    return True
