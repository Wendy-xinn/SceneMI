"""Fixed family-balanced validation for configuration screening, never final selection."""
from collections import defaultdict
import copy
import json
from pathlib import Path
import numpy as np
import torch
from scenemi.config import ROOT
from scenemi.data.amass_dataset import AmassDataset,collate
from scenemi.data.amass_prepare import atomic_json,sha256
from scenemi.data.checked_scene_dataset import SceneSelection,checked_ranges
from scenemi.data.exo_initialization import CONDITIONS,repair_example,repair_payload
from scenemi.data.native_repair import NativeRepairDataset
from scenemi.data.scene_dataset import valid_window_grid
from scenemi.exo_denoise import sample_denoising
from scenemi.metrics_batch import metric_rows,repair_rows
from scenemi.native_joint_data import SceneFreeEgoDataset,ego_payload,validation_cases
from scenemi.selection import validation_manifest
from scenemi.training import now
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from .data import CORPORA,normalize,ArrayCache


def prepare(model,paths,c,out):
    bundle=SceneSelection(ROOT/c['selection']);rng=np.random.default_rng(2026102805)
    data={};cases={};manifest=[];count=c['pilot']['validation_families_per_source']
    for mode in ('ego','exo'):
        native=SceneFreeEgoDataset.__new__(SceneFreeEgoDataset) if mode=='ego' else NativeRepairDataset.__new__(NativeRepairDataset)
        native.split='validation';native.common_pelvis=model.base.shape.rest_template[0].detach().cpu().numpy()
        native.selection_bundle=bundle;native.bundle=bundle;native.array=ArrayCache(bundle.array);native.records=[]
        if mode=='exo':native.bank=copy.deepcopy(model.base.native_body.bank).cpu().eval()
        groups=defaultdict(lambda:defaultdict(list))
        for key in ('native_manifest','trumans_exo_manifest','grail_validation'):
            for row in bundle.rows(key):
                if row['split']!='validation' or row['frames']<128:continue
                source='grail' if key=='grail_validation' else row['dataset']
                if source not in CORPORA[1:]:continue
                if source!='grail':
                    if mode not in row['allowed_modes'] or row['diagnostic_only']:continue
                    if mode=='ego' and key!='native_manifest':continue
                    if mode=='exo' and ((source=='trumans')!=(key=='trumans_exo_manifest')):continue
                family=row.get('object_family_id') or row.get('recording') or row['sequence_id']
                groups[source][family].append(row)
        chosen=[]
        for source in CORPORA[1:]:
            families=sorted(groups[source]);rng.shuffle(families);accepted=0
            for family in families:
                row=groups[source][family][0];meta=bundle.metadata(row,'validation')
                starts=checked_ranges(row,'validation',mode=mode) if source=='grail' else valid_window_grid(bundle.array(meta,'observation_valid'),128,1)
                if mode=='exo':
                    valid=np.asarray(bundle.array(meta,'exo_position_available')).any(-1);prefix=np.r_[0,np.cumsum(valid)]
                    starts=starts[prefix[starts+128]-prefix[starts]>=c['data']['minimum_tracked_frames_per_window']]
                if not len(starts):continue
                record=dict(path=row['path'],dataset=source,family=family,starts=starts)
                windows=validation_cases([record]);start=windows[len(windows)//2]['start']
                member=len(native.records);native.records.append(record)
                chosen.append(dict(source=source,family=family,member=member,start=start,case_id=len(chosen)))
                manifest.append(dict(mode=mode,source=source,family=family,path=row['path'],start=start,metadata_sha256=sha256(Path(row['path'])/'metadata.json')))
                accepted+=1
                if accepted==count:break
            if accepted!=count:raise ValueError('Insufficient distinct eligible validation families: '+source)
        amass=AmassDataset(ROOT/'runs/amass128/index/complete.json',paths,'val');all_cases=validation_manifest(amass.members)
        families=sorted({v['family'] for v in all_cases});rng.shuffle(families)
        for family in families[:count]:
            options=[v for v in all_cases if v['family']==family];case=options[len(options)//2]
            chosen.append(dict(source='amass',**case));manifest.append(dict(mode=mode,source='amass',**case))
        data[mode]=(native,amass);cases[mode]=chosen
    atomic_json(out/'PILOT_CASES.json',dict(scope='configuration_screen_only_not_checkpoint_selection',families_per_source=count,
        cases=manifest,full_validation=False,test_used=False,selection_sha256=sha256(ROOT/c['selection'])))
    return data,cases


def pairs_for(data,cases,mode,condition,c):
    native,amass=data;values=[]
    for case in cases:
        rng=np.random.default_rng(777+case['case_id']);source=case['source']
        if source=='amass':
            if mode=='ego':pair=amass.sample(rng,'ego',member=case['member'],start=case['start'],reflected=False,yaw=0.,temporal_dropout=None)
            else:pair=repair_example(amass,rng,c['frontend'],case=case,condition=condition)
        else:
            kw=dict(member=case['member'],start=case['start'],yaw=0.)
            pair=native.sample(rng,**kw) if mode=='ego' else native.sample(rng,c['frontend'],condition=condition,**kw)
        values.append(normalize(*pair,source,mode))
    return values


@torch.inference_mode()
def evaluate(model,data,cases,c,out,name):
    model.eval();rows=[];result={'ego':{},'exo':{}}
    for mode in ('ego','exo'):
        for condition in (('ego',) if mode=='ego' else CONDITIONS):
            batch_size=4 if mode=='ego' else 8
            for source in CORPORA:
                selected=[v for v in cases[mode] if v['source']==source];bucket=defaultdict(list)
                for start in range(0,len(selected),batch_size):
                    pairs=pairs_for(data[mode],selected[start:start+batch_size],mode,condition,c)
                    batch=collate([p[0] for p in pairs],'cuda')
                    truth=forward_kinematics(batch['motion'],batch['target_rest'])
                    if mode=='ego':
                        payload=ego_payload(batch);motion=sample_native30(model.base,payload,steps=20,seed=777,use_scene=False)
                        _,details=model.base(motion,torch.zeros(len(pairs),device='cuda',dtype=torch.long),payload,return_details=True)
                        variants={'repair':forward_kinematics(motion,details['rest'])}
                    else:
                        payload=repair_payload(batch);motion=sample_denoising(model,payload,steps=8,start_timestep=200,seed=777)
                        initial=forward_kinematics(batch['initial_motion'],batch['initial_rest'])
                        variants={'input':initial,'repair':forward_kinematics(motion,batch['initial_rest'])}
                    for variant,points in variants.items():
                        if mode=='ego':metrics=metric_rows(points,truth)
                        else:
                            visible=(batch['joint_visible']>0)&(batch['estimate_available'][...,None]>0)
                            metrics=repair_rows(points,truth,initial,visible,condition)
                        bucket[variant].extend(metrics)
                        rows.extend(dict(mode=mode,source=source,condition=condition,variant=variant,identity=p[1],metrics=m) for p,m in zip(pairs,metrics))
                target=result['ego'] if mode=='ego' else result['exo'].setdefault(condition,{})
                target[source]={v:{key:float(np.mean([r[key] for r in items])) for key in items[0]} for v,items in bucket.items()}
                atomic_json(out/'STATUS.json',dict(state='screening',candidate=name,mode=mode,condition=condition,source=source,updated_at_utc=now(),full_validation=False))
    result.update(full_validation=False,test_used=False,identity_max_change_mm=max(r['metrics']['max_change_from_initial_mm'] for r in rows if r['mode']=='exo' and r['condition']=='identity' and r['variant']=='repair'))
    folder=out/name;folder.mkdir(exist_ok=True)
    with (folder/'metrics.jsonl').open('w') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    atomic_json(folder/'SCREEN.json',result);return result


def compare(result,baseline):
    checks={'identity_preserved':result['identity_max_change_mm']<=.001}
    for source in CORPORA:
        value=result['ego'][source]['repair']['mpjpe_mm'];base=baseline['ego'][source]['repair']['mpjpe_mm']
        checks['ego_'+source]=value<=base+(5 if source=='amass' else 0)
    for condition in ('tracking','occlusion','gap'):
        for source in CORPORA:
            repair=result['exo'][condition][source]['repair'];initial=result['exo'][condition][source]['input']
            for key in ('mpjpe_mm','root_error_mm','tracked_visible_mpjpe_mm'):
                checks[f'{condition}_{source}_{key}']=repair[key]<=initial[key]+1
            if source=='amass':checks[condition+'_amass_retained']=repair['repair_region_mpjpe_mm']<=baseline['exo'][condition][source]['repair']['repair_region_mpjpe_mm']+1
    ego=np.mean([result['ego'][s]['repair']['mpjpe_mm']/baseline['ego'][s]['repair']['mpjpe_mm'] for s in CORPORA[1:]])
    exo=np.mean([result['exo'][v][s]['repair']['repair_region_mpjpe_mm']/baseline['exo'][v][s]['repair']['repair_region_mpjpe_mm'] for v in ('tracking','occlusion','gap') for s in CORPORA[1:]])
    return dict(checks=checks,screen_eligible=all(checks.values()),score=float(.5*ego+.5*exo),full_validation=False)
