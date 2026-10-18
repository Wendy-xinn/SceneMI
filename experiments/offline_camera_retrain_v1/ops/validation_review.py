"""Persistent paired VAL checkpoint review, with independent CPU completion.

The immutable cohort and original model/config identities are prepared on CPU.
One GPU job evaluates case/regime/model pairs and commits each prediction before
advancing. Completion releases the GPU lock and invokes a fresh CPU verifier.
This review never trains, opens TEST outputs, or promotes a checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from contextlib import contextmanager
import datetime as dt
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .dependencies import dependency_closure,sha256

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[1]
DEFAULT_OUT=BASE/'runs/native30_checkpoint_review_v1'
MAIN=BASE/'runs/native30_gait_v2_ego55k_oct15'
GPU_LOCK=Path('/tmp/scenemi-native30-gpu.lock')
ADAPTER_LOCK=Path('/tmp/scenemi-native30-adapter.lock')
MODULE='experiments.offline_camera_retrain_v1.ops.validation_review'
PROTOCOL='native30-paired-expanded-validation-review-v1'
LABELS=('candidate41k','final55k')


def now():
    return dt.datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def require(condition,message):
    if not condition:
        raise ValueError(message)


def status(out,state,**fields):
    write(out/'STATUS.json',dict(protocol=PROTOCOL,status=state,checked_at=now(),pid=os.getpid(),
        optimizer_updates=0,TEST_used=False,automatic_checkpoint_promotion=False,**fields))


@contextmanager
def gpu_lock():
    # The bounded adapter stage can intentionally bypass the old global flock.
    # Acquire its admission lock first (the same order as adapter_trial), so a
    # queued review cannot accidentally become a third GPU job after TEST ends.
    with ADAPTER_LOCK.open('a') as admission,GPU_LOCK.open('a') as shared:
        fcntl.flock(admission,fcntl.LOCK_EX)
        try:
            fcntl.flock(shared,fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(shared,fcntl.LOCK_UN)
        finally:fcntl.flock(admission,fcntl.LOCK_UN)


def closure():
    """Only explicit modules and their real repository-local import closure."""
    from vit_pytorch import vit
    return dependency_closure([Path(__file__),BASE/'ops/validation_cohort.py'],
        extra_files=[ROOT.parent/'diffusion-motion-inbetweening/sample/virtual_head_camera.py',Path(vit.__file__)],
        import_roots=[ROOT.parent/'diffusion-motion-inbetweening'])


def cpu_setup():
    import torch
    require(not torch.cuda.is_initialized(),'CPU preparation/recheck cannot initialize CUDA')
    torch.set_num_threads(2)


def configurations(out):
    config=read(out/'config.json')
    require(config['frames']==192 and config['fps']==30 and config['sampling']['ddim_steps']==20,
            'Review uses original native192/30Hz/DDIM20')
    require(config['physical_guidance']['enabled'] is True,'Original physics must remain ON')
    require(config['ego_head_only_probability']==1,'EGO review is head-only')
    require(config['scene']['memory_protocol']=='observed-static-recording-union-v3','Scene memory protocol differs')
    return config


def validation_data(config):
    from experiments.offline_camera_retrain_v1.native30_data import Native30Data
    from experiments.offline_camera_retrain_v1.prepare_native30 import window_options
    from experiments.offline_camera_retrain_v1.native30_surface_cache import CanonicalSurfaceCache
    data=Native30Data(BASE/config['train_manifest'],'validation',config['seed']+1,
        frames=192,memory_protocol=config['scene']['memory_protocol'],quality_config=config['gt_reliability'],
        exo_max_empty_frames=config.get('exo_max_empty_frames'),**window_options(config))
    data.scene_memory.cached_streams=1
    data.surface_cache=CanonicalSurfaceCache(data,limit=1)
    return data


def observation_builder(config,device='cpu'):
    from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
    from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
    return Native30ObservationBuilder(Native30SkinAnchorCache(device=device,limit=4),read(BASE/config['mount_defaults']),
                                      config['observation']['input_noise'])


def validate_entries(data,entries,config):
    require(data.split=='validation','Only official VAL may enter checkpoint selection review')
    require(len(entries)==306,'Registered review is RICH206 + TRUMANS50 + EgoBody50 windows')
    seen=set();counts=Counter();recordings=defaultdict(set)
    for index,row in enumerate(entries):
        require(row['case']==index and row['mode']=='ego' and list(row['joints'])==[15],'Noncanonical case or head-only mode')
        group=row['group'];dataset='egobody' if group in ('camera_wearer','interactee') else group
        require(row['dataset']==dataset and dataset in ('rich','trumans','egobody'),'Dataset/group mismatch')
        meta,_=data.members[group][row['member']]
        require(meta['split']=='validation' and row['sequence_id']==meta['sequence_id'],'VAL source identity differs')
        require(Path(row['path']).resolve()==Path(meta['path']).resolve(),'VAL bundle path differs')
        require(row['start'] in data.starts_for(group,row['member'],'ego'),'Illegal VAL indexed start')
        require(row['source_first']==meta['source_first']+row['start'],'VAL source clock differs')
        require(row['recording']==(meta.get('recording') or meta['sequence_id']),'VAL recording identity differs')
        key=(group,meta['sequence_id'],row['start']);require(key not in seen,'Duplicate VAL window');seen.add(key)
        seedkey=json.dumps([group,'ego',meta['sequence_id'],row['source_first']],separators=(',',':'))
        expected=(config['validation']['seed']+int.from_bytes(hashlib.sha256(seedkey.encode()).digest()[:4],'little'))%2**32
        require(row['case_seed']==expected,'VAL seed differs from original identity-derived rule')
        counts[dataset]+=1;recordings[dataset].add(row['recording'])
    require(dict(counts)==dict(rich=206,trumans=50,egobody=50),'Registered dataset coverage differs')
    require(len(recordings['rich'])==27,'All27 eligible RICH VAL recordings are required')
    require(all(row['dataset']=='rich' for row in entries[:206]),'RICH must execute before other datasets')
    require(len({row['recording'] for row in entries[:27]})==27,'First27 windows must cover27 different RICH recordings')
    return dict(windows=dict(counts),recordings={k:len(v) for k,v in recordings.items()},
                warning='Window counts are not independent recording counts; adjacent windows can overlap.')


def metadata_identity(data,config):
    paths=[data.manifest,*data.window_index_files,BASE/config['mount_defaults']]
    # Native30Data has already checked official VAL membership, original clock,
    # and persisted legal windows. Every lazily read source array is verified
    # against the corresponding immutable metadata output SHA by data.load.
    paths += [Path(meta['path'])/'metadata.json' for meta in data.scene_memory.entries]
    # Explicit native topology assets affect both visible skin anchors and the
    # known body profile. Seal only the exact model/gender pairs in VAL.
    for model,gender in {(meta['native_body']['model'],meta['native_body']['gender']) for meta in data.scene_memory.entries}:
        if model=='smpl':paths.append(ROOT.parent/'ProtoMotions/data/smpl'/f'SMPL_{gender.upper()}.pkl')
        elif model=='smplx':paths.append(ROOT/'body_models/smplx'/f'SMPLX_{gender.upper()}.npz')
        else:raise ValueError('Unsupported declared native body model: '+model)
    require(all(meta['split']=='validation' for meta in data.scene_memory.entries),'Non-VAL scene-memory source')
    return {str(Path(path).resolve()):sha256(path) for path in sorted(set(paths))}


def model_declarations(path=None):
    source=read(path or MAIN/'final_review/MODELS.json')
    aliases={'best41k':'candidate41k','last55k':'final55k'}
    result={aliases.get(label,label):dict(row) for label,row in source.items()}
    require(set(result)==set(LABELS),'Declare only original41k and final55k')
    for label,step in zip(LABELS,(41000,55000)):
        row=result[label]
        require(row['expected_step']==step and isinstance(row['checkpoint_sha256'],str) and len(row['checkpoint_sha256'])==64,
                'Checkpoint step and exact SHA must be explicit')
        row['checkpoint']=str(Path(row['checkpoint']).resolve())
        require(sha256(row['checkpoint'])==row['checkpoint_sha256'],'Declared checkpoint bytes differ: '+label)
    return result


def prepare(out,models_path=None):
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model
    from experiments.offline_camera_retrain_v1.train_native30 import checkpoint_config,model_digest
    cpu_setup();out.mkdir(parents=True,exist_ok=True)
    require(not (out/'PLAN.json').exists(),'An immutable prepared PLAN already exists; evaluate or resume it')
    status(out,'preparing_cpu')
    config=configurations(out);cohort=read(out/'COHORT.json');entries=cohort['entries']
    data=validation_data(config);coverage=validate_entries(data,entries,config)
    metadata=metadata_identity(data,config);models=model_declarations(models_path)
    for path,expected in cohort['source_sha256'].items():
        require(sha256(path)==expected,'Cohort source changed before preparation: '+path)
    directory=out/'checkpoints';directory.mkdir(exist_ok=True)
    for label in LABELS:
        declaration=models[label];source=Path(declaration['checkpoint']);target=directory/(label+'.pt')
        # A killed CPU preparation may have already pinned one candidate. Its
        # exact published SHA is sufficient to reuse it; different bytes fail.
        if not target.exists():os.link(source,target)
        require(sha256(target)==declaration['checkpoint_sha256'],'Pinned checkpoint bytes differ')
        state=torch.load(target,map_location='cpu',mmap=True,weights_only=False)
        require(int(state['step'])==declaration['expected_step'] and checkpoint_config(state)==config,
                'Actual checkpoint step/config differs')
        model=make_model(config);model.load_state_dict(state['model'],strict=True)
        require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()),'Nonfinite checkpoint')
        declaration.update(source_checkpoint=str(source),checkpoint=str(target.resolve()),
                           model_tensor_sha256=model_digest(model),strict_CPU_load=True,finite=True)
        del model,state;gc.collect()
    input_probe=probe_inputs(data,observation_builder(config),entries[0],config)
    plan=dict(protocol=PROTOCOL,prepared_at=now(),cohort_sha256=sha256(out/'COHORT.json'),
        config_sha256=sha256(out/'config.json'),config=config,models=models,model_order=list(LABELS),
        coverage=coverage,windows=306,regimes=['clean','noisy'],planned_predictions=1224,
        frames=192,fps=30,ddim_steps=20,scene='on',physical_guidance='on',mode='ego',joints=[15],
        CPU_input_probe=input_probe,
        execution_order='First27 RICH recordings one window each; remainingRICH;TRUMANS;EgoBody. case/regime/model interleaved.',
        metadata_sha256=metadata,cohort_source_sha256=cohort['source_sha256'],source_sha256=closure(),
        array_validation='Native30Data uses published per-array SHA and content-addressed verified mirrors. Existing DrvFS mirrors are reused without rehash; no claim of a fresh full-array byte audit. No TEST arrays opened.',
        selection=dict(primary='RICH27-recording macro WorldMPJPE, arithmetic mean of clean and noisy',
            secondary='Separate clean/noisy, root-relative/PA, visible/hidden, paired per-recording differences and foot guards',
            foot_speed_tolerance='final<=41k+max(.01m/s,5%) for GT low-speed3D foot speed per regime',
            foot_articulation_tolerance='Bilateral RMS and stride errors final<=41k+max(5mm,10%) per regime',
            automatic_promotion=False,partial_results_select_checkpoint=False),TEST_used=False,optimizer_updates=0)
    write(out/'PLAN.json',plan)
    status(out,'prepared',planned_predictions=1224,completed_predictions=0,coverage=coverage)
    require(not torch.cuda.is_initialized(),'CPU preparation unexpectedly initialized CUDA')
    return plan


def verify_plan(out,plan,*,data=None):
    require(plan['protocol']==PROTOCOL and plan['planned_predictions']==1224,'Unexpected review protocol/budget')
    require(sha256(out/'COHORT.json')==plan['cohort_sha256'],'Cohort changed after preparation')
    require(sha256(out/'config.json')==plan['config_sha256'],'Config changed after preparation')
    require(configurations(out)==plan['config'],'Config identity differs')
    for path,expected in {**plan['source_sha256'],**plan['metadata_sha256'],**plan['cohort_source_sha256']}.items():
        require(sha256(path)==expected,'Sealed source/VAL metadata changed: '+path)
    for label,declaration in plan['models'].items():
        require(sha256(declaration['checkpoint'])==declaration['checkpoint_sha256'],'Frozen checkpoint changed: '+label)
    if data is not None:
        require(metadata_identity(data,plan['config'])==plan['metadata_sha256'],'Current VAL metadata coverage changed')
        validate_entries(data,read(out/'COHORT.json')['entries'],plan['config'])


def schedule(entries,labels=LABELS):
    return [(entry,regime,label) for entry in entries for regime in ('clean','noisy') for label in labels]


def row_key(row):
    return row['case'],row['regime'],row['model']


def committed_rows(out,plan,entries):
    path=out/'rows.jsonl'
    if not path.exists():
        return []
    content=path.read_bytes();lines=content.splitlines(keepends=True)
    # A killed process can leave only the final JSON line uncommitted. Retain
    # those bytes separately before truncating; completed NPZs remain reusable.
    if lines and not lines[-1].endswith(b'\n'):
        tail=lines.pop();digest=hashlib.sha256(tail).hexdigest()
        (out/('TRUNCATED_ROW_TAIL_'+digest+'.bin')).write_bytes(tail)
        write(out/'TRUNCATED_ROW_TAIL.json',dict(checked_at=now(),bytes=len(tail),sha256=digest))
        with path.open('r+b') as stream:stream.truncate(sum(map(len,lines)));stream.flush();os.fsync(stream.fileno())
    rows=[json.loads(line) for line in lines]
    expected={ (e['case'],regime,label):(e,regime,label) for e,regime,label in schedule(entries,plan['model_order']) }
    seen=set()
    for row in rows:
        key=row_key(row);require(key in expected and key not in seen,'Unexpected/duplicate committed prediction');seen.add(key)
        entry,regime,label=expected[key]
        require(row['checkpoint_sha256']==plan['models'][label]['checkpoint_sha256'],'Result model identity differs')
        require(row['checkpoint_step']==plan['models'][label]['expected_step'],'Result checkpoint step differs')
        require(row['seed']==entry['case_seed'] and all(row[k]==entry[k] for k in ('dataset','group','recording','sequence_id','source_first','start')),
                'Result input identity differs')
        artifact=out/'motions'/f"case{entry['case']:04d}_{regime}_{label}.npz"
        require(Path(row['artifact']).resolve()==artifact.resolve(),'Committed artifact path differs')
        require(sha256(artifact)==row['artifact_sha256'],'Committed prediction artifact changed')
    return rows


def tensor_fingerprint(known):
    digest=hashlib.sha256()
    for key,value in tensor_leaves(known):
        array=value.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([key,str(array.dtype),list(array.shape)],separators=(',',':')).encode())
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def tensor_leaves(value,path=''):
    import torch
    if isinstance(value,torch.Tensor):yield path,value
    elif isinstance(value,dict):
        for key in sorted(value):yield from tensor_leaves(value[key],path+'.'+str(key))
    elif isinstance(value,(tuple,list)):
        for index,item in enumerate(value):yield from tensor_leaves(item,path+f'[{index}]')


def assert_input_device(known,device):
    import torch
    expected=torch.device(device)
    if expected.type=='cuda' and expected.index is None:expected=torch.device('cuda',torch.cuda.current_device())
    for path,value in tensor_leaves(known):
        require(value.device==expected,'Inference tensor device differs: '+path)


def collated_inputs(config,observed,scene,asset,device):
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    # The original collator moves sample tensors but deliberately passes native
    # skin-assets through. The builder uses the requested device, and this
    # assertion also checks nested skinning tensors before any model call.
    batch=collate_observations([(observed,scene,asset,'ego')],device=device)
    known=inference_inputs(config,batch);assert_input_device(known,device)
    return batch,known


def inference_inputs(config,batch):
    """Use the original explicit model-input whitelist; never a full sample."""
    from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs
    known=model_inputs(config,batch)
    require(not any(key in known for key in ('motion','joints','contact_target','contact_valid')) and
            not any(key.startswith('gt_') for key in known),'Inference contains hidden body labels')
    return known


def probe_inputs(data,builder,entry,config):
    import numpy as np
    import torch
    result={}
    for regime in ('clean','noisy'):
        sample,observed,scene,asset=source_observation(data,builder,entry,regime)
        require(all(not key.startswith('gt_') and key not in ('motion','joints','contact_target','contact_valid') for key in observed),
                'Observation builder retained GT-only labels')
        available=observed['observation_meta'][...,:2]>0
        require(not available[:,[j for j in range(22) if j!=15]].any(),'Non-head joint sensor in EGO review')
        batch,known=collated_inputs(config,observed,scene,asset,'cpu')
        require(all(bool(torch.isfinite(value).all()) for _,value in tensor_leaves(known)),
                'Nonfinite CPU inference input')
        result[regime]=dict(case=entry['case'],source_first=entry['source_first'],CPU_input_sha256=tensor_fingerprint(known),
            tensor_keys=sorted(k for k,v in known.items() if isinstance(v,torch.Tensor)),
            head_position_available_frames=int(available[:,15,0].sum()),head_only=True,GT_labels_removed=True)
    require(not torch.cuda.is_initialized(),'CPU input preflight unexpectedly initialized CUDA')
    return result


def source_observation(data,builder,entry,regime):
    import numpy as np
    from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
    sample,identity=data.sample(entry['group'],'ego',member=entry['member'],start=entry['start'],yaw=0)
    # Restrict inference construction to sensors/profile; hidden body labels
    # are retained separately only for persisted evaluation references.
    sensors={k:v for k,v in sample.items() if k not in ('motion','joints','contact_target','contact_valid') and not k.startswith('gt_')}
    selected=builder if regime=='clean' else Native30ObservationBuilder(builder.skin_cache,builder.mounts,
                                     dict(builder.noise_config,clean_probability=0.))
    observed,scene,asset=selected.build(data,sensors,identity,'ego',(15,),
                        noise_rng=None if regime=='clean' else np.random.default_rng(entry['case_seed']))
    return sample,observed,scene,asset


def metrics_for(saved):
    import numpy as np
    import torch
    from .support_metrics import support_metrics
    from .foot_trial_evaluation import extended_metrics
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    prediction,truth,rest,observed,meta=(saved[key] for key in ('prediction','truth','rest','observations','observation_meta'))
    result=support_metrics(prediction,truth,rest,observed,meta)
    result.update(extended_metrics(prediction,truth,rest))
    result['visible_mpjpe_mm']=result['conditioned_mpjpe_mm'];result['hidden_mpjpe_mm']=result['unconditioned_mpjpe_mm']
    p,g=(torch.from_numpy(np.array(value,dtype=np.float32,copy=True))[None] for value in (prediction,truth))
    r=torch.from_numpy(np.array(rest,dtype=np.float32,copy=True))[None]
    with torch.no_grad():
        jp,jg=(forward_kinematics(value,r)[0].numpy().astype(np.float64) for value in (p,g))
    vp,vg=(np.diff(value[:,[10,11]],axis=0)*30 for value in (jp,jg));low=np.linalg.norm(vg,axis=-1)<.2
    result['foot_joint_speed_gt_low_m_s']=float(np.linalg.norm(vp,axis=-1)[low].mean()) if low.any() else None
    T=len(prediction);visible=int((meta[...,0]>0).sum());lowcount=int(low.sum())
    counts=dict(w_mpjpe_mm=T*22,mpjpe_mm=T*22,root_relative_mpjpe_mm=T*22,pa_mpjpe_mm=T*22,
        conditioned_mpjpe_mm=visible,visible_mpjpe_mm=visible,unconditioned_mpjpe_mm=T*22-visible,hidden_mpjpe_mm=T*22-visible,
        root_position_error_mm=T,head_orientation_deg=T,root_orientation_deg=T,
        mpjve_mm_s=(T-1)*22,acceleration_error_mm_s2=(T-2)*22,
        foot_joint_speed_gt_low_m_s=lowcount,foot_speed_during_GT_stance_m_s=lowcount,
        foot_joint_horizontal_speed_gt_low_m_s=lowcount,foot_joint_vertical_speed_gt_low_m_s=lowcount,
        foot_joint_y_abs_error_gt_low_mm=lowcount,foot_joint_y_signed_error_gt_low_mm=lowcount,
        foot_joint_y_abs_error_mm=T*2,foot_joint_y_signed_error_mm=T*2,
        bilateral_motion_rms_error_mm=1,bilateral_stride_change_abs_error_mm=1)
    return result,counts


def aggregate(rows):
    groups={}
    for model in LABELS:
        for dataset in ('rich','trumans','egobody','all'):
            for regime in ('clean','noisy'):
                chosen=[r for r in rows if r['model']==model and r['regime']==regime and (dataset=='all' or r['dataset']==dataset)]
                if not chosen:continue
                metrics={}
                for key in sorted({key for row in chosen for key in row['metrics']}):
                    valid=[row for row in chosen if row['metrics'].get(key) is not None]
                    if not valid:continue
                    recs=defaultdict(list)
                    for row in valid:recs[(row['dataset'],row['recording'])].append(row)
                    def mean(items):
                        counts=[row.get('counts',{}).get(key) for row in items]
                        return (sum(row['metrics'][key]*count for row,count in zip(items,counts))/sum(counts)
                                if all(c is not None and c>0 for c in counts) else statistics.mean(row['metrics'][key] for row in items))
                    counted=[row for row in valid if row.get('counts',{}).get(key,0)>0]
                    total=sum(row['counts'][key] for row in counted)
                    metrics[key]=dict(window_mean=statistics.mean(row['metrics'][key] for row in valid),
                        recording_macro=statistics.mean(mean(items) for items in recs.values()),windows=len(valid),recordings=len(recs),
                        count_weighted=sum(row['metrics'][key]*row['counts'][key] for row in counted)/total if total else None,
                        count=total if total else None)
                groups['/'.join((model,dataset,regime))]=dict(windows=len(chosen),recordings=len({r['recording'] for r in chosen}),metrics=metrics)
    return groups


def equal_dataset_macro(groups):
    result={}
    for model in LABELS:
        for regime in ('clean','noisy'):
            selected={dataset:groups[model+'/'+dataset+'/'+regime] for dataset in ('rich','trumans','egobody')
                      if model+'/'+dataset+'/'+regime in groups}
            if len(selected)!=3:continue
            keys=set.intersection(*(set(group['metrics']) for group in selected.values()))
            result[model+'/'+regime]={key:statistics.mean(group['metrics'][key]['recording_macro'] for group in selected.values()) for key in sorted(keys)}
    return result


def paired_recording_differences(rows):
    """Use only identical case/regime pairs, then aggregate within recording."""
    indexed={row_key(row):row for row in rows};paired=[]
    for (case,regime,label),a in indexed.items():
        if label!=LABELS[0] or (case,regime,LABELS[1]) not in indexed:continue
        b=indexed[case,regime,LABELS[1]]
        require(a['input_sha256']==b['input_sha256'],'Paired full input hashes differ')
        require(a['counts']==b['counts'],'Paired metric denominators differ')
        paired.extend((a,b))
    result=[]
    recording_names=sorted({(r['dataset'],r['recording'],r['regime']) for r in paired})
    for dataset,recording,regime in recording_names:
        selected=[r for r in paired if (r['dataset'],r['recording'],r['regime'])==(dataset,recording,regime)]
        scoped=aggregate(selected)
        first=scoped[LABELS[0]+'/'+dataset+'/'+regime];last=scoped[LABELS[1]+'/'+dataset+'/'+regime]
        metrics={key:dict(candidate41k=first['metrics'][key]['recording_macro'],final55k=last['metrics'][key]['recording_macro'],
                    delta_final_minus_candidate=last['metrics'][key]['recording_macro']-first['metrics'][key]['recording_macro'])
                 for key in first['metrics'] if key in last['metrics']}
        result.append(dict(dataset=dataset,recording=recording,regime=regime,paired_windows=first['windows'],metrics=metrics))
    return result


def comparison(groups,complete):
    result=dict(status='complete' if complete else 'partial_do_not_select',automatic_promotion=False,
                primary='RICH recording-macro WorldMPJPE, clean/noisy arithmetic mean',models={})
    for label in LABELS:
        keys=[label+'/rich/'+regime for regime in ('clean','noisy')]
        if all(key in groups for key in keys):
            result['models'][label]=dict(primary_world_mpjpe_mm=statistics.mean(groups[key]['metrics']['w_mpjpe_mm']['recording_macro'] for key in keys),
                                        regimes={regime:groups[label+'/rich/'+regime]['metrics']['w_mpjpe_mm']['recording_macro'] for regime in ('clean','noisy')})
    if len(result['models'])==2:
        result['delta_final55k_minus_candidate41k_mm']=result['models']['final55k']['primary_world_mpjpe_mm']-result['models']['candidate41k']['primary_world_mpjpe_mm']
    guards=[]
    for regime in ('clean','noisy'):
        a=groups.get('candidate41k/rich/'+regime);b=groups.get('final55k/rich/'+regime)
        if not a or not b:continue
        for metric,absolute,relative in [('foot_joint_speed_gt_low_m_s',.01,.05),('bilateral_motion_rms_error_mm',5.,.1),('bilateral_stride_change_abs_error_mm',5.,.1)]:
            if metric not in a['metrics'] or metric not in b['metrics']:
                guards.append(dict(regime=regime,metric=metric,passed=None,status='no_valid_samples'));continue
            x=a['metrics'][metric]['recording_macro'];y=b['metrics'][metric]['recording_macro'];limit=x+max(absolute,relative*x)
            guards.append(dict(regime=regime,metric=metric,candidate41k=x,final55k=y,maximum_final=limit,passed=y<=limit))
    result['RICH_foot_guards']=guards
    return result


def summarize(out,plan,rows,verified=False):
    complete=len(rows)==plan['planned_predictions'];groups=aggregate(rows)
    result=dict(protocol=PROTOCOL,status='complete_cpu_verified' if complete and verified else 'sampling_complete_pending_cpu' if complete else 'partial',
        checked_at=now(),completed_predictions=len(rows),planned_predictions=plan['planned_predictions'],groups=groups,
        comparison=comparison(groups,complete and verified),scope=plan['coverage'],TEST_used=False,
        equal_dataset_macro=equal_dataset_macro(groups),paired_per_recording=paired_recording_differences(rows),
        aggregation='Count-weighted only with explicit valid denominators; equal-window means; count-weighted within recording then equal-recording macro. all pools recordings; equal_dataset_macro averages the three dataset recording macros equally.',
        overlapping_windows_are_not_independent=True,automatic_checkpoint_promotion=False)
    write(out/'SUMMARY.json',result)
    lines=['# 扩大 VAL 双 checkpoint 配对复核','',
        f"状态：{result['status']}，{len(rows)}/{plan['planned_predictions']} 次预测。没有训练或自动替换 best。",'',
        'RICH 为全部206窗/27个录制；TRUMANS、EgoBody各50窗，实际独立录制数另列。每窗 clean/noisy、原41k/最终55k成对评价。',
        '输入固定头部观测、可见场景、已知体型；30Hz/192帧/DDIM20，原物理修正开启。TEST不参与。','',
        '| 模型 | 数据 | 输入 | 窗数 | 录制数 | World录制宏均值mm | PA录制宏均值mm | GT低速3D脚速m/s |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for label in LABELS:
        for dataset in ('rich','trumans','egobody'):
            for regime in ('clean','noisy'):
                group=groups.get('/'.join((label,dataset,regime)))
                if group:
                    m=group['metrics'];fmt=lambda key:f"{m[key]['recording_macro']:.3f}" if key in m else 'n/a'
                    lines.append(f"| {label} | {dataset} | {regime} | {group['windows']} | {group['recordings']} | {fmt('w_mpjpe_mm')} | {fmt('pa_mpjpe_mm')} | {fmt('foot_joint_speed_gt_low_m_s')} |")
    lines+=['','主判断是RICH27录制WorldMPJPE宏均值的clean/noisy平均；同时检查脚速、双足运动RMS误差和stride误差。',
            '部分结果不用于选checkpoint；GT低速只是运动学代理，不能作为真实接触或全mesh穿模率。完整输出须CPU_RECHECK通过。']
    (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    return result


def progress(out,plan,rows,elapsed):
    recent=rows[-min(12,len(rows)):]
    per_prediction=statistics.mean(row['sampling_s']+row['metric_s']+row['preparation_share_s'] for row in recent) if recent else None
    remaining=plan['planned_predictions']-len(rows)
    result=dict(checked_at=now(),completed_predictions=len(rows),planned_predictions=plan['planned_predictions'],elapsed_this_process_s=elapsed,
        timing_predictions=len(recent),recent_seconds_per_prediction=per_prediction,
        estimated_remaining_sampling_s=remaining*per_prediction if per_prediction is not None else None,
        estimate_scope='Measured recent DDIM+metric+shared input prep; CPU final recheck and queue/startup excluded. Throughput can change.',
        unique_windows_started=len({row['case'] for row in rows}),RICH_recordings_started=len({row['recording'] for row in rows if row['dataset']=='rich'}))
    write(out/'PROGRESS.json',result)
    if len(rows)==12:write(out/'FIRST_12_TIMING.json',result)
    return result


def evaluate(out,resume=False,max_predictions=0,memory_fraction=.5):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model
    from experiments.offline_camera_retrain_v1.train_native30 import model_digest
    from experiments.offline_camera_retrain_v1.native30_refinement import sample_motion
    plan=read(out/'PLAN.json');verify_plan(out,plan);entries=read(out/'COHORT.json')['entries']
    require(0<memory_fraction<=.5,'Review GPU memory fraction must be at most one half')
    require(resume or not (out/'rows.jsonl').exists(),'Existing results require explicit resume')
    rows=committed_rows(out,plan,entries);done={row_key(row):row for row in rows}
    if len(rows)==plan['planned_predictions']:
        return True
    status(out,'waiting_for_gpu',completed_predictions=len(rows),planned_predictions=plan['planned_predictions'])
    with gpu_lock():
            verify_plan(out,plan)
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
            torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(memory_fraction)
            torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
            config=plan['config'];data=validation_data(config);verify_plan(out,plan,data=data);builder=observation_builder(config,device='cuda')
            models={}
            for label in plan['model_order']:
                declaration=plan['models'][label];state=torch.load(declaration['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
                model=make_model(config);model.load_state_dict(state['model'],strict=True)
                require(model_digest(model)==declaration['model_tensor_sha256'],'Strict model tensor digest differs')
                models[label]=model.cuda().eval();del state
            started=time.monotonic();generated=0;folder=out/'motions';folder.mkdir(exist_ok=True)
            for entry in entries:
                for regime in plan['regimes']:
                    if all((entry['case'],regime,label) in done for label in plan['model_order']):continue
                    prepare_started=time.monotonic();sample,observed,scene,asset=source_observation(data,builder,entry,regime)
                    batch,known=collated_inputs(config,observed,scene,asset,'cuda')
                    input_sha=tensor_fingerprint(known);prep_share=(time.monotonic()-prepare_started)/len(plan['model_order'])
                    paired=[row for key,row in done.items() if key[:2]==(entry['case'],regime)]
                    require(all(row['input_sha256']==input_sha for row in paired),'Resumed matched input tensors differ')
                    for label in plan['model_order']:
                        key=entry['case'],regime,label
                        if key in done:continue
                        status(out,'sampling',completed_predictions=len(rows),planned_predictions=plan['planned_predictions'],
                               current_case=entry['case'],dataset=entry['dataset'],regime=regime,model=label)
                        before=time.monotonic()
                        with torch.no_grad():prediction=sample_motion(models[label],known,config,steps=20,seed=entry['case_seed'],use_scene=True)
                        torch.cuda.synchronize();sampling_s=time.monotonic()-before
                        require(bool(torch.isfinite(prediction).all()),'Nonfinite DDIM prediction')
                        artifact=folder/f"case{entry['case']:04d}_{regime}_{label}.npz";temporary=artifact.with_suffix('.tmp')
                        with temporary.open('wb') as stream:
                            np.savez_compressed(stream,prediction=prediction[0].detach().cpu().numpy(),truth=sample['motion'],rest=sample['rest'],
                                observations=observed['trajectory'],observation_meta=observed['observation_meta'])
                            stream.flush();os.fsync(stream.fileno())
                        temporary.replace(artifact);before=time.monotonic()
                        with np.load(artifact,allow_pickle=False) as saved:metrics,counts=metrics_for(saved)
                        row=dict(case=entry['case'],dataset=entry['dataset'],group=entry['group'],mode='ego',model=label,
                            recording=entry['recording'],sequence_id=entry['sequence_id'],source_first=entry['source_first'],start=entry['start'],
                            regime=regime,seed=entry['case_seed'],checkpoint_step=plan['models'][label]['expected_step'],
                            checkpoint_sha256=plan['models'][label]['checkpoint_sha256'],input_sha256=input_sha,
                            artifact=str(artifact.resolve()),artifact_sha256=sha256(artifact),metrics=metrics,counts=counts,
                            sampling_s=sampling_s,metric_s=time.monotonic()-before,preparation_share_s=prep_share,checked_at=now())
                        with (out/'rows.jsonl').open('a') as stream:
                            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                        rows.append(row);done[key]=row;generated+=1
                        progress(out,plan,rows,time.monotonic()-started)
                        if len(rows)%12==0 or len(rows)==plan['planned_predictions']:summarize(out,plan,rows)
                        if max_predictions and generated>=max_predictions:
                            summarize(out,plan,rows);status(out,'partial_requested',completed_predictions=len(rows),planned_predictions=plan['planned_predictions'])
                            return False
                    del sample,observed,scene,asset,batch,known
            for label,model in models.items():
                require(model_digest(model)==plan['models'][label]['model_tensor_sha256'],'Frozen model tensors changed during review')
                require(all(parameter.grad is None for parameter in model.parameters()),'Unexpected inference model gradient')
            verify_plan(out,plan)
            write(out/'SAMPLING_VERIFY.json',dict(status='passed',checked_at=now(),predictions=len(rows),
                model_tensor_digests_unchanged=True,optimizer_updates=0,source_sha256_matches=True))
            del models,model,prediction,data,builder;gc.collect();torch.cuda.empty_cache()
    summarize(out,plan,rows);status(out,'sampling_completed_pending_cpu',completed_predictions=len(rows),planned_predictions=plan['planned_predictions'])
    return True


def independent_core(saved):
    """NumPy core formulas, separate from raw_metrics/standard_motion_metrics."""
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    p,g=(torch.from_numpy(np.array(saved[k],dtype=np.float32,copy=True))[None] for k in ('prediction','truth'))
    rest=torch.from_numpy(np.array(saved['rest'],dtype=np.float32,copy=True))[None]
    with torch.no_grad():jp,jg=(forward_kinematics(value,rest)[0].numpy().astype(np.float64) for value in (p,g))
    error=np.linalg.norm(jp-jg,axis=-1)*1000;visible=saved['observation_meta'][...,0]>0
    mean=lambda value,mask:float(value[mask].mean()) if mask.any() else None
    vp,vg=(np.diff(value[:,[10,11]],axis=0)*30 for value in (jp,jg));low=np.linalg.norm(vg,axis=-1)<.2
    return dict(w_mpjpe_mm=float(error.mean()),visible_mpjpe_mm=mean(error,visible),hidden_mpjpe_mm=mean(error,~visible),
        root_relative_mpjpe_mm=float(np.linalg.norm((jp-jp[:,:1])-(jg-jg[:,:1]),axis=-1).mean()*1000),
        foot_joint_speed_gt_low_m_s=mean(np.linalg.norm(vp,axis=-1),low))


def recheck(out):
    import numpy as np
    import torch
    from .foot_trial_recheck import independent_metrics
    cpu_setup();plan=read(out/'PLAN.json');verify_plan(out,plan);entries=read(out/'COHORT.json')['entries']
    data=validation_data(plan['config']);verify_plan(out,plan,data=data);builder=observation_builder(plan['config'])
    rows=committed_rows(out,plan,entries);require(len(rows)==plan['planned_predictions'],'Full review not yet sampled')
    indexed={row_key(row):row for row in rows};maximum=independent_maximum=0.;checked=0
    status(out,'cpu_rechecking',completed_predictions=len(rows),planned_predictions=plan['planned_predictions'],verified_predictions=0)
    for entry in entries:
        for regime in plan['regimes']:
            sample,observed,scene,asset=source_observation(data,builder,entry,regime)
            paired_arrays=None;paired_counts=None;paired_fingerprint=None
            for label in plan['model_order']:
                row=indexed[entry['case'],regime,label]
                with np.load(row['artifact'],allow_pickle=False) as saved:
                    require(np.array_equal(saved['truth'],sample['motion']) and np.array_equal(saved['rest'],sample['rest']),
                            'Saved GT/rest differs from independently reloaded VAL source')
                    require(np.array_equal(saved['observations'],observed['trajectory']) and np.array_equal(saved['observation_meta'],observed['observation_meta']),
                            'Saved observed input differs from independently rebuilt VAL input')
                    inputs={key:saved[key].copy() for key in ('truth','rest','observations','observation_meta')}
                    actual,counts=metrics_for(saved);direct,_,denominators=independent_metrics(saved['prediction'],saved['truth'],saved['rest'],saved['observations'],saved['observation_meta'])
                    direct.update(independent_core(saved))
                require(counts==row['counts'],'Saved metric denominators differ')
                for key,value in row['metrics'].items():
                    require((value is None)==(actual[key] is None),'Metric missingness differs')
                    if value is not None:maximum=max(maximum,abs(value-actual[key]))
                for key,value in direct.items():
                    if key in actual:
                        require((value is None)==(actual[key] is None),'Independent metric missingness differs')
                        if value is not None:independent_maximum=max(independent_maximum,abs(value-actual[key]))
                if paired_arrays is not None:
                    require(all(np.array_equal(value,paired_arrays[key]) for key,value in inputs.items()),'Pair GT/rest/observations differ')
                    require(counts==paired_counts and row['input_sha256']==paired_fingerprint,'Pair denominator/full input fingerprint differs')
                paired_arrays=inputs;paired_counts=counts;paired_fingerprint=row['input_sha256'];checked+=1
            del sample,observed,scene,asset
            if checked%24==0:status(out,'cpu_rechecking',completed_predictions=len(rows),planned_predictions=plan['planned_predictions'],verified_predictions=checked)
    require(maximum<1e-6 and independent_maximum<1e-3,'CPU metric verification differs')
    require(not torch.cuda.is_initialized(),'CPU verifier unexpectedly initialized CUDA')
    verify_plan(out,plan)
    write(out/'CPU_RECHECK.json',dict(status='independent_cpu_passed',checked_at=now(),predictions=checked,
        planned_predictions=plan['planned_predictions'],CUDA_initialized=torch.cuda.is_initialized(),
        maximum_recompute_difference=maximum,maximum_independent_formula_difference=independent_maximum,
        source_checkpoint_metadata_SHA_matches=True,original_GT_rest_observations_exact=True,
        same_window_full_input_fingerprints_and_denominators_exact=True,rows_sha256=sha256(out/'rows.jsonl'),
        metric_scope='All saved metrics recomputed; independent NumPy World/PA/root-relative/visible/hidden/foot formulas after unchanged native FK.',
        TEST_used=False,optimizer_updates=0,automatic_checkpoint_promotion=False))
    summarize(out,plan,rows,verified=True);status(out,'completed',completed_predictions=checked,planned_predictions=plan['planned_predictions'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','evaluate','resume','recheck','status','run'))
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--models',type=Path)
    parser.add_argument('--memory-fraction',type=float,default=.5)
    parser.add_argument('--max-predictions',type=int,default=0)
    args=parser.parse_args(argv);out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.command=='status':
        print(json.dumps(read(out/'STATUS.json') if (out/'STATUS.json').exists() else dict(status='not_prepared'),ensure_ascii=False,indent=2));return
    acquired=False
    try:
        complete=False
        with (out/'job.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            acquired=True
            if args.command=='prepare':prepare(out,args.models)
            elif args.command=='recheck':recheck(out)
            else:
                if args.command=='run' and not (out/'PLAN.json').exists():prepare(out,args.models)
                complete=evaluate(out,resume=args.command in ('resume','run'),max_predictions=args.max_predictions,memory_fraction=args.memory_fraction)
        if complete:
            # Both GPU and job flocks are released before the fresh CPU child
            # takes its own job lock; no parent must remain interactively alive.
            subprocess.run([sys.executable,'-m',MODULE,'recheck','--out',str(out)],check=True)
    except BaseException as exc:
        if acquired:status(out,'failed',operation=args.command,error=repr(exc))
        raise


if __name__=='__main__':
    main()
