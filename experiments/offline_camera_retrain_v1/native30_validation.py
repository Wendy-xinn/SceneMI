"""Fixed validation windows, original-GT metrics, preserved training RNG.

This is short-trial monitoring, not full-recording benchmark evaluation. The
visibility grouping is conditioned/unconditioned joints, not inferred image
visibility of the unseen ego wearer. GT candidate metrics never hide raw errors.
"""
import json
import hashlib
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs,sample_motion
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations


def fixed_plan(data,config,head_only=False):
    if data.split!='validation':raise ValueError('Monitoring must use validation, never TRAIN/test')
    result=[]
    for group,members in data.members.items():
        for mode in ('ego','exo'):
            eligible=sorted([i for i,(m,_) in enumerate(members) if mode in m.get('allowed_modes',['ego','exo']) and len(data.starts_for(group,i,mode))],
                            key=lambda i:members[i][0]['sequence_id'])
            for i in eligible[:config['members_per_group_mode']]:
                meta,_=members[i];starts=data.starts_for(group,i,mode);start=int(starts[len(starts)//2])
                for joints in (((15,),) if head_only else ((15,),(15,20,21))) if mode=='ego' else (None,):
                    result.append(dict(group=group,mode=mode,member=i,start=start,joints=joints,
                        sequence_id=meta['sequence_id'],source_first=meta['source_first']+start))
    if not result:raise ValueError('Empty validation plan')
    return result


SUBSET_PROTOCOL='fixed-plus-rotating-recordings-v1'


def monitoring_plan(data,config,step):
    """Outcome-independent recording panels; deterministic even after resume."""
    options=config['validation']
    if options.get('subset_protocol') is None:
        return fixed_plan(data,options,head_only=config['ego_head_only_probability']==1)
    if data.split!='validation' or options['subset_protocol']!=SUBSET_PROTOCOL:
        raise ValueError('Rotating monitoring must use registered official VAL protocol')
    if config['ego_head_only_probability']!=1:
        raise ValueError('This monitoring protocol registers head-only ego')
    result=[];round_index=step//options['interval_steps']
    fixed=options['fixed_recordings_per_dataset_mode'];rotating=options['rotating_recordings_per_dataset_mode']
    if min(fixed,rotating)<1:raise ValueError('Fixed and rotating panel sizes must be positive')
    for dataset in ('trumans','egobody','rich'):
        groups=('camera_wearer','interactee') if dataset=='egobody' else (dataset,)
        for mode in ('ego','exo'):
            records={}
            for group in groups:
                for i,(meta,_) in enumerate(data.members[group]):
                    if mode in meta.get('allowed_modes',['ego','exo']) and len(data.starts_for(group,i,mode)):
                        records.setdefault(meta.get('recording') or meta['sequence_id'],[]).append((group,i))
            seed=options['seed']+int.from_bytes(hashlib.sha256((dataset+'/'+mode).encode()).digest()[:4],'little')
            ordered=np.random.default_rng(seed).permutation(sorted(records)).tolist()
            if len(ordered)<fixed+rotating:raise ValueError('Insufficient distinct VAL recordings for '+dataset+'/'+mode)
            remaining=ordered[fixed:]
            selected=[('fixed',r,0) for r in ordered[:fixed]]
            selected += [('rotating',remaining[(round_index*rotating+j)%len(remaining)],
                          (round_index*rotating+j)//len(remaining)) for j in range(rotating)]
            for panel,record,cycle in selected:
                members=sorted(records[record]);roles=sorted({g for g,_ in members})
                role=roles[(round_index+cycle)%len(roles)] if panel=='rotating' else roles[0]
                members=[v for v in members if v[0]==role];group,i=members[cycle%len(members)]
                starts=data.starts_for(group,i,mode);meta,_=data.members[group][i]
                offset=(len(starts)//2+cycle)%len(starts);start=int(starts[offset])
                entry=dict(group=group,mode=mode,member=i,start=start,joints=(15,) if mode=='ego' else None,
                    sequence_id=meta['sequence_id'],source_first=meta['source_first']+start,
                    dataset=dataset,recording=record,panel=panel,monitor_round=round_index)
                key=json.dumps([group,mode,meta['sequence_id'],entry['source_first']],separators=(',',':'))
                entry['case_seed']=(options['seed']+int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little'))%(2**32)
                result.append(entry)
    return result


def aggregate_monitoring(rows):
    """Count-weighted metrics, keeping changing panels and input modes separate."""
    buckets={}
    for row in rows:
        key='/'.join((row.get('panel','fixed'),row['mode'],row['regime'],row.get('scene_variant','scene_on')))
        target=buckets.setdefault(key,{})
        for name,value in row['metrics'].items():
            if value['count']:
                sums=target.setdefault(name,[0.,0]);sums[0]+=value['value']*value['count'];sums[1]+=value['count']
    return {key:{name:dict(value=sums[0]/sums[1],count=sums[1]) for name,sums in values.items()} for key,values in buckets.items()}


def raw_metrics(prediction,batch,conditioned):
    joints=forward_kinematics(prediction,batch['rest']);target=forward_kinematics(batch['motion'],batch['rest'])
    error=(joints-target).norm(dim=-1)*1000
    def metric(values,mask):
        mask=mask.bool();n=int(mask.sum())
        return dict(value=float(values[mask].mean()) if n else None,count=n)
    quality=batch.get('gt_position_weight',torch.ones_like(error))
    rotation=global_rotations(prediction);truth_rotation=global_rotations(batch['motion'])
    cosine=((rotation*truth_rotation).sum((-1,-2))-1)/2
    angle=torch.rad2deg(torch.acos(cosine.clamp(-1,1)))
    rotation_quality=batch.get('gt_global_rotation_weight',torch.ones_like(error))
    velocity=torch.diff(joints,dim=1)*30;true_velocity=torch.diff(target,dim=1)*30
    stance=true_velocity[:,:,(10,11)].norm(dim=-1)<.2
    return dict(mpjpe_mm=metric(error,torch.ones_like(conditioned)),
        conditioned_mpjpe_mm=metric(error,conditioned),unconditioned_mpjpe_mm=metric(error,~conditioned),
        candidate_position_mpjpe_mm=metric(error,quality<1),regular_position_mpjpe_mm=metric(error,quality>=1),
        root_position_error_mm=metric(error[:,:,0],torch.ones_like(conditioned[:,:,0])),
        root_orientation_deg=metric(angle[:,:,0],torch.ones_like(conditioned[:,:,0])),
        head_orientation_deg=metric(angle[:,:,15],torch.ones_like(conditioned[:,:,15])),
        candidate_rotation_deg=metric(angle,rotation_quality<1),regular_rotation_deg=metric(angle,rotation_quality>=1),
        foot_speed_during_GT_stance_m_s=metric(velocity[:,:,(10,11)].norm(dim=-1),stance))


def validate(model,data,builder,config,folder,step):
    options=config['validation'];plan=monitoring_plan(data,config,step);rows=[];saved={};identities=[]
    device=next(model.parameters()).device;devices=[device.index or 0] if device.type=='cuda' else []
    was_training=model.training
    # DDIM seeds may otherwise change training dropout/noise. Dataset sampling
    # specifies member/start/yaw and uses no sampler RNG; observation RNG is local.
    with torch.random.fork_rng(devices=devices),torch.no_grad():
        try:
            model.eval()
            for i,entry in enumerate(plan):
                seed=entry.get('case_seed',options['seed']+i)
                sample,identity=data.sample(entry['group'],entry['mode'],member=entry['member'],start=entry['start'],yaw=0)
                clean,scene,asset=builder.build(data,sample,identity,entry['mode'],entry['joints'])
                for regime in ('clean','noisy'):
                    observed=clean
                    if regime=='noisy':
                        # Fixed application branch; TRAIN has the configured
                        # 50% clean mixture, validation reports both separately.
                        from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
                        noise=dict(builder.noise_config,clean_probability=0.)
                        other=Native30ObservationBuilder(builder.skin_cache,builder.mounts,noise)
                        observed,scene,asset=other.build(data,sample,identity,entry['mode'],entry['joints'],noise_rng=np.random.default_rng(seed))
                    batch=collate_observations([(observed,scene,asset,entry['mode'])],device=device)
                    conditioned=torch.from_numpy(observed['observation_meta'][...,0]>0)[None].to(device)
                    known=model_inputs(config,batch)
                    prediction=sample_motion(model,known,config,steps=config['sampling']['ddim_steps'],seed=seed)
                    if not torch.isfinite(prediction).all():raise FloatingPointError('Nonfinite DDIM validation output')
                    available=np.asarray(observed['observation_meta'])[...,0]>0
                    confidence=np.asarray(observed['observation_meta'])[...,2]
                    input_stats=dict(available_joint_fraction=float(available.mean()),
                        mean_position_confidence=float(confidence[available].mean()) if available.any() else None)
                    rows.append(dict(case=i,step=step,regime=regime,**entry,input_stats=input_stats,
                        metrics=raw_metrics(prediction,batch,conditioned)))
                    if 'reference_motion' in known:
                        rows[-1]['reference_metrics']=raw_metrics(known['reference_motion'],batch,conditioned)
                        saved[f'{i}_{regime}_reference_motion']=known['reference_motion'][0].cpu().numpy()
                    saved[f'{i}_{regime}_prediction']=prediction[0].cpu().numpy()
                    saved[f'{i}_{regime}_observations']=observed['trajectory']
                    saved[f'{i}_{regime}_observation_meta']=observed['observation_meta']
                    interval=options.get('scene_ablation_interval_steps',0)
                    if interval and (step==1 or step%interval==0) and entry.get('panel')=='fixed':
                        off=sample_motion(model,known,config,steps=config['sampling']['ddim_steps'],seed=seed,use_scene=False)
                        if not torch.isfinite(off).all():raise FloatingPointError('Nonfinite scene-OFF validation output')
                        rows.append(dict(case=i,step=step,regime=regime,**entry,scene_variant='scene_off',input_stats=input_stats,
                            metrics=raw_metrics(off,batch,conditioned)))
                        saved[f'{i}_{regime}_scene_off_prediction']=off[0].cpu().numpy()
                saved[f'{i}_truth']=sample['motion'];saved[f'{i}_rest']=sample['rest'];identities.append(identity)
        finally:model.train(was_training)
    folder=Path(folder)
    with (folder/'validation.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(step=step,scope=options.get('scope','fixed windows; original unweighted GT'),
            cases=rows,summary=aggregate_monitoring(rows)),allow_nan=False)+'\n')
    saved['identities_json']=np.array(json.dumps(identities));saved['step']=np.array(step)
    temporary=folder/'validation_latest.tmp.npz';np.savez_compressed(temporary,**saved);temporary.replace(folder/'validation_latest.npz')
    (folder/'validation_plan.json').write_text(json.dumps(dict(seed=options['seed'],plan=plan,GT_metrics_unweighted=True),indent=2)+'\n')
    return rows
