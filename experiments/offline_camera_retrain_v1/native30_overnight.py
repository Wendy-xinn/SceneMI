"""Bounded overnight pilot, matched scene control, and evidence-gated full training.

TRAIN windows are selected before fitting, official validation before evaluation.
No test set is opened. A failed scene gate never starts full-corpus training.
Only one resumable atomic last.pt per independent model is retained.
"""
import argparse
import copy
import datetime as dt
import json
import math
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from experiments.offline_camera_retrain_v1.data import HERE,rotation6d
from experiments.offline_camera_retrain_v1.prepare_native30 import (
    read_config,make_model,window_options,write_json,code_hashes,dataset_digest,sample_item)
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import (
    Native30ObservationBuilder,inverse_head,sanitize_observations)
from experiments.offline_camera_retrain_v1.native30_input_noise import apply_input_noise
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs,training_motion,sample_motion
from experiments.offline_camera_retrain_v1.native30_validation import raw_metrics
from experiments.offline_camera_retrain_v1.native30_losses import objective
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1.unified_geometry import occupancy_from_points,skin_points
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D,forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas


def scoped_data(config,split):
    return Native30Data(HERE/config['train_manifest'],split,config['seed'],frames=config['frames'],
        memory_protocol=config['scene']['memory_protocol'],quality_config=config['gt_reliability'],
        exo_max_empty_frames=config['exo_max_empty_frames'],**window_options(config))


def window_plan(data,members_per_group,windows_per_member):
    """Distinct recordings/targets; fixed temporal quantiles, no outcome selection."""
    result=[];recordings=set()
    for group,members in data.members.items():
        streams={}
        for i,(meta,_) in enumerate(members):
            key=meta.get('recording') or meta['sequence_id']
            for mode in ('ego','exo'):
                if len(data.starts_for(group,i,mode)):streams.setdefault(key,{}).setdefault(mode,[]).append(i)
        candidates=sorted(key for key,modes in streams.items() if all(mode in modes for mode in ('ego','exo')))
        # Choose a spread across scene IDs before repeats, while retaining the
        # fixed source ordering. EgoBody's two roles use distinct recordings.
        scenes=set();chosen=[]
        for repeat in (False,True):
            for key in candidates:
                if key in recordings:continue
                meta,_=members[streams[key]['ego'][0]]
                scene=meta.get('scene_id')
                if not repeat and scene in scenes:continue
                chosen.append(key);recordings.add(key);scenes.add(scene)
                if len(chosen)==members_per_group:break
            if len(chosen)==members_per_group:break
        if len(chosen)!=members_per_group:raise ValueError('Insufficient independent sources: '+group)
        for key in chosen:
            for mode in ('ego','exo'):
                # Real PV and master full are separate bundles with independent
                # clocks/validity grids; never require both modes in one file.
                i=max(streams[key][mode],key=lambda ix:len(data.starts_for(group,ix,mode)))
                starts=data.starts_for(group,i,mode)
                selected=[]
                for q in np.linspace(.25,.75,windows_per_member) if windows_per_member>1 else [.5]:
                    start=int(starts[int(q*(len(starts)-1))])
                    if any(abs(start-s)<data.frames for s in selected):
                        alternatives=[int(s) for s in starts if all(abs(int(s)-v)>=data.frames for v in selected)]
                        if not alternatives:continue
                        start=min(alternatives,key=lambda v:abs(v-start))
                    selected.append(start)
                    meta,_=members[i]
                    result.append(dict(group=group,mode=mode,member=i,start=start,
                        sequence_id=meta['sequence_id'],recording=meta.get('recording'),split=data.split))
    return result


def build_bank(data,builder,plan,config=None):
    bank=[];identities=[]
    for i,entry in enumerate(plan):
        sample,identity=data.sample(entry['group'],entry['mode'],member=entry['member'],start=entry['start'],yaw=0)
        item=builder.build(data,sample,identity,entry['mode'],(15,20,21) if entry['mode']=='ego' else None)
        if config and config.get('scene_relations',{}).get('enabled'):
            from experiments.offline_camera_retrain_v1.native30_scene_support import attach_support_supervision
            item=(attach_support_supervision(item[0],item[1],item[2]),item[1],item[2])
        bank.append((*item,entry['mode']));identities.append(identity)
        print('BANK',data.split,i+1,len(plan),entry['group'],entry['mode'],entry['sequence_id'],entry['start'],flush=True)
    return bank,identities


def audit_bank(bank,identities,builder,config):
    maximum=0.;rows=[];intervals={}
    for i,(item,identity) in enumerate(zip(bank,identities)):
        observed=item[0];available=observed['observation_meta'][...,0]>0
        rows.append(dict(case=i,group=identity['group'],mode=identity['mode'],
            mean_observed_joints=float(available.sum(-1).mean()),missing_joint_fraction=float((~available).mean()),
            static_points=len(item[1].points),contact_label_entries=int(observed['contact_valid'].sum())))
        target=(identity['dataset'],identity.get('recording') or identity['sequence_id'],identity['group'])
        start=identity['source_start_30fps'];intervals.setdefault(target,[]).append((start,start+192))
        augmented=augment(item,identity,builder,config,np.random.default_rng(500+i))
        motion=torch.from_numpy(augmented[0]['motion'])[None];rest=torch.from_numpy(augmented[0]['rest'])[None]
        error=float((forward_kinematics(motion,rest)[0]-torch.from_numpy(augmented[0]['joints'])*2).abs().max())
        maximum=max(maximum,error)
        if error>2e-5:raise AssertionError('Common-world yaw changed GT/FK alignment')
    unique=0
    for parts in intervals.values():
        merged=[]
        for lo,hi in sorted(parts):
            if merged and lo<=merged[-1][1]:merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
            else:merged.append((lo,hi))
        unique+=sum(hi-lo for lo,hi in merged)
    return dict(rows=rows,condition_windows=len(bank),window_frame_slots=len(bank)*192,
        unique_target_30hz_times=unique,distinct_target_streams=len(intervals),yaw_fk_max_error_m=maximum)


class YawSurface:
    """Rigid augmentation reuses base neighbor trees and preserves surface masks."""
    def __init__(self,base,rotation):
        self.base,self.rotation=base,rotation
        self.points=(base.points@rotation.T).astype(np.float32)
        self.dynamic_features=base.dynamic_features.copy()
        self.dynamic_features[...,:3]=base.dynamic_features[...,:3]@rotation.T
        for t,cloud in enumerate(base.dynamic):
            for j,owner in enumerate(base.dynamic_owner_ids):
                points=cloud['points'][cloud['owners']==owner]@rotation.T
                if len(points):self.dynamic_features[t,j,3:6]=points.max(0)-points.min(0)
        self.dynamic_features[...,6:9]=base.dynamic_features[...,6:9]@rotation.T
        self.dynamic_valid=base.dynamic_valid;self.dynamic_owner_ids=base.dynamic_owner_ids

    def query(self,vertices,rotations,neighbors=8):
        result=self.base.query(vertices@self.rotation,self.rotation.T@rotations,neighbors)
        for key in ('points','normals','velocities'):result[key]=result[key]@self.rotation.T
        return result


def augment(item,identity,builder,config,rng,*,noisy=True,yaw=True,head_only=None):
    base,scene,asset,mode=item;sample={k:v.copy() for k,v in base.items()}
    selected=(15,) if mode=='ego' and (head_only if head_only is not None else rng.random()<config['ego_head_only_probability']) else ((15,20,21) if mode=='ego' else tuple(range(22)))
    mask=np.zeros(22,bool);mask[list(selected)]=True
    sample['observation_meta'][...,:4]*=mask[None,:,None]
    if noisy:
        apply_input_noise(sample['trajectory'],sample['observation_meta'],sample['camera'],sample['camera_available'],
            sample['camera_confidence'],sample['observation_uncertainty'],mode,selected,rng,builder.noise_config)
        if mode=='ego':
            mount=identity['actual_virtual_mount'] or builder.mounts['camera_wearer']
            head,available,confidence=inverse_head(sample['camera'],sample['camera_available'],sample['camera_confidence'],mount)
            sample['trajectory'][:,15]=head;sample['observation_meta'][:,15,:2]=available;sample['observation_meta'][:,15,2:4]=confidence
            sample['observation_uncertainty'][:,15,0]+=np.linalg.norm(mount['translation_m'])*sample['observation_uncertainty'][:,15,1]
    trajectory,meta=sanitize_observations(torch.from_numpy(sample['trajectory']),torch.from_numpy(sample['observation_meta']))
    sample['trajectory'],sample['observation_meta']=trajectory.numpy(),meta.numpy()
    if yaw:
        r=Rotation.from_rotvec([0,float(rng.uniform(-math.pi,math.pi)),0]).as_matrix().astype(np.float32)
        sample['motion'][...,:3]=((sample['motion'][...,:3]*2+sample['rest'][0])@r.T-sample['rest'][0])/2
        six=sample['motion'][...,3:9]+np.asarray(IDENTITY_6D)
        root=np.stack((six[...,:3],six[...,3:],np.cross(six[...,:3],six[...,3:])), -1)
        sample['motion'][...,3:9]=rotation6d(r@root)-np.asarray(IDENTITY_6D)
        sample['motion'][...,135:]=(sample['motion'][...,135:].reshape(-1,22,3)@r.T).reshape(-1,66)
        sample['joints']=sample['joints']@r.T
        for key in ('support_points','support_normals','support_velocities','support_vertices'):
            if key in sample:sample[key]=sample[key]@r.T
        for key in ('trajectory','camera'):
            values=sample[key];values[...,:3]=values[...,:3]@r.T
            six=values[...,3:];matrix=np.stack((six[...,:3],six[...,3:],np.cross(six[...,:3],six[...,3:])), -1)
            values[...,3:]=rotation6d(r@matrix)
        sample['trajectory'],sample['observation_meta']=(x.numpy() for x in sanitize_observations(torch.from_numpy(sample['trajectory']),torch.from_numpy(sample['observation_meta'])))
        scene=YawSurface(scene,r);sample['occupancy']=occupancy_from_points(scene.points)
    return sample,scene,asset,mode


def mean_metrics(rows,mode,regime,key,field='metrics'):
    selected=[row[field][key] for row in rows if row['mode']==mode and row['regime']==regime]
    count=sum(m['count'] for m in selected)
    return sum(m['value']*m['count'] for m in selected if m['count'])/count if count else None


def surface_metrics(motion,batch):
    """Observed-plane/contact diagnostics; no closed-scene collision claim."""
    asset=batch['skin_assets'][0];scene=batch['surface_windows'][0]
    vertices=skin_points(motion[0],asset);labels=asset['labels']
    rotation=global_rotations(motion)[0,:,labels]
    query=scene.query(vertices.cpu().numpy(),rotation.cpu().numpy())
    point=torch.as_tensor(query['points'][:,:,2,0],device=motion.device)
    normal=torch.as_tensor(query['normals'][:,:,2,0],device=motion.device)
    valid=torch.as_tensor(query['valid'][:,:,2,0],device=motion.device)
    delta=vertices-point;distance=delta.norm(dim=-1);signed=(delta*normal).sum(-1)
    tangent=(delta-signed[...,None]*normal).norm(dim=-1)
    reliable=valid&torch.as_tensor(query['normal_reliable'][:,:,2,0],device=motion.device)&(distance<.25)&(tangent<.09)
    annotated=batch['contact_valid'][0,:,labels]&(batch['contact_target'][0,:,labels]>.5)
    def metric(values,mask):
        count=int(mask.sum());return dict(value=float(values[mask].mean()) if count else None,count=count)
    return dict(observed_plane_behind_proxy_mm=metric((-signed-.02).clamp_min(0)*1000,reliable),
        annotated_contact_distance_mm=metric(distance*1000,annotated&valid),
        annotated_contact_query_coverage=metric(valid.float(),annotated),
        query_anchor_coverage=metric(valid.float(),torch.ones_like(valid)),
        limitation='partial camera-facing planes; unknown regions have no collision measurement')


def evaluate(model,bank,identities,builder,config,folder,step,*,steps=20,seeds=(2026101401,),ablate=True,scene_modes=None):
    rows=[];saved={};folder.mkdir(parents=True,exist_ok=True)
    local_noise=copy.deepcopy(builder.noise_config);local_noise['clean_probability']=0.
    noisy_builder=Native30ObservationBuilder(builder.skin_cache,builder.mounts,local_noise)
    with torch.random.fork_rng(devices=[0]),torch.no_grad():
        model.eval()
        for i,(item,identity) in enumerate(zip(bank,identities)):
            for regime in ('clean','noisy'):
                observed=augment(item,identity,noisy_builder,config,np.random.default_rng(2026101402+i),
                    noisy=regime=='noisy',yaw=False,head_only=True)
                batch=collate_observations([observed]);known=model_inputs(config,batch)
                conditioned=batch['observation_meta'][...,0]>0
                reference_metrics=raw_metrics(known['reference_motion'],batch,conditioned)
                reference_scene=surface_metrics(known['reference_motion'],batch)
                truth_scene=surface_metrics(batch['motion'],batch)
                for seed in seeds:
                    modes=scene_modes or (('full','off') if ablate else ('off',))
                    for scene_mode in modes:
                        start=time.monotonic()
                        prediction=sample_motion(model,known,config,steps=steps,seed=seed+i,use_scene=scene_mode=='full')
                        if not torch.isfinite(prediction).all():raise FloatingPointError('Nonfinite validation motion')
                        rows.append(dict(case=i,group=identity['group'],mode=identity['mode'],sequence_id=identity['sequence_id'],
                            source_start=identity['source_start_30fps'],split='validation',regime=regime,seed=seed,
                            scene=scene_mode,metrics=raw_metrics(prediction,batch,conditioned),reference_metrics=reference_metrics,
                            scene_metrics=surface_metrics(prediction,batch),reference_scene_metrics=reference_scene,GT_scene_metrics=truth_scene))
                        print('EVAL',step,i,regime,seed,scene_mode,round(rows[-1]['metrics']['mpjpe_mm']['value'],2),
                            's',round(time.monotonic()-start,2),flush=True)
                        if seed==seeds[0] and scene_mode==('full' if ablate else 'off'):
                            saved[f'{i}_{regime}_prediction']=prediction[0].cpu().numpy()
                            saved[f'{i}_{regime}_observations']=batch['trajectory'][0].cpu().numpy()
                            saved[f'{i}_{regime}_observation_meta']=batch['observation_meta'][0].cpu().numpy()
                            saved[f'{i}_{regime}_reference_motion']=known['reference_motion'][0].cpu().numpy()
                        if seed==seeds[0]:saved[f'{i}_{regime}_{scene_mode}_prediction']=prediction[0].cpu().numpy()
                saved[f'{i}_truth']=batch['motion'][0].cpu().numpy();saved[f'{i}_rest']=batch['rest'][0].cpu().numpy()
    saved.update(identities_json=np.array(json.dumps(identities)),step=np.array(step))
    temp=folder/'validation_latest.tmp.npz';np.savez_compressed(temp,**saved);temp.replace(folder/'validation_latest.npz')
    report=dict(step=step,ddim_steps=steps,seeds=list(seeds),rows=rows,split='validation',GT_alignment=False)
    write_json(folder/'evaluation_latest.json',report)
    with (folder/'evaluation.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
    model.train();return report


def scene_gate(candidate,control):
    """Predeclared thresholds; benefit must survive both regimes and seeds."""
    checks={};full=[r for r in candidate['rows'] if r['scene']=='full'];off=[r for r in candidate['rows'] if r['scene']=='off']
    for mode in ('ego','exo'):
        for regime in ('clean','noisy'):
            key='mpjpe_mm' if mode=='ego' else 'unconditioned_mpjpe_mm'
            value=mean_metrics(full,mode,regime,key);initial=mean_metrics(full,mode,regime,key,'reference_metrics')
            without=mean_metrics(off,mode,regime,key);matched=mean_metrics(control['rows'],mode,regime,key)
            checks[f'{mode}/{regime}/better_than_initial']=dict(passed=value<=initial*.95,value=value,reference=initial,required_fraction=.95)
            checks[f'{mode}/{regime}/scene_gain']=dict(passed=value<=without*.98,value=value,reference=without,required_fraction=.98)
            checks[f'{mode}/{regime}/matched_control_gain']=dict(passed=value<=matched*.98,value=value,reference=matched,required_fraction=.98)
            for metric,tolerance in [('mpjpe_mm',5.),('conditioned_mpjpe_mm',5.),('root_position_error_mm',10.)]:
                if mode=='exo':
                    v=mean_metrics(full,mode,regime,metric);ref=mean_metrics(full,mode,regime,metric,'reference_metrics')
                    checks[f'{mode}/{regime}/guard/{metric}']=dict(passed=v<=ref+tolerance,value=v,reference=ref,tolerance=tolerance)
            # A pooled gain cannot conceal a reversal for a fixed DDIM seed.
            for seed in candidate['seeds']:
                f=[r for r in full if r['seed']==seed];o=[r for r in off if r['seed']==seed]
                v=mean_metrics(f,mode,regime,key);ref=mean_metrics(o,mode,regime,key)
                checks[f'{mode}/{regime}/seed/{seed}']=dict(passed=v<ref,value=v,reference=ref)
            positive=[]
            for corpus in ('egobody','trumans','rich'):
                groups=('camera_wearer','interactee') if corpus=='egobody' else (corpus,)
                v=mean_metrics([r for r in full if r['group'] in groups],mode,regime,key)
                ref=mean_metrics([r for r in off if r['group'] in groups],mode,regime,key)
                if v<ref:positive.append(corpus)
            checks[f'{mode}/{regime}/two_corpora']=dict(passed=len(positive)>=2,positive=positive)
    return dict(passed=all(row['passed'] for row in checks.values()),checks=checks,
        policy='held-out official VAL; two DDIM seeds; both regimes; scene-off and matched-training controls; trusted/root guards',
        candidate_step=candidate['step'],control_step=control['step'])


def train_steps(model,opt,config,coefficients,folder,start,stop,rng,counts,deadline,*,
                bank=None,identities=None,builder=None,data=None,scene=True,checkpoint_meta=None,monitor=None):
    alphas=cosine_alphas().cuda();began=time.monotonic();folder.mkdir(parents=True,exist_ok=True)
    full_plan=data.sampling_plan(.7) if data is not None else None
    by_mode={mode:[i for i,v in enumerate(bank or []) if v[-1]==mode] for mode in ('ego','exo')}
    def checkpoint(step):
        save_checkpoint(dict(step=step,model=model.state_dict(),optimizer=opt.state_dict(),config=config,counts=counts,
            condition_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            data_rng=data.sampler_state() if data is not None else None,meta=checkpoint_meta),folder/'last.pt')
        write_json(folder/'progress.json',dict(step=step,status='running',counts=counts,elapsed_s=time.monotonic()-began,
            full_corpus=data is not None,free_disk_bytes=shutil.disk_usage(HERE).free))
    with (folder/'trace.jsonl').open('a') as trace:
        step=start
        for step in range(start+1,stop+1):
            # Reserve 45 minutes for DDIM20 comparisons and native meshes.
            if time.time()>=deadline-45*60:
                step-=1;checkpoint(step);return step
            items=[];ids=[]
            for b in range(config['batch_size']):
                # Exact 7:3 in each ten sample slots, including mixed batches.
                slot=(step-1)*config['batch_size']+b;mode='ego' if slot%10<7 else 'exo'
                if data is None:
                    i=int(rng.choice(by_mode[mode]));items.append(augment(bank[i],identities[i],builder,config,rng));ids.append(
                        dict(group=identities[i]['group'],sequence_id=identities[i]['sequence_id'],start=identities[i]['start'],mode=mode))
                else:
                    plan=[dict(row,probability=row['probability']/sum(r['probability'] for r in full_plan if r['mode']==mode)) for row in full_plan if row['mode']==mode]
                    item,identity=sample_item(data,builder,config,plan,rng);items.append(item);ids.append(
                        dict(group=identity['group'],sequence_id=identity['sequence_id'],start=identity['start'],mode=mode))
            batch=collate_observations(items);known=model_inputs(config,batch);target=training_motion(config,batch,known)
            # Matched models get identical data, perturbations, t and Gaussian
            # noise even though scene dropout consumes different torch RNG.
            diffusion_rng=torch.Generator(device='cuda').manual_seed(config['seed']+step*31)
            t=torch.randint(1000,(len(items),),device='cuda',generator=diffusion_rng)
            low=torch.rand(len(items),device='cuda',generator=diffusion_rng)<config['noise']['extra_low_noise_probability']
            t=torch.where(low,torch.randint(config['noise']['low_noise_steps'],t.shape,device='cuda',generator=diffusion_rng),t)
            signal=alphas[t];noise=torch.randn(target.shape,device=target.device,generator=diffusion_rng)
            noisy=signal[:,None,None].sqrt()*target+(1-signal[:,None,None]).sqrt()*noise
            scene_draw=rng.random();enabled=scene and scene_draw>=config['scene']['dropout_probability']
            model.train();opt.zero_grad(set_to_none=True)
            prediction,details=model(noisy,t,known,use_scene=enabled,return_details=True)
            loss,terms=objective(prediction,batch['motion'],batch,signal,details,config,coefficients)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            lr=config['optimizer']['lr']*min(1.,step/100)*max(.2,.5*(1+math.cos(math.pi*min(step,12000)/12000)))
            for group in opt.param_groups:group['lr']=lr
            opt.step()
            for identity in ids:
                key=f"{identity['group']}/{identity['mode']}";counts[key]=counts.get(key,0)+1
            row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),lr=lr,identities=ids,scene=enabled,timesteps=t.tolist(),
                observation_checksum=float(batch['trajectory'].double().sum()),
                terms={group:{key:float(v.detach()) for key,v in values.items()} for group,values in terms.items()})
            trace.write(json.dumps(row)+'\n');trace.flush()
            if step==1 or step%25==0:print('STEP',folder.name,step,round(row['loss'],5),'seconds',round(time.monotonic()-began,1),counts,flush=True)
            if step==1 or step%100==0:checkpoint(step)
            if monitor is not None and step in (500,1000,2000):monitor(model,step)
        checkpoint(step);return step


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-utc',required=True)
    p.add_argument('--pilot-steps',type=int,default=3000);p.add_argument('--extension-steps',type=int,default=6000)
    p.add_argument('--full-steps',type=int,default=15000)
    p.add_argument('--continue-pilot',type=Path,help='Continue this official TRAIN-only pilot equally in candidate/control')
    args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Use a new trial directory; automatic restart cannot overwrite an existing run')
    config=read_config(args.config);deadline=dt.datetime.fromisoformat(args.deadline_utc).timestamp()
    prepared=HERE/config['prepared_output'];effective=json.loads((prepared/'effective_config.json').read_text())
    checks=json.loads((prepared/'input_checks.json').read_text());hashes=code_hashes()
    if not checks['training_ready'] or effective['config']!=config or effective['code_sha256']!=hashes:
        raise ValueError('Current full zero-step preparation must pass before any optimizer update')
    protocol=dict(status='building_banks',config=config,code_sha256=hashes,deadline_utc=args.deadline_utc,
        initialization='from scratch for matched candidate/control',pilot_steps=args.pilot_steps,extension_steps=args.extension_steps,
        full_steps=args.full_steps,ego_exo='exact 7:3 sample slots',test_used=False,
        physical_metrics='original unweighted world GT, no alignment; whole observed static recording and current dynamics',
        free_disk_before=shutil.disk_usage(HERE).free,checkpoint_policy='one atomic resumable last.pt per independent trial')
    write_json(out/'protocol.json',protocol)
    torch.set_num_threads(4);torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    data=scoped_data(config,'train');val=scoped_data(config,'validation')
    if dataset_digest(data)|dataset_digest(val)!=effective['data_sha256']:raise ValueError('Cache changed after preparation')
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),json.loads((HERE/config['mount_defaults']).read_text()),config['observation']['input_noise'])
    plan=window_plan(data,4,2);vplan=window_plan(val,2,1)
    write_json(out/'window_plan.json',dict(train=plan,validation=vplan,frames=192,stride=64,
        train_condition_windows=len(plan),train_window_frame_slots=len(plan)*192,
        note='ego/exo may share body time; frame slots are not unique source frames'))
    trainbank,trainids=build_bank(data,builder,plan,config);valbank,valids=build_bank(val,builder,vplan)
    write_json(out/'identities.json',dict(train=trainids,validation=valids))
    write_json(out/'bank_audit.json',dict(train=audit_bank(trainbank,trainids,builder,config),
        validation=audit_bank(valbank,valids,builder,config)))
    torch.manual_seed(config['seed']);model=make_model(config).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=config['optimizer']['lr'],weight_decay=.01)
    rng=np.random.default_rng(config['seed']+991);counts={};start=0
    meta=dict(effective_config=effective,train_plan=plan,validation_plan=vplan,scene=True)
    warm=None
    if args.continue_pilot:
        from experiments.offline_camera_retrain_v1.prepare_native30 import sha256
        warm=torch.load(args.continue_pilot,map_location='cpu',weights_only=False)
        if warm['config'].get('version')!='native30-official-overnight-observation-anchor-v2' or any(
                entry['split']!='train' for entry in warm['meta']['train_plan']):
            raise ValueError('Only this official TRAIN-only overnight pilot is an admitted continuation')
        if warm['meta']['train_plan']!=plan or warm['meta']['validation_plan']!=vplan:
            raise ValueError('Continuation must keep the fixed pilot and validation plans')
        model.load_state_dict(warm['model'],strict=True);opt.load_state_dict(warm['optimizer'])
        start=warm['step'];counts=dict(warm['counts']);rng.bit_generator.state=warm['condition_rng']
        torch.set_rng_state(warm['torch_rng']);torch.cuda.set_rng_state(warm['cuda_rng'])
        protocol['initialization']=f"continued own TRAIN-only pilot at step {start}; both models share weights, Adam and RNG"
        protocol['continuation_source']=dict(path=str(args.continue_pilot),sha256=sha256(args.continue_pilot),step=start)
        meta['continuation_source']=protocol['continuation_source'];write_json(out/'protocol.json',protocol)
    monitor_ids=[next(i for i,m in enumerate(valids) if m['group']==group and m['mode']==mode)
        for group in data.members for mode in ('ego','exo')]
    def monitor(current,step):
        evaluate(current,[valbank[i] for i in monitor_ids],[valids[i] for i in monitor_ids],builder,config,
            out/'early_validation',step,steps=8,seeds=(2026101401,))
    for budget in (args.pilot_steps,args.extension_steps):
        protocol.update(status='pilot_training',phase_budget=budget);write_json(out/'protocol.json',protocol)
        start=train_steps(model,opt,config,effective['observation_coefficients'],out/'candidate',start,budget,rng,counts,deadline,
            bank=trainbank,identities=trainids,builder=builder,checkpoint_meta=meta,monitor=monitor)
        # Run both full and disabled scene with identical fixed observations,
        # seeds, DDIM20 and confidence constraints. Evaluation preserves RNG.
        protocol.update(status='candidate_validation',candidate_step=start);write_json(out/'protocol.json',protocol)
        candidate=evaluate(model,valbank,valids,builder,config,out/'candidate',start,seeds=(2026101401,2026101403))
        try:
            from experiments.offline_camera_retrain_v1.native30_overnight_mesh import render
            for regime in ('clean','noisy'):print('MESH',render(out/'candidate/validation_latest.npz',out/'candidate',regime),flush=True)
        except Exception:
            write_json(out/'mesh_failure.json',dict(traceback=traceback.format_exc(),step=start))
        # Release GPU weights/Adam while fitting the matched no-scene control.
        del opt,model;torch.cuda.empty_cache()
        torch.manual_seed(config['seed']);model=make_model(config).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=config['optimizer']['lr'],weight_decay=.01)
        crng=np.random.default_rng(config['seed']+991);ccounts={};cstart=0
        if warm is not None:
            model.load_state_dict(warm['model'],strict=True);opt.load_state_dict(warm['optimizer'])
            cstart=warm['step'];ccounts=dict(warm['counts']);crng.bit_generator.state=warm['condition_rng']
            torch.set_rng_state(warm['torch_rng']);torch.cuda.set_rng_state(warm['cuda_rng'])
        if (out/'control'/'last.pt').exists():
            state=torch.load(out/'control'/'last.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);cstart=state['step'];ccounts=state['counts']
            crng.bit_generator.state=state['condition_rng'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);del state
        protocol.update(status='matched_control_training');write_json(out/'protocol.json',protocol)
        cstart=train_steps(model,opt,config,effective['observation_coefficients'],out/'control',cstart,start,crng,ccounts,deadline,
            bank=trainbank,identities=trainids,builder=builder,scene=False,checkpoint_meta=dict(meta,scene=False))
        control=evaluate(model,valbank,valids,builder,config,out/'control',cstart,seeds=(2026101401,2026101403),ablate=False)
        gate=scene_gate(candidate,control) if cstart==start else dict(passed=False,reason='deadline prevented matched training budget')
        with (out/'candidate/trace.jsonl').open() as first,(out/'control/trace.jsonl').open() as second:
            paired=all(all(a[key]==b[key] for key in ('step','identities','timesteps','observation_checksum'))
                for a,b in zip(map(json.loads,first),map(json.loads,second))) and cstart==start
        gate['paired_training_inputs']=paired;gate['passed']=gate['passed'] and paired
        write_json(out/f'scene_gate_step{start}.json',gate);protocol['last_gate']=gate
        print('SCENE_GATE',start,gate['passed'],flush=True)
        del opt,model;torch.cuda.empty_cache()
        model=make_model(config).cuda();opt=torch.optim.AdamW(model.parameters(),lr=config['optimizer']['lr'],weight_decay=.01)
        state=torch.load(out/'candidate'/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);del state
        if gate['passed']:
            protocol.update(status='full_training',full_training_authorized_by_gate=True);write_json(out/'protocol.json',protocol)
            # The selected pilot continues on the entire training pool; one
            # last.pt retains the earlier pilot optimizer and RNG lineage.
            start=train_steps(model,opt,config,effective['observation_coefficients'],out/'candidate',start,args.full_steps,rng,counts,
                deadline, builder=builder,data=data,checkpoint_meta=dict(meta,stage='full',scene_gate=gate))
            protocol.update(status='final_validation',candidate_step=start);write_json(out/'protocol.json',protocol)
            evaluate(model,valbank,valids,builder,config,out/'candidate',start,seeds=(2026101401,2026101403))
            for regime in ('clean','noisy'):print('MESH',render(out/'candidate/validation_latest.npz',out/'candidate',regime),flush=True)
            protocol.update(status='completed',candidate_step=start,full_training_started=True);break
        if time.time()>=deadline-45*60 or budget==args.extension_steps:
            protocol.update(status='completed_gate_failed',candidate_step=start,full_training_started=False);break
    protocol.update(free_disk_after=shutil.disk_usage(HERE).free,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write_json(out/'protocol.json',protocol);print('FINISHED',protocol['status'],flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--output',type=Path);args,_=parser.parse_known_args()
        if args.output:
            write_json(args.output/'failure.json',dict(traceback=traceback.format_exc(),utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        raise
