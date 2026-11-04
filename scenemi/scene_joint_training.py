"""Fresh scene training with explicit camera units and separate branch rates.

The historical trainer and checkpoints remain immutable. Each optimizer step
accumulates 64 EGO and 64 EXO windows in bounded homogeneous microbatches.
"""
import argparse
import ast
from contextlib import nullcontext
import fcntl
import hashlib
import json
import math
import os
import copy
import importlib.util
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.checked_scene_dataset import CheckedLocalSceneDataset
from scenemi.data.local_scene import inference_payload as local_payload
from scenemi.data.mapped_arrays import numeric_mmap_backend
from scenemi.data.scene_prefetch import ScenePrefetch
from scenemi.native_joint_data import ego_payload
from scenemi.exo_denoise import channel_reliability, denoising_loss
from scenemi.parallel_resources import configure_cuda
from scenemi.replay.comparison_geometry import cpu_observed_geometry
from scenemi.replay.data import clone_dataset
from scenemi.scene_model import native_motion_loss
from scenemi.training import (contract, deterministic, ensure_bounded, now,
                              restore_rng, rng_state, tensor_digest)
from scenemi import scene_joint_replay as historical
from experiments.offline_camera_retrain_v1.ops import resources
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

CONFIG=ROOT/'configs/scene_joint128_conditioning_corrected.yaml'
REVISION='corrected-exo-intrinsics-separate-lr-fresh-scene-v1'


def settings(path):
    c=historical.settings(path)
    if (c.get('training_revision')!=REVISION or
        c['data'].get('exo_intrinsics_policy')!='export_normalized_fxW_fyH_cxW_cyH_no_extra_scale' or
        c['output']=='runs/scene_joint128_conditioning' or c['runtime']['geometry_device']!='cpu'):
        raise ValueError('Require a distinct, explicitly corrected fresh run')
    micro=c['training']['microbatch_size']
    if micro<1 or any(c['training'][mode+'_batch_size']%micro for mode in ('ego','exo')):
        raise ValueError('Each homogeneous mode batch must divide into microbatches')
    if c['optimizer']['lr_ego']<=0 or c['optimizer']['lr_exo']<=0:
        raise ValueError('Both branch learning rates must be positive')
    return c


def export_intrinsics(meta):
    camera=meta['exo_camera'];k=np.asarray(camera['K'],np.float64)
    width,height=camera['source_size']
    if k.shape!=(3,3) or width<=0 or height<=0 or not np.isfinite(k).all():
        raise ValueError('Invalid declared EXO optical calibration')
    return np.array([k[0,0]/width,k[1,1]/height,k[0,2]/width,k[1,2]/height],np.float32)


class CorrectedSceneExoDataset(historical.SceneExoDataset):
    def sample(self,rng,policy,**kwargs):
        sample,identity=super().sample(rng,policy,**kwargs)
        # Undo ONLY the historical EXO loader's extra power-of-two feature
        # scaling. Source calibration, rendered points and camera poses stay
        # intact. Assert against the original array and declared K / size.
        sample['camera_intrinsics']=sample['camera_intrinsics']*self.config['data']['intrinsics_scale_pixels']
        row=self.records[identity['member']];meta=self.bundle.metadata(row,self.split)
        raw=np.asarray(self.bundle.array(meta,'exo_intrinsics'),np.float32)
        raw=np.broadcast_to(raw,(128,4)) if raw.ndim==1 else raw[identity['start']:identity['start']+128]
        if not np.array_equal(sample['camera_intrinsics'],raw):
            raise ValueError('EXO input differs from exported normalized intrinsics')
        if not np.allclose(raw,export_intrinsics(meta),rtol=0,atol=2e-6):
            raise ValueError('EXO export disagrees with declared optical calibration')
        identity['intrinsics_policy']='export_normalized_no_extra_scale'
        return sample,identity


def make_data(c,model,split):
    common=model.base.shape.rest_template[:1].detach().cpu().numpy()
    ego=CheckedLocalSceneDataset(ROOT/c['paths']['scene_paths'],common,split=split,
        config=c,selection=ROOT/c['paths']['selection'])
    exo_config=dict(c,selection=c['paths']['selection'])
    exo=CorrectedSceneExoDataset(exo_config,split);exo.config=exo_config
    return dict(ego=clone_dataset(ego),exo=clone_dataset(exo))


def optimizer_for(model,c):
    ego=list(model.ego.adapter.parameters())
    exo=list(model.exo.condition.parameters())+list(model.exo.projections.parameters())
    exo+=list(model.exo.scene_adapter.parameters())
    ids=[id(p) for p in ego+exo]
    actual={id(p) for p in model.parameters() if p.requires_grad}
    if len(ids)!=len(set(ids)) or set(ids)!=actual or any(p.requires_grad for p in model.base.parameters()):
        raise ValueError('Trainable branch groups overlap, omit parameters, or include the frozen prior')
    groups=[dict(params=ego,name='ego',base_lr=c['optimizer']['lr_ego'],lr=c['optimizer']['lr_ego']),
            dict(params=exo,name='exo',base_lr=c['optimizer']['lr_exo'],lr=c['optimizer']['lr_exo'])]
    return torch.optim.AdamW(groups,weight_decay=c['optimizer']['weight_decay'])


def schedule(optimizer,c,step):
    factor=min(step/c['optimizer']['warmup_updates'],1.)*(.1+.9*(1+math.cos(
        math.pi*step/c['training']['planned_updates']))/2)
    for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
    return {group['name']:group['lr'] for group in optimizer.param_groups}


def reduce_terms(terms,coefficients,fraction,overrides):
    if set(terms)-{'total'}!=set(coefficients):
        raise ValueError('Accumulation must preserve exactly the original loss terms')
    return sum(coefficients[key]*value*overrides.get(key,fraction)
               for key,value in terms.items() if key!='total')


def reduction_totals(pairs,mode):
    samples=[p[0] for p in pairs]
    result=dict(windows=len(samples))
    if mode=='ego':
        result['humans']=sum(int(s['body_format']) in (1,2) for s in samples)
    else:
        result['hidden']=sum(float((1-s['joint_visible']*s['estimate_available'][...,None]).sum()) for s in samples)
        result['missing']=sum(float((1-s['estimate_available']).sum()) for s in samples)
    return result


def mode_loss(model,batch,mode,c,loss_config,device,totals,draws=None):
    fraction=len(batch['motion'])/totals['windows']
    if mode=='ego':
        t,noise=draws if draws is not None else (torch.randint(0,1000,(len(batch['motion']),),device=device),torch.randn_like(batch['motion']))
        alpha=model.alphas[t]
        noisy=alpha[:,None,None].sqrt()*batch['motion']+(1-alpha[:,None,None]).sqrt()*noise
        prediction,details=model(noisy,t,local_payload(batch),return_details=True)
        terms=native_motion_loss(prediction,batch['motion'],batch,alpha.sqrt(),details=details,config=loss_config)
        joints=forward_kinematics(prediction,details['rest'])
        baseline=forward_kinematics(details['baseline_motion'],details['rest']).detach()
        terms['preserve_head']=(joints[:,:,15]-baseline[:,:,15]).square().mean()
        terms['preserve_root']=(joints[:,:,0]-baseline[:,:,0]).square().mean()
        coefficients=dict(loss_config['loss']['coefficients'],shape_betas=0.,native_shape_betas=.05,
            native_template=.01,preserve_head=c['adapter']['head_consistency_weight'],
            preserve_root=c['adapter']['root_consistency_weight'])
        humans=int(((batch['body_format']==1)|(batch['body_format']==2)).sum())/max(totals['humans'],1)
        return reduce_terms(terms,coefficients,fraction,dict(native_shape_betas=humans,native_template=humans))
    t,noise=draws if draws is not None else (torch.randint(0,201,(len(batch['motion']),),device=device),torch.randn_like(batch['motion']))
    alpha=model.alphas[t]
    clean=alpha[:,None,None].sqrt()*batch['motion']+(1-alpha[:,None,None]).sqrt()*noise
    observed=alpha[:,None,None].sqrt()*model.exo.anchor(batch)+(1-alpha[:,None,None]).sqrt()*noise
    noisy=torch.lerp(clean,observed,channel_reliability(batch))
    prediction,details=model(noisy,t,batch,return_details=True)
    terms=denoising_loss(prediction,batch,details)
    hidden=float((1-batch['joint_visible']*batch['estimate_available'][...,None]).sum())/max(totals['hidden'],1)
    missing=float((1-batch['estimate_available']).sum())/max(totals['missing'],1)
    coefficients=dict(position=10.,hidden_position=10.,root=5.,hidden_root=5.,pose=2.,hidden_pose=4.,
                      velocity=.01,preserve=1.)
    return reduce_terms(terms,coefficients,fraction,dict(hidden_position=hidden,hidden_pose=hidden,hidden_root=missing))


def update(model,optimizer,data,rng,c,device,step,loss_config,*,counts=None):
    model.train();optimizer.zero_grad(set_to_none=True);rates=schedule(optimizer,c,step)
    counts=counts or {m:c['training'][m+'_batch_size'] for m in ('ego','exo')}
    metrics={};identities=[];micro=c['training']['microbatch_size'];data_seconds=compute_seconds=0.
    with cpu_observed_geometry():
        for mode in ('ego','exo'):
            total=0.;tick=time.monotonic()
            all_pairs=[data[mode].sample(rng,**({'policy':c['frontend']} if mode=='exo' else {}))
                       for _ in range(counts[mode])]
            totals=reduction_totals(all_pairs,mode)
            data_seconds+=time.monotonic()-tick;tick=time.monotonic()
            logical=c['training'].get('noise_batch_size',8)
            # Preserve the original CUDA generator call shapes even when
            # packing a larger physical microbatch. Denoisers have no random
            # layers in their frozen eval path or scene adapters.
            draws=[]
            for position in range(0,counts[mode],logical):
                length=min(logical,counts[mode]-position)
                t=torch.randint(0,1000 if mode=='ego' else 201,(length,),device=device)
                noise=torch.randn((length,128,201),device=device)
                draws.append((t,noise))
            times=torch.cat([x[0] for x in draws]);noise=torch.cat([x[1] for x in draws])
            if mode=='ego' and any(not np.isfinite(p[0]['camera']).all() or
                not np.all(p[0]['camera_available']==1) or not np.all(p[0]['camera_confidence']==1) for p in all_pairs):
                raise ValueError('EGO accumulation requires the declared continuous confidence-one camera input')
            for start in range(0,counts[mode],micro):
                count=min(micro,counts[mode]-start)
                pairs=all_pairs[start:start+count]
                identities.extend(dict(p[1],mode=mode) for p in pairs)
                batch=collate([p[0] for p in pairs],device)
                loss=mode_loss(model,batch,mode,c,loss_config,device,totals,(times[start:start+count],noise[start:start+count]))
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite '+mode+' loss')
                (.5*loss).backward()
                total+=float(loss.detach())
                del batch,loss,pairs
            metrics[mode+'_loss']=total
            compute_seconds+=time.monotonic()-tick
            del all_pairs
    params=[p for group in optimizer.param_groups for p in group['params']]
    norm=torch.nn.utils.clip_grad_norm_(params,c['optimizer']['clip_norm'],error_if_nonfinite=True)
    optimizer.step()
    return dict(step=step,**metrics,total=.5*(metrics['ego_loss']+metrics['exo_loss']),
        gradient_norm=float(norm),lr_ego=rates['ego'],lr_exo=rates['exo'],mode_batch=counts,
        sampled_identities_sha256=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest(),
        performance=dict(data_wait_seconds=data_seconds,compute_seconds=compute_seconds))


def durable_checkpoint(file,model,optimizer,rng,c,seal,step,device):
    tmp=file.with_suffix('.partial')
    with tmp.open('wb') as stream:
        torch.save(dict(step=step,parent_sha256=c['parent']['sha256'],revision=REVISION,
            adapter=historical.adapter_state(model),optimizer=optimizer.state_dict(),
            rng=rng_state(rng,device),seal_sha256=seal['sha256']),stream)
        stream.flush();os.fsync(stream.fileno())
    tmp.replace(file)
    fd=os.open(file.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def restore_checkpoint(file,model,optimizer,rng,seal,device):
    state=torch.load(file,map_location='cpu',weights_only=False)
    if state['revision']!=REVISION or state['seal_sha256']!=seal['sha256']:
        raise ValueError('Checkpoint revision or source seal changed')
    historical.load_adapter_state(model,state['adapter']);optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'],rng,device)
    return state['step']


def setup(c,device):
    deterministic(c['seed'],device)
    model=historical.make_model(c,device);data=make_data(c,model,'train')
    paths=load_paths(ROOT/'configs/paths.local.yaml')
    seal=contract(c,paths,ROOT/c['paths']['selection'])
    assets=[c['parent']['checkpoint'],c['paths']['scene_paths'],c['paths']['loss'],str(CONFIG),
            'configs/paths.local.yaml','runs/data_inventory/CURRENT_SELECTION_CHECK.json']
    assets+=list(data['exo'].bundle.sources.values())
    predecessor=ROOT/c['output']/'performance/recovery_source'
    if c.get('execution',{}).get('predecessor'):
        assets+=[str(predecessor/name) for name in ('SEAL.json','last.pt','scene_joint_training.py')]
        seal['sources'][str((predecessor/'scene_joint_training.py').relative_to(ROOT))]=sha256(predecessor/'scene_joint_training.py')
    seal['assets']={str(p):sha256(ROOT/p) for p in assets}
    seal['protocol']=REVISION
    seal['sha256']=hashlib.sha256(json.dumps({k:v for k,v in seal.items() if k!='sha256'},sort_keys=True).encode()).hexdigest()
    return model,data,seal


def pin_seal(out,seal):
    path=out/'SEAL.json'
    if path.exists() and json.loads(path.read_text())!=seal:raise ValueError('Prepared source closure changed')
    atomic_json(path,seal)
    if not (out/'sources.tar.gz').exists():
        with tarfile.open(out/'sources.tar.gz','w:gz') as archive:
            for name in seal['sources']:
                p=Path(name) if Path(name).is_absolute() else ROOT/name
                archive.add(p,arcname=name if not Path(name).is_absolute() else 'external/'+name.lstrip('/'))


def audit(c,model,data,out):
    rows=[];datasets_by_split={'train':data,'validation':make_data(c,model,'validation')}
    for split,datasets in datasets_by_split.items():
        exo=datasets['exo'];counts={s:0 for s in c['data']['datasets']}
        for row in exo.records:
            meta=exo.bundle.metadata(row,split);raw=np.asarray(exo.bundle.array(meta,'exo_intrinsics'))
            expected=export_intrinsics(meta)
            if raw.shape[-1:]!=(4,) or not np.isfinite(raw).all() or not np.allclose(raw,expected,rtol=0,atol=2e-6):
                raise ValueError('EXO calibration mismatch: '+row['path'])
            counts[row['dataset']]+=1
        for source in c['data']['datasets']:
            for mode in ('ego','exo'):
                ds=datasets[mode];index=next(i for i,r in enumerate(ds.records) if r['dataset']==source)
                row=ds.records[index];kw=dict(member=index,start=int(row['starts'][0]),yaw=0.)
                if mode=='exo':kw.update(policy=c['frontend'],condition='gap')
                sample,identity=ds.sample(np.random.default_rng(c['seed']),**kw)
                if mode=='ego':
                    meta=ds.selection_bundle.metadata(row,split)
                    raw=np.asarray(ds.array(meta,'camera_intrinsics_128x96'))[int(row['starts'][0])]
                    if not np.array_equal(sample['camera_intrinsics'][0],raw/c['data']['intrinsics_scale_pixels']):
                        raise ValueError('EGO pixel feature scaling changed')
                rows.append(dict(split=split,mode=mode,source=source,
                    first_frame_network_intrinsics=sample['camera_intrinsics'][0].tolist(),
                    identity=identity,exported_exo_sequences_checked=counts.get(source) if mode=='exo' else None))
        atomic_json(out/'INPUT_AUDIT_PROGRESS.json',dict(state='auditing',split=split,exo_sequences=counts,updated_at_utc=now()))
    result=dict(state='passed',full_export_units_audit=True,all_eligible_exo_train_and_validation_sequences=True,
        test_opened=False,rows=rows,ego_policy='pixel source /256 as in parent',
        exo_policy='export-normalized source unchanged; checked against declared K and source_size',updated_at_utc=now())
    atomic_json(out/'INPUT_AUDIT.json',result)
    return result


def preflight(c,model,data,seal,out,device):
    rng=np.random.default_rng(c['seed']);opt=optimizer_for(model,c)
    loss_config={'loss':yaml.safe_load((ROOT/c['paths']['loss']).read_text())['loss']}
    frozen=tensor_digest(model.base.state_dict());tick=time.monotonic()
    counts=None if device=='cuda' else dict(ego=2,exo=2)
    first=update(model,opt,data,rng,c,device,1,loss_config,counts=counts)
    target=out/('preflight_'+device+'.pt');durable_checkpoint(target,model,opt,rng,c,seal,1,device)
    a=update(model,opt,data,rng,c,device,2,loss_config,counts=dict(ego=2,exo=2))
    expected=tensor_digest(historical.adapter_state(model));expected_opt=tensor_digest(opt.state_dict())
    restore_checkpoint(target,model,opt,rng,seal,device)
    b=update(model,opt,data,rng,c,device,2,loss_config,counts=dict(ego=2,exo=2))
    exact=({k:v for k,v in a.items() if k!='performance'}=={k:v for k,v in b.items() if k!='performance'} and
        tensor_digest(historical.adapter_state(model))==expected and tensor_digest(opt.state_dict())==expected_opt)
    if not exact or tensor_digest(model.base.state_dict())!=frozen:
        raise ValueError('Resume is not exact or frozen shared prior changed')
    result=dict(state='passed',device=device,first_update=first,resume_bitwise_exact=exact,
        frozen_parent_unchanged=True,actual_optimizer_groups=[dict(name=g['name'],base_lr=g['base_lr'],
            scheduled_lr=g['lr'],parameters=sum(p.numel() for p in g['params'])) for g in opt.param_groups],
        full_effective_batch_verified=device=='cuda',microbatch_size=c['training']['microbatch_size'],
        seal_sha256=seal['sha256'],peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        elapsed_seconds=time.monotonic()-tick,updates_discarded=True,updated_at_utc=now())
    atomic_json(out/('PREFLIGHT_'+device+'.json'),result);target.unlink()
    return result


def reconcile_log(path,step):
    if not path.exists():return
    original=path.read_bytes();retained=[];discarded=[]
    for line in original.splitlines():
        try:
            row=json.loads(line)
            if row['step']<=step and row['step']==len(retained)+1:retained.append(line);continue
        except (ValueError,KeyError):pass
        discarded.append(line)
    if len(retained)!=step:raise ValueError('Committed training log has missing or repeated steps')
    if discarded:
        archive=path.with_name('uncommitted_'+hashlib.sha256(original).hexdigest()[:12]+'.jsonl')
        archive.write_bytes(b'\n'.join(discarded)+b'\n');path.write_bytes(b'\n'.join(retained)+b'\n')


def train(c,model,data,seal,out,device,resume):
    for name in ('INPUT_AUDIT.json','PREFLIGHT_cpu.json','PREFLIGHT_cuda.json'):
        proof=json.loads((out/name).read_text())
        if proof['state']!='passed' or ('seal_sha256' in proof and proof['seal_sha256']!=seal['sha256']):
            raise ValueError('Fresh exact CPU/CUDA/input acceptance required')
    folder=out/'training';folder.mkdir(exist_ok=True);last=folder/'last.pt';rng=np.random.default_rng(c['seed'])
    opt=optimizer_for(model,c);step=0
    if last.exists():
        if not resume:raise ValueError('Existing corrected checkpoint requires --resume')
        step=restore_checkpoint(last,model,opt,rng,seal,device)
    elif resume:raise ValueError('No corrected checkpoint to resume')
    else:durable_checkpoint(last,model,opt,rng,c,seal,0,device)
    log=folder/'updates.jsonl';reconcile_log(log,step)
    stopping=[False]
    def stop(signum,frame):stopping[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    loss_config={'loss':yaml.safe_load((ROOT/c['paths']['loss']).read_text())['loss']}
    prefetch=ScenePrefetch(data,rng,dict(ego=64,exo=64),c['frontend']) if c['training'].get('prefetch') else None
    if prefetch:data=prefetch.datasets
    started=time.monotonic();initial=step;checkpoint_step=step;rec=None
    status=dict(state='starting',pid=os.getpid(),completed_updates=step,committed_updates=step,
        planned_updates=c['training']['planned_updates'],seal_sha256=seal['sha256'],updated_at_utc=now())
    atomic_json(folder/'STATUS.json',status)
    try:
        with log.open('a') as stream:
            while step<c['training']['planned_updates'] and not stopping[0]:
                tick=time.monotonic();rec=update(model,opt,data,rng,c,device,step+1,loss_config);step+=1
                rec.update(seconds=time.monotonic()-tick)
                stream.write(json.dumps(rec)+'\n');stream.flush()
                if step%c['training']['checkpoint_every']==0 or stopping[0] or step==c['training']['planned_updates']:
                    os.fsync(stream.fileno())
                    durable_checkpoint(last,model,opt,rng,c,seal,step,device);checkpoint_step=step
                status=dict(state='training',pid=os.getpid(),completed_updates=step,committed_updates=checkpoint_step,
                    planned_updates=c['training']['planned_updates'],effective_batch_size=128,mode_batch=dict(ego=64,exo=64),
                    microbatch_size=c['training']['microbatch_size'],latest=rec,seal_sha256=seal['sha256'],
                    updates_per_second=(step-initial)/max(time.monotonic()-started,1e-6),updated_at_utc=now())
                atomic_json(folder/'STATUS.json',status)
                print(json.dumps(status),flush=True)
        if step!=checkpoint_step:durable_checkpoint(last,model,opt,rng,c,seal,step,device)
        status.update(state='training_complete' if step==c['training']['planned_updates'] else 'stopped_at_safe_boundary',
            committed_updates=step,updated_at_utc=now());atomic_json(folder/'STATUS.json',status)
    except BaseException as error:
        atomic_json(folder/'STATUS.json',dict(state='failed',pid=os.getpid(),last_completed_update=step,
            committed_updates=checkpoint_step,error=repr(error),updated_at_utc=now()))
        raise
    finally:
        if prefetch:prefetch.close()


def known_gpu_job(pid):
    try:
        p=Path('/proc')/str(pid);words=p.joinpath('cmdline').read_bytes().split(b'\0')
        words=[s.decode() for s in words if s];module=words[words.index('-m')+1]
        action=words[words.index('-m')+2]
        return p.stat().st_uid==os.getuid() and p.joinpath('cwd').resolve()==ROOT and action in (
            'run','preflight','train','pipeline','optimize') and module in ('scenemi.replay.comparison_geometry','scenemi.scene_joint_training')
    except (OSError,ValueError,IndexError):return False


def admission(c):
    from scenemi.gpu_policy import admission as current
    return current('cuda',c['runtime']['memory_gib'])


def reuse_input_audit(out,seal):
    previous=out/'input_audit_sources'
    proof=json.loads((out/'INPUT_AUDIT.json').read_text())
    old=json.loads((previous/'SEAL.json').read_text())
    if proof['state']!='passed' or old['config']['data']!=seal['config']['data']:
        raise ValueError('Input unit audit does not match the corrected data contract')
    # The completed audit used the identical loader. Training orchestration and
    # mathematically equivalent accumulation can evolve before acceptance.
    def definitions(source):
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(source).body
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in (
                    'export_intrinsics','CorrectedSceneExoDataset','make_data')}
    with tarfile.open(previous/'sources.tar.gz','r:gz') as archive:
        original=archive.extractfile('scenemi/scene_joint_training.py').read().decode()
    if definitions(original)!=definitions(Path(__file__).read_text()):
        raise ValueError('Input loader changed after the complete units audit')
    for name,digest in old['sources'].items():
        if name=='scenemi/scene_joint_training.py':continue
        if seal['sources'].get(name)!=digest:raise ValueError('Input audit dependency changed: '+name)
    atomic_json(out/'INPUT_AUDIT_ACCEPTANCE.json',dict(state='passed',audit_seal_sha256=old['sha256'],
        training_seal_sha256=seal['sha256'],loader_definitions_identical=True,other_sources_identical=True,updated_at_utc=now()))


def pipeline(c,model,data,seal,out,resume):
    reuse_input_audit(out,seal)
    initial=historical.adapter_state(model)
    preflight(c,model,data,seal,out,'cpu')
    historical.load_adapter_state(model,initial);deterministic(c['seed'],'cuda')
    record=admission(c);atomic_json(out/'ADMISSION_before_cuda.json',record)
    if not record['admitted']:raise RuntimeError('Fresh GPU admission refused after CPU acceptance')
    model.to('cuda');torch.cuda.reset_peak_memory_stats()
    preflight(c,model,data,seal,out,'cuda')
    historical.load_adapter_state(model,initial);deterministic(c['seed'],'cuda')
    return train(c,model,data,seal,out,'cuda',resume)


def replay_crash_inputs(c,model,data,out,predecessor):
    saved=torch.load(predecessor/'last.pt',map_location='cpu',weights_only=False)
    rng=np.random.default_rng();rng.bit_generator.state=saved['rng']['data']
    logs={r['step']:r for r in [json.loads(s) for s in (out/'training/updates.jsonl').read_text().splitlines() if s.strip()]}
    rows=[];tick=time.monotonic()
    for step in range(saved['step']+1,41):
        identities=[]
        for mode in ('ego','exo'):
            for _ in range(c['training'][mode+'_batch_size']):
                _,identity=data[mode].sample(rng,**({'policy':c['frontend']} if mode=='exo' else {}))
                identities.append(dict(identity,mode=mode))
        digest=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest()
        if step in logs and digest!=logs[step]['sampled_identities_sha256']:
            raise ValueError('Input replay changed original sample order at step '+str(step))
        rows.append(dict(step=step,samples=128,identity_sha256=digest,
            matches_historical_record=step in logs,original_crash_boundary_completed=step==40))
        atomic_json(out/'performance/INPUT_REPLAY.json',dict(state='running',rows=rows,updated_at_utc=now()))
        print(json.dumps(dict(input_replay_completed=step,source_checksums_enforced=True)),flush=True)
    result=dict(state='passed',rows=rows,samples_replayed=len(rows)*128,
        all_historical_identities_match=True,original_crash_boundary_completed=True,
        conclusion='The same sampled inputs pass declared checksums and direct NPY mapping; file corruption and hardware failure are not established.',
        elapsed_seconds=time.monotonic()-tick,updated_at_utc=now())
    atomic_json(out/'performance/INPUT_REPLAY.json',result)


def optimize(c,model,data,seal,out):
    work=out/'performance/optimization';work.mkdir(exist_ok=True)
    predecessor=out/'performance/recovery_source';old=json.loads((predecessor/'SEAL.json').read_text())
    for field in ('parent','data','optimizer','adapter','exo','frontend','local'):
        if old['config'][field]!=c[field]:raise ValueError('Execution optimization changed scientific configuration: '+field)
    preflight(c,model,data,seal,work,'cpu')
    replay_crash_inputs(c,model,data,out,predecessor)
    record=admission(c);atomic_json(work/'ADMISSION.json',record)
    if not record['admitted']:raise RuntimeError('GPU admission refused after input replay')
    model.to('cuda');opt=optimizer_for(model,c);rng=np.random.default_rng(c['seed'])
    spec=importlib.util.spec_from_file_location('scene_execution_reference',predecessor/'scene_joint_training.py')
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    loss_config={'loss':yaml.safe_load((ROOT/c['paths']['loss']).read_text())['loss']}
    def restore():
        return reference.restore_checkpoint(predecessor/'last.pt',model,opt,rng,old,'cuda')
    start=restore();reference_config=copy.deepcopy(old['config']);reference_config['training']['microbatch_size']=8
    tick=time.monotonic();baseline=reference.update(model,opt,data,rng,reference_config,'cuda',start+1,loss_config)
    torch.cuda.synchronize();baseline_seconds=time.monotonic()-tick
    expected_adapter=historical.adapter_state(model)
    expected_gradients=[p.grad.detach().cpu().clone() if p.grad is not None else None
                       for group in opt.param_groups for p in group['params']]
    expected_optimizer=tensor_digest(opt.state_dict());expected_rng=tensor_digest(rng_state(rng,'cuda'))
    original_logs={r['step']:r for r in [json.loads(s) for s in (out/'training/updates.jsonl').read_text().splitlines() if s.strip()]}
    previous=original_logs.get(start+1)
    baseline_matches_log=previous is not None and all(baseline[k]==previous[k] for k in baseline)
    if previous is None or baseline['sampled_identities_sha256']!=previous['sampled_identities_sha256'] or any(
        abs(baseline[k]-previous[k])>1e-7 for k in ('ego_loss','exo_loss','gradient_norm')):
        raise ValueError('Original checkpoint/input replay did not reproduce the recorded update')
    results=[]
    for micro in (8,16,32,64):
        restore();candidate=copy.deepcopy(c);candidate['training']['microbatch_size']=micro
        counts=dict(ego=64,exo=64);tick=time.monotonic()
        prepared=ScenePrefetch(data,rng,counts,c['frontend'],max_batches=2)
        torch.cuda.reset_peak_memory_stats()
        try:
            actual=update(model,opt,prepared.datasets,rng,candidate,'cuda',start+1,loss_config)
            torch.cuda.synchronize();seconds=time.monotonic()-tick
        finally:prepared.close()
        parameters=historical.adapter_state(model)
        def difference(a,b):
            return max(float((a[k]-b[k]).abs().max()) if torch.is_tensor(a[k]) else difference(a[k],b[k]) for k in a)
        delta=difference(parameters,expected_adapter)
        actual_gradients=[p.grad.detach().cpu() if p.grad is not None else None
                          for group in opt.param_groups for p in group['params']]
        grad_max=0.;grad_square=reference_square=0.
        for actual_gradient,expected_gradient in zip(actual_gradients,expected_gradients):
            if (actual_gradient is None)!=(expected_gradient is None):raise ValueError('Gradient parameter coverage changed')
            if actual_gradient is not None:
                difference_gradient=actual_gradient-expected_gradient
                grad_max=max(grad_max,float(difference_gradient.abs().max()))
                grad_square+=float(difference_gradient.double().square().sum())
                reference_square+=float(expected_gradient.double().square().sum())
        grad_relative=math.sqrt(grad_square/max(reference_square,1e-30))
        loss_delta=abs(actual['total']-baseline['total'])
        same_rng=tensor_digest(rng_state(rng,'cuda'))==expected_rng
        exact=(tensor_digest(parameters)==tensor_digest(expected_adapter) and tensor_digest(opt.state_dict())==expected_optimizer)
        passed=(actual['sampled_identities_sha256']==baseline['sampled_identities_sha256'] and same_rng and
            loss_delta<=1e-6 and delta<=2e-6 and grad_max<=1e-6 and grad_relative<=1e-4 and (micro!=8 or exact))
        row=dict(microbatch=micro,seconds=seconds,data_wait_seconds=actual['performance']['data_wait_seconds'],
            compute_seconds=actual['performance']['compute_seconds'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            loss_difference=loss_delta,max_adapter_parameter_difference=delta,same_rng=same_rng,
            max_gradient_difference=grad_max,relative_gradient_l2_difference=grad_relative,
            same_sample_identities=actual['sampled_identities_sha256']==baseline['sampled_identities_sha256'],
            bitwise_same_adapter_and_optimizer=exact,passed=passed)
        results.append(row);atomic_json(work/'BENCHMARK.json',dict(state='running',baseline_seconds=baseline_seconds,
            original_update_matches_log=baseline_matches_log,rows=results,updated_at_utc=now()))
        print(json.dumps(row),flush=True)
    eligible=[r for r in results if r['passed']]
    if not eligible:raise ValueError('No execution candidate passed numerical equivalence')
    chosen=min(eligible,key=lambda r:r['seconds'])
    result=dict(state='passed',baseline_seconds=baseline_seconds,original_update_matches_log=baseline_matches_log,
        rows=results,chosen_microbatch=chosen['microbatch'],speedup=baseline_seconds/chosen['seconds'],
        numerical_limits=dict(max_loss_difference=1e-6,max_adapter_parameter_difference=2e-6),
        all_benchmark_updates_discarded=True,updated_at_utc=now())
    atomic_json(work/'BENCHMARK.json',result)
    # The configuration is selected BEFORE the new formal source seal is
    # published. Old checkpoints and seals remain preserved for rollback.
    c['training']['microbatch_size']=chosen['microbatch']
    CONFIG.write_text(yaml.safe_dump(c,sort_keys=False))
    restore();candidate=copy.deepcopy(c);counts=dict(ego=2,exo=2)
    a=update(model,opt,data,rng,candidate,'cuda',start+1,loss_config,counts=counts)
    temp=work/'resume_test.pt';durable_checkpoint(temp,model,opt,rng,c,seal,start+1,'cuda')
    b=update(model,opt,data,rng,candidate,'cuda',start+2,loss_config,counts=counts)
    expected=tensor_digest(historical.adapter_state(model));expected_opt=tensor_digest(opt.state_dict())
    restore_checkpoint(temp,model,opt,rng,seal,'cuda')
    d=update(model,opt,data,rng,candidate,'cuda',start+2,loss_config,counts=counts)
    if (tensor_digest(historical.adapter_state(model))!=expected or tensor_digest(opt.state_dict())!=expected_opt or
        {k:v for k,v in b.items() if k!='performance'}!={k:v for k,v in d.items() if k!='performance'}):
        raise ValueError('Optimized execution checkpoint resume is not exact')
    temp.unlink();restore()
    # Rebuild the seal after selecting the measured execution batch. Explicit
    # model/data/loss invariants above guard this metadata-only migration.
    paths=load_paths(ROOT/'configs/paths.local.yaml');new=contract(c,paths,ROOT/c['paths']['selection'])
    new['assets']=dict(seal['assets']);new['assets'][str(CONFIG)]=sha256(CONFIG)
    new['sources'][str((predecessor/'scene_joint_training.py').relative_to(ROOT))]=sha256(predecessor/'scene_joint_training.py')
    new['protocol']=REVISION;new['sha256']=hashlib.sha256(json.dumps({k:v for k,v in new.items() if k!='sha256'},sort_keys=True).encode()).hexdigest()
    for name in ('SEAL.json','sources.tar.gz','PREFLIGHT_cpu.json','PREFLIGHT_cuda.json'):
        if (out/name).exists():(out/name).replace(work/('before_'+name))
    pin_seal(out,new)
    for name in ('cpu','cuda'):
        atomic_json(out/('PREFLIGHT_'+name+'.json'),dict(state='passed',seal_sha256=new['sha256'],
            original_real_data_acceptance=str(work/'PREFLIGHT_cpu.json') if name=='cpu' else str(predecessor/'PREFLIGHT_cuda.json'),
            full_effective_batch_verified=name=='cuda',resume_bitwise_exact=True,
            optimization_benchmark=str(work/'BENCHMARK.json'),updates_discarded=True,updated_at_utc=now()))
    migration_before=dict(adapter=tensor_digest(historical.adapter_state(model)),optimizer=tensor_digest(opt.state_dict()),rng=tensor_digest(rng_state(rng,'cuda')))
    durable_checkpoint(out/'training/last.pt',model,opt,rng,c,new,start,'cuda')
    migrated=torch.load(out/'training/last.pt',map_location='cpu',weights_only=False)
    migration_after=dict(adapter=tensor_digest(migrated['adapter']),optimizer=tensor_digest(migrated['optimizer']),rng=tensor_digest(migrated['rng']))
    if migration_before!=migration_after:raise ValueError('Execution migration changed checkpoint state')
    atomic_json(work/'MIGRATION.json',dict(state='passed',step=start,before=migration_before,after=migration_after,
        old_seal_sha256=old['sha256'],new_seal_sha256=new['sha256'],no_scientific_configuration_change=True,
        old_checkpoint_preserved=str(predecessor/'last.pt'),updated_at_utc=now()))
    return train(c,model,data,new,out,'cuda',True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','audit','preflight','train','pipeline','optimize','launch'))
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--launch-action',choices=('audit','preflight','train','pipeline','optimize'),default='preflight')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();c=settings(args.config);out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
    if args.action in ('check','launch'):
        record=admission(c) if args.device=='cuda' else resources.admission('cpu',c['runtime']['memory_gib'])
        atomic_json(out/('ADMISSION_'+args.device+'.json'),record);print(json.dumps(record),flush=True)
        if not record['admitted']:raise RuntimeError('Resource admission refused')
        if args.action=='check':return
        command=[sys.executable,'-u','-m','scenemi.scene_joint_training',args.launch_action,
                 '--config',str(args.config),'--device',args.device]
        if args.resume:command.append('--resume')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if args.device=='cuda' else '')
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[key]=str(c['runtime']['cpu_count'])
        code=subprocess.call(resources.scope_command(command,c['runtime']['memory_gib'],c['runtime']['cpu_count']),env=env)
        if code:
            file=out/'training/STATUS.json'
            status=json.loads(file.read_text()) if file.exists() else {}
            status.update(state='interrupted',launcher_returncode=code,child_exit_observed=True,updated_at_utc=now())
            atomic_json(file,status)
        raise SystemExit(code)
    ensure_bounded()
    if args.device=='cuda':
        record=admission(c);atomic_json(out/'ADMISSION_child.json',record)
        if not record['admitted']:raise RuntimeError('Fresh child resource admission refused')
        os.environ['SCENEMI_ALLOCATOR_GIB']=str(c['runtime']['allocator_gib']);configure_cuda()
    with (out/'.lock').open('w') as lock,numeric_mmap_backend(out/'CURRENT_ARRAY_READ.json'):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        model,data,seal=setup(c,'cpu' if args.action in ('pipeline','optimize') else args.device)
        if args.action=='optimize':return optimize(c,model,data,seal,out)
        pin_seal(out,seal)
        if args.action=='audit':result=audit(c,model,data,out)
        elif args.action=='preflight':result=preflight(c,model,data,seal,out,args.device)
        elif args.action=='pipeline':return pipeline(c,model,data,seal,out,args.resume)
        else:return train(c,model,data,seal,out,args.device,args.resume)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
