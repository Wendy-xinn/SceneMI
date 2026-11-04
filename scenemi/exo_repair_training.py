"""Bounded, resumable initialization-repair pilot with paired AMASS controls."""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import AmassDataset, collate, inference_payload
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.exo_initialization import CONDITIONS, repair_example, repair_payload
from scenemi.exo_diagnostics import selected_cases
from scenemi.exo_repair import InitializedExoPrior, repair_loss, sample_initialized
from scenemi.exo_training import INDEX, make_model, settings
from scenemi.metrics import motion_metrics
from scenemi.parallel_resources import configure_cuda
from scenemi.selection import validation_manifest
from scenemi.training import contract, deterministic, ensure_bounded, now, restore_rng, rng_state, tensor_digest
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d

CONFIG = ROOT/'configs/amass_exo_repair.yaml'


def verify(seal):
    for path,digest in seal['sources'].items():
        if sha256(ROOT/path)!=digest:
            raise ValueError('Sealed dependency changed: '+path)
    for path,digest in seal['assets'].items():
        if sha256(ROOT/path)!=digest:
            raise ValueError('Required asset changed: '+path)


def save(model, optimizer, rng, device, step, seal, out, log_bytes=0):
    save_checkpoint(dict(step=step, repair=model.repair_state(), optimizer=optimizer.state_dict(),
        rng=rng_state(rng,device), device=device, contract=seal, log_bytes=log_bytes,
        required_base=seal['config']['base']),out)


def restore(model, optimizer, rng, device, seal, path):
    state = torch.load(path,map_location='cpu',weights_only=False)
    if state['contract']!=seal or state['device']!=device:
        raise ValueError('Recovery contract/device mismatch')
    model.load_repair_state(state['repair'])
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'],rng,device)
    return state


def batch_for(data,rng,config,device,forced=False):
    pairs = [repair_example(data,rng,config['frontend'],
        condition=CONDITIONS[i%4] if forced else None) for i in range(config['training']['batch_size'])]
    return collate([sample for sample,_ in pairs],device),[identity for _,identity in pairs]


def update(model,optimizer,data,rng,config,device,step):
    batch,identities=batch_for(data,rng,config,device,forced=step<=2)
    timestep=torch.randint(0,config['diffusion']['start_timestep']+1,(len(batch['motion']),),device=device)
    alpha=model.alphas[timestep,None,None]
    noisy=alpha.sqrt()*batch['motion']+(1-alpha).sqrt()*torch.randn_like(batch['motion'])
    model.train();optimizer.zero_grad(set_to_none=True)
    prediction,details=model(noisy,timestep,repair_payload(batch),return_details=True)
    terms=repair_loss(prediction,batch,details)
    if not all(torch.isfinite(term).all() for term in terms.values()):
        raise FloatingPointError('Nonfinite repair objective')
    terms['total'].backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
        config['optimizer']['clip_norm'],error_if_nonfinite=True)
    gradient=float(model.readout.weight.grad.norm())
    lr=config['optimizer']['lr']*min(step/config['optimizer']['warmup_updates'],1.)*(
        .2+.8*.5*(1+math.cos(math.pi*step/config['training']['updates'])))
    for group in optimizer.param_groups:group['lr']=lr
    optimizer.step()
    return dict(step=step,identities=identities,timesteps=timestep.tolist(),lr=lr,
        terms={key:float(value.detach()) for key,value in terms.items()},grad_norm=float(norm),readout_gradient=gradient)


@torch.no_grad()
def ego_check(model,data,cases,device):
    pairs=[data.sample(np.random.default_rng(777+c['case_id']),'ego',member=c['member'],
        start=c['start'],reflected=False,yaw=0.)[0] for c in cases[:4]]
    batch=collate(pairs,device);payload=inference_payload(batch)
    noisy=torch.zeros_like(batch['motion']);times=torch.full((len(noisy),),500,device=device,dtype=torch.long)
    expected=model.base(noisy,times,payload)
    actual=model(noisy,times,payload)
    if not torch.equal(expected,actual) or any(p.requires_grad or p.grad is not None for p in model.base.parameters()):
        raise AssertionError('EGO frozen path changed')
    return dict(exact_original_ego=True,windows=len(pairs),max_difference=float((actual-expected).abs().max()))


@torch.no_grad()
def smooth_initialization(payload):
    initial=payload['initial_motion']
    values=initial[:,:,:135].transpose(1,2)
    kernel=values.new_tensor([1,2,3,2,1]).reshape(1,1,5)/9
    smooth=F.conv1d(F.pad(values,(2,2),mode='replicate'),kernel.expand(135,1,5),groups=135).transpose(1,2)
    result=initial.clone()
    result[:,:,:3]=torch.lerp(initial[:,:,:3],smooth[:,:,:3],1-payload['initial_root_reliability'][...,None])
    weight=(1-payload['initial_reliability'])[...,None].expand(-1,-1,-1,6).flatten(2)
    result[:,:,3:135]=torch.lerp(initial[:,:,3:135],smooth[:,:,3:135],weight)
    moved=reconcile_motion(result,payload['initial_rest'])
    fixed=(payload['initial_root_reliability']>=1)&(payload['initial_reliability']>=1).all(-1)
    return torch.where(fixed[...,None],initial,moved)


def aggregate(rows):
    output={}
    for field in rows[0]['metrics']:
        groups=defaultdict(list)
        for row in rows:
            if row['metrics'][field] is not None:
                groups[row['family']].append(row['metrics'][field])
        values=[np.mean(v) for v in groups.values()]
        output[field]=float(np.mean(values)) if values else None
    return dict(windows=len(rows),families=len({r['family'] for r in rows}),family_macro_mean=output)


@torch.no_grad()
def evaluate(model,data,cases,config,device,out,step,full=False):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULT.json').exists():
        raise ValueError('Refuse to overwrite completed evaluation')
    rows=[];started=time.monotonic();saved=rng_state(np.random.default_rng(),device)
    was_training=model.training;model.eval()
    size=config['validation']['batch_size']
    try:
        with (out/'metrics.jsonl').open('w') as stream:
            for condition in CONDITIONS:
                for offset in range(0,len(cases),size):
                    chunk=cases[offset:offset+size]
                    pairs=[repair_example(data,np.random.default_rng(777+c['case_id']),config['frontend'],
                        case=c,condition=condition) for c in chunk]
                    batch=collate([pair[0] for pair in pairs],device);payload=repair_payload(batch)
                    sampled=sample_initialized(model,payload,steps=config['diffusion']['ddim_steps'],
                        start_timestep=config['diffusion']['start_timestep'],seed=config['diffusion']['seed'])
                    variants=dict(initial=batch['initial_motion'],temporal_smoothing=smooth_initialization(payload),repair=sampled)
                    truth=forward_kinematics(batch['motion'],batch['target_rest'])
                    initial_points=forward_kinematics(batch['initial_motion'],batch['initial_rest'])
                    for variant,motion in variants.items():
                        points=forward_kinematics(motion,batch['initial_rest'])
                        for i,case in enumerate(chunk):
                            visible=(batch['joint_visible'][i]>0)&(batch['estimate_available'][i,:,None]>0)
                            region=(~visible) if condition in ('occlusion','gap') else torch.ones_like(visible)
                            error=(points[i]-truth[i]).norm(dim=-1)*1000
                            metrics=motion_metrics(points[i],truth[i])
                            metrics.update(repair_region_mpjpe_mm=float(error[region].mean()),
                                tracked_visible_mpjpe_mm=float(error[visible].mean()) if visible.any() else None,
                                max_change_from_initial_mm=float((points[i]-initial_points[i]).norm(dim=-1).max()*1000))
                            row=dict(case_id=case['case_id'],family=pairs[i][1]['family'],condition=condition,
                                variant=variant,identity=pairs[i][1],metrics=metrics)
                            rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                    stream.flush()
                    atomic_json(out/'STATUS.json',dict(state='evaluating',step=step,full_validation=full,
                        condition=condition,completed=min(offset+size,len(cases)),total=len(cases),
                        pid=os.getpid(),updated_at_utc=now(),elapsed_seconds=time.monotonic()-started))
    finally:
        model.train(was_training);restore_rng(saved,np.random.default_rng(),device)
    summaries={condition:{variant:aggregate([r for r in rows if r['condition']==condition and r['variant']==variant])
        for variant in ('initial','temporal_smoothing','repair')} for condition in CONDITIONS}
    policy=config['validation'];checks={}
    identity=[r for r in rows if r['condition']=='identity' and r['variant']=='repair']
    checks['identity_preserved']=max(r['metrics']['max_change_from_initial_mm'] for r in identity)<=policy['identity_max_change_mm']
    for condition in ('tracking','occlusion','gap'):
        values={v:summaries[condition][v]['family_macro_mean'] for v in summaries[condition]}
        checks[condition+'_overall_noninferiority']=values['repair']['mpjpe_mm']<=values['initial']['mpjpe_mm']+policy['visible_noninferiority_mm']
        checks[condition+'_visible_noninferiority']=values['repair']['tracked_visible_mpjpe_mm']<=values['initial']['tracked_visible_mpjpe_mm']+policy['visible_noninferiority_mm']
        if condition in policy['required_improvement_conditions']:
            for control in ('initial','temporal_smoothing'):
                checks[condition+'_beats_'+control]=values['repair']['repair_region_mpjpe_mm']<values[control]['repair_region_mpjpe_mm']
    result=dict(state='complete',step=step,conditions=summaries,checks=checks,candidate_gate_passed=all(checks.values()),
        full_validation=full,families=len({r['family'] for r in rows}),windows_per_condition=len(cases),
        total_model_windows=len(cases)*len(CONDITIONS),test_used=False,selection='fixed_final_budget_not_best_subset',
        elapsed_seconds=time.monotonic()-started,completed_at_utc=now())
    atomic_json(out/'RESULT.json',result);atomic_json(out/'STATUS.json',result)
    print(json.dumps(dict(evaluation=str(out),step=step,candidate_gate_passed=result['candidate_gate_passed'],checks=checks)),flush=True)
    return result


def preflight(model,data,validation,cases,config,device,out,seal):
    folder=out/'preflight';folder.mkdir(exist_ok=True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],**{
        key:config['optimizer'][key] for key in ('lr','weight_decay')})
    rng=np.random.default_rng(config['seed']+91)
    original=tensor_digest(model.base.state_dict())
    before=ego_check(model,validation,cases,device)
    first=update(model,optimizer,data,rng,config,device,1)
    if first['readout_gradient']<=0:raise AssertionError('Zero repair-head gradient')
    temporary=folder/f'.{device}.pt'
    save(model,optimizer,rng,device,1,seal,temporary)
    expected=update(model,optimizer,data,rng,config,device,2)
    digest=tensor_digest((model.repair_state(),optimizer.state_dict(),rng_state(rng,device)))
    restore(model,optimizer,rng,device,seal,temporary)
    actual=update(model,optimizer,data,rng,config,device,2)
    if expected!=actual or digest!=tensor_digest((model.repair_state(),optimizer.state_dict(),rng_state(rng,device))):
        raise AssertionError('Repair exact resume failed')
    temporary.unlink()
    if tensor_digest(model.base.state_dict())!=original:raise AssertionError('Frozen prior changed')
    batch,_=batch_for(data,rng,config,device,forced=True)
    payload=repair_payload(batch);noisy=torch.zeros_like(batch['motion']);times=torch.full((len(noisy),),100,device=device,dtype=torch.long)
    model.eval()
    with torch.no_grad():
        prediction=model(noisy,times,payload)
        poisoned=dict(payload,motion=torch.full_like(batch['motion'],float('nan')),
            target_rest=torch.full_like(batch['target_rest'],float('nan')),
            target_betas=torch.full_like(batch['target_betas'],float('nan')))
        if not torch.equal(prediction,model(noisy,times,poisoned)):raise AssertionError('Private labels leaked')
        identity=batch['initial_reliability'].amin((-1,-2))>=1
        if not torch.equal(prediction[identity],batch['initial_motion'][identity]):raise AssertionError('Reliable identity changed')
        points=forward_kinematics(prediction,batch['initial_rest'])
        torch.testing.assert_close(points,prediction[:,:,135:].reshape(len(noisy),128,22,3)*2,atol=3e-5,rtol=0)
        rotations=rotation_from_6d(prediction[:,:,3:135].reshape(len(noisy),128,22,6))
        torch.testing.assert_close(rotations.transpose(-1,-2)@rotations,
            torch.eye(3,device=device).expand_as(rotations),atol=3e-5,rtol=0)
        # Synthetic tracked gaps are interpolated without consulting hidden data.
        missing=batch['estimate_available']==0
        if batch['observation_meta'][missing][...,:4].any() or batch['trajectory'][missing].any():
            raise AssertionError('Missing frontend estimates leaked into observations')
        sampled=sample_initialized(model,payload,steps=2,start_timestep=200)
        if not torch.isfinite(sampled).all():raise AssertionError('Nonfinite DDIM repair')
    after=ego_check(model,validation,cases,device);verify(seal)
    result=dict(state='passed',contract_sha256=seal['sha256'],device=device,batch_size=len(noisy),
        exact_resume=True,frozen_prior_unchanged=True,ego_before=before,ego_after=after,
        private_labels_isolated=True,identity_preserved=True,fk_so3_valid=True,
        missing_estimates_removed=True,ddim_finite=True,first_update=first,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,completed_at_utc=now())
    atomic_json(folder/f'{device}.json',result);print(json.dumps(result),flush=True)


def train(model,data,validation,subset,cases,config,device,out,seal,resume):
    if device!='cuda':raise ValueError('Formal pilot requires CUDA acceptance')
    for name in ('cpu','cuda'):
        record=json.loads((out/'preflight'/f'{name}.json').read_text())
        if record['state']!='passed' or record['contract_sha256']!=seal['sha256']:
            raise ValueError('Missing matching preflight: '+name)
    folder=out/'training';folder.mkdir(exist_ok=True)
    lock=(folder/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    last=folder/'last.pt';log=folder/'updates.jsonl'
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],**{
        key:config['optimizer'][key] for key in ('lr','weight_decay')})
    rng=np.random.default_rng(config['seed']+91);start=0
    if last.exists()!=resume:raise ValueError('Existing checkpoint requires --resume')
    if resume:
        state=restore(model,optimizer,rng,device,seal,last);start=state['step']
        with log.open('r+b') as stream:
            if stream.seek(0,2)<state['log_bytes']:raise ValueError('Committed log missing')
            stream.truncate(state['log_bytes'])
    elif log.exists() and log.stat().st_size:raise ValueError('Uncommitted log exists')
    stopping=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopping.append(True))
    started=time.monotonic();step=start
    status=dict(state='starting',pid=os.getpid(),completed_updates=step,planned_updates=config['training']['updates'],
        contract_sha256=seal['sha256'],started_at_utc=now(),training_mode='exo_initialized_repair_frozen_ego',
        status_note='Synthetic native-parameter initialization; richer inputs than partial-joint baseline; real export not evaluated')
    def report(phase,**extra):
        status.update(state=phase,completed_updates=step,updated_at_utc=now(),elapsed_seconds=time.monotonic()-started,**extra)
        atomic_json(out/'STATUS.json',status);atomic_json(folder/'STATUS.json',status)
    try:
        with log.open('a',buffering=1) as stream:
            def commit():
                stream.flush();os.fsync(stream.fileno())
                save(model,optimizer,rng,device,step,seal,last,stream.tell())
            if not resume:
                commit();report('baseline_subset_validation')
                evaluate(model,validation,subset,config,device,out/'validation/step_000000',0)
            for step in range(start+1,config['training']['updates']+1):
                row=update(model,optimizer,data,rng,config,device,step)
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                if step%config['training']['checkpoint_every']==0 or stopping:commit()
                if step in config['validation']['subset_steps']:
                    report('subset_validation');verify(seal)
                    evaluate(model,validation,subset,config,device,out/f'validation/step_{step:06d}',step)
                    status['ego_preservation']=ego_check(model,validation,cases,device)
                if step%config['training']['log_every']==0 or stopping:
                    report('training',last_terms=row['terms'],peak_cuda_bytes=torch.cuda.max_memory_allocated())
                    print(json.dumps(status),flush=True)
                if stopping:report('interrupted');return
            commit();verify(seal)
            report('final_full_validation')
            result=evaluate(model,validation,cases,config,device,out/'full_validation',step,True)
            preservation=ego_check(model,validation,cases,device)
            verify(seal)
            report('complete',candidate_gate_passed=result['candidate_gate_passed'],ego_preservation=preservation,
                final_checkpoint_sha256=sha256(last),full_validation_result=str(out/'full_validation/RESULT.json'))
    except BaseException as error:
        report('failed',error=repr(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','train'])
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();ensure_bounded()
    config=yaml.safe_load(args.config.read_text())
    if (config['version']!='amass-exo-initialization-repair-pilot-v1' or config['validation']['test_used']
            or config['validation']['split']!='val' or config['validation']['checkpoint_selection']!='fixed_final_2000'):
        raise ValueError('Unexpected pilot protocol')
    deterministic(config['seed'],args.device)
    if args.device=='cuda':
        if float(os.environ['SCENEMI_ALLOCATOR_GIB'])!=config['runtime']['allocator_gib']:
            raise ValueError('Allocator budget differs from declared configuration')
        configure_cuda()
    paths=load_paths(ROOT/'configs/paths.local.yaml')
    old_config,old_reference,_=settings(ROOT/'configs/amass_exo128.yaml')
    if config['base']!=old_config['base']:raise ValueError('Unexpected frozen EGO initializer')
    original=make_model(old_config,old_reference,paths,args.device)
    base=original.base;del original
    model=InitializedExoPrior(base,**config['model']).to(args.device)
    data=AmassDataset(INDEX,paths,'train');validation=AmassDataset(INDEX,paths,'val')
    cases=validation_manifest(validation.members);subset=selected_cases(validation,config['validation']['families_per_corpus'])
    if len(cases)!=2222 or len(validation.members)!=337:raise ValueError('Full validation membership changed')
    out=ROOT/config['output'];out.mkdir(parents=True,exist_ok=True)
    seal=contract(config,paths,INDEX)
    old_seal=json.loads((ROOT/'runs/amass_exo128/PROTOCOL.json').read_text())
    seal['sources'].update(old_seal['sources'])
    for path in (args.config.resolve(),ROOT/'scenemi/initialized_runtime.py'):
        seal['sources'][str(path.relative_to(ROOT))]=sha256(path)
    seal.update(protocol=config['version'],validation_cases=cases,subset_cases=subset,
        assets={str(INDEX.relative_to(ROOT)):sha256(INDEX),config['base']['checkpoint']:config['base']['sha256'],
            'runs/amass_exo128/SHARING_AUTHORIZATION.json':sha256(ROOT/'runs/amass_exo128/SHARING_AUTHORIZATION.json')},
        test_used=False,full_native_estimated_initialization=True,ground_truth_initialization=False)
    seal.pop('sha256');seal['sha256']=hashlib.sha256(json.dumps(seal,sort_keys=True).encode()).hexdigest()
    protocol=out/'PROTOCOL.json'
    if protocol.exists() and json.loads(protocol.read_text())!=seal:raise ValueError('Pilot protocol already sealed differently')
    atomic_json(protocol,seal);verify(seal)
    if args.action=='preflight':preflight(model,data,validation,cases,config,args.device,out,seal)
    else:train(model,data,validation,subset,cases,config,args.device,out,seal,args.resume)


if __name__=='__main__':
    main()
