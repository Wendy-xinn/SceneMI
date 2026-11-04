"""Alternating EGO generation / EXO completion on a shared AMASS denoiser.

Warm adaptation and a genuine from-scratch exposure control share this entry.
Use joint_runtime for admission, preflight on CPU and CUDA, then train. Every
checkpoint-selection evaluation is synchronous and covers the complete val
manifest; training never reads the independent test split.
"""
import argparse
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
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import AmassDataset, collate, inference_payload
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.exo_initialization import CONDITIONS, repair_example, repair_payload
from scenemi.exo_completion import BoundaryEncoder
from scenemi.exo_denoise import channel_reliability, denoising_loss, sample_denoising
from scenemi.exo_repair_training import aggregate, verify
from scenemi.exo_training import verify_initialization_seal
from scenemi.joint_model import JointMotionPrior
from scenemi import joint_runtime as _runtime_dependency
from scenemi.metrics import motion_metrics
from scenemi.models import build_model, motion_loss
from scenemi.parallel_resources import configure_cuda
from scenemi.selection import evaluate_candidate, validation_manifest
from scenemi.training import (contract, deterministic, ensure_bounded, now,
    rng_state, restore_rng, tensor_digest, input_isolation_check)
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d

CONFIG = ROOT/'configs/amass_joint.yaml'


def configuration(path, variant):
    config = yaml.safe_load(path.read_text())
    if config['version'] != 'amass-joint-generation-repair-v1':
        raise ValueError('Wrong joint training protocol')
    selected = config.pop('variants')[variant]
    config.update(selected, variant=variant)
    if config['training']['sample_order'] != ['ego', 'exo']*4:
        raise ValueError('Require exactly four alternating EGO/EXO pairs per update')
    if config['training']['scene'] or config['validation']['test_used']:
        raise ValueError('Joint AMASS uses no scene or test data')
    config['validation']['every_updates'] = config['full_validation_every']
    return config


def setup(config, path, device):
    ensure_bounded(); deterministic(config['seed'], device)
    if device == 'cuda':
        if float(os.environ['SCENEMI_ALLOCATOR_GIB']) != config['runtime']['allocator_gib']:
            raise ValueError('Allocator differs from configuration')
        configure_cuda()
    paths = load_paths(ROOT/'configs/paths.local.yaml')
    reference = yaml.safe_load((ROOT/config['reference_config']).read_text())
    base = build_model(reference, paths['smplh_models']/'neutral/model.npz')
    if config['variant'] == 'warm':
        if sha256(ROOT/config['base']['checkpoint']) != config['base']['sha256']:
            raise ValueError('EGO initializer changed')
        original_seal = json.loads((ROOT/config['base']['checkpoint']).with_name('CONTRACT.json').read_text())
        verify_initialization_seal(original_seal, reference, paths)
        state = torch.load(ROOT/config['base']['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        if state['step'] != config['base']['step'] or state['contract'] != original_seal:
            raise ValueError('EGO initializer contract/step changed')
        base.load_state_dict(state['model'], strict=True); del state
    model = JointMotionPrior(base, **config['model'])
    if config['variant'] == 'warm':
        parent = config['boundary_parent']
        if sha256(ROOT/parent['checkpoint']) != parent['sha256']:
            raise ValueError('Boundary initializer changed')
        state = torch.load(ROOT/parent['checkpoint'], map_location='cpu', weights_only=False)
        if state['step'] != parent['step']:
            raise ValueError('Boundary initializer step changed')
        verify(state['contract']); model.load_repair_state(state['repair']); del state
    model.to(device)
    index = ROOT/config['index']
    data, validation = AmassDataset(index, paths, 'train'), AmassDataset(index, paths, 'val')
    cases = validation_manifest(validation.members)
    if len(cases) != 2222 or len(validation.members) != 337:
        raise ValueError('Full AMASS validation membership changed')
    seal = contract(config, paths, index)
    assets = {name:sha256(ROOT/name) for name in config['baselines'].values()}
    if config['variant'] == 'warm':
        for value in (config['base'], config['boundary_parent']):
            assets[value['checkpoint']] = value['sha256']
    for item in (path.resolve(), ROOT/'scenemi/joint_runtime.py'):
        seal['sources'][str(item.relative_to(ROOT))] = sha256(item)
    seal.update(protocol=config['version'], assets=assets, validation_cases=cases,
        pretrained_weights_loaded=config['variant']=='warm', test_used=False,
        reference_gradient='full_difference_both_passes_same_trainable_weights',
        one_trainable_denoiser=True, validation_baselines_used_as_training_targets=False,
        modes_per_optimizer_update={'ego':4, 'exo':4},
        sampling='Independent family/variant/start draws with replacement in EGO,EXO order')
    seal.pop('sha256');seal['sha256']=hashlib.sha256(json.dumps(seal,sort_keys=True).encode()).hexdigest()
    out = ROOT/config['output'];out.mkdir(parents=True, exist_ok=True)
    previous = out/'PROTOCOL.json'
    if previous.exists() and json.loads(previous.read_text()) != seal:
        raise ValueError('Started run is immutable; use a new configuration/output')
    atomic_json(previous, seal); verify(seal)
    deterministic(config['seed'], device)
    return model, data, validation, cases, reference, seal, out


def optimizer_for(model, config):
    return torch.optim.AdamW([
        dict(params=[p for p in model.base.parameters() if p.requires_grad],
             lr=config['backbone_lr'], initial_lr=config['backbone_lr'], name='shared_prior'),
        dict(params=list(model.condition.parameters())+list(model.projections.parameters()),
             lr=config['adapter_lr'], initial_lr=config['adapter_lr'], name='exo_boundary')],
        weight_decay=config['optimizer']['weight_decay'])


def batch_for(data, rng, config, device, *, forced=False):
    ego, exo, identities = [], [], []
    for i in range(4):
        sample, identity = data.sample(rng, 'ego', temporal_dropout=None)
        ego.append(sample); identities.append(identity)
        sample, identity = repair_example(data, rng, config['frontend'],
            condition=CONDITIONS[i] if forced else None)
        exo.append(sample); identities.append(identity)
    if [row['mode'] for row in identities] != config['training']['sample_order']:
        raise AssertionError('Training observation order changed')
    return collate(ego, device), collate(exo, device), identities


def update(model, optimizer, data, rng, config, reference, device, step):
    ego, exo, identities = batch_for(data, rng, config, device, forced=step<=2)
    model.train();optimizer.zero_grad(set_to_none=True)
    t = torch.randint(0, 1000, (4,), device=device)
    alpha = model.alphas[t, None, None]
    noisy = alpha.sqrt()*ego['motion']+(1-alpha).sqrt()*torch.randn_like(ego['motion'])
    prediction, details = model(noisy, t, inference_payload(ego), return_details=True)
    ego_terms = motion_loss(prediction, ego['motion'], ego, alpha[:, 0, 0].sqrt(),
                           details=details, config=reference)
    if not all(torch.isfinite(v).all() for v in ego_terms.values()):
        raise FloatingPointError('Nonfinite EGO objective')
    (.5*ego_terms['total']).backward()
    shape_gradient = sum(float(p.grad.detach().square().sum()) for p in
        model.base.shape_encoder.parameters() if p.grad is not None)**.5
    t_exo = torch.randint(0, config['diffusion']['start_timestep']+1, (4,), device=device)
    alpha_exo = model.alphas[t_exo, None, None]
    payload = repair_payload(exo);noise = torch.randn_like(exo['motion'])
    target = alpha_exo.sqrt()*exo['motion']+(1-alpha_exo).sqrt()*noise
    observed = alpha_exo.sqrt()*model.anchor(payload)+(1-alpha_exo).sqrt()*noise
    noisy = torch.lerp(target, observed, channel_reliability(payload))
    prediction, details = model(noisy, t_exo, payload, return_details=True)
    exo_terms = denoising_loss(prediction, exo, details)
    if not all(torch.isfinite(v).all() for v in exo_terms.values()):
        raise FloatingPointError('Nonfinite EXO objective')
    (.5*exo_terms['total']).backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
        config['optimizer']['clip_norm'], error_if_nonfinite=True)
    factor = min(step/config['optimizer']['warmup_updates'], 1.)*max(
        config['optimizer']['lr_floor_fraction'], .5*(1+math.cos(math.pi*step/config['updates'])))
    for group in optimizer.param_groups:group['lr'] = group['initial_lr']*factor
    optimizer.step()
    return dict(step=step, identities=identities, ego_timesteps=t.tolist(), exo_timesteps=t_exo.tolist(),
        ego_terms={k:float(v.detach()) for k,v in ego_terms.items()},
        exo_terms={k:float(v.detach()) for k,v in exo_terms.items()},
        shape_gradient=shape_gradient, grad_norm=float(norm),
        learning_rates={g['name']:g['lr'] for g in optimizer.param_groups})


def save(model, optimizer, rng, device, step, seal, path, log_bytes=0):
    # Unlike frozen-adapter checkpoints, this saves the UPDATED shared model.
    save_checkpoint(dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
        rng=rng_state(rng, device), device=device, contract=seal, log_bytes=log_bytes), path)


def restore(model, optimizer, rng, device, seal, path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['contract'] != seal or state['device'] != device:
        raise ValueError('Resume source/config/data/device identity changed')
    model.load_state_dict(state['model'], strict=True);optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'], rng, device)
    return state


def reduce_exo(rows):
    return {c:{v:aggregate([r for r in rows if r['condition']==c and r['variant']==v])
        for v in ('initial', 'temporal_smoothing', 'repair')} for c in CONDITIONS}


@torch.no_grad()
def evaluate_exo(model, data, cases, config, device, out, step):
    out.mkdir(parents=True, exist_ok=True)
    cached = [json.loads(line) for line in (ROOT/config['baselines']['exo_rows']).read_text().splitlines()]
    cached = [r for r in cached if r['variant'] in ('initial', 'temporal_smoothing')]
    lookup = {(r['condition'], r['case_id'], r['variant']):r for r in cached}
    rows=[];started=time.monotonic();model.eval()
    with (out/'metrics.jsonl').open('w') as stream:
        for condition in CONDITIONS:
            for offset in range(0, len(cases), config['validation']['batch_size']):
                chunk=cases[offset:offset+config['validation']['batch_size']]
                pairs=[repair_example(data,np.random.default_rng(777+c['case_id']),config['frontend'],
                    case=c,condition=condition) for c in chunk]
                batch=collate([s for s,_ in pairs],device);payload=repair_payload(batch)
                sampled=sample_denoising(model,payload,
                    steps=config['diffusion']['ddim_steps'],start_timestep=config['diffusion']['start_timestep'],
                    seed=config['diffusion']['seed'])
                points=forward_kinematics(sampled,batch['initial_rest'])
                truth=forward_kinematics(batch['motion'],batch['target_rest'])
                initial=forward_kinematics(batch['initial_motion'],batch['initial_rest'])
                for i,case in enumerate(chunk):
                    for variant in ('initial','temporal_smoothing'):
                        cached_row=lookup[condition,case['case_id'],variant]
                        if cached_row['identity']!=pairs[i][1]:raise ValueError('Cached frontend identity differs')
                        rows.append(cached_row);stream.write(json.dumps(cached_row,allow_nan=False)+'\n')
                    visible=(batch['joint_visible'][i]>0)&(batch['estimate_available'][i,:,None]>0)
                    region=~visible if condition in ('occlusion','gap') else torch.ones_like(visible)
                    error=(points[i]-truth[i]).norm(dim=-1)*1000
                    metrics=motion_metrics(points[i],truth[i])
                    metrics.update(repair_region_mpjpe_mm=float(error[region].mean()),
                        tracked_visible_mpjpe_mm=float(error[visible].mean()) if visible.any() else None,
                        max_change_from_initial_mm=float((points[i]-initial[i]).norm(dim=-1).max()*1000))
                    row=dict(case_id=case['case_id'],family=pairs[i][1]['family'],condition=condition,
                        variant='repair',identity=pairs[i][1],metrics=metrics)
                    rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                stream.flush();atomic_json(out/'STATUS.json',dict(state='evaluating',step=step,
                    condition=condition,completed=offset+len(chunk),total=len(cases),updated_at_utc=now()))
    families=len({c['family'] for c in cases})
    result=dict(conditions=reduce_exo(rows),full_validation=len(cases)==2222 and families==337,families=families,
        windows_per_condition=len(cases),test_used=False,elapsed_seconds=time.monotonic()-started)
    atomic_json(out/'RESULT.json',result)
    return result


def selection_result(ego, exo, config, step):
    initial_ego=json.loads((ROOT/config['baselines']['ego']).read_text())['family_macro_mean']['mpjpe_mm']
    initial_exo=json.loads((ROOT/config['baselines']['exo']).read_text())['conditions']
    conditions=exo['conditions'];policy=config['validation'];checks={}
    checks['identity_preserved']=conditions['identity']['repair']['family_macro_mean']['max_change_from_initial_mm']<=policy['identity_max_change_mm']
    ratios=[ego['family_macro_mean']['mpjpe_mm']/initial_ego]
    for c in ('tracking','occlusion','gap'):
        values=conditions[c];repair=values['repair']['family_macro_mean'];inputs=values['initial']['family_macro_mean']
        ratios.append(repair['repair_region_mpjpe_mm']/initial_exo[c]['repair']['family_macro_mean']['repair_region_mpjpe_mm'])
        for key in ('mpjpe_mm','root_error_mm','tracked_visible_mpjpe_mm'):
            checks[c+'_'+key+'_input_noninferiority']=repair[key]<=inputs[key]+policy['visible_noninferiority_mm']
        if config['variant']=='warm':
            checks[c+'_parent_region_noninferiority']=repair['repair_region_mpjpe_mm']<=initial_exo[c]['repair']['family_macro_mean']['repair_region_mpjpe_mm']+policy['warm_exo_region_max_regression_mm']
    if config['variant']=='warm':
        checks['ego_parent_noninferiority']=ego['family_macro_mean']['mpjpe_mm']<=initial_ego+policy['warm_ego_max_regression_mm']+1e-6
    return dict(step=step,ego=ego,exo=exo,checks=checks,selection_eligible=all(checks.values()),
        selection_score=float(np.mean(ratios)),normalization_baselines=dict(ego_mm=initial_ego,
            exo_regions_mm={c:initial_exo[c]['repair']['family_macro_mean']['repair_region_mpjpe_mm']
                            for c in ('tracking','occlusion','gap')}),
        full_validation=True,families=337,windows_per_condition=2222,test_used=False,
        completed_at_utc=now())


def validate(model, validation, cases, config, device, out, step):
    folder=out/f'selection/step_{step:06d}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'RESULT.json').exists():result=json.loads((folder/'RESULT.json').read_text())
    else:
        saved=rng_state(np.random.default_rng(),device);was_training=model.training
        try:
            if step==0 and config['variant']=='warm':
                for name in ('cpu','cuda'):
                    parity=json.loads((out/'preflight'/f'{name}.json').read_text())
                    if not parity['initializer_exact']:raise ValueError('Unverified baseline reuse')
                ego=json.loads((ROOT/config['baselines']['ego']).read_text())
                exo=json.loads((ROOT/config['baselines']['exo']).read_text())
                result=selection_result(ego,exo,config,step);result['baseline_reused_exact_initializer']=True
            else:
                ego_out=folder/'ego';ego_out.mkdir(exist_ok=True)
                ego=evaluate_candidate(model.base,validation,cases,dict(batch_size=config['validation']['ego_batch_size'],
                    seed=777,ddim_steps=config['validation']['ego_ddim_steps']),device,ego_out,dict(step=step,mode='ego'))
                atomic_json(ego_out/'RESULT.json',ego)
                exo=evaluate_exo(model,validation,cases,config,device,folder/'exo',step)
                result=selection_result(ego,exo,config,step)
            atomic_json(folder/'RESULT.json',result)
        finally:model.train(was_training);restore_rng(saved,np.random.default_rng(),device)
    best_path=out/'selection/BEST.json'
    best=json.loads(best_path.read_text()) if best_path.exists() else None
    if result['selection_eligible'] and (best is None or result['selection_score']<best['selection_score']):
        target=out/'selection/best.pt';temporary=target.with_name('.best.pt');temporary.unlink(missing_ok=True)
        os.link(out/'training/last.pt',temporary);os.replace(temporary,target)
        best=dict(step=step,selection_score=result['selection_score'],checkpoint=str(target),
            checkpoint_sha256=sha256(target),metrics=str(folder/'RESULT.json'),full_validation=True,test_used=False)
        atomic_json(best_path,best)
    return result,best


def preflight(model, data, validation, cases, config, reference, device, seal, out):
    folder=out/'preflight';folder.mkdir(exist_ok=True)
    model.eval();rng=np.random.default_rng(config['seed']+91)
    ego,exo,identities=batch_for(data,rng,config,device,forced=True)
    with torch.no_grad():
        payload=repair_payload(exo)
        prediction=sample_denoising(model,payload,steps=8,start_timestep=200)
        before_ego=model.base(torch.zeros_like(ego['motion']),torch.full((4,),500,device=device),inference_payload(ego))
        expected=prediction
        # SAME weights through the original frozen forward must give identical
        # inference. Only the gradient policy, not the warm initializer, changes.
        from scenemi.exo_denoise import DenoisingExoPrior
        original_forward=model.forward
        try:
            model.forward=lambda *args,**kwargs:DenoisingExoPrior.forward(model,*args,**kwargs)
            expected=sample_denoising(model,payload,steps=8,start_timestep=200)
        finally:model.forward=original_forward
        if not torch.equal(prediction,expected):raise AssertionError('Changed EXO initializer')
        if config['variant']=='warm':
            # Compare EGO directly with the immutable 55k initializer too.
            original_state=torch.load(ROOT/config['base']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
            own={k:v.detach().cpu() for k,v in model.base.state_dict().items()}
            if tensor_digest(own)!=tensor_digest(original_state['model']):raise AssertionError('Changed EGO initializer')
            del original_state,own
    # A zero adapter must cancel the shared-prior derivative exactly. This
    # detects the incorrect asymmetric stop-gradient reference on unfreezing.
    projection_state={k:v.detach().clone() for k,v in model.projections.state_dict().items()}
    with torch.no_grad():
        for parameter in model.projections.parameters():parameter.zero_()
    model.zero_grad(set_to_none=True)
    zero_prediction,zero_details=model(torch.zeros_like(exo['motion']),
        torch.full((4,),100,device=device),repair_payload(exo),return_details=True)
    if not zero_details['reference_x0'].requires_grad:raise AssertionError('Reference detached from shared prior')
    zero_prediction[...,:3].mean().backward()
    cancellation=max(float(p.grad.abs().max()) for p in model.base.parameters() if p.grad is not None)
    if cancellation>1e-6:raise AssertionError('Zero adapter created a spurious shared-prior gradient')
    model.projections.load_state_dict(projection_state);model.zero_grad(set_to_none=True)
    del zero_prediction,zero_details,projection_state
    base_before=tensor_digest(model.base.state_dict());geometry_before=tensor_digest(model.geometry.state_dict())
    optimizer=optimizer_for(model,config)
    first=update(model,optimizer,data,rng,config,reference,device,1)
    if first['shape_gradient']<=0:raise AssertionError('No camera-only EGO shape gradient')
    if tensor_digest(model.base.state_dict())==base_before:raise AssertionError('Shared prior did not update')
    temporary=folder/f'.{device}.pt';save(model,optimizer,rng,device,1,seal,temporary)
    expected=update(model,optimizer,data,rng,config,reference,device,2)
    digest=tensor_digest((model.state_dict(),optimizer.state_dict(),rng_state(rng,device)))
    restore(model,optimizer,rng,device,seal,temporary)
    actual=update(model,optimizer,data,rng,config,reference,device,2)
    if expected!=actual or digest!=tensor_digest((model.state_dict(),optimizer.state_dict(),rng_state(rng,device))):
        raise AssertionError('Joint exact recovery failed')
    temporary.unlink()
    if tensor_digest(model.geometry.state_dict())!=geometry_before:raise AssertionError('Fixed temporal anchor trained')
    isolation=input_isolation_check(model.base,ego)
    model.eval()
    with torch.no_grad():
        payload=repair_payload(exo);times=torch.full((4,),100,device=device)
        prediction=model(torch.zeros_like(exo['motion']),times,payload)
        poisoned=dict(payload,motion=torch.full_like(exo['motion'],float('nan')),
            target_rest=torch.full_like(exo['target_rest'],float('nan')),
            target_betas=torch.full_like(exo['target_betas'],float('nan')))
        if not torch.equal(prediction,model(torch.zeros_like(exo['motion']),times,poisoned)):
            raise AssertionError('Private EXO labels leaked')
        fixed=(exo['initial_reliability'].amin((-1,-2))>=1)&(exo['initial_root_reliability'].amin(-1)>=1)
        if not torch.equal(prediction[fixed],exo['initial_motion'][fixed]):raise AssertionError('Perfect input changed')
        points=forward_kinematics(prediction,exo['initial_rest'])
        torch.testing.assert_close(points,prediction[...,135:].reshape(4,128,22,3)*2,atol=3e-5,rtol=0)
        rot=rotation_from_6d(prediction[...,3:135].reshape(4,128,22,6))
        torch.testing.assert_close(rot.transpose(-1,-2)@rot,torch.eye(3,device=device).expand_as(rot),atol=3e-5,rtol=0)
        missing=exo['estimate_available']==0
        if exo['trajectory'][missing].any() or exo['observation_meta'][missing][...,:4].any():
            raise AssertionError('Missing EXO raw estimates leaked')
        if not torch.isfinite(sample_denoising(model,payload,steps=8,start_timestep=200)).all():
            raise AssertionError('Nonfinite repaired DDIM')
    # Exercise the actual EGO and EXO evaluator paths before waiting for 5k.
    # This is a diagnostic subset, never a checkpoint-selection result.
    smoke_cases=[cases[i] for i in (0,100,800,len(cases)-1)]
    smoke=folder/f'diagnostic_{device}';(smoke/'ego').mkdir(parents=True,exist_ok=True)
    ego_smoke=evaluate_candidate(model.base,validation,smoke_cases,
        dict(batch_size=4,seed=777,ddim_steps=20),device,smoke/'ego',dict(diagnostic_only=True))
    exo_smoke=evaluate_exo(model,validation,smoke_cases,config,device,smoke/'exo',-1)
    atomic_json(smoke/'RESULT.json',dict(diagnostic_only=True,used_for_selection=False,
        cases=[c['case_id'] for c in smoke_cases],ego_windows=ego_smoke['windows'],
        exo_windows_per_condition=exo_smoke['windows_per_condition']))
    verify(seal)
    result=dict(state='passed',device=device,contract_sha256=seal['sha256'],initializer_exact=True,
        batch_size=8,actual_samples_per_mode=4,alternating_sample_order_verified=True,
        exact_resume=True,shared_prior_updated=True,ego_shape_gradient=True,geometry_frozen=True,
        reference_gradient_connected=True,zero_adapter_shared_gradient_max=cancellation,
        actual_ego_exo_evaluators_checked=True,
        ego_input_isolation=isolation,exo_private_labels_isolated=True,identity_preserved=True,
        fk_so3_valid=True,missing_estimates_removed=True,ddim_steps_checked=8,
        pretrained_weights_loaded=config['variant']=='warm',first_update=first,
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,completed_at_utc=now())
    atomic_json(folder/f'{device}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='first_update'}),flush=True)


def train(model, data, validation, cases, config, reference, device, seal, out, resume):
    if device!='cuda':raise ValueError('Formal training requires CUDA acceptance')
    for name in ('cpu','cuda'):
        record=json.loads((out/'preflight'/f'{name}.json').read_text())
        if record['state']!='passed' or record['contract_sha256']!=seal['sha256']:
            raise ValueError('Missing matching acceptance: '+name)
    folder=out/'training';folder.mkdir(exist_ok=True)
    lock=(folder/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    last,log=folder/'last.pt',folder/'updates.jsonl'
    optimizer=optimizer_for(model,config);rng=np.random.default_rng(config['seed']+91);start=0
    if last.exists()!=resume:raise ValueError('Existing checkpoint requires explicit resume')
    if resume:
        state=restore(model,optimizer,rng,device,seal,last);start=state['step']
        with log.open('r+b') as stream:
            if stream.seek(0,2)<state['log_bytes']:raise ValueError('Committed log missing')
            stream.truncate(state['log_bytes'])
    elif log.exists() and log.stat().st_size:raise ValueError('Uncommitted log exists')
    stopping=[]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopping.append(True))
    started=time.monotonic();step=start
    best_path=out/'selection/BEST.json'
    best=json.loads(best_path.read_text()) if best_path.exists() else None
    status=dict(state='starting',pid=os.getpid(),started_at_utc=now(),variant=config['variant'],
        planned_updates=config['updates'],contract_sha256=seal['sha256'],shared_prior_trainable=True,
        initialization=config['initialization'],reference_gradient=seal['reference_gradient'],
        automatic_external_monitor=False,automatic_three_corpus_launch=False)
    def report(phase,**extra):
        status.update(state=phase,completed_updates=step,samples_per_mode=step*4,updated_at_utc=now(),
            elapsed_seconds=time.monotonic()-started,**extra)
        atomic_json(out/'STATUS.json',status);atomic_json(folder/'STATUS.json',status)
    try:
        with log.open('a',buffering=1) as stream:
            def commit():
                stream.flush();os.fsync(stream.fileno());save(model,optimizer,rng,device,step,seal,last,stream.tell())
            if not resume:commit()
            if (config['variant']=='warm' and step==0) or (resume and step>0 and (
                    step%config['full_validation_every']==0 or step==config['updates'])):
                report('full_validation');result,best=validate(model,validation,cases,config,device,out,step)
                status['best']=best
            for step in range(start+1,config['updates']+1):
                tick=time.monotonic();row=update(model,optimizer,data,rng,config,reference,device,step)
                row.update(update_seconds=time.monotonic()-tick,updated_at_utc=now());stream.write(json.dumps(row,allow_nan=False)+'\n')
                full=step%config['full_validation_every']==0 or step==config['updates']
                if step%config['training']['checkpoint_every']==0 or full or stopping:commit()
                if full:
                    report('full_validation');verify(seal)
                    result,best=validate(model,validation,cases,config,device,out,step)
                    status.update(best=best,last_validation=dict(step=step,score=result['selection_score'],
                        eligible=result['selection_eligible'],failed_checks=[k for k,v in result['checks'].items() if not v]))
                    with (out/'selection/history.jsonl').open('a') as history:
                        history.write(json.dumps(dict(step=step,selection_score=result['selection_score'],
                            eligible=result['selection_eligible'],ego_mpjpe_mm=result['ego']['family_macro_mean']['mpjpe_mm'],
                            exo_regions_mm={c:result['exo']['conditions'][c]['repair']['family_macro_mean']['repair_region_mpjpe_mm']
                                            for c in ('tracking','occlusion','gap')}))+'\n')
                if step%config['training']['log_every']==0 or full or stopping:
                    report('training',last_update_seconds=row['update_seconds'],
                        last_loss_ego=row['ego_terms']['total'],last_loss_exo=row['exo_terms']['total'],
                        peak_cuda_bytes=torch.cuda.max_memory_allocated());print(json.dumps(status),flush=True)
                if stopping:report('interrupted');return
            verify(seal)
            report('complete',best=best,final_checkpoint_sha256=sha256(last))
            atomic_json(out/'HANDOFF.json',dict(state='amass_complete_requires_native_preflight',
                best=best,training_status=str(folder/'STATUS.json'),new_training_started=False,
                next_stage_datasets=['egobody','trumans','grail'],excluded_datasets=['rich'],
                required=['Full EGO/EXO best metrics review','Checked native joint observation/shape adaptation',
                          'Fresh CPU/CUDA acceptance','Resource admission'],updated_at_utc=now()))
    except BaseException as error:
        report('failed',error=repr(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('preflight','train'))
    parser.add_argument('--variant',required=True,choices=('warm','scratch'))
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();config=configuration(args.config,args.variant)
    model,data,validation,cases,reference,seal,out=setup(config,args.config,args.device)
    if args.action=='preflight':preflight(model,data,validation,cases,config,reference,args.device,seal,out)
    else:train(model,data,validation,cases,config,reference,args.device,seal,out,args.resume)


if __name__=='__main__':main()
