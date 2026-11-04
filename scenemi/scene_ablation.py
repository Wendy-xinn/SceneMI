"""Matched scene-free adaptation control, using the same EGO-only AMASS prior.

Run through scenemi.runtime and `scenemi.benchmark train-control`. Keep the
active local trainer immutable. Reuse its initialization, dataset, losses,
checkpoint serialization and selection helpers, with no geometry in inference.
The selection loop here permits future adaptation controls without modifying
the sealed original experiment or changing its checkpoint policy.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi import local_training as local
from scenemi import scene_training as engine
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.local_scene import LocalSceneDataset
from scenemi.metrics import motion_metrics
from scenemi.observations import INPUT_KEYS
from scenemi.parallel_resources import configure_cuda
from scenemi.scene_model import native_motion_loss
from scenemi.scene_selection import aggregate_scene_rows, capture_candidate, finalists, validation_manifests
from scenemi.training import deterministic, ensure_bounded, now, rng_state, tensor_digest
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

INPUTS = INPUT_KEYS + ('body_format', 'sensor_profile')
SCENE_MODULES = ('patch_projection', 'patch_position', 'motion_query', 'region_embedding',
    'patch_attention', 'patch_condition', 'owner_embedding', 'dynamic_embedding',
    'dynamic_attention', 'dynamic_condition', 'surface_embedding', 'anchor_embedding',
    'anchor_query', 'surface_attention', 'region_condition', 'contact_readout', 'contact_pose',
    'correspondence_readout', 'feedback_temporal', 'root_feedback', 'rotation_feedback')


def inference_payload(batch):
    return {key: batch[key] for key in INPUTS}


def settings(path):
    config = yaml.safe_load(Path(path).read_text())
    reference = local.settings(ROOT/config['control']['reference_config'])
    expected = json.loads(json.dumps(reference))
    expected.update(version='native-scene-ego128-none-v1', output='runs/scene_ego128_noscene',
        model=dict(body_format=reference['model']['body_format'], scene='disabled', local_skin_feedback='disabled'),
        control=config['control'], runtime=dict(allocator_gib=6, memory_gib=10, cpu_count=2))
    config['loss'] = reference['loss']
    if config != expected or config['control']['kind'] != 'scene_free_adaptation':
        raise ValueError('Control must match the local branch except scene paths, output and resource limits')
    return config


def disable_scene(model):
    model.patch_enabled = model.local_enabled = model.feedback_enabled = False
    for name in SCENE_MODULES:
        getattr(model, name).requires_grad_(False)
    model.core.scene_embedding.requires_grad_(False)
    model.local_bps_gain.requires_grad_(False)
    return model


def make_model(config, paths, scene_paths):
    # Construct exactly the reference initializer, preserving initial shared
    # weights and random-number consumption before disabling unused branches.
    return disable_scene(local.make_model(config, paths, scene_paths))


def checkpoint_state(model, optimizer, rng, device, step, seal):
    value = local.checkpoint_state(model, optimizer, rng, device, step, seal)
    value.update(scene_enabled=False, local_skin_feedback=False, control='scene_free_adaptation')
    return value


def update(model, optimizer, data, rng, config, device, step, forced=None):
    examples = [data.sample(rng, member=forced[i % len(forced)] if forced else None)
                for i in range(config['training']['batch_size'])]
    batch = collate([s for s, _ in examples], device)
    timestep = torch.randint(0, 1000, (len(examples),), device=device)
    alpha = model.alphas[timestep]
    noisy = alpha[:, None, None].sqrt()*batch['motion'] + (1-alpha[:, None, None]).sqrt()*torch.randn_like(batch['motion'])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction, details = model(noisy, timestep, inference_payload(batch), use_scene=False, return_details=True)
    terms = native_motion_loss(prediction, batch['motion'], batch, alpha.sqrt(), details=details, config=config)
    if not all(torch.isfinite(v).all() for v in terms.values()):
        raise FloatingPointError('Nonfinite control loss; no update applied')
    terms['total'].backward()
    norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
        config['optimizer']['clip_norm'], error_if_nonfinite=True)
    shape_norm = sum(float(p.grad.square().sum()) for p in model.native_body.coefficients.parameters()
                     if p.grad is not None)**.5
    fraction = step/config['training']['planned_updates']
    lr = config['optimizer']['lr']*min(1., step/config['optimizer']['warmup_updates'])*(.1+.9*(1+math.cos(math.pi*fraction))/2)
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.step()
    return dict(step=step, losses={k:float(v.detach()) for k,v in terms.items()},
        gradient_norm=float(norm), native_shape_gradient=shape_norm, lr=lr,
        scene_enabled=False, identities=[identity for _, identity in examples])


@torch.inference_mode()
def evaluate(model, data, cases, config, device, progress=None):
    from scenemi.training import restore_rng
    saved = rng_state(np.random.default_rng(0), device)
    was_training, rows = model.training, []
    started = time.monotonic()
    model.eval()
    try:
        for case in cases:
            sample, identity = data.sample(np.random.default_rng(config['validation']['seed']),
                member=case['member'], start=case['start'], yaw=0.)
            if identity['synthetic_chair'] or identity['dataset']=='rich':
                raise ValueError('Invalid control validation condition')
            identity['category'] = case['category']
            batch = collate([sample], device)
            payload = inference_payload(batch)
            prediction = sample_native30(model, payload, steps=config['validation']['ddim_steps'],
                seed=config['validation']['seed'], use_scene=False)
            _, details = model(prediction, torch.zeros(1, device=device, dtype=torch.long),
                payload, use_scene=False, return_details=True)
            rows.append(dict(identity=identity, metrics=motion_metrics(
                forward_kinematics(prediction, details['rest'])[0],
                forward_kinematics(batch['motion'], batch['target_rest'])[0])))
            if progress is not None and (len(rows)%10==0 or len(rows)==len(cases)):
                progress(len(rows),len(cases),time.monotonic()-started)
    finally:
        model.train(was_training)
        restore_rng(saved,np.random.default_rng(0),device)
    result = aggregate_scene_rows(rows)
    elapsed = time.monotonic()-started
    result.update(elapsed_seconds=elapsed, seconds_per_window=elapsed/len(rows), scene_enabled=False)
    return result


def verify_reference(config, manifests):
    reference = ROOT/config['control']['reference_run']
    seal = json.loads((reference/'training/CONTRACT.json').read_text())
    if seal['config'] != local.settings(ROOT/config['control']['reference_config']):
        raise ValueError('Reference scene training configuration changed')
    for name, digest in seal['sources'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'Active reference source changed: {name}')
    if manifests != json.loads((reference/'VALIDATION_MANIFESTS.json').read_text()):
        raise ValueError('Control must reuse the exact reference validation windows and categories')
    return seal


def preflight(model, data, validation, cases, config, paths, device, out, seal, scene_paths):
    folder = out/'preflight'
    folder.mkdir(exist_ok=True)
    file, checkpoint = folder/f'{device}.json', folder/f'.{device}.pt'
    record = dict(state='running',device=device,started_at_utc=now(),contract_sha256=seal['sha256'],
                  formal_training_started=False,scene_enabled=False)
    atomic_json(file,record)
    started = time.monotonic()
    rng = np.random.default_rng(config['seed']+31)
    optimizer = engine.optimizer_for(model,config)
    try:
        reference = ROOT/config['control']['reference_run']
        candidate = reference/'selection/candidates/step_000000.pt'
        identity = json.loads(candidate.with_suffix('.json').read_text())
        if sha256(candidate) != identity['checkpoint_sha256']:
            raise ValueError('Reference zero-step candidate changed')
        state = torch.load(candidate,map_location='cpu',weights_only=False,mmap=True)
        if tensor_digest(model.state_dict()) != tensor_digest(state['model']):
            raise AssertionError('Control does not begin with exactly the same model tensors')
        del state
        members = engine.forced_members(data)
        examples = [data.sample(rng,member=member,yaw=0.) for member in members]
        batch = collate([s for s,_ in examples],device)
        payload = inference_payload(batch)
        noisy = torch.randn_like(batch['motion'])
        times = torch.full((len(examples),),100,device=device,dtype=torch.long)
        model.eval()
        with torch.no_grad():
            expected = model(noisy,times,payload,use_scene=False)
            poison = dict(payload,**{key:torch.full_like(value,float('nan'))
                for key,value in batch.items() if key not in INPUTS and torch.is_floating_point(value)})
            poison['target_template'] = torch.full_like(batch['target_template'],999)
            if not torch.equal(expected,model(noisy,times,poison,use_scene=False)):
                raise AssertionError('Scene geometry or target labels influenced no-scene inference')
            truth = forward_kinematics(batch['motion'],batch['target_rest'])
            if not torch.allclose(truth,batch['motion'][...,135:].reshape(len(examples),128,22,3)*2,atol=3e-4,rtol=0):
                raise AssertionError('Native target FK differs')
        first = update(model,optimizer,data,rng,config,device,1,members)
        if first['native_shape_gradient']<=0:
            raise AssertionError('Native predicted-shape adapter has no gradient')
        if any(p.grad is not None for name in SCENE_MODULES for p in getattr(model,name).parameters()):
            raise AssertionError('Disabled scene module received a gradient')
        save_checkpoint(checkpoint_state(model,optimizer,rng,device,1,seal),checkpoint)
        uninterrupted = update(model,optimizer,data,rng,config,device,2,members)
        digest = tensor_digest((model.state_dict(),optimizer.state_dict(),rng_state(rng,device)))
        del model,optimizer
        gc.collect()
        if device=='cuda': torch.cuda.empty_cache()
        model = make_model(config,paths,scene_paths).to(device)
        optimizer = engine.optimizer_for(model,config)
        engine.load_state(checkpoint,model,optimizer,rng,device,seal)
        resumed = update(model,optimizer,data,rng,config,device,2,members)
        if uninterrupted != resumed or digest != tensor_digest((model.state_dict(),optimizer.state_dict(),rng_state(rng,device))):
            raise AssertionError('Control recovery differs from uninterrupted training')
        checks = evaluate(model,validation,[next(c for c in cases if c['dataset']==name)
            for name in ('egobody','trumans','grail')],config,device)
        record.update(state='passed',batch_size=config['training']['batch_size'],exact_resume=True,
            scene_and_label_poison_invariant=True,initial_weights_bitwise_equal_to_local=True,
            frozen_scene_gradients_absent=True,initial_update=first,ddim_checks=checks,
            resumable_checkpoint_bytes=checkpoint.stat().st_size,elapsed_seconds=time.monotonic()-started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,completed_at_utc=now())
        checkpoint.unlink()
        atomic_json(file,record)
        return record
    except BaseException as error:
        record.update(state='failed',error=repr(error),updated_at_utc=now())
        atomic_json(file,record)
        raise


def train(model,data,validation,manifests,config,device,out,seal,resume):
    for kind in ('cpu','cuda'):
        checked = json.loads((out/'preflight'/f'{kind}.json').read_text())
        if checked['state']!='passed' or checked['contract_sha256']!=seal['sha256']:
            raise ValueError('Matching CPU and CUDA preflights are required')
    estimated = checked['resumable_checkpoint_bytes']*(config['training']['planned_updates']//1000+1)
    if not resume and shutil.disk_usage(out).free<estimated*1.1+20*2**30:
        raise ValueError('Insufficient disk for control candidates plus 20GiB reserve')
    folder = out/'training'
    folder.mkdir(exist_ok=True)
    if (folder/'last.pt').exists() and not resume:
        raise ValueError('Existing control requires explicit --resume')
    atomic_json(folder/'CONTRACT.json',seal)
    optimizer = engine.optimizer_for(model,config)
    rng = np.random.default_rng(config['seed']+41)
    step = engine.load_state(folder/'last.pt',model,optimizer,rng,device,seal) if resume else 0
    stop=[]
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda signum,frame:stop.append(signum))
    selection=out/'selection'
    quick=selection/'quick'
    quick.mkdir(parents=True,exist_ok=True)
    best=json.loads((selection/'BEST.json').read_text()) if (selection/'BEST.json').exists() else None
    started,initial=time.monotonic(),step
    counts=Counter()
    def status(state,**extra):
        value=dict(state=state,completed_updates=step,planned_updates=config['training']['planned_updates'],
            updates_per_second=(step-initial)/max(time.monotonic()-started,1e-6),datasets=dict(counts),
            scene_enabled=False,local_skin_feedback=False,mode='ego',rich_excluded=True,
            contract_sha256=seal['sha256'],pid=os.getpid(),updated_at_utc=now(),**extra)
        atomic_json(folder/'STATUS.json',value)
        return value
    def score():
        nonlocal best
        candidate=capture_candidate(folder/'last.pt',selection,step,seal['sha256'])
        identity=json.loads(candidate.with_suffix('.json').read_text())
        status('validation_quick',validation_completed=0,validation_windows=len(manifests['quick']))
        result=evaluate(model,validation,manifests['quick'],config,device,
            progress=lambda done,total,elapsed:status('validation_quick',validation_completed=done,
                validation_windows=total,validation_elapsed_seconds=elapsed))
        if step==0:
            baseline=json.loads((ROOT/config['control']['reference_run']/'selection/quick/step_000000.json').read_text())
            if (len(result['rows'])!=len(baseline['rows']) or any(a['identity']!=b['identity'] or
                    any(abs(v-b['metrics'][k])>1e-4 for k,v in a['metrics'].items())
                    for a,b in zip(result['rows'],baseline['rows']))):
                raise ValueError('Zero-step control must reproduce reference zero-scene-gate validation')
        result.update(step=step,contract_sha256=seal['sha256'],checkpoint_sha256=identity['checkpoint_sha256'],
            selection_tier='quick',provisional=True,completed_at_utc=now())
        atomic_json(quick/f'step_{step:06d}.json',result)
        if best is None or result['selection_score_mm']<best['selection_score_mm']:
            temp=selection/'.best.pt'
            temp.unlink(missing_ok=True)
            os.link(candidate,temp)
            os.replace(temp,selection/'best.pt')
            best={key:result[key] for key in ('step','selection_score_mm','checkpoint_sha256',
                'contract_sha256','selection_tier','provisional')}
            atomic_json(selection/'BEST.json',best)
    try:
        if not resume:
            save_checkpoint(checkpoint_state(model,optimizer,rng,device,step,seal),folder/'last.pt')
            score()
        elif step%1000==0 and not (quick/f'step_{step:06d}.json').exists(): score()
        with (folder/'updates.jsonl').open('a') as stream:
            while step<config['training']['planned_updates'] and not stop:
                step+=1
                record=update(model,optimizer,data,rng,config,device,step)
                counts.update(r['dataset'] for r in record['identities'])
                stream.write(json.dumps(record,allow_nan=False)+'\n')
                stream.flush()
                if step%20==0: status('training',latest_loss=record['losses']['total'])
                if step%1000==0 or step==config['training']['planned_updates']:
                    save_checkpoint(checkpoint_state(model,optimizer,rng,device,step,seal),folder/'last.pt')
                    score()
        save_checkpoint(checkpoint_state(model,optimizer,rng,device,step,seal),folder/'last.pt')
        if step==config['training']['planned_updates'] and not stop:
            results=[json.loads(p.read_text()) for p in sorted(quick.glob('step_*.json'))]
            if {r['step'] for r in results}!=set(range(0,step+1,1000)) or any(r['contract_sha256']!=seal['sha256'] for r in results):
                raise ValueError('Missing or foreign quick evaluations')
            selected=finalists(results,step)
            atomic_json(selection/'FINALISTS.json',dict(steps=selected,policy='quick_top3_plus_final',
                contract_sha256=seal['sha256'],test_used=False))
            comprehensive=selection/'comprehensive'
            comprehensive.mkdir(exist_ok=True)
            full=[]
            for candidate_step in selected:
                if stop: break
                candidate=selection/'candidates'/f'step_{candidate_step:06d}.pt'
                identity=json.loads(candidate.with_suffix('.json').read_text())
                if sha256(candidate)!=identity['checkpoint_sha256'] or identity['contract_sha256']!=seal['sha256']:
                    raise ValueError('Control finalist changed')
                path=comprehensive/f'step_{candidate_step:06d}.json'
                if path.exists():
                    result=json.loads(path.read_text())
                    if result['checkpoint_sha256']!=identity['checkpoint_sha256'] or result['contract_sha256']!=seal['sha256']:
                        raise ValueError('Foreign control comprehensive evaluation')
                else:
                    state=torch.load(candidate,map_location='cpu',weights_only=False,mmap=True)
                    if state['contract']!=seal or state['step']!=candidate_step:
                        raise ValueError('Control finalist content changed')
                    model.load_state_dict(state['model'],strict=True)
                    del state
                    gc.collect()
                    status('validation_comprehensive',evaluating_step=candidate_step,validation_completed=0,
                        validation_windows=len(manifests['comprehensive']))
                    result=evaluate(model,validation,manifests['comprehensive'],config,device,
                        progress=lambda done,total,elapsed:status('validation_comprehensive',evaluating_step=candidate_step,
                            validation_completed=done,validation_windows=total,validation_elapsed_seconds=elapsed))
                    result.update(step=candidate_step,contract_sha256=seal['sha256'],
                        checkpoint_sha256=identity['checkpoint_sha256'],selection_tier='comprehensive',
                        provisional=False,completed_at_utc=now())
                    atomic_json(path,result)
                full.append(result)
            if len(full)==len(selected):
                winner=min(full,key=lambda r:(r['selection_score_mm'],r['step']))
                temp=selection/'.best.pt'
                temp.unlink(missing_ok=True)
                os.link(selection/'candidates'/f"step_{winner['step']:06d}.pt",temp)
                os.replace(temp,selection/'best.pt')
                best={key:winner[key] for key in ('step','selection_score_mm','checkpoint_sha256',
                    'contract_sha256','selection_tier','provisional')}
                best.update(evaluated_finalists=selected,metrics=f"comprehensive/step_{winner['step']:06d}.json",
                    selection_scope='Best among comprehensive finalists, not every training update.')
                atomic_json(selection/'BEST.json',best)
                atomic_json(selection/'RESULT.json',dict(state='complete',best=best,
                    ranked_finalists=[{key:r[key] for key in ('step','selection_score_mm','checkpoint_sha256')}
                        for r in sorted(full,key=lambda r:(r['selection_score_mm'],r['step']))],
                    coverage=manifests['coverage'],test_used=False,candidate_retention='Retained for selection review.'))
        return status('complete' if step==config['training']['planned_updates'] and not stop else
                      'selection_pending' if step==config['training']['planned_updates'] else 'stopped',best=best)
    except BaseException as error:
        status('failed',error=repr(error))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('preflight','train'))
    parser.add_argument('--config',type=Path,default=ROOT/'configs/scene_ego128_noscene.yaml')
    parser.add_argument('--paths',type=Path,default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--scene-paths',type=Path,default=ROOT/'configs/scene_paths.local.yaml')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    ensure_bounded()
    config=settings(args.config)
    paths=load_paths(args.paths)
    if args.device=='cuda': configure_cuda()
    deterministic(config['seed'],args.device)
    model=make_model(config,paths,args.scene_paths).to(args.device)
    pelvis=model.shape.rest_template[:1].detach().cpu().numpy()
    data=LocalSceneDataset(args.scene_paths,pelvis,config=config)
    validation=LocalSceneDataset(args.scene_paths,pelvis,split='validation',config=config)
    manifests=validation_manifests(validation,config['validation']['seed'],config['validation']['quick_grail_families_per_category'])
    reference=verify_reference(config,manifests)
    seal=engine.contract(config,args.scene_paths,paths,manifests)
    seal.update(control_reference_sha256=reference['sha256'],scene_enabled=False,
        inference_keys=list(INPUTS),all_initial_weights_match_reference=True)
    for name in ('scenemi/runtime.py','scenemi/benchmark.py','scenemi/scene_ablation.py'):
        seal['sources'][name]=sha256(ROOT/name)
    seal.pop('sha256')
    seal['sha256']=hashlib.sha256(json.dumps(seal,sort_keys=True).encode()).hexdigest()
    out=ROOT/config['output']
    out.mkdir(parents=True,exist_ok=True)
    prepared=out/'PREPARED_CONTRACT.json'
    if prepared.exists() and json.loads(prepared.read_text())!=seal:
        raise ValueError('Prepared control sources, dependencies or protocol changed')
    atomic_json(prepared,seal)
    atomic_json(out/'VALIDATION_MANIFESTS.json',manifests)
    result=(preflight(model,data,validation,manifests['quick'],config,paths,args.device,out,seal,args.scene_paths)
            if args.action=='preflight' else train(model,data,validation,manifests,config,args.device,out,seal,args.resume))
    print(json.dumps({k:v for k,v in result.items() if k not in ('initial_update','ddim_checks')},indent=2),flush=True)


if __name__=='__main__': main()
