"""Train scene adapters with the independently adapted camera prior frozen.

Stable entry: python -m scenemi.benchmark scene repair preflight|train.
Checkpoint files contain only adapters, optimizer and RNG, and pin the full
immutable base by SHA256. Validation and native shape conventions are reused.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi import local_training as local, scene_training as engine
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.local_scene import LocalSceneDataset, inference_payload
from scenemi.ego_training import load_ego_config
from scenemi.evaluation import verify_training_seal
from scenemi.parallel_resources import configure_cuda
from scenemi.scene_ablation import settings as control_settings
from scenemi.scene_model import native_motion_loss
from scenemi.scene_residual import FrozenSceneResidual
from scenemi.scene_selection import capture_candidate, finalists, validation_manifests
from scenemi.training import deterministic, ensure_bounded, now, restore_rng, rng_state, tensor_digest
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def settings(path):
    raw = yaml.safe_load(Path(path).read_text())
    c = local.settings(ROOT/raw['reference_config'])
    c.update(raw)
    if (c['version']!='native-scene-frozen-residual-v1' or
            c['training']['batch_size']!=8 or c['training']['validation_every']!=1000 or
            c['residual']['frozen_backbone_eval'] is not True or
            c['data']['datasets']!=['egobody','trumans','grail'] or
            c['data']['modes']!=['ego'] or c['validation']['test_used'] is not False):
        raise ValueError('Unexpected frozen scene repair protocol')
    if not (0<c['residual']['root_limit_m']<=.08 and 0<c['residual']['rotation_6d_limit']<=.2):
        raise ValueError('Scene residual exceeds its declared bounds')
    return c


def verify_base(c, scene_paths):
    base = c['base']; path = ROOT/base['checkpoint']
    if sha256(path)!=base['sha256']:
        raise ValueError('Pinned independent no-scene base checkpoint changed')
    seal = json.loads((ROOT/base['contract']).read_text())
    value = {k:v for k,v in seal.items() if k!='sha256'}
    if (hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()!=seal['sha256'] or
            seal['protocol']!='native-scene-ego128-none-v1' or
            seal['config']!=control_settings(ROOT/'configs/scene_ego128_noscene.yaml')):
        raise ValueError('Invalid independent no-scene base contract')
    inputs = yaml.safe_load(Path(scene_paths).read_text())
    for names in (seal['sources'], seal['assets']):
        for name,digest in names.items():
            if sha256(ROOT/name)!=digest:
                raise ValueError('Base source or body asset changed: '+name)
    for name,digest in seal['inputs'].items():
        if sha256(inputs[name])!=digest:
            raise ValueError('Base input membership changed: '+name)
    return seal


def make_model(c, paths, scene_paths):
    base_seal = verify_base(c, scene_paths)
    reference = local.settings(ROOT/c['reference_config'])
    # Explicit AMASS initialization preserves the original constructor and
    # verifies its body/coordinate contract before loading the adapted base.
    parent = json.loads((ROOT/reference['prior_contract']).read_text())
    verify_training_seal(parent,load_ego_config(ROOT/reference['prior_config']),paths,
                         paths['output']/'index/complete.json')
    if sha256(ROOT/reference['prior_checkpoint'])!=reference['prior_sha256']:
        raise ValueError('AMASS constructor initializer changed')
    prior = torch.load(ROOT/reference['prior_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    if prior['contract']!=parent or prior['step']!=reference['prior_step']:
        raise ValueError('Wrong AMASS constructor state')
    mount = json.loads(Path(yaml.safe_load(Path(scene_paths).read_text())['pv_mount']).read_text())['camera_wearer']
    model = FrozenSceneResidual(paths['smplh_models']/'neutral/model.npz',mount,
                                prior_state=prior['model'],local=c['local'])
    del prior
    state = torch.load(ROOT/c['base']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    if state['contract']!=base_seal or state['step']!=c['base']['step']:
        raise ValueError('Wrong scene-free adapted base content')
    model.load_state_dict(state['model'],strict=True); del state
    return model.set_adaptation(c['residual'])


def frozen_digest(model):
    trained = {n for n,p in model.named_parameters() if p.requires_grad}
    return tensor_digest({n:p for n,p in model.state_dict().items() if n not in trained})


def checkpoint_state(model, optimizer, rng, device, step, seal):
    return dict(adapter=model.adapter_state(), optimizer=optimizer.state_dict(),
        rng=rng_state(rng,device), step=step, contract=seal, base_sha256=seal['config']['base']['sha256'],
        storage='adapter_optimizer_rng_with_pinned_base')


def load_state(path, model, optimizer, rng, device, seal):
    state = torch.load(path,map_location='cpu',weights_only=False)
    if state['contract']!=seal or state['base_sha256']!=seal['config']['base']['sha256']:
        raise ValueError('Adapter checkpoint contract/base mismatch')
    model.load_adapter_state(state['adapter'])
    optimizer.load_state_dict(state['optimizer']); restore_rng(state['rng'],rng,device)
    return state['step']


def reconcile_journal(folder, step):
    """Keep only checkpointed updates; preserve crash-tail evidence separately."""
    path = folder/'updates.jsonl'
    if not path.exists():
        if step:
            raise ValueError('A nonzero resume checkpoint has no update journal')
        return
    lines = path.read_text().splitlines(keepends=True)
    if len(lines)<step:
        raise ValueError('Update journal is shorter than the resume checkpoint')
    if [json.loads(line)['step'] for line in lines[:step]]!=list(range(1,step+1)):
        raise ValueError('Checkpointed update identities are not contiguous')
    if len(lines)>step:
        tail=''.join(lines[step:])
        digest=hashlib.sha256(tail.encode()).hexdigest()
        (folder/f'interrupted_tail_{step:06d}_{digest[:12]}.jsonl').write_text(tail)
        temporary=path.with_suffix('.tmp'); temporary.write_text(''.join(lines[:step]))
        os.replace(temporary,path)


def update(model, optimizer, data, rng, c, device, step, forced=None):
    examples = [data.sample(rng,member=forced[i%len(forced)] if forced else None)
                for i in range(c['training']['batch_size'])]
    batch = collate([s for s,_ in examples],device)
    timestep = torch.randint(0,1000,(len(examples),),device=device)
    alpha = model.alphas[timestep]
    noisy = alpha[:,None,None].sqrt()*batch['motion']+(1-alpha[:,None,None]).sqrt()*torch.randn_like(batch['motion'])
    model.train(); optimizer.zero_grad(set_to_none=True)
    prediction,details = model(noisy,timestep,inference_payload(batch),return_details=True)
    terms = native_motion_loss(prediction,batch['motion'],batch,alpha.sqrt(),details=details,config=c)
    joints = forward_kinematics(prediction,details['rest'])
    baseline = forward_kinematics(details['baseline_motion'],details['rest']).detach()
    terms['preserve_head'] = (joints[:,:,15]-baseline[:,:,15]).square().mean()
    terms['preserve_root'] = (joints[:,:,0]-baseline[:,:,0]).square().mean()
    terms['total'] = terms['total']+c['residual']['head_consistency_weight']*terms['preserve_head']+c['residual']['root_consistency_weight']*terms['preserve_root']
    if not all(torch.isfinite(v).all() for v in terms.values()):
        raise FloatingPointError('Nonfinite repair loss; no update applied')
    terms['total'].backward()
    gradients = {name:float(getattr(model,name).weight.grad.norm()) for name in
                 ('patch_condition','dynamic_condition','region_condition','root_feedback','rotation_feedback')}
    norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                          c['optimizer']['clip_norm'],error_if_nonfinite=True)
    fraction = step/c['training']['planned_updates']
    lr = c['optimizer']['lr']*min(1.,step/c['optimizer']['warmup_updates'])*(.1+.9*(1+math.cos(math.pi*fraction))/2)
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.step()
    return dict(step=step, losses={k:float(v.detach()) for k,v in terms.items()},
        adapter_gradients=gradients, gradient_norm=float(norm),lr=lr,
        identities=[i for _,i in examples])


def preflight(model, data, validation, manifests, c, device, out, seal):
    folder = out/'preflight'; folder.mkdir(exist_ok=True)
    record = dict(state='running',device=device,contract_sha256=seal['sha256'],formal_training_started=False)
    atomic_json(folder/f'{device}.json',record)
    started = time.monotonic(); temporary = folder/f'.{device}.pt'
    rng = np.random.default_rng(c['seed']+31); optimizer = engine.optimizer_for(model,c)
    members = engine.forced_members(data); frozen = frozen_digest(model)
    try:
        examples = [data.sample(rng,member=m,yaw=0.) for m in members]
        batch = collate([s for s,_ in examples],device); payload = inference_payload(batch)
        noisy = torch.randn_like(batch['motion']); times = torch.full((len(examples),),100,device=device,dtype=torch.long)
        with torch.no_grad():
            model.eval()
            base = model(noisy,times,payload,use_scene=False)
            scene,details = model(noisy,times,payload,return_details=True)
            if not torch.equal(base,scene):
                raise AssertionError('Zero scene adapter does not exactly reproduce the independent prior')
            model.train()
            if model.core.training or model.shape_encoder.training or model.native_body.training:
                raise AssertionError('Frozen backbone or shape decoder entered training mode')
            if not torch.equal(base,model(noisy,times,payload)):
                raise AssertionError('Training mode changed the zero-adapter base output')
            model.eval()
            poisoned = dict(payload,target_rest=torch.full_like(batch['target_rest'],float('nan')),
                target_native_betas=torch.full_like(batch['target_native_betas'],float('nan')),
                target_template=torch.full_like(batch['target_template'],999))
            if not torch.equal(scene,model(noisy,times,poisoned)):
                raise AssertionError('Target body labels leaked into inference')
        first = update(model,optimizer,data,rng,c,device,1,members)
        if not any(v>0 for v in first['adapter_gradients'].values()):
            raise AssertionError('No scene adapter receives a motion-loss gradient')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise AssertionError('A frozen parameter received gradients')
        if frozen_digest(model)!=frozen:
            raise AssertionError('The base prior changed during adaptation')
        save_checkpoint(checkpoint_state(model,optimizer,rng,device,1,seal),temporary)
        expected = update(model,optimizer,data,rng,c,device,2,members)
        digest = tensor_digest((model.adapter_state(),optimizer.state_dict(),rng_state(rng,device)))
        load_state(temporary,model,optimizer,rng,device,seal)
        actual = update(model,optimizer,data,rng,c,device,2,members)
        if expected!=actual or tensor_digest((model.adapter_state(),optimizer.state_dict(),rng_state(rng,device)))!=digest:
            raise AssertionError('Adapter exact-resume failed')
        model.eval()
        with torch.no_grad():
            changed,details = model(noisy,times,payload,return_details=True)
            root_delta = (changed[...,:3]-details['baseline_motion'][...,:3])*2
            if root_delta.norm(dim=-1).max()>c['residual']['root_limit_m']+1e-6:
                raise AssertionError('Scene correction exceeded its root bound')
            fk = forward_kinematics(changed,details['rest'])
            if not torch.allclose(fk,changed[...,135:].reshape(*changed.shape[:2],22,3)*2,atol=2e-5,rtol=0):
                raise AssertionError('Adaptation broke FK consistency')
            if torch.equal(changed,base):
                raise AssertionError('Trained scene paths do not affect motion')
        if frozen_digest(model)!=frozen:
            raise AssertionError('The base changed after resume')
        # Real 20-step DDIM, one window per corpus, not a synthetic-only check.
        checks = local.evaluate(model,validation,[next(r for r in manifests['quick'] if r['dataset']==name)
            for name in ('egobody','trumans','grail')],c,device)
        record.update(state='passed',exact_zero_initializer=True,base_unchanged=True,
            target_poison_invariant=True,exact_resume=True,fk_consistent=True,
            scene_effective=True,root_bounded=True,frozen_sha256=frozen,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            initial_update=first,ddim_checks=checks,elapsed_seconds=time.monotonic()-started,
            checkpoint_bytes=temporary.stat().st_size,
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,completed_at_utc=now())
        atomic_json(folder/f'{device}.json',record)
        return record
    except BaseException as error:
        record.update(state='failed',error=repr(error),updated_at_utc=now())
        atomic_json(folder/f'{device}.json',record); raise
    finally:
        temporary.unlink(missing_ok=True)


def train(model, data, validation, manifests, c, device, out, seal, resume):
    for kind in ('cpu','cuda'):
        check = json.loads((out/'preflight'/f'{kind}.json').read_text())
        if check['state']!='passed' or check['contract_sha256']!=seal['sha256']:
            raise ValueError('Fresh matching CPU and CUDA preflights are mandatory')
    folder = out/'training'; folder.mkdir(exist_ok=True)
    contract_file = folder/'CONTRACT.json'
    if contract_file.exists() and json.loads(contract_file.read_text())!=seal:
        raise ValueError('Cannot change an existing repair training contract')
    if (folder/'last.pt').exists() and not resume:
        raise ValueError('Use --resume for an existing repair run')
    atomic_json(contract_file,seal)
    optimizer = engine.optimizer_for(model,c); rng = np.random.default_rng(c['seed']+41)
    step = load_state(folder/'last.pt',model,optimizer,rng,device,seal) if resume else 0
    reconcile_journal(folder,step)
    frozen = frozen_digest(model); stop=[]
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda signum,frame:stop.append(signum))
    selection = out/'selection'; selection.mkdir(exist_ok=True)
    quick = selection/'quick'; quick.mkdir(exist_ok=True)
    started = time.monotonic(); initial = step; counts = Counter()
    def status(state, **kw):
        value = dict(state=state,completed_updates=step,planned_updates=c['training']['planned_updates'],
            updates_per_second=(step-initial)/max(time.monotonic()-started,1e-6),datasets=dict(counts),
            pid=os.getpid(),updated_at_utc=now(),contract_sha256=seal['sha256'],
            frozen_base_sha256=c['base']['sha256'],rich_excluded=True,**kw)
        atomic_json(folder/'STATUS.json',value); return value
    def score():
        candidate = capture_candidate(folder/'last.pt',selection,step,seal['sha256'])
        digest = sha256(candidate)
        status('validation_quick',validation_completed=0,validation_windows=len(manifests['quick']))
        result = local.evaluate(model,validation,manifests['quick'],c,device,
            progress=lambda done,total,elapsed:status('validation_quick',
                validation_completed=done,validation_windows=total,validation_elapsed_seconds=elapsed))
        if step==0:
            reference = json.loads((ROOT/c['base']['quick_reference']).read_text())
            from scenemi.scene_evaluation import check_parity
            for a,b in zip(result['rows'],reference['rows']):
                if a['identity']!=b['identity']:
                    raise ValueError('Initial independent control identity changed')
                check_parity(a['metrics'],b['metrics'])
        if frozen_digest(model)!=frozen:
            raise ValueError('Frozen base changed during training')
        result.update(step=step,contract_sha256=seal['sha256'],checkpoint_sha256=digest,
                      selection_tier='quick',provisional=True,completed_at_utc=now())
        atomic_json(quick/f'step_{step:06d}.json',result)
        all_results = [json.loads(p.read_text()) for p in quick.glob('step_*.json')]
        best = min(all_results,key=lambda r:(r['selection_score_mm'],r['step']))
        atomic_json(selection/'BEST.json',{k:best[k] for k in ('step','selection_score_mm',
            'checkpoint_sha256','contract_sha256','selection_tier','provisional')})
        print(json.dumps(dict(event='quick_result',step=step,score_mm=result['selection_score_mm'],
                              best_step=best['step'],best_mm=best['selection_score_mm'])),flush=True)
    try:
        if not resume:
            save_checkpoint(checkpoint_state(model,optimizer,rng,device,step,seal),folder/'last.pt')
        if step%1000==0 and not (quick/f'step_{step:06d}.json').exists():
            score()
        with (folder/'updates.jsonl').open('a') as stream:
            while step<c['training']['planned_updates'] and not stop:
                step+=1; record=update(model,optimizer,data,rng,c,device,step)
                counts.update(i['dataset'] for i in record['identities'])
                stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush()
                if step%20==0:
                    value=status('training',latest_loss=record['losses']['total'])
                    if step%200==0:
                        print(json.dumps(value),flush=True)
                if step%c['training']['checkpoint_every']==0 or step==c['training']['planned_updates']:
                    save_checkpoint(checkpoint_state(model,optimizer,rng,device,step,seal),folder/'last.pt')
                if step%c['training']['validation_every']==0:
                    score()
        save_checkpoint(checkpoint_state(model,optimizer,rng,device,step,seal),folder/'last.pt')
        if stop:
            return status('stopped')
        results=[json.loads(p.read_text()) for p in quick.glob('step_*.json')]
        if {r['step'] for r in results}!=set(range(0,c['training']['planned_updates']+1,1000)):
            raise ValueError('Missing predeclared quick validation evaluations')
        selected=finalists(results,step)
        atomic_json(selection/'FINALISTS.json',dict(steps=selected,policy='quick_top3_plus_final',
            contract_sha256=seal['sha256'],test_used=False))
        comprehensive=selection/'comprehensive'; comprehensive.mkdir(exist_ok=True)
        full=[]
        for candidate_step in selected:
            candidate=selection/'candidates'/f'step_{candidate_step:06d}.pt'
            digest=sha256(candidate)
            result_file=comprehensive/f'step_{candidate_step:06d}.json'
            if result_file.exists():
                result=json.loads(result_file.read_text())
                if result['contract_sha256']!=seal['sha256'] or result['checkpoint_sha256']!=digest:
                    raise ValueError('Changed comprehensive candidate')
            else:
                load_state(candidate,model,optimizer,rng,device,seal)
                status('validation_comprehensive',evaluating_step=candidate_step,validation_completed=0,
                       validation_windows=len(manifests['comprehensive']))
                result=local.evaluate(model,validation,manifests['comprehensive'],c,device,
                    progress=lambda done,total,elapsed:status('validation_comprehensive',
                        evaluating_step=candidate_step,validation_completed=done,validation_windows=total,
                        validation_elapsed_seconds=elapsed))
                result.update(step=candidate_step,contract_sha256=seal['sha256'],checkpoint_sha256=digest,
                    selection_tier='comprehensive',provisional=False,completed_at_utc=now())
                atomic_json(result_file,result)
            full.append(result)
        best=min(full,key=lambda r:(r['selection_score_mm'],r['step']))
        identity={k:best[k] for k in ('step','selection_score_mm','checkpoint_sha256','contract_sha256',
                                     'selection_tier','provisional')}
        identity.update(storage='adapters; load with pinned no-scene base',evaluated_finalists=selected)
        candidate=selection/'candidates'/f"step_{best['step']:06d}.pt"
        target=selection/'.best.pt'; target.unlink(missing_ok=True);os.link(candidate,target)
        os.replace(target,selection/'best.pt'); atomic_json(selection/'BEST.json',identity)
        baseline=json.loads((ROOT/c['base']['comprehensive_reference']).read_text())
        if [r['identity'] for r in best['rows']]!=[r['identity'] for r in baseline['rows']]:
            raise ValueError('Final baseline and scene identities differ')
        result=dict(state='complete',best=identity,independent_noscene_score_mm=baseline['selection_score_mm'],
            baseline_minus_scene_mm=baseline['selection_score_mm']-best['selection_score_mm'],
            ranked_finalists=[{k:r[k] for k in ('step','selection_score_mm','checkpoint_sha256')}
                for r in sorted(full,key=lambda r:(r['selection_score_mm'],r['step']))],
            test_used=False,interpretation='Exploratory validation repair; base has additional adapter training.')
        atomic_json(selection/'RESULT.json',result)
        return status('complete',result=result)
    except BaseException as error:
        status('failed',error=repr(error));raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('preflight','train'))
    p.add_argument('--config',type=Path,default=ROOT/'configs/scene_ego128_residual.yaml')
    p.add_argument('--paths',type=Path,default=ROOT/'configs/paths.local.yaml')
    p.add_argument('--scene-paths',type=Path,default=ROOT/'configs/scene_paths.local.yaml')
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(argv); ensure_bounded()
    if a.device=='cuda': configure_cuda()
    c=settings(a.config); deterministic(c['seed'],a.device)
    paths=load_paths(a.paths); model=make_model(c,paths,a.scene_paths).to(a.device)
    pelvis=model.shape.rest_template[:1].detach().cpu().numpy()
    data=LocalSceneDataset(a.scene_paths,pelvis,config=c)
    validation=LocalSceneDataset(a.scene_paths,pelvis,split='validation',config=c)
    manifests=validation_manifests(validation,c['validation']['seed'],c['validation']['quick_grail_families_per_category'])
    original=json.loads((ROOT/c['base']['contract']).read_text())['validation_cases']
    if manifests!=original:
        raise ValueError('Repair must use exactly the original selection windows')
    seal=engine.contract(c,a.scene_paths,paths,manifests)
    for name in ('scenemi/runtime.py','scenemi/benchmark.py','scenemi/scene_evaluation.py',
                 c['reference_config'],'configs/scene_ego128_noscene.yaml'):
        seal['sources'][name]=sha256(ROOT/name)
    seal['base_references']={k:sha256(ROOT/c['base'][k]) for k in
                            ('contract','quick_reference','comprehensive_reference')}
    seal.pop('sha256'); seal['sha256']=hashlib.sha256(json.dumps(seal,sort_keys=True).encode()).hexdigest()
    out=ROOT/c['output']; out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'PREPARED_CONTRACT.json',seal);atomic_json(out/'VALIDATION_MANIFESTS.json',manifests)
    result=(preflight(model,data,validation,manifests,c,a.device,out,seal) if a.action=='preflight'
            else train(model,data,validation,manifests,c,a.device,out,seal,a.resume))
    print(json.dumps({k:v for k,v in result.items() if k not in ('initial_update','ddim_checks')},indent=2),flush=True)


if __name__=='__main__':
    main()
