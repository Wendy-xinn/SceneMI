"""Prepare and train predicted-skin scene feedback after GRAIL review.

The previous global scene run remains an immutable control. Reuse its trainer
utilities; this stage owns its explicit input payload, updates and validation.
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
import shutil
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_config, load_paths
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.local_scene import LocalSceneDataset, inference_payload
from scenemi.ego_training import load_ego_config
from scenemi.local_model import LocalScenePrior
from scenemi.scene_model import native_motion_loss
from scenemi import scene_training as engine
from scenemi.parallel_resources import configure_cuda
from scenemi.training import deterministic, ensure_bounded, now, rng_state, tensor_digest, restore_rng
from scenemi.metrics import motion_metrics
from scenemi.evaluation import verify_training_seal
from scenemi.scene_selection import (aggregate_scene_rows, capture_candidate, finalists,
                                    validation_manifests)
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def settings(path):
    c = yaml.safe_load(Path(path).read_text())
    if (c['version']!='native-scene-ego128-local-v3' or c['data']['datasets']!=['egobody', 'trumans', 'grail']
            or c['data']['excluded_datasets']!=['rich'] or c['data']['modes']!=['ego']
            or c['data']['individual_shape_input'] is not False or c['validation']['test_used'] is not False):
        raise ValueError('Wrong local scene protocol')
    if not (0<c['local']['feedback_radius_m']<=c['local']['query_radius_m']<=.6):
        raise ValueError('Invalid observed-neighbourhood radii')
    if (c['training']['checkpoint_every']!=1000 or c['training']['validation_every']!=1000 or
            c['validation']['comprehensive_windows_per_unit']!=2 or
            c['validation']['finalists']!='quick_top3_plus_final' or
            c['validation']['primary']!='dataset_macro_grail_category_family_mpjpe_mm'):
        raise ValueError('Expected frozen two-stage scene checkpoint selection')
    c['loss'] = load_config(ROOT/c['base_config'])['loss']
    return c


def make_model(c, paths, scene_paths):
    checkpoint = ROOT/c['prior_checkpoint']
    if sha256(checkpoint)!=c['prior_sha256']:
        raise ValueError('Selected AMASS EGO-only initializer changed')
    parent = json.loads((ROOT/c['prior_contract']).read_text())
    verify_training_seal(parent,load_ego_config(ROOT/c['prior_config']),paths,paths['output']/'index/complete.json')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if (state['contract']!=parent or state['step']!=c['prior_step'] or
            parent['config']['observation']['mode_probabilities']!={'ego':1.,'exo':0.}):
        raise ValueError('Expected the validation-selected sealed EGO-only AMASS prior')
    mount = json.loads(Path(yaml.safe_load(Path(scene_paths).read_text())['pv_mount']).read_text())['camera_wearer']
    if mount['fitting_split']!='train':
        raise ValueError('PV population calibration must be train-only')
    model = LocalScenePrior(paths['smplh_models']/'neutral/model.npz', mount,
        prior_state=state['model'], local=c['local'])
    del state
    return model


def checkpoint_state(model,optimizer,rng,device,step,seal):
    state = engine.checkpoint_state(model,optimizer,rng,device,step,seal)
    state.update(initialized_from_amass55k=False, initialization=dict(
        source='amass_ego_only_validation_best',step=seal['config']['prior_step'],
        checkpoint=seal['config']['prior_checkpoint'],sha256=seal['config']['prior_sha256']))
    return state


def update(model, optimizer, data, rng, c, device, step, forced=None):
    examples = [data.sample(rng, member=forced[i%len(forced)] if forced else None)
                for i in range(c['training']['batch_size'])]
    batch = collate([s for s,_ in examples], device)
    t = torch.randint(0, 1000, (len(examples),), device=device)
    alpha = model.alphas[t]
    noisy = alpha[:, None, None].sqrt()*batch['motion']+(1-alpha[:, None, None]).sqrt()*torch.randn_like(batch['motion'])
    model.train(); optimizer.zero_grad(set_to_none=True)
    pred, details = model(noisy, t, inference_payload(batch), return_details=True)
    terms = native_motion_loss(pred, batch['motion'], batch, alpha.sqrt(), details=details, config=c)
    if not all(torch.isfinite(v).all() for v in terms.values()):
        raise FloatingPointError('Nonfinite local scene loss; no update applied')
    terms['total'].backward()
    gradients = {name:float(getattr(model,name).weight.grad.norm()) for name in
                 ('region_condition', 'root_feedback', 'rotation_feedback')}
    norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
        c['optimizer']['clip_norm'], error_if_nonfinite=True)
    fraction = step/c['training']['planned_updates']
    lr = c['optimizer']['lr']*min(1.,step/c['optimizer']['warmup_updates'])*(.1+.9*(1+math.cos(math.pi*fraction))/2)
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.step()
    return dict(step=step, losses={k:float(v.detach()) for k,v in terms.items()},
        gradient_norm=float(norm), local_gradients=gradients, lr=lr,
        local_coverage=details['local_coverage'].detach().cpu().tolist(), identities=[i for _,i in examples])


@torch.inference_mode()
def evaluate(model, data, cases, c, device, progress=None):
    saved = rng_state(np.random.default_rng(0), device)
    rows, was_training = [], model.training
    started = time.monotonic()
    model.eval()
    try:
        for case in cases:
            sample, identity = data.sample(np.random.default_rng(c['validation']['seed']),
                member=case['member'], start=case['start'], yaw=0.)
            if identity['synthetic_chair'] or identity['dataset']=='rich':
                raise ValueError('Invalid validation condition')
            identity['category'] = case['category']
            batch = collate([sample], device); payload = inference_payload(batch)
            pred = sample_native30(model, payload, steps=c['validation']['ddim_steps'],
                seed=c['validation']['seed'], use_scene=True)
            _, details = model(pred, torch.zeros(1,device=device,dtype=torch.long), payload, return_details=True)
            rows.append(dict(identity=identity,metrics=motion_metrics(
                forward_kinematics(pred,details['rest'])[0], forward_kinematics(batch['motion'],batch['target_rest'])[0])))
            if progress is not None and (len(rows)%10==0 or len(rows)==len(cases)):
                progress(len(rows),len(cases),time.monotonic()-started)
    finally:
        model.train(was_training); restore_rng(saved,np.random.default_rng(0),device)
    result = aggregate_scene_rows(rows)
    elapsed = time.monotonic()-started
    result.update(elapsed_seconds=elapsed,seconds_per_window=elapsed/len(rows))
    return result


def preflight(model, data, validation, cases, c, paths, device, out, seal):
    folder = out/'preflight'; folder.mkdir(parents=True,exist_ok=True)
    file = folder/f'{device}.json'; checkpoint = folder/f'.{device}.pt'
    started = time.monotonic()
    record = dict(state='running',device=device,contract_sha256=seal['sha256'],formal_training_started=False)
    atomic_json(file,record)
    rng = np.random.default_rng(c['seed']+31)
    optimizer = engine.optimizer_for(model,c)
    members = engine.forced_members(data)
    try:
        # Cover physical PV wearer, virtual interactee, SMPL and fixed G1.
        examples = [data.sample(rng,member=member,yaw=0.) for member in members]
        batch = collate([s for s,_ in examples],device); payload = inference_payload(batch)
        noisy = torch.randn_like(batch['motion']); times = torch.full((len(examples),),100,device=device,dtype=torch.long)
        model.eval()
        with torch.no_grad():
            model.local_enabled = False; control = model(noisy,times,payload)
            model.local_enabled = True; local,details = model(noisy,times,payload,return_details=True)
            if not torch.equal(control,local):
                raise AssertionError('Zero-initialized local paths change the global scene initializer')
            poisoned = dict(payload,target_rest=torch.full_like(batch['target_rest'],float('nan')),
                target_native_betas=torch.full_like(batch['target_native_betas'],float('nan')),
                target_template=torch.full_like(batch['target_template'],999))
            if not torch.equal(local,model(noisy,times,poisoned)):
                raise AssertionError('Target shape/template entered local inference')
            # Independently compare batched predicted-shape LBS with the frozen
            # per-window implementation, including the common pelvis offset.
            from scenemi.geometry.native_shape import FORMATS
            from experiments.offline_camera_retrain_v1.unified_geometry import skin_points
            actual_skin,_ = model.skin(batch['motion'],details,payload['body_format'])
            skin_error=0.
            for i,code in enumerate(payload['body_format'].tolist()):
                kind=FORMATS[code]
                template=int(details['predicted_template'][i]) if code!=3 else 0
                beta=details['predicted_native_betas'][i] if code!=3 else torch.zeros_like(details['predicted_native_betas'][i])
                def get(key):return getattr(model.skin,f'{kind}_{key}')[template]
                vertices=get('vertices')+torch.einsum('aci,i->ac',get('directions'),beta)
                joints=get('joints')+torch.einsum('jci,i->jc',get('joint_directions'),beta)
                asset=dict(vertices=vertices,joints=joints,weights=get('weights'),
                    parents=get('parents'),posedirs=get('posedirs'),extra_rotations=get('extra'),scale=1.)
                motion=batch['motion'][i].clone()
                motion[:,:3]+=(details['rest'][i,0]-joints[0])/2
                expected_skin=skin_points(motion,asset)
                skin_error=max(skin_error,float((actual_skin[i]-expected_skin).abs().max()))
            if skin_error>2e-5:
                raise AssertionError('Predicted native LBS differs from the frozen independent formula')
            truth = forward_kinematics(batch['motion'],batch['target_rest'])
            if not torch.allclose(truth,batch['motion'][...,135:].reshape(len(examples),128,22,3)*2,atol=3e-4,rtol=0):
                raise AssertionError('Native target FK mismatch')
            # Test active feedback without allowing unsupported surfaces to gate it.
            root_bias = model.root_feedback.bias.clone(); rot_bias = model.rotation_feedback.bias.clone()
            model.root_feedback.bias.fill_(.3); model.rotation_feedback.bias.fill_(.2)
            empty = dict(payload,local_history_valid=torch.zeros_like(payload['local_history_valid']),
                local_frame_owner=torch.full_like(payload['local_frame_owner'],-1))
            _, no_surface = model(noisy,times,empty,return_details=True)
            if no_surface['root_delta_m'].abs().max()!=0 or no_surface['rotation_delta_rad'].abs().max()!=0:
                raise AssertionError('Unknown surface triggered feedback')
            # Controlled *observed* surfaces at the generated skin ensure the
            # feedback path actually moves motion rather than just auxiliary logits.
            conditioned, _, shape, formats = model.conditions(payload,True,None)
            blank = (noisy.new_zeros(len(noisy),model.core.latent_dim),None,None,None)
            from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
            proposal = reconcile_motion(model._core_pass(noisy,times,conditioned,blank,None,False,True),shape['rest'])
            anchors,_ = model.skin(proposal,shape,formats)
            observed = dict(empty,local_frame_points=anchors,
                local_frame_owner=torch.zeros(anchors.shape[:-1],device=device,dtype=torch.long))
            shifted, active = model(noisy,times,observed,return_details=True)
            if not (active['root_delta_m'].abs().max()>0 and active['rotation_delta_rad'].abs().max()>0):
                raise AssertionError('Observed skin context cannot affect root and rotation')
            if active['root_delta_m'].norm(dim=-1).max()>c['local']['root_limit_m']+1e-6:
                raise AssertionError('Feedback exceeded root limit')
            if active['rotation_delta_rad'].norm(dim=-1).max()>c['local']['rotation_limit_rad']+1e-6:
                raise AssertionError('Feedback exceeded rotation limit')
            fk = forward_kinematics(shifted,active['rest'])
            if not torch.allclose(fk,shifted[...,135:].reshape(*shifted.shape[:2],22,3)*2,atol=2e-5,rtol=0):
                raise AssertionError('Feedback broke FK consistency')
            model.root_feedback.bias.copy_(root_bias); model.rotation_feedback.bias.copy_(rot_bias)
        first = update(model,optimizer,data,rng,c,device,1,members)
        if any(g<=0 for g in first['local_gradients'].values()):
            raise AssertionError('New local path has no body-loss gradient')
        save_checkpoint(checkpoint_state(model,optimizer,rng,device,1,seal),checkpoint)
        expected = update(model,optimizer,data,rng,c,device,2,members)
        digest = tensor_digest((model.state_dict(),optimizer.state_dict(),rng_state(rng,device)))
        engine.load_state(checkpoint,model,optimizer,rng,device,seal)
        actual = update(model,optimizer,data,rng,c,device,2,members)
        if expected!=actual or digest!=tensor_digest((model.state_dict(),optimizer.state_dict(),rng_state(rng,device))):
            raise AssertionError('Local scene exact-resume check failed')
        checks = evaluate(model,validation,[next(x for x in cases if x['dataset']==name)
            for name in ('egobody','trumans','grail')],c,device)
        record.update(state='passed',batch_size=c['training']['batch_size'],exact_resume=True,
            zero_local_equal=True,target_poison_invariant=True,unsupported_feedback_zero=True,
            observed_feedback_effective=True,feedback_bounded=True,native_fk_consistent=True,
            skin_lbs_max_error_m=skin_error,
            initial_update=first,ddim_checks=checks,elapsed_seconds=time.monotonic()-started,
            resumable_checkpoint_bytes=checkpoint.stat().st_size,
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,completed_at_utc=now())
        checkpoint.unlink()
        atomic_json(file,record)
        return record
    except BaseException as error:
        record.update(state='failed',error=repr(error),updated_at_utc=now());atomic_json(file,record)
        raise


def train(model,data,validation,manifests,c,device,out,seal,resume):
    if c['grail_review']!='accepted':
        raise ValueError('Formal local scene training waits for the user GRAIL review')
    for device_name in ('cpu','cuda'):
        checked=json.loads((out/'preflight'/f'{device_name}.json').read_text())
        if checked['state']!='passed' or checked['contract_sha256']!=seal['sha256']:
            raise ValueError('Current CPU and CUDA preflights required')
    estimated=checked['resumable_checkpoint_bytes']*(c['training']['planned_updates']//1000+1)
    if not resume and shutil.disk_usage(out).free < estimated*1.1+20*2**30:
        raise ValueError('Insufficient disk for retained checkpoint candidates plus 20GiB reserve')
    folder=out/'training';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'last.pt').exists() and not resume:
        raise ValueError('Existing run needs explicit resume')
    atomic_json(folder/'CONTRACT.json',seal)
    optimizer=engine.optimizer_for(model,c);rng=np.random.default_rng(c['seed']+41)
    step=engine.load_state(folder/'last.pt',model,optimizer,rng,device,seal) if resume else 0
    stop=[]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda signum,frame:stop.append(signum))
    selection=out/'selection';selection.mkdir(exist_ok=True)
    quick_folder=selection/'quick';quick_folder.mkdir(exist_ok=True)
    best=json.loads((selection/'BEST.json').read_text()) if (selection/'BEST.json').exists() else None
    counts=Counter();started=time.monotonic();initial=step
    def status(state,**kw):
        value=dict(state=state,completed_updates=step,planned_updates=c['training']['planned_updates'],
            updates_per_second=(step-initial)/max(time.monotonic()-started,1e-6),datasets=dict(counts),
            mode='ego',rich_excluded=True,local_skin_feedback=True,contract_sha256=seal['sha256'],
            pid=os.getpid(),updated_at_utc=now(),**kw)
        atomic_json(folder/'STATUS.json',value);return value
    def score():
        nonlocal best
        candidate=capture_candidate(folder/'last.pt',selection,step,seal['sha256'])
        digest=json.loads(candidate.with_suffix('.json').read_text())['checkpoint_sha256']
        status('validation_quick',validation_completed=0,validation_windows=len(manifests['quick']))
        result=evaluate(model,validation,manifests['quick'],c,device,
            progress=lambda done,total,elapsed:status('validation_quick',
                validation_completed=done,validation_windows=total,validation_elapsed_seconds=elapsed))
        result.update(step=step,contract_sha256=seal['sha256'],checkpoint_sha256=digest,
                      selection_tier='quick',provisional=True,completed_at_utc=now())
        atomic_json(quick_folder/f'step_{step:06d}.json',result)
        if best is None or result['selection_score_mm']<best['selection_score_mm']:
            target=selection/'.best.pt';target.unlink(missing_ok=True);os.link(candidate,target)
            os.replace(target,selection/'best.pt')
            best=dict(step=step,selection_score_mm=result['selection_score_mm'],checkpoint_sha256=digest,
                      contract_sha256=seal['sha256'],selection_tier='quick',provisional=True)
            atomic_json(selection/'BEST.json',best)
    if not resume:
        save_checkpoint(checkpoint_state(model,optimizer,rng,device,step,seal),folder/'last.pt');score()
    elif step%1000==0 and not (quick_folder/f'step_{step:06d}.json').exists():
        # A crash between durable checkpoint capture and completed validation
        # must resume that validation before advancing the training sequence.
        score()
    with (folder/'updates.jsonl').open('a') as stream:
        while step<c['training']['planned_updates'] and not stop:
            step+=1;record=update(model,optimizer,data,rng,c,device,step)
            counts.update(r['dataset'] for r in record['identities'])
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
            if step%20==0:status('training',latest_loss=record['losses']['total'])
            if step%c['training']['checkpoint_every']==0 or step==c['training']['planned_updates']:
                save_checkpoint(checkpoint_state(model,optimizer,rng,device,step,seal),folder/'last.pt')
            if step%c['training']['validation_every']==0 or step==c['training']['planned_updates']:score()
    save_checkpoint(checkpoint_state(model,optimizer,rng,device,step,seal),folder/'last.pt')
    if step==c['training']['planned_updates'] and not stop:
        results=[json.loads(p.read_text()) for p in sorted(quick_folder.glob('step_*.json'))]
        expected=set(range(0,c['training']['planned_updates']+1,1000))
        if {r['step'] for r in results}!=expected or any(r['contract_sha256']!=seal['sha256'] for r in results):
            raise ValueError('Missing or foreign quick checkpoint evaluations; candidates remain retained')
        selected=finalists(results,step)
        atomic_json(selection/'FINALISTS.json',dict(steps=selected,policy='quick_top3_plus_final',
            contract_sha256=seal['sha256'],test_used=False))
        comprehensive=selection/'comprehensive';comprehensive.mkdir(exist_ok=True)
        full_results=[]
        for candidate_step in selected:
            if stop:break
            candidate=selection/'candidates'/f'step_{candidate_step:06d}.pt'
            identity=json.loads(candidate.with_suffix('.json').read_text())
            if sha256(candidate)!=identity['checkpoint_sha256'] or identity['contract_sha256']!=seal['sha256']:
                raise ValueError('Retained candidate changed before comprehensive validation')
            result_file=comprehensive/f'step_{candidate_step:06d}.json'
            if result_file.exists():
                result=json.loads(result_file.read_text())
                if (result['checkpoint_sha256']!=identity['checkpoint_sha256'] or
                        result['contract_sha256']!=seal['sha256']):
                    raise ValueError('Foreign comprehensive checkpoint evaluation')
            else:
                state=torch.load(candidate,map_location='cpu',weights_only=False,mmap=True)
                if state['contract']!=seal or state['step']!=candidate_step:
                    raise ValueError('Retained candidate content does not match its identity')
                model.load_state_dict(state['model'],strict=True);del state;gc.collect()
                status('validation_comprehensive',evaluating_step=candidate_step,validation_completed=0,
                       validation_windows=len(manifests['comprehensive']))
                result=evaluate(model,validation,manifests['comprehensive'],c,device,
                    progress=lambda done,total,elapsed:status('validation_comprehensive',
                        evaluating_step=candidate_step,validation_completed=done,validation_windows=total,
                        validation_elapsed_seconds=elapsed))
                result.update(step=candidate_step,contract_sha256=seal['sha256'],
                    checkpoint_sha256=identity['checkpoint_sha256'],selection_tier='comprehensive',
                    provisional=False,completed_at_utc=now())
                atomic_json(result_file,result)
            full_results.append(result)
        if len(full_results)==len(selected):
            winner=min(full_results,key=lambda r:(r['selection_score_mm'],r['step']))
            candidate=selection/'candidates'/f"step_{winner['step']:06d}.pt"
            target=selection/'.best.pt';target.unlink(missing_ok=True);os.link(candidate,target)
            os.replace(target,selection/'best.pt')
            best={key:winner[key] for key in ('step','selection_score_mm','checkpoint_sha256',
                       'contract_sha256','selection_tier','provisional')}
            best.update(evaluated_finalists=selected,metrics=f"comprehensive/step_{winner['step']:06d}.json",
                selection_scope='Best among comprehensive finalists, not every training update.')
            atomic_json(selection/'BEST.json',best)
            atomic_json(selection/'RESULT.json',dict(state='complete',best=best,
                ranked_finalists=[{k:r[k] for k in ('step','selection_score_mm','checkpoint_sha256')}
                    for r in sorted(full_results,key=lambda r:(r['selection_score_mm'],r['step']))],
                coverage=manifests['coverage'],test_used=False,
                candidate_retention='Retained for selection review and possible additional validation.'))
    state=('complete' if step==c['training']['planned_updates'] and not stop else
           'selection_pending' if step==c['training']['planned_updates'] else 'stopped')
    return status(state,best=best)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['preflight','train'])
    p.add_argument('--config',type=Path,default=ROOT/'configs/scene_ego128_local.yaml')
    p.add_argument('--paths',type=Path,default=ROOT/'configs/paths.local.yaml')
    p.add_argument('--scene-paths',type=Path,default=ROOT/'configs/scene_paths.local.yaml')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();ensure_bounded();c=settings(a.config);paths=load_paths(a.paths)
    if a.action=='train' and c['grail_review']!='accepted':
        p.error('Formal training is held pending the user GRAIL review')
    if a.device=='cuda':configure_cuda()
    deterministic(c['seed'],a.device)
    model=make_model(c,paths,a.scene_paths).to(a.device)
    pelvis=model.shape.rest_template[:1].detach().cpu().numpy()
    data=LocalSceneDataset(a.scene_paths,pelvis,config=c)
    validation=LocalSceneDataset(a.scene_paths,pelvis,split='validation',config=c)
    manifests=validation_manifests(validation,c['validation']['seed'],
        c['validation']['quick_grail_families_per_category'])
    seal=engine.contract(c,a.scene_paths,paths,manifests)
    seal['sources']['scenemi/runtime.py']=sha256(ROOT/'scenemi/runtime.py')
    seal.pop('sha256');seal['sha256']=hashlib.sha256(json.dumps(seal,sort_keys=True).encode()).hexdigest()
    out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'PREPARED_CONTRACT.json',seal)
    atomic_json(out/'VALIDATION_MANIFESTS.json',manifests)
    result=(preflight(model,data,validation,manifests['quick'],c,paths,a.device,out,seal) if a.action=='preflight'
            else train(model,data,validation,manifests,c,a.device,out,seal,a.resume))
    print(json.dumps({k:v for k,v in result.items() if k not in ('initial_update','ddim_checks')},indent=2),flush=True)


if __name__=='__main__':main()
