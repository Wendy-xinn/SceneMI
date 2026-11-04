"""Conservative scene-free continuation with parent AdamW/RNG retained.

Uses the original independent scene-free model, original motion objectives and
input payload. No LR restart, new scene parameters or extra geometry labels.
The final step is evaluated on the original comprehensive validation manifest.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi import scene_ablation as control, scene_training as engine, scene_repair as repair
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.local_scene import LocalSceneDataset
from scenemi.geometry.native_shape import FORMATS
from scenemi.local_model import LocalScenePrior
from scenemi.scene_evaluation import check_parity
from scenemi.scene_model import native_motion_loss
from scenemi.training import deterministic, ensure_bounded, now, restore_rng, rng_state, tensor_digest
from scenemi.parallel_resources import configure_cuda
from experiments.offline_camera_retrain_v1.native30_cache_io import verified_sha256
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint


def settings(path):
    p=yaml.safe_load(Path(path).read_text())
    if (p['version']!='native-noscene-parent-optimizer-continuation-v1' or p['optimizer']!='restore_parent_adamw'
            or p['rng']!='restore_parent_all_rng' or p['test_used'] is not False or p['parent_step']!=30000
            or p['additional_updates']!=3000 or p['learning_rate']!=5e-6):
        raise ValueError('Unexpected conservative continuation protocol')
    c=control.settings(ROOT/p['reference_config']);c.update(version=p['version'],output=p['output'],continuation=p)
    c['training']=dict(c['training'],planned_updates=p['additional_updates'],checkpoint_every=p['checkpoint_every'])
    c['runtime']=p['runtime'];return c


def make_model(c,paths,scene_paths):
    p=c['continuation'];parent=json.loads((ROOT/p['parent_contract']).read_text())
    value={k:v for k,v in parent.items() if k!='sha256'}
    if hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()!=parent['sha256']:
        raise ValueError('Invalid original scene-free parent seal')
    if parent['config']!=control.settings(ROOT/p['reference_config']):raise ValueError('Parent protocol changed')
    for group in (parent['sources'],parent['assets']):
        for name,expected in group.items():verified_sha256(ROOT/name,expected)
    inputs=yaml.safe_load(Path(scene_paths).read_text())
    for name,expected in parent['inputs'].items():verified_sha256(inputs[name],expected)
    verified_sha256(ROOT/p['parent_checkpoint'],p['parent_sha256'])
    state=torch.load(ROOT/p['parent_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    if state['step']!=p['parent_step'] or state['contract']!=parent:raise ValueError('Wrong scene-free parent content')
    weights=state['model']
    def bases(prefix):
        return {kind:{name[len(prefix+kind+'_'):]:value for name,value in weights.items()
                if name.startswith(prefix+kind+'_')} for kind in FORMATS[1:]}
    prior={name:value for name,value in weights.items()
        if name!='local_bps_gain' and not name.startswith(('native_body.','skin.','pv_'))}
    mount=json.loads(Path(inputs['pv_mount']).read_text())['camera_wearer']
    if mount['fitting_split']!='train':raise ValueError('PV calibration must remain train-only')
    model=LocalScenePrior(paths['smplh_models']/'neutral/model.npz',mount,prior_state=prior,
        local=c['local'],bases=bases('native_body.bank.'),skin_bases=bases('skin.'))
    model.load_state_dict(weights,strict=True);model.requires_grad_(True);control.disable_scene(model)
    if any(not torch.equal(value,weights[name]) for name,value in model.state_dict().items()):
        raise AssertionError('Continuation initializer changed parent state')
    return model,parent


def restore_parent(model,c,device):
    p=c['continuation'];optimizer=engine.optimizer_for(model,c)
    state=torch.load(ROOT/p['parent_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    optimizer.load_state_dict(state['optimizer'])
    for group in optimizer.param_groups:
        if group['lr']!=p['learning_rate']:raise ValueError('Declared LR does not match parent final LR')
        for parameter in group['params']:
            entry=optimizer.state.get(parameter,{})
            if 'exp_avg' in entry and entry['exp_avg'].shape!=parameter.shape:
                raise ValueError('Parent optimizer parameter order mismatch')
    rng=np.random.default_rng(c['seed']);restore_rng(state['rng'],rng,device)
    return optimizer,rng


def update(model,optimizer,data,rng,c,device,step,forced=None):
    examples=[data.sample(rng,member=forced[i%len(forced)] if forced else None)
              for i in range(c['training']['batch_size'])]
    batch=collate([s for s,_ in examples],device)
    t=torch.randint(0,1000,(len(examples),),device=device);alpha=model.alphas[t]
    noisy=alpha[:,None,None].sqrt()*batch['motion']+(1-alpha[:,None,None]).sqrt()*torch.randn_like(batch['motion'])
    model.train();optimizer.zero_grad(set_to_none=True)
    pred,details=model(noisy,t,control.inference_payload(batch),use_scene=False,return_details=True)
    terms=native_motion_loss(pred,batch['motion'],batch,alpha.sqrt(),details=details,config=c)
    if not all(torch.isfinite(v).all() for v in terms.values()):raise FloatingPointError('Nonfinite conservative continuation')
    terms['total'].backward()
    norm=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),c['optimizer']['clip_norm'],error_if_nonfinite=True)
    for group in optimizer.param_groups:
        if group['lr']!=c['continuation']['learning_rate']:raise AssertionError('Continuation LR changed')
    optimizer.step()
    return dict(step=step,losses={k:float(v.detach()) for k,v in terms.items()},
        lr=optimizer.param_groups[0]['lr'],gradient_norm=float(norm),identities=[i for _,i in examples])


def save(model,optimizer,rng,device,step,seal,path):
    save_checkpoint(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(rng,device),
        step=step,contract=seal,parent_step=seal['config']['continuation']['parent_step']),path)


def load(path,model,optimizer,rng,device,seal):
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if state['contract']!=seal:raise ValueError('Conservative continuation resume seal changed')
    model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'],rng,device);return state['step']


def preflight(model,data,validation,c,device,seal,out):
    folder=out/'preflight';folder.mkdir(exist_ok=True)
    optimizer,rng=restore_parent(model,c,device);members=engine.forced_members(data)
    record=dict(state='running',contract_sha256=seal['sha256'],device=device,formal_training_started=False)
    temporary=folder/f'.{device}.pt';atomic_json(folder/f'{device}.json',record)
    try:
        initial_lr=[g['lr'] for g in optimizer.param_groups]
        parity=None
        if device=='cuda':
            cases=seal['validation_cases']['quick'];chosen=[next(r for r in cases if r['dataset']==d) for d in ('egobody','trumans','grail')]
            result=control.evaluate(model,validation,chosen,c,device)
            reference=json.loads((ROOT/'runs/scene_ego128_noscene/selection/quick/step_030000.json').read_text())['rows']
            for actual in result['rows']:
                expected=next(r for r in reference if r['identity']==actual['identity']);check_parity(actual['metrics'],expected['metrics'])
            parity=True
        first=update(model,optimizer,data,rng,c,device,1,members)
        save(model,optimizer,rng,device,1,seal,temporary)
        expected=update(model,optimizer,data,rng,c,device,2,members)
        digest=tensor_digest((model.state_dict(),optimizer.state_dict(),rng_state(rng,device)))
        load(temporary,model,optimizer,rng,device,seal)
        actual=update(model,optimizer,data,rng,c,device,2,members)
        if expected!=actual or digest!=tensor_digest((model.state_dict(),optimizer.state_dict(),rng_state(rng,device))):
            raise AssertionError('Conservative continuation exact resume failed')
        if initial_lr!=[g['lr'] for g in optimizer.param_groups]:raise AssertionError('LR was restarted')
        record.update(state='passed',parent_optimizer_and_rng_restored=True,constant_parent_lr=initial_lr,
            exact_resume=True,cuda_parent_ddim_parity=parity,initial_update=first,
            inference_keys=list(control.INPUTS),peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,completed_at_utc=now())
        atomic_json(folder/f'{device}.json',record);return record
    except BaseException as error:
        record.update(state='failed',error=repr(error));atomic_json(folder/f'{device}.json',record);raise
    finally:temporary.unlink(missing_ok=True)


def train(model,data,validation,c,device,seal,out,resume):
    for kind in ('cpu','cuda'):
        report=json.loads((out/'preflight'/f'{kind}.json').read_text())
        if report['state']!='passed' or report['contract_sha256']!=seal['sha256']:raise ValueError('Matching preflights required')
    folder=out/'training';folder.mkdir(exist_ok=True);file=folder/'CONTRACT.json'
    if file.exists() and json.loads(file.read_text())!=seal:raise ValueError('Started continuation protocol changed')
    atomic_json(file,seal);last=folder/'last.pt'
    if last.exists() and not resume:raise ValueError('Use --resume')
    optimizer,rng=restore_parent(model,c,device);step=load(last,model,optimizer,rng,device,seal) if resume else 0
    repair.reconcile_journal(folder,step);stop=[];started=time.monotonic();initial=step;counts=Counter()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda signum,frame:stop.append(signum))
    def status(state,**extra):
        value=dict(state=state,completed_updates=step,planned_updates=c['training']['planned_updates'],
            parent_step=c['continuation']['parent_step'],pid=os.getpid(),contract_sha256=seal['sha256'],
            datasets=dict(counts),updates_per_second=(step-initial)/max(time.monotonic()-started,1e-6),updated_at_utc=now(),**extra)
        atomic_json(folder/'STATUS.json',value);return value
    try:
        status('training')
        with (folder/'updates.jsonl').open('a') as stream:
            while step<c['training']['planned_updates'] and not stop:
                step+=1;record=update(model,optimizer,data,rng,c,device,step)
                counts.update(i['dataset'] for i in record['identities'])
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                if step%20==0:status('training',latest_loss=record['losses']['total'],lr=record['lr'])
                if step%c['training']['checkpoint_every']==0:save(model,optimizer,rng,device,step,seal,last)
                if step%c['continuation']['quick_every']==0:
                    status('validation_quick')
                    result=control.evaluate(model,validation,seal['validation_cases']['quick'],c,device)
                    result.update(step=step,contract_sha256=seal['sha256'],test_used=False,selection_tier='fixed_step_quick')
                    atomic_json(out/'evaluation/quick'/f'step_{step:06d}.json',result)
        save(model,optimizer,rng,device,step,seal,last)
        if stop:return status('stopped')
        status('validation_comprehensive',completed_windows=0,total_windows=len(seal['validation_cases']['comprehensive']))
        result=control.evaluate(model,validation,seal['validation_cases']['comprehensive'],c,device,
            progress=lambda done,total,elapsed:status('validation_comprehensive',completed_windows=done,total_windows=total,elapsed_seconds=elapsed))
        result.update(step=step,contract_sha256=seal['sha256'],checkpoint_sha256=sha256(last),test_used=False,
            selection_tier='fixed_step_comprehensive',interpretation='Matched extra-update step on validation; scene 3k was selected after 8k search. Not best-to-best or an independent test.')
        atomic_json(out/'evaluation/METRICS.json',result);return status('complete',selection_score_mm=result['selection_score_mm'])
    except BaseException as error:status('failed',error=repr(error));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('preflight','train'))
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--resume',action='store_true')
    p.add_argument('--config',type=Path,default=ROOT/'configs/scene_noscene_continue.yaml')
    p.add_argument('--paths',type=Path,default=ROOT/'configs/paths.local.yaml')
    p.add_argument('--scene-paths',type=Path,default=ROOT/'configs/scene_paths.local.yaml')
    a=p.parse_args();ensure_bounded()
    if a.device=='cuda':configure_cuda()
    c=settings(a.config);deterministic(c['seed'],a.device)
    model,parent=make_model(c,load_paths(a.paths),a.scene_paths);model=model.to(a.device)
    data=LocalSceneDataset(a.scene_paths,model.shape.rest_template[:1].cpu().numpy(),config=c)
    validation=LocalSceneDataset(a.scene_paths,model.shape.rest_template[:1].cpu().numpy(),split='validation',config=c)
    seal=engine.contract(c,a.scene_paths,load_paths(a.paths),parent['validation_cases']);seal['sources'].update(parent['sources'])
    seal['sources'][str(a.config.resolve().relative_to(ROOT))]=sha256(a.config);seal['assets']=parent['assets']
    seal.update(parent_checkpoint_sha256=c['continuation']['parent_sha256'],optimizer_and_rng='restore_parent',learning_rate='constant_parent_final_lr')
    seal.pop('sha256');seal['sha256']=hashlib.sha256(json.dumps(seal,sort_keys=True).encode()).hexdigest()
    out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True);file=out/'PREPARED_CONTRACT.json'
    if file.exists() and json.loads(file.read_text())!=seal:raise ValueError('Prepared continuation changed')
    atomic_json(file,seal)
    result=(preflight(model,data,validation,c,a.device,seal,out) if a.action=='preflight'
        else train(model,data,validation,c,a.device,seal,out,a.resume))
    print(json.dumps({k:v for k,v in result.items() if k!='initial_update'},indent=2),flush=True)


if __name__=='__main__':main()
