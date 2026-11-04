"""Resolve replay batch/LR with three matched short runs and fixed validation."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from scenemi import native_joint_training as stage
from scenemi.config import ROOT
from scenemi.data.amass_prepare import atomic_json,sha256
from scenemi.geometry.temporal import NativeJointMotionPrior
from scenemi.joint_model import JointMotionPrior
from scenemi.models import AmassMotionPrior
from scenemi.native_joint_data import ego_payload
from scenemi.data.exo_initialization import repair_payload
from scenemi.parallel_resources import configure_cuda
from scenemi.selection import pin_checkpoint
from scenemi.training import contract,deterministic,ensure_bounded,now,tensor_digest
from .benchmark import setup as data_setup
from .optimization import update
from .stream import SOURCE_WEIGHTS,CachedStream,build_cache
from .screen import prepare as prepare_screen,evaluate,compare


def initialize(settings,out):
    # Pin one already fully evaluated parent for all pilots. This is not the
    # final parent selection for formal training while scratch is still active.
    parent=out/'parent.pt';parent_info=out/'PARENT.json'
    if not parent.exists():
        best=json.loads((ROOT/settings['parent_run']/'selection/BEST.json').read_text())
        pin_checkpoint(Path(best['checkpoint']),parent)
        if sha256(parent)!=best['checkpoint_sha256']:raise ValueError('Parent changed during pin; rerun preparation')
        atomic_json(parent_info,dict(best,final_parent=False,pinned_at_utc=now()))
    info=json.loads(parent_info.read_text())
    if sha256(parent)!=info['checkpoint_sha256']:raise ValueError('Pinned pilot parent changed')
    state=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
    if state['step']!=info['step'] or state['contract']['config']['variant']!='scratch':raise ValueError('Expected validated scratch parent')
    deterministic(settings['seed'],'cpu');torch.set_num_threads(1)
    previous,discard,sampler,c,reference,paths=data_setup()
    bank=previous.base.native_body.bank
    bases={kind:{key:getattr(bank,kind+'_'+key).clone() for key in ('rest_template','rest_directions','eye_template','eye_directions')} for kind in ('smpl','smplx','grail_g1')}
    local=ROOT/'configs/scene_paths.local.yaml';mount=json.loads(Path(yaml.safe_load(local.read_text())['pv_mount']).read_text())['camera_wearer']
    prior={k[5:]:v for k,v in state['model'].items() if k.startswith('base.')}
    deterministic(settings['seed'],'cpu');torch.set_num_threads(1)
    base=stage.NativeMotionPrior(paths['smplh_models']/'neutral/model.npz',mount,prior_state=prior,bases=bases)
    c.update(seed=settings['seed'],source_weights=settings['source_weights'],pilot=settings['pilot'])
    stage.configure_trainable(base,c);model=NativeJointMotionPrior(base,**c['model'])
    adapter={k:v for k,v in state['model'].items() if not k.startswith('base.')}
    missing,unexpected=model.load_state_dict(adapter,strict=False)
    if unexpected or any(not k.startswith('base.') for k in missing):raise ValueError('Parent boundary transfer mismatch')
    for k,v in adapter.items():
        if not torch.equal(model.state_dict()[k],v):raise ValueError('Parent EXO/geometry transfer not exact')
    initial={k:v.clone() for k,v in model.state_dict().items()}
    del previous,discard,bank,bases;gc.collect()
    return model,initial,sampler,c,reference,paths,state,info


@torch.no_grad()
def parity(model,state,manifest,c,paths,out):
    old=JointMotionPrior(AmassMotionPrior(paths['smplh_models']/'neutral/model.npz'),**c['model'])
    old.load_state_dict(state['model'],strict=True);old.cuda().eval();model.eval()
    stream=CachedStream(manifest)
    try:ego,exo,_=stream.get(0,128)
    finally:stream.close()
    result={}
    for mode,batch,payload_fn in [('ego',ego,ego_payload),('exo',exo,repair_payload)]:
        mask=batch['dataset_id']==0;batch={k:v[mask] for k,v in batch.items()}
        noisy=batch['motion'];t=torch.full((len(noisy),),100,device='cuda',dtype=torch.long);payload=payload_fn(batch)
        a=model(noisy,t,payload);b=old(noisy,t,payload);difference=float((a-b).abs().max())
        if difference>1e-6:raise AssertionError(f'AMASS {mode} transfer differs: {difference}')
        poisoned=dict(payload,target_betas=torch.full_like(batch['target_betas'],float('nan')),
            target_rest=torch.full_like(batch['target_rest'],float('nan')),motion=torch.full_like(noisy,float('nan')),
            dataset_id=torch.full_like(batch['dataset_id'],99))
        if not torch.equal(a,model(noisy,t,poisoned)):raise AssertionError('Private labels affect inference')
        result[mode]=dict(max_abs_difference=difference,private_labels_ignored=True)
    del old;gc.collect();torch.cuda.empty_cache()
    result.update(state='passed',new_native_heads_are_observation_only=True,parent_step=state['step'])
    atomic_json(out/'TRANSFER_PREFLIGHT.json',result)


def candidate_config(c,settings,candidate):
    value=copy.deepcopy(c);batch=candidate['batch']
    value['training'].update(batch_size=batch,sampled_windows=settings['formal_training']['sampled_windows'],updates=settings['formal_training']['sampled_windows']//batch)
    value['optimizer']['warmup_windows']=settings['formal_training']['warmup_windows']
    for group in ('backbone','native_shape','exo'):
        value['optimizer'][group+'_lr']*=candidate.get('group_lr_multipliers',{}).get(group,candidate['lr_multiplier'])
    if 'source_loss_weights' in candidate:
        weights=candidate['source_loss_weights']
        if len(weights)!=4 or any(v<=0 for v in weights) or abs(sum(weights)-1)>1e-8:raise ValueError('Invalid source loss weights')
        value['source_weights']=weights
    return value


def resume_acceptance(model,initial,manifest,c,reference,seal,out):
    rng=np.random.default_rng(c['seed']);optimizer=stage.optimizer_for(model,c);stream=CachedStream(manifest)
    try:
        ego,exo,_=stream.get(0,128);update(model,optimizer,ego,exo,c,reference,1,diagnostic=True)
        path=out/'resume_probe.pt';stage.save(model,optimizer,rng,'cuda',1,seal,path)
        ego,exo,_=stream.get(128,128);first=update(model,optimizer,ego,exo,c,reference,2);expected=tensor_digest(model.state_dict())
        stage.restore(model,optimizer,rng,'cuda',seal,path)
        second=update(model,optimizer,ego,exo,c,reference,2)
        if first!=second or tensor_digest(model.state_dict())!=expected:raise AssertionError('Optimizer resume not exact')
        path.unlink();atomic_json(out/'RESUME_PREFLIGHT.json',dict(state='passed',model_and_optimizer_resume_exact=True,matched_noises=True))
    finally:
        stream.close();model.load_state_dict(initial,strict=True);model.zero_grad(set_to_none=True)
        del optimizer;gc.collect();torch.cuda.empty_cache()


def run_candidate(model,initial,manifest,c,reference,seal,out,candidate):
    folder=out/candidate['name'];folder.mkdir(exist_ok=True)
    model.load_state_dict(initial,strict=True);model.zero_grad(set_to_none=True)
    deterministic(c['seed'],'cuda');torch.set_num_threads(1)
    optimizer=stage.optimizer_for(model,c);rng=np.random.default_rng(c['seed']);stream=CachedStream(manifest)
    batch=candidate['batch'];updates=c['pilot']['windows']//batch;step=0;last=folder/'last.pt';log=folder/'updates.jsonl'
    if last.exists():
        state=stage.restore(model,optimizer,rng,'cuda',seal,last);step=state['step']
        with log.open('r+b') as handle:handle.truncate(state['log_bytes'])
    started=time.monotonic();completed_before=step
    try:
        with log.open('a',buffering=1) as handle:
            for step in range(step+1,updates+1):
                tick=time.monotonic();ego,exo,identities=stream.get((step-1)*batch,batch)
                value=update(model,optimizer,ego,exo,c,reference,step,diagnostic=step in (1,updates) or step%32==0)
                value.update(step=step,windows=step*batch,seconds=time.monotonic()-tick)
                handle.write(json.dumps(value,allow_nan=False)+'\n')
                if step%32==0 or step==updates:
                    handle.flush();os.fsync(handle.fileno());stage.save(model,optimizer,rng,'cuda',step,seal,last,handle.tell())
                    atomic_json(out/'STATUS.json',dict(state='pilot_training',candidate=candidate['name'],completed_updates=step,planned_updates=updates,
                        completed_windows=step*batch,planned_windows=c['pilot']['windows'],updated_at_utc=now(),formal_training_started=False))
                    print(json.dumps(dict(candidate=candidate['name'],step=step,windows=step*batch,loss_ego=value['loss_ego'],loss_exo=value['loss_exo'])),flush=True)
        return dict(optimizer_updates=updates,windows=updates*batch,resumed_from=completed_before,train_seconds=time.monotonic()-started,
            checkpoint_sha256=sha256(last),batch=batch,learning_rates={g['name']:g['initial_lr'] for g in optimizer.param_groups})
    finally:stream.close()


def run(args):
    ensure_bounded();settings=yaml.safe_load(args.config.read_text())
    if settings['version']!='replay-configuration-screen-v1' or settings['source_weights']!=list(SOURCE_WEIGHTS):raise ValueError('Unexpected screening protocol')
    out=ROOT/settings['output'];out.mkdir(exist_ok=True,parents=True)
    atomic_json(out/'STATUS.json',dict(state='preparing',started_at_utc=now(),formal_training_started=False))
    model,initial,sampler,c,reference,paths,parent,info=initialize(settings,out)
    data,cases=prepare_screen(model,paths,c,out)
    # Import dynamic CPU/CUDA entry before sealing; no old active source edits.
    from . import runtime
    seal=contract(settings,paths,ROOT/'runs/amass128/index/complete.json')
    seal['assets']={str(args.config.resolve().relative_to(ROOT)):sha256(args.config),'parent.pt':info['checkpoint_sha256'],
        c['selection']:sha256(ROOT/c['selection']),'PILOT_CASES.json':sha256(out/'PILOT_CASES.json')}
    seal['scope']='short_configuration_screen_not_formal_checkpoint_selection'
    seal['sha256']=hashlib.sha256(json.dumps({k:v for k,v in seal.items() if k!='sha256'},sort_keys=True).encode()).hexdigest()
    protocol=out/'PROTOCOL.json'
    if protocol.exists() and json.loads(protocol.read_text())!=seal:raise ValueError('Pilot protocol changed; preserve prior output')
    atomic_json(protocol,seal)
    cache=ROOT/settings['cache_source'] if 'cache_source' in settings else out/'train_cache'
    cache_manifest=cache/'MANIFEST.json'
    if cache_manifest.exists():manifest=json.loads(cache_manifest.read_text())
    else:manifest=build_cache(sampler,cache,c['pilot']['windows'],settings['seed'])
    if manifest['windows']!=c['pilot']['windows'] or manifest['seed']!=settings['seed']:raise ValueError('Training stream mismatch')
    configure_cuda();model.cuda();deterministic(settings['seed'],'cuda');torch.set_num_threads(1)
    parity(model,parent,manifest,c,paths,out)
    del parent,sampler;gc.collect()
    probe=candidate_config(c,settings,settings['pilot']['candidates'][min(1,len(settings['pilot']['candidates'])-1)])
    resume_acceptance(model,initial,manifest,probe,reference,seal,out)
    baseline_file=out/'baseline/SCREEN.json'
    if 'baseline_source' in settings and not baseline_file.exists():
        import shutil
        source=ROOT/settings['baseline_source']
        if (source/'PILOT_CASES.json').read_bytes()!=(out/'PILOT_CASES.json').read_bytes() or json.loads((source/'PARENT.json').read_text())['checkpoint_sha256']!=info['checkpoint_sha256']:
            raise ValueError('Reused baseline cases or parent changed')
        shutil.copytree(source/'baseline',out/'baseline')
    baseline=json.loads(baseline_file.read_text()) if baseline_file.exists() else evaluate(model,data,cases,c,out,'baseline')
    results={}
    for candidate in settings['pilot']['candidates']:
        folder=out/candidate['name'];result_file=folder/'RESULT.json';cfg=candidate_config(c,settings,candidate)
        if result_file.exists():result=json.loads(result_file.read_text())
        else:
            timing=run_candidate(model,initial,manifest,cfg,reference,seal,out,candidate)
            metrics=evaluate(model,data,cases,cfg,out,candidate['name']);result=dict(candidate=candidate,training=timing,metrics=metrics,**compare(metrics,baseline))
            atomic_json(result_file,result)
        results[candidate['name']]=result
    eligible=[v for v in results.values() if v['screen_eligible']]
    chosen=min(eligible,key=lambda v:v['score']) if eligible else None
    resolution=dict(state='complete',chosen=chosen['candidate'] if chosen else None,results={name:{k:v for k,v in value.items() if k!='metrics'} for name,value in results.items()},
        selection_scope='configuration_screen_only',full_validation=False,test_used=False,formal_training_started=False,parent_step=info['step'],completed_at_utc=now())
    if chosen:
        resolved=candidate_config(c,settings,chosen['candidate']);resolved.update(version='native-joint-replay-v1',output='runs/scene_joint128_replay',
            parent=dict(run=settings['parent_run'],require_complete=True,selection='final_full_validation_best'),
            resolution=dict(state='screen_passed_formal_preflight_pending',pilot_result=str((out/'RESULT.json').relative_to(ROOT)),pilot_parent_step=info['step'],candidate=chosen['candidate']))
        resolved.pop('base',None);resolved.pop('recovery',None);resolved.pop('batching',None)
        resolved['data']['datasets']=list(('amass','egobody','trumans','grail'))
        resolved['training'].pop('sample_order',None);resolved['training']['checkpoint_every']=max(1,4096//chosen['candidate']['batch'])
        resolved['optimizer'].pop('warmup_updates',None)
        resolved['validation'].pop('every_updates',None);resolved['validation'].pop('early_updates',None)
        resolved['data']['sampling']='fixed_source_quotas_then_family_then_recording_then_all_valid_starts_with_replacement'
        resolved['sampling_source_weights']=list(SOURCE_WEIGHTS)
        resolved['validation']['sampled_windows']=settings['formal_training']['validation_windows'];resolved['validation']['include_amass_exo']=True
        resolved['runtime']=settings['runtime'];resolved['pilot']=settings['pilot']
        (ROOT/'configs/scene_joint128_replay.yaml').write_text(yaml.safe_dump(resolved,sort_keys=False,allow_unicode=True))
    for name,digest in seal['sources'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Pilot source changed while running: '+name)
    atomic_json(out/'RESULT.json',resolution);atomic_json(out/'STATUS.json',resolution)
    print(json.dumps(resolution),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['pilot'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/scene_joint128_replay_pilot.yaml');args=parser.parse_args()
    try:run(args)
    except BaseException as error:
        settings=yaml.safe_load(args.config.read_text());out=ROOT/settings['output'];out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/'STATUS.json',dict(state='failed',error=repr(error),updated_at_utc=now(),formal_training_started=False));raise
