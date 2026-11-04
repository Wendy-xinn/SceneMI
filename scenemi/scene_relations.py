"""Observed-surface supervision pilot, forked from the selected scene adapter.

This is an exploratory local-only continuation, not a matched training ablation
or a test-set result. The frozen no-scene base and selected 3k adapter are pinned
separately. Existing runs and their dependency closures remain unchanged.
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
import sys
import time

import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi import local_training as local, scene_training as engine, scene_repair as repair
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.local_scene import LocalSceneDataset, inference_payload
from scenemi.geometry.observed_relations import supervision, observed_planes
from scenemi.geometry.native_shape import FORMATS
from scenemi.metrics import motion_metrics
from scenemi.scene_evaluation import SCENE_KEYS, check_parity
from scenemi.scene_followup import change_scene, verify_sources
from scenemi.scene_model import native_motion_loss
from scenemi.scene_selection import aggregate_scene_rows
from scenemi.scene_residual import FrozenSceneResidual
from scenemi.training import deterministic, ensure_bounded, now, restore_rng, rng_state, tensor_digest
from scenemi.parallel_resources import configure_cuda
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.native30_cache_io import verified_sha256
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


LOCAL_MODULES = ('surface_embedding', 'anchor_embedding', 'anchor_query', 'surface_attention',
                 'region_condition', 'contact_readout', 'contact_pose', 'correspondence_readout',
                 'feedback_temporal', 'root_feedback', 'rotation_feedback')


def settings(path):
    p=yaml.safe_load(Path(path).read_text())
    if (p['version']!='native-observed-relations-pilot-v1' or p['test_used'] is not False
            or p['planned_updates']!=500 or p['parent_step']!=3000):
        raise ValueError('Unexpected observed-relation pilot protocol')
    c=repair.settings(ROOT/p['repair_config'])
    c.update(version=p['version'], output=p['output'], relation_pilot=p)
    c['training']=dict(c['training'],planned_updates=p['planned_updates'],batch_size=p['batch_size'],
                       checkpoint_every=p['checkpoint_every'])
    c['optimizer']=dict(c['optimizer'],lr=p['learning_rate'],warmup_updates=p['warmup_updates'])
    c['runtime']=p['runtime']
    return c


def pilot_indices(rows, seed, count):
    rng=np.random.default_rng(seed);chosen=[]
    for dataset in ('egobody','trumans','grail'):
        pool=[i for i,r in enumerate(rows) if r['identity']['dataset']==dataset]
        order=list(map(int,rng.permutation(pool)));selected=[];families=set()
        if dataset=='grail':
            categories=sorted({rows[i]['identity']['category'] for i in order})
            for category in categories:
                i=next(i for i in order if rows[i]['identity']['category']==category)
                selected.append(i);families.add(rows[i]['identity']['family'])
        for i in order:
            if len(selected)>=count:break
            if rows[i]['identity']['family'] not in families:
                selected.append(i);families.add(rows[i]['identity']['family'])
        if len(selected)!=count:raise ValueError('Insufficient independent pilot families')
        chosen.extend(selected)
    return sorted(chosen)


def make_model(c, paths, scene_paths):
    p=c['relation_pilot'];parent=json.loads((ROOT/p['parent_contract']).read_text());verify_sources(parent)
    # A stale buffered read must never become a new accepted identity. Reuse
    # the existing exact-digest retry path, which can refresh this file only.
    base=json.loads((ROOT/c['base']['contract']).read_text())
    value={k:v for k,v in base.items() if k!='sha256'}
    if (hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()!=base['sha256']
            or base['config']!=repair.control_settings(ROOT/'configs/scene_ego128_noscene.yaml')):
        raise ValueError('Invalid independent no-scene base contract')
    for group in (base['sources'],base['assets']):
        for name,expected in group.items():verified_sha256(ROOT/name,expected)
    inputs=yaml.safe_load(Path(scene_paths).read_text())
    for name,expected in base['inputs'].items():verified_sha256(inputs[name],expected)
    verified_sha256(ROOT/c['base']['checkpoint'],c['base']['sha256'])
    verified_sha256(ROOT/p['parent_checkpoint'],p['parent_sha256'])
    state=torch.load(ROOT/c['base']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    if state['contract']!=base or state['step']!=c['base']['step']:
        raise ValueError('Wrong independent no-scene checkpoint content')
    weights=state['model']
    # These public body-model bases are already serialized in the pinned full
    # checkpoint. Rehydrate them rather than repeatedly unpickling ~250MB
    # legacy assets. Full strict reload below verifies the resulting model.
    def bases(prefix):
        return {kind:{name[len(prefix+kind+'_'):]:value for name,value in weights.items()
                      if name.startswith(prefix+kind+'_')} for kind in FORMATS[1:]}
    prior={name:value for name,value in weights.items()
           if name!='local_bps_gain' and not name.startswith(('native_body.','skin.','pv_'))}
    mount=json.loads(Path(inputs['pv_mount']).read_text())['camera_wearer']
    if mount['fitting_split']!='train':raise ValueError('PV calibration must remain train-only')
    model=FrozenSceneResidual(paths['smplh_models']/'neutral/model.npz',mount,prior_state=prior,
        local=c['local'],bases=bases('native_body.bank.'),skin_bases=bases('skin.'))
    model.load_state_dict(weights,strict=True);model.set_adaptation(c['residual'])
    del prior
    state=torch.load(ROOT/p['parent_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    if state['contract']!=parent or state['step']!=p['parent_step']:
        raise ValueError('Wrong scene-parent state')
    model.load_adapter_state(state['adapter'])
    for name,value in model.state_dict().items():
        expected=state['adapter'].get(name,weights[name])
        if not torch.equal(value,expected):raise AssertionError('Serialized-base initializer changed parent state: '+name)
    model._verified_parent_state=True
    del state,weights
    model.requires_grad_(False)
    model._adapter_modules=LOCAL_MODULES
    for name in LOCAL_MODULES:model.get_submodule(name).requires_grad_(True)
    model.local_bps_gain.requires_grad_(True)
    model.eval()
    return model


def prepare(c, paths, scene_paths, config):
    model=make_model(c,paths,scene_paths)
    pelvis=model.shape.rest_template[:1].cpu().numpy()
    data=LocalSceneDataset(scene_paths,pelvis,config=c)
    validation=LocalSceneDataset(scene_paths,pelvis,split='validation',config=c)
    p=c['relation_pilot'];parent=json.loads((ROOT/p['parent_contract']).read_text())
    reference=json.loads((ROOT/'runs/scene_ego128_residual/selection/quick/step_003000.json').read_text())
    ids=pilot_indices(reference['rows'],p['evaluation_seed'],p['pilot_families_per_dataset'])
    manifests=parent['validation_cases'];seal=engine.contract(c,scene_paths,paths,manifests)
    seal['sources'].update(parent['sources'])
    # Record the asset identities actually verified above; a legacy buffered
    # hash in engine.contract cannot silently replace those pinned identities.
    seal['assets']=json.loads((ROOT/c['base']['contract']).read_text())['assets']
    for name in ('scenemi/scene_relations.py','scenemi/geometry/observed_relations.py',str(Path(config).resolve().relative_to(ROOT))):
        seal['sources'][name]=sha256(ROOT/name)
    seal.update(parent_scene_sha256=p['parent_sha256'],parent_scene_step=p['parent_step'],
        frozen_noscene_sha256=c['base']['sha256'],pilot_case_ids=ids,
        selection_policy='Fixed final 500 updates; pilot diagnostics only; no pilot checkpoint search',
        trainable_modules=list(LOCAL_MODULES)+['local_bps_gain'],
        initializer='Verified full no-scene checkpoint with serialized native/skin bases, strict reload, then pinned selected scene adapter',
        contact_labels='Weak proximity/correspondence from observed local planes; not gold contact')
    seal.pop('sha256');seal['sha256']=hashlib.sha256(json.dumps(seal,sort_keys=True).encode()).hexdigest()
    out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
    file=out/'PREPARED_CONTRACT.json'
    if file.exists() and json.loads(file.read_text())!=seal:raise ValueError('Started pilot protocol changed')
    atomic_json(file,seal)
    return model,data,validation,reference,seal


def update(model,optimizer,data,rng,c,device,step,forced=None):
    examples=[data.sample(rng,member=forced[i%len(forced)] if forced else None)
              for i in range(c['training']['batch_size'])]
    batch=collate([s for s,_ in examples],device)
    t=torch.randint(0,1000,(len(examples),),device=device);alpha=model.alphas[t]
    noisy=alpha[:,None,None].sqrt()*batch['motion']+(1-alpha[:,None,None]).sqrt()*torch.randn_like(batch['motion'])
    optimizer.zero_grad(set_to_none=True);model.train()
    prediction,details=model(noisy,t,inference_payload(batch),return_details=True)
    terms=native_motion_loss(prediction,batch['motion'],batch,alpha.sqrt(),details=details,config=c)
    joints=forward_kinematics(prediction,details['rest'])
    baseline=forward_kinematics(details['baseline_motion'],details['rest']).detach()
    terms['preserve_head']=(joints[:,:,15]-baseline[:,:,15]).square().mean()
    terms['preserve_root']=(joints[:,:,0]-baseline[:,:,0]).square().mean()
    terms['total']=terms['total']+c['residual']['head_consistency_weight']*terms['preserve_head']+c['residual']['root_consistency_weight']*terms['preserve_root']
    geometric,coverage=supervision(model,prediction,details,batch,c['relation_pilot']['relation'])
    terms.update(geometric)
    for key,weight in c['relation_pilot']['weights'].items():terms['total']=terms['total']+weight*terms[key]
    if not all(torch.isfinite(value).all() for value in terms.values()):raise FloatingPointError('Nonfinite relation update')
    terms['total'].backward()
    gradient={name:float(getattr(model,name).weight.grad.norm()) for name in
              ('region_condition','root_feedback','rotation_feedback','contact_readout','correspondence_readout')}
    norm=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
        c['optimizer']['clip_norm'],error_if_nonfinite=True)
    lr=c['optimizer']['lr']*min(1.,step/c['optimizer']['warmup_updates'])*(.1+.9*(1+math.cos(math.pi*step/c['training']['planned_updates']))/2)
    for group in optimizer.param_groups:group['lr']=lr
    optimizer.step()
    return dict(step=step,losses={k:float(v.detach()) for k,v in terms.items()},
        gradients=gradient,gradient_norm=float(norm),lr=lr,coverage=coverage,identities=[i for _,i in examples])


def save(model,optimizer,rng,device,step,seal,path):
    value=repair.checkpoint_state(model,optimizer,rng,device,step,seal)
    value['parent_scene_sha256']=seal['parent_scene_sha256']
    save_checkpoint(value,path)


def load(path,model,optimizer,rng,device,seal):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['contract']!=seal or state['parent_scene_sha256']!=seal['parent_scene_sha256']:
        raise ValueError('Relation resume protocol/parent changed')
    model.load_adapter_state(state['adapter']);optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'],rng,device)
    return state['step']


def preflight(model,data,validation,reference,c,device,seal):
    out=ROOT/c['output']/'preflight';out.mkdir(exist_ok=True)
    record=dict(state='running',device=device,contract_sha256=seal['sha256'],formal_training_started=False)
    atomic_json(out/f'{device}.json',record);started=time.monotonic()
    rng=np.random.default_rng(c['relation_pilot']['training_seed']);optimizer=engine.optimizer_for(model,c)
    members=engine.forced_members(data);frozen=repair.frozen_digest(model)
    temporary=out/f'.{device}.pt'
    try:
        examples=[data.sample(rng,member=m,yaw=0.) for m in members]
        batch=collate([s for s,_ in examples],device);payload=inference_payload(batch)
        noisy=torch.randn_like(batch['motion']);t=torch.full((len(examples),),100,device=device,dtype=torch.long)
        model.eval()
        with torch.no_grad():
            prediction,details=model(noisy,t,payload,return_details=True)
            rest_errors=[]
            for i,code in enumerate(batch['body_format'].tolist()):
                decoded=model.native_body.bank(FORMATS[code],batch['target_native_betas'][i:i+1],batch['target_template'][i:i+1])
                rest_errors.append(float((decoded['rest']-batch['target_rest'][i:i+1]).abs().max()))
            if max(rest_errors)>2e-5:raise AssertionError('Native supervision body/coordinate mismatch')
            poisoned=dict(payload,target_rest=batch['target_rest']*float('nan'),
                target_template=batch['target_template']+999,target_native_betas=batch['target_native_betas']*float('nan'))
            if not torch.equal(prediction,model(noisy,t,poisoned)):
                raise AssertionError('Target labels leaked into relation-model inference')
            terms,coverage=supervision(model,prediction,details,batch,c['relation_pilot']['relation'])
            empty=change_scene(payload,'empty')
            if not torch.equal(model(noisy,t,empty),model(noisy,t,payload,use_scene=False)):
                raise AssertionError('Empty scene does not exactly reproduce the frozen prior')
        # Rehydrating bases must preserve the selected parent's actual DDIM
        # outputs, not merely accept a state dictionary with matching shapes.
        reference_rows=[next(r for r in reference['rows'] if r['identity']['dataset']==name)
                        for name in ('egobody','trumans','grail')]
        parity=local.evaluate(model,validation,[r['identity'] for r in reference_rows],c,device)
        atomic_json(out/f'{device}_parent_ddim.json',parity)
        if device=='cuda':
            for actual,expected_row in zip(parity['rows'],reference_rows):
                if actual['identity']!=expected_row['identity']:raise AssertionError('Parent parity identity changed')
                check_parity(actual['metrics'],expected_row['metrics'])
        first=update(model,optimizer,data,rng,c,device,1,members)
        if not any(v>0 for v in first['coverage']['anchor_fraction']):raise AssertionError('No geometric supervision on real data')
        if not all(first['gradients'][name]>0 for name in ('contact_readout','correspondence_readout')):
            raise AssertionError('Weak geometric gate targets do not receive gradients')
        save(model,optimizer,rng,device,1,seal,temporary)
        expected=update(model,optimizer,data,rng,c,device,2,members)
        digest=tensor_digest((model.adapter_state(),optimizer.state_dict(),rng_state(rng,device)))
        load(temporary,model,optimizer,rng,device,seal)
        actual=update(model,optimizer,data,rng,c,device,2,members)
        if expected!=actual or digest!=tensor_digest((model.adapter_state(),optimizer.state_dict(),rng_state(rng,device))):
            raise AssertionError('Relation exact resume failed')
        if repair.frozen_digest(model)!=frozen:raise AssertionError('Frozen global/motion/shape changed')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise AssertionError('A frozen parameter received gradients')
        record.update(state='passed',target_poison_invariant=True,empty_equals_prior=True,exact_resume=True,
            exact_parent_state_identity=model._verified_parent_state,
            frozen_motion_shape_global_unchanged=True,initial_coverage=coverage,initial_update=first,
            target_native_rest_max_error_m=max(rest_errors),
            selected_parent_ddim_parity=(True if device=='cuda' else 'CPU sampling only; reference uses CUDA RNG'),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
            elapsed_seconds=time.monotonic()-started,completed_at_utc=now())
        atomic_json(out/f'{device}.json',record);return record
    except BaseException as error:
        record.update(state='failed',error=repr(error),updated_at_utc=now());atomic_json(out/f'{device}.json',record);raise
    finally:temporary.unlink(missing_ok=True)


def scene_difference(correct,other,target,policy):
    # Independent diagnostic: quantify actual local geometry change. It does
    # not filter or replace primary validation windows.
    a=observed_planes(target,correct,policy);b=observed_planes(target,other,policy)
    da=((target-a['centre'])*a['normal']).sum(-1).abs()
    db=((target-b['centre'])*b['normal']).sum(-1).abs()
    common=a['valid']&b['valid']
    x=correct['occupancy']>0;y=other['occupancy']>0
    union=(x|y).sum()
    return dict(occupancy_iou=float((x&y).sum()/union) if union else None,
        correct_planar_coverage=float(a['valid'].float().mean()),mismatched_planar_coverage=float(b['valid'].float().mean()),
        changed_observed_region_fraction=float((a['valid']^b['valid']).float().mean()),
        shared_plane_offset_difference_mm=float((da-db).abs()[common].mean()*1000) if common.any() else None)


def evaluate(model,validation,reference,c,device,seal):
    out=ROOT/c['output']/'evaluation';out.mkdir(exist_ok=True)
    protocol=json.loads((ROOT/'runs/scene_ego128_residual_checks/PROTOCOL.json').read_text())
    ids=seal['pilot_case_ids'];rows={name:[] for name in ('correct','empty','mismatched')};differences=[]
    model.eval();started=time.monotonic();saved=[]
    for index,i in enumerate(ids):
        identity=reference['rows'][i]['identity'];donor_id=protocol['donors'][i];donor_identity=reference['rows'][donor_id]['identity']
        def get(row):
            sample,actual=validation.sample(np.random.default_rng(c['validation']['seed']),
                member=row['member'],start=row['start'],yaw=0.)
            if actual['path']!=row['path'] or actual['start']!=row['start']:raise ValueError('Pilot identity changed')
            return collate([sample],device)
        batch=get(identity);other=get(donor_identity);payload=inference_payload(batch);donor=inference_payload(other)
        target_shape=dict(rest=batch['target_rest'],predicted_native_betas=batch['target_native_betas'],predicted_template=batch['target_template'])
        with torch.no_grad():
            skin,_=model.skin(batch['motion'][:,::4],target_shape,batch['body_format'])
        def compact_scene(x):return dict(x,local_frame_points=x['local_frame_points'][:,::4],local_frame_owner=x['local_frame_owner'][:,::4])
        geometric_difference=scene_difference(compact_scene(payload),compact_scene(donor),skin,c['relation_pilot']['relation'])
        predictions={}
        with torch.inference_mode():
            for variant in rows:
                altered=payload if variant=='correct' else change_scene(payload,variant,donor)
                motion=sample_native30(model,altered,steps=c['validation']['ddim_steps'],seed=c['validation']['seed'],use_scene=True)
                _,details=model(motion,torch.zeros(1,device=device,dtype=torch.long),altered,return_details=True)
                joints=forward_kinematics(motion,details['rest'])[0]
                truth=forward_kinematics(batch['motion'],batch['target_rest'])[0]
                rows[variant].append(dict(identity=identity,metrics=motion_metrics(joints,truth)))
                predictions[variant]=joints.cpu().numpy()
        delta=np.linalg.norm(predictions['correct']-predictions['mismatched'],axis=-1)*1000
        differences.append(dict(case_id=i,identity=identity,donor_identity=donor_identity,
            scene_difference=geometric_difference,prediction_change_mean_mm=float(delta.mean()),
            prediction_change_feet_mm=float(delta[:,[7,8,10,11]].mean()),prediction_change_hips_mm=float(delta[:,[0,1,2]].mean())))
        saved.append(predictions)
        atomic_json(out/'STATUS.json',dict(state='running',completed=index+1,total=len(ids),pid=os.getpid(),updated_at_utc=now()))
    result=dict(state='complete',contract_sha256=seal['sha256'],final_step=c['training']['planned_updates'],
        variants={name:dict(rows=value,**aggregate_scene_rows(value)) for name,value in rows.items()},
        parent_correct=dict(rows=[reference['rows'][i] for i in ids],**aggregate_scene_rows([reference['rows'][i] for i in ids])),
        sensitivity=differences,pilot_only=True,test_used=False,elapsed_seconds=time.monotonic()-started,
        interpretation='Predeclared 24-family validation pilot, exploratory continuation from the selected 3k scene adapter. Not a full benchmark, matched loss ablation, or independent test.',completed_at_utc=now())
    verify_sources(seal);atomic_json(out/'RESULT.json',result)
    np.savez_compressed(out/'paired_joints.npz',**{name:np.stack([r[name] for r in saved]) for name in rows})
    atomic_json(out/'STATUS.json',dict(state='complete',completed=len(ids),total=len(ids),updated_at_utc=now()))
    return result


def train(model,data,validation,reference,c,device,seal,resume):
    out=ROOT/c['output'];folder=out/'training';folder.mkdir(exist_ok=True)
    for kind in ('cpu','cuda'):
        check=json.loads((out/'preflight'/f'{kind}.json').read_text())
        if check['state']!='passed' or check['contract_sha256']!=seal['sha256']:raise ValueError('Matching preflights required')
    path=folder/'CONTRACT.json'
    if path.exists() and json.loads(path.read_text())!=seal:raise ValueError('Training seal changed')
    atomic_json(path,seal);last=folder/'last.pt'
    if last.exists() and not resume:raise ValueError('Use --resume for existing pilot')
    optimizer=engine.optimizer_for(model,c);rng=np.random.default_rng(c['relation_pilot']['training_seed'])
    step=load(last,model,optimizer,rng,device,seal) if resume else 0
    repair.reconcile_journal(folder,step);frozen=repair.frozen_digest(model);stop=[]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda signum,frame:stop.append(signum))
    started=time.monotonic();initial=step;counts=Counter()
    def status(state,**extra):
        value=dict(state=state,completed_updates=step,planned_updates=c['training']['planned_updates'],
            pid=os.getpid(),contract_sha256=seal['sha256'],datasets=dict(counts),
            updates_per_second=(step-initial)/max(time.monotonic()-started,1e-6),updated_at_utc=now(),**extra)
        atomic_json(folder/'STATUS.json',value);return value
    try:
        status('training')
        with (folder/'updates.jsonl').open('a') as stream:
            while step<c['training']['planned_updates'] and not stop:
                step+=1;record=update(model,optimizer,data,rng,c,device,step)
                counts.update(i['dataset'] for i in record['identities'])
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                if step%10==0:status('training',latest_loss=record['losses']['total'],latest_coverage=record['coverage'])
                if step%c['training']['checkpoint_every']==0:save(model,optimizer,rng,device,step,seal,last)
        save(model,optimizer,rng,device,step,seal,last)
        if repair.frozen_digest(model)!=frozen:raise AssertionError('Frozen parent changed during pilot')
        if stop:return status('stopped')
        status('evaluating_pilot');evaluate(model,validation,reference,c,device,seal)
        return status('complete')
    except BaseException as error:status('failed',error=repr(error));raise


def bounded():
    from scenemi import parallel_resources
    parallel_resources.MODULES |= {'scenemi.scene_followup','scenemi.scene_checks','scenemi.scene_relations'}
    sys.argv=[__file__]+sys.argv[2:]
    parallel_resources.main()


def main():
    if len(sys.argv)>1 and sys.argv[1]=='bounded':return bounded()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('preflight','train'))
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/scene_relations.yaml')
    parser.add_argument('--paths',type=Path,default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--scene-paths',type=Path,default=ROOT/'configs/scene_paths.local.yaml')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();ensure_bounded()
    if args.device=='cuda':configure_cuda()
    c=settings(args.config);deterministic(c['seed'],args.device)
    model,data,validation,reference,seal=prepare(c,load_paths(args.paths),args.scene_paths,args.config)
    model=model.to(args.device)
    result=(preflight(model,data,validation,reference,c,args.device,seal) if args.action=='preflight'
            else train(model,data,validation,reference,c,args.device,seal,args.resume))
    print(json.dumps({k:v for k,v in result.items() if k!='initial_update'},indent=2),flush=True)


if __name__=='__main__':main()
