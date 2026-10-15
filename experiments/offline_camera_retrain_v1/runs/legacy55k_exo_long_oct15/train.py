"""Background continuation of the retained EXO residual; absolute cap55000.

Preserves parent1000 model/Adam/RNG; explicitly changes temporal inference and
supervision protocol. Samples official TRAIN into a renewing bounded replay
pool. Full pool, sampler and RNG are stored in one atomic resumable last.pt.
"""
import argparse
from collections import Counter
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parents[3]))
import numpy as np
import torch
from temporal_guard import SmoothResidual,smooth_reference,alternation_loss,PILOT
from stream_data import StreamData
import run_probe as common
from experiments.offline_camera_retrain_v1.prepare_native30 import code_hashes,write_json,sha256,window_options
from experiments.offline_camera_retrain_v1.native30_sampling import StratifiedRecordingSampler
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.legacy55k_exo_residual import frozen_initial_pose,scene_features,residual_inputs,correction_payload
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.native30_gt_reliability import derivative_valid

PARENT=PILOT/'reference_anchor/prior_scene/last.pt'
CAP=55000
SEED=2026101556
STOP=False


def stop(signum,frame):
    global STOP
    STOP=True


def sources():
    result=code_hashes()
    paths=[ROOT/'train.py',ROOT/'temporal_guard.py',ROOT/'stream_data.py',
        PILOT/'run_probe.py',PILOT/'run_reference_probe.py']
    result.update({str(p.resolve()):sha256(p) for p in paths})
    return result


def model_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def dataset(config,split):
    return StreamData(common.HERE/config['train_manifest'],split,SEED,frames=192,
        memory_protocol=config['scene']['memory_protocol'],quality_config=config['gt_reliability'],
        exo_max_empty_frames=config['exo_max_empty_frames'],read_cache=ROOT/'verified_read_cache',**window_options(config))


@torch.no_grad()
def prepare(data,builder,prior,rows,rng,validation=False):
    items=[];identities=[]
    for row in rows:
        seed=int(row['seed']) if validation else int(rng.integers(0,2**32-1))
        regime=row['regime'] if validation else ('noisy' if rng.random()<.5 else 'clean')
        sample,identity=data.sample(row['group'],'exo',member=row['member'],start=row['start'],yaw=.6 if validation else None)
        sample,scene,asset=builder.build(data,sample,identity,'exo',noise_rng=np.random.default_rng(seed) if regime=='noisy' else None)
        items.append((sample,scene,asset,'exo'))
        identities.append(dict(group=row['group'],member=row['member'],start=row['start'],regime=regime,seed=seed,
            sequence_id=identity['sequence_id'],recording=identity.get('recording',identity['sequence_id']),
            source_start_30fps=identity['source_start_30fps'],yaw=identity['yaw_augmentation_rad'],
            panel=row.get('panel','train'),unseen_recording=row.get('unseen_recording'),
            sampling={k:row[k] for k in ('recording_key','cooldown_fallback','overlap_fallback') if k in row}))
    batch=common.collate_observations(items)
    prior_seed=int(rows[0]['seed'])-int(rows[0]['regime']=='noisy') if validation else int(rng.integers(0,2**32-1))
    legacy,reference=frozen_initial_pose(prior,batch,prior_seed)
    initial=smooth_reference(reference,batch['rest'])
    joints=forward_kinematics(initial,batch['rest']).cpu().numpy()
    geometry=torch.from_numpy(np.stack([scene_features(item[1],j) for item,j in zip(items,joints)])).cuda()
    known=residual_inputs(initial,batch,geometry)
    known['features']=torch.cat((known['features'],legacy-initial),-1)
    keep={k:batch[k] for k in ('motion','rest','trajectory','observation_meta','gt_position_weight','gt_local_rotation_weight','gt_global_rotation_weight')}
    natural=[]
    for row,identity in zip(rows,identities):
        meta,_=data.members[row['group']][row['member']]
        natural.append(~np.asarray(data.load(meta['path'],'exo_position_available')[row['start']:row['start']+192],bool))
    keep.update(known,natural_missing=torch.from_numpy(np.stack(natural)).cuda())
    output=[]
    for i,identity in enumerate(identities):
        identity.update(prior_batch_seed=prior_seed,prior_batch_offset=i,prior_batch_size=len(items))
        output.append(({k:v[i:i+1].detach().cpu().clone() for k,v in keep.items()},identity))
    return output


def validation_plan(data):
    original=json.loads((PILOT/'IDENTITIES.json').read_text())['validation']
    plan=[dict(r,panel='fixed',unseen_recording=False) for r in original]
    seen={('egobody' if r['group'] in ('camera_wearer','interactee') else r['group'],r['recording']) for r in original}
    report={}
    for group,members in data.members.items():
        pool=[]
        for member,(meta,_) in enumerate(members):
            starts=data.starts_for(group,member,'exo')
            if not len(starts):continue
            start=int(starts[len(starts)//2]);record=meta.get('recording',meta['sequence_id'])
            domain='egobody' if group in ('camera_wearer','interactee') else group
            if any(r['sequence_id']==meta['sequence_id'] and abs(r['start']-start)<192 for r in original):continue
            pool.append(dict(group=group,member=member,start=start,sequence_id=meta['sequence_id'],recording=record,
                unseen_recording=(domain,record) not in seen))
        unseen=[r for r in pool if r['unseen_recording']]
        eligible=sorted(unseen or pool,key=lambda r:r['sequence_id'])
        picks=[eligible[i] for i in np.linspace(0,len(eligible)-1,min(2,len(eligible)),dtype=int)] if eligible else []
        report[group]=dict(eligible_sources=len(pool),unseen_recording_sources=len(unseen),selected=picks)
        for index,row in enumerate(picks):
            for regime in ('clean','noisy'):
                plan.append(dict(row,panel='confirmation',regime=regime,
                    seed=SEED+100000+len(plan)*2+int(regime=='noisy')))
    write_json(ROOT/'VALIDATION_PLAN.json',dict(cases=plan,confirmation_coverage=report,
        rule='Old fixed panel retained; new central windows prefer unseen recordings, else nonoverlapping source intervals; no prediction/GT-pose selection'))
    return plan


def stack(pool,ids,device='cuda'):
    return {k:torch.cat([pool[i][0][k] for i in ids]).to(device) for k in pool[0][0]}


def objective(pred,batch,details):
    loss,terms=common.fit_loss(pred,batch,details)
    phase=alternation_loss(pred,batch['motion'],batch['rest'],batch['gt_position_weight'])
    return loss+.05*phase,dict(terms,foot_alternation=phase)


@torch.no_grad()
def validate(model,bank,step,scene=True,prior_on=True):
    model.eval();rows=[];outputs=[]
    for i,(_,identity) in enumerate(bank):
        batch=stack(bank,[i]);p,_=model(correction_payload(batch),use_scene=scene,use_prior=prior_on)
        m=common.metrics(p,batch)
        j=forward_kinematics(p,batch['rest']);t=forward_kinematics(batch['motion'],batch['rest'])
        a=j.diff(n=2,dim=1)*900;ta=t.diff(n=2,dim=1)*900
        valid=derivative_valid(batch['gt_position_weight'],2)>0
        m['acceleration_RMS_m_s2']=float(a.norm(dim=-1)[valid].square().mean().sqrt()) if valid.any() else None
        m['acceleration_error_RMS_m_s2']=float((a-ta).norm(dim=-1)[valid].square().mean().sqrt()) if valid.any() else None
        rows.append(dict(identity,metrics=m));outputs.append(p.cpu())
    aggregate={panel:common.aggregate([r for r in rows if r['panel']==panel]) for panel in sorted({r['panel'] for r in rows})}
    return dict(step=step,scene=scene,prior=prior_on,cases=rows,aggregate=aggregate),torch.cat(outputs)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.28)
    torch.manual_seed(SEED);np.random.seed(SEED)
    audit=json.loads((ROOT/'DATA_AUDIT.json').read_text());assert audit['status']=='passed'
    own=ROOT/'last.pt';existing=own.exists()
    if existing and not args.resume:raise RuntimeError('Existing last.pt requires explicit --resume')
    state=torch.load(own if existing else PARENT,map_location='cpu',weights_only=False)
    assert 'optimizer' in state and not state.get('evaluation_only',False)
    if existing:
        protocol=json.loads((ROOT/'PROTOCOL.json').read_text());assert sources()==protocol['code_sha256']
    else:
        assert state['step']==1000
        protocol=dict(status='preparing',created_at=datetime.datetime.now().astimezone().isoformat(),seed=SEED,
            pid=os.getpid(),
            parent_checkpoint=str(PARENT),parent_checkpoint_sha256=sha256(PARENT),parent_model_sha256=model_digest(state['model']),
            initialization='Retained residual1000 weights+Adam+Torch/CUDA/NumPy RNG; frozen legacy55k external prior unchanged',
            base_legacy_checkpoint=str(common.OLD),base_legacy_checkpoint_sha256=sha256(common.OLD),
            optimizer_reset=False,stage_changed=True,start_step=1000,max_step=CAP,batch_size=8,fps=30,frames=192,
            lr='Continuous2e-4 at step1001; cosine over remaining54000 updates to2e-5 at55000',
            model_parameters=1150341,precision='FP32',reference='observation-only5-frame intrinsic SO3/root Gaussian sigma0.8frames',
            residual='7-frame sigma1 tangent filter;4-frame cosine permission interior ramp; same protected output in train/eval',
            objective='Original base+30Hz gait+2 hiddenFK+2 observedFK+.001 residual; additional0.05 dedicated body-local left/right foot relation',
            targets_smoothed=False,GT_in_inference=False,RGB_frontend='ideal visible positions only; known native shape',
            legacy_TEST_overlap_recordings=12,scope='Legacy-prior development continuation, not clean official paper benchmark',
            replay=dict(initial_windows=256,max_windows=512,refresh_windows=8,refresh_every_updates=20,
                source='All eligible official TRAIN EXO windows via recording-cycle sampler; rolling CPU replay;50% clean/noisy, random yaw',
                frozen_feature_cache='Per-window ref/prior/scene computed once; whole active pool saved in atomic checkpoint'),
            checkpoint='One atomic last.pt every500 steps and graceful stop; weights/Adam/pool/sampler/all RNG; no best.pt',
            validation='Fixed plus additional confirmation source windows every1000; scene/prior inputOFF every5000; explicit recording overlap scope',
            code_sha256=sources(),data_audit_sha256=sha256(ROOT/'DATA_AUDIT.json'))
        write_json(ROOT/'PROTOCOL.json',protocol)
    write_json(ROOT/'STATUS.json',dict(status='preparing',pid=os.getpid(),latest_update_step=int(state['step']),
        checkpoint_step=int(state['step']),own_checkpoint_exists=existing,max_step=CAP,
        observed_at=datetime.datetime.now().astimezone().isoformat()))
    config=common.read_config(common.HERE/'native30_fresh_ego_config.json')
    data=dataset(config,'train');valdata=dataset(config,'validation')
    # Confirm current source metadata/manifest fingerprints still match the
    # just-completed full original-array audit before any update.
    for path,digest in audit['data_sha256'].items():assert sha256(path)==digest,('Data metadata drift',path)
    options=copy.deepcopy(config['sampler']);options['training_mode']='exo'
    plan=[r for r in data.sampling_plan(.5) if r['mode']=='exo'];total=sum(r['probability'] for r in plan)
    for row in plan:row['probability']/=total
    data.training_sampler=StratifiedRecordingSampler(data,plan,options,SEED+371)
    builder=common.Native30ObservationBuilder(common.Native30SkinAnchorCache(),json.loads((common.HERE/config['mount_defaults']).read_text()),config['observation']['input_noise'])
    prior=OfflineSceneMI(256,(1,2,4),body_conditioning=True,contact_prediction=True).cuda().eval()
    legacy=torch.load(common.OLD,map_location='cpu',weights_only=False,mmap=True);assert legacy['step']==55000
    prior.load_state_dict(legacy['model'],strict=True);del legacy
    for p in prior.parameters():p.requires_grad_(False)
    model=SmoothResidual().cuda();model.load_state_dict(state['model'],strict=True)
    assert model_digest(model.state_dict())==(model_digest(state['model']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01);optimizer.load_state_dict(state['optimizer'])
    sample_rng=np.random.default_rng(SEED+71);replay_rng=np.random.default_rng(SEED+72)
    if existing:
        pool=state['pool'];cursor=state['pool_cursor'];data.load_sampler_state(state['data_sampler_state'])
        sample_rng.bit_generator.state=state['sample_rng'];replay_rng.bit_generator.state=state['replay_rng']
        bank=torch.load(ROOT/'validation_inputs.pt',map_location='cpu',weights_only=False)
        assert sha256(ROOT/'validation_inputs.pt')==state['validation_inputs_sha256']
    else:
        pool=[];cursor=0
        for _ in range(32):
            rows=[data.training_sampler.draw() for _ in range(8)]
            pool.extend(prepare(data,builder,prior,rows,sample_rng))
            print('prepared TRAIN pool',len(pool),flush=True)
            write_json(ROOT/'STATUS.json',dict(status='preparing_TRAIN_pool',pid=os.getpid(),latest_update_step=1000,
                checkpoint_step=1000,own_checkpoint_exists=False,max_step=CAP,prepared_train_windows=len(pool),
                planned_initial_pool=256,observed_at=datetime.datetime.now().astimezone().isoformat()))
        vplan=validation_plan(valdata);bank=[]
        write_json(ROOT/'STATUS.json',dict(status='preparing_VAL',pid=os.getpid(),latest_update_step=1000,
            checkpoint_step=1000,own_checkpoint_exists=False,max_step=CAP,prepared_train_windows=len(pool),
            planned_validation_windows=len(vplan),observed_at=datetime.datetime.now().astimezone().isoformat()))
        for row in vplan:
            bank.extend(prepare(valdata,builder,prior,[row],sample_rng,validation=True))
        save_checkpoint(bank,ROOT/'validation_inputs.pt')
        preflight,_=validate(model,bank,1000)
        write_json(ROOT/'PREFLIGHT_VALIDATION.json',preflight)
        old=json.loads((PILOT/'reference_anchor/prior_scene/GUARDED_VALIDATION.json').read_text())['aggregate']
        checks={}
        for regime,current in preflight['aggregate']['fixed'].items():
            checks[regime]=dict(visible=current['visible_mm']<=old[regime]['visible_mm']+5,
                occluded=current['occluded_mm']<=old[regime]['occluded_mm']*1.15,
                foot_velocity=current['foot_velocity_rmse_m_s']<=old[regime]['foot_velocity_rmse_m_s']*1.15,
                acceleration=current['acceleration_RMS_m_s2']<35.)
        write_json(ROOT/'PREFLIGHT_GATES.json',dict(checks=checks,scope='Parent1000 before new optimizer updates; changed reference and protected temporal inference'))
        if not all(all(v.values()) for v in checks.values()):raise RuntimeError('Temporal preflight quality gates failed before optimizer updates')
        if not all(torch.isfinite(v).all() for item,_ in pool for v in item.values() if v.is_floating_point()):raise RuntimeError('Nonfinite prepared TRAIN feature')
        write_json(ROOT/'INITIALIZATION_AUDIT.json',dict(status='passed',source_step=1000,
            parent_weights_equal=True,optimizer_reset=False,optimizer_steps=sorted({float(v['step']) for v in optimizer.state.values()}),
            TRAIN_pool_windows=len(pool),validation_feature_windows=len(bank),
            eligible_TRAIN_exo_windows=sum(len(data.starts_for(g,i,'exo')) for g,m in data.members.items() for i in range(len(m))),
            targets_untouched=True,inference_GT_whitelist=True))
    del valdata
    # Restore parent learning RNG after feature/model construction. New data
    # and replay streams are explicitly stage-specific, then checkpointed.
    torch.set_rng_state(state['torch_rng_state']);torch.cuda.set_rng_state_all(state['cuda_rng_state'])
    np.random.set_state(state['numpy_rng_state'])
    step=int(state['step']);checkpoint_step=step;fresh_draws=data.training_sampler.draws
    completed_stage_before=state.get('stage_elapsed_s',0.);del state
    val_hash=sha256(ROOT/'validation_inputs.pt')
    if existing and (ROOT/'training_log.jsonl').exists():
        path=ROOT/'training_log.jsonl';lines=path.read_text().splitlines();keep=[line for line in lines if json.loads(line)['step']<=step]
        if len(keep)!=len(lines):
            path.rename(ROOT/('interrupted_trace_'+str(time.time_ns())+'.jsonl'));path.write_text('\n'.join(keep)+'\n')
    began=time.monotonic();recent=[];last_validation=None
    def checkpoint():
        nonlocal checkpoint_step
        if sources()!=protocol['code_sha256']:raise RuntimeError('Sealed training source changed')
        save_checkpoint(dict(step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),config=protocol,
            torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),numpy_rng_state=np.random.get_state(),
            pool=pool,pool_cursor=cursor,data_sampler_state=data.sampler_state(),sample_rng=sample_rng.bit_generator.state,
            replay_rng=replay_rng.bit_generator.state,validation_inputs_sha256=val_hash,
            stage_elapsed_s=completed_stage_before+time.monotonic()-began),own)
        checkpoint_step=step
    def status(phase):
        rate=float(np.mean(recent[-200:])) if recent else None
        write_json(ROOT/'STATUS.json',dict(status=phase,pid=os.getpid(),observed_at=datetime.datetime.now().astimezone().isoformat(),
            latest_update_step=step,checkpoint_step=checkpoint_step,max_step=CAP,remaining_updates=CAP-step,
            seconds_per_update_including_refresh=rate,estimated_remaining_hours=(CAP-step)*rate/3600 if rate else None,
            train_pool_windows=len(pool),fresh_TRAIN_draws=data.training_sampler.draws,latest_validation=last_validation,
            Linux_free_GiB=shutil.disk_usage(ROOT).free/2**30))
    checkpoint();protocol.update(status='training',started_at=datetime.datetime.now().astimezone().isoformat())
    write_json(ROOT/'PROTOCOL.json',protocol);status('training')
    try:
        while step<CAP and not STOP:
            tick=time.monotonic()
            if (step-1000)%20==0:
                rows=[data.training_sampler.draw() for _ in range(8)]
                fresh=prepare(data,builder,prior,rows,sample_rng)
                for item in fresh:
                    if len(pool)<512:pool.append(item)
                    else:pool[cursor]=item;cursor=(cursor+1)%512
            ids=replay_rng.choice(len(pool),8,replace=False).tolist();batch=stack(pool,ids)
            torch.cuda.synchronize();compute=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True)
            next_step=step+1;progress=(next_step-1001)/(CAP-1001)
            lr=2e-5+(2e-4-2e-5)*.5*(1+math.cos(math.pi*progress))
            for group in optimizer.param_groups:group['lr']=lr
            pred,details=model(correction_payload(batch));loss,terms=objective(pred,batch,details)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            torch.cuda.synchronize();step=next_step
            elapsed=time.monotonic()-tick;recent.append(elapsed)
            if len(recent)>200:recent=recent[-200:]
            row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(grad),lr=lr,loop_seconds=elapsed,
                compute_seconds=time.monotonic()-compute,pool_indices=ids,
                samples=[pool[i][1] for i in ids],**{k:float(v.detach()) for k,v in terms.items()})
            with (ROOT/'training_log.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            if step<=1005 or step%100==0:print('TRAIN',step,row['loss'],'loop_s',elapsed,'fresh',data.training_sampler.draws,flush=True)
            if step%500==0:checkpoint()
            if step%1000==0:
                status('validating');report,output=validate(model,bank,step)
                write_json(ROOT/'VALIDATION_LATEST.json',report)
                with (ROOT/'validation_log.jsonl').open('a') as stream:stream.write(json.dumps(dict(step=step,aggregate=report['aggregate']))+'\n')
                save_checkpoint(dict(step=step,prediction=output,identities=[i for _,i in bank]),ROOT/'validation_latest.pt')
                last_validation=dict(step=step,aggregate=report['aggregate'])
                if step%5000==0:
                    for tag,scene_on,prior_on in (('scene_OFF',False,True),('prior_OFF',True,False)):
                        ablation,_=validate(model,bank,step,scene=scene_on,prior_on=prior_on)
                        write_json(ROOT/(tag+'_LATEST.json'),ablation)
                        with (ROOT/(tag+'_log.jsonl')).open('a') as stream:stream.write(json.dumps(dict(step=step,aggregate=ablation['aggregate']))+'\n')
                print('VALIDATION',step,json.dumps(report['aggregate']),flush=True)
            if step<=1005 or step%20==0:status('training')
            if shutil.disk_usage(ROOT).free<12*2**30:raise RuntimeError('Space guard: less than12GiB Linux free; checkpoint before stopping')
        checkpoint();phase='completed' if step==CAP else 'stopped_at_update_boundary'
        status(phase);protocol.update(status=phase,last_step=step);write_json(ROOT/'PROTOCOL.json',protocol)
    except BaseException as exc:
        try:checkpoint()
        finally:
            status('failed_saved');write_json(ROOT/'ERROR.json',dict(step=step,error=repr(exc),saved_step=checkpoint_step))
        raise


if __name__=='__main__':main()
