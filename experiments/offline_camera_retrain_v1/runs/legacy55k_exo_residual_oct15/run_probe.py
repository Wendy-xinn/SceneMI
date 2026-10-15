"""Bounded 55k-prior EXO residual pilot with matched scene/no-scene arms."""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config,window_options,write_json,code_hashes,sha256
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.legacy55k_exo_residual import ExoResidual,frozen_initial_pose,scene_features,residual_inputs,correction_payload
from experiments.offline_camera_retrain_v1.native30_losses import base_losses
from experiments.offline_camera_retrain_v1.native30_gait_supervision import DEFAULT
from experiments.offline_camera_retrain_v1.native30_gt_reliability import derivative_valid
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,rotation_from_6d
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint

ROOT=Path(__file__).resolve().parent
OLD=HERE/'runs/native_dynamic_scene20_contact_55k_oct07/last.pt'
SEED=2026101555
UPDATES=1000
BATCH=8
LR=2e-4


def dataset(config,split):
    return Native30Data(HERE/config['train_manifest'],split,SEED,frames=192,
        memory_protocol=config['scene']['memory_protocol'],quality_config=config['gt_reliability'],
        exo_max_empty_frames=config['exo_max_empty_frames'],**window_options(config))


def select_windows(data,per_group):
    """One central legal window per bundle, stratify by INPUT missing fraction.

    Select quantiles before generating any predictions. No pose/error-based
    cherry-picking; every represented group gets the same number of windows.
    """
    selected=[];screen=[]
    for group,members in data.members.items():
        pool=[]
        for member,(meta,_) in enumerate(members):
            starts=data.starts_for(group,member,'exo')
            if not len(starts):continue
            start=int(starts[len(starts)//2])
            available=np.asarray(data.load(meta['path'],'exo_position_available')[start:start+192],bool)
            fraction=float(1-available.mean())
            pool.append(dict(group=group,member=member,start=start,sequence_id=meta['sequence_id'],
                recording=meta.get('recording',meta['sequence_id']),missing_fraction=fraction))
        pool.sort(key=lambda x:(x['missing_fraction'],x['sequence_id']))
        eligible=[r for r in pool if .01<=r['missing_fraction']<=.95]
        if not eligible:eligible=pool
        ids=np.linspace(0,len(eligible)-1,min(per_group,len(eligible)),dtype=int) if eligible else []
        selected.extend(eligible[i] for i in ids);screen.extend(pool)
    if not selected:raise ValueError('No eligible EXO sources')
    return selected,screen


@torch.no_grad()
def make_bank(data,builder,plan,prior,validation=False):
    rows=[];identities=[]
    for index,row in enumerate(plan):
        regimes=('clean','noisy') if validation else (('noisy',) if index%2 else ('clean',))
        for regime in regimes:
            seed=SEED+(100000 if validation else 0)+index*100+int(regime=='noisy')
            sample,identity=data.sample(row['group'],'exo',member=row['member'],start=row['start'],yaw=.6)
            natural_missing=~np.asarray(data.load(identity['path'],'exo_position_available')[row['start']:row['start']+192],bool)
            prepared,scene,asset=builder.build(data,sample,identity,'exo',noise_rng=np.random.default_rng(seed) if regime=='noisy' else None)
            batch=collate_observations([(prepared,scene,asset,'exo')])
            # Pair clean/noisy diffusion noise, and freeze the same prior for all arms.
            initial,reference=frozen_initial_pose(prior,batch,seed-int(regime=='noisy'))
            joint=forward_kinematics(initial,batch['rest'])
            geometry=torch.from_numpy(scene_features(scene,joint[0].cpu().numpy()))[None].cuda()
            known=residual_inputs(initial,batch,geometry)
            keep={k:batch[k] for k in ('motion','rest','trajectory','observation_meta',
                'gt_position_weight','gt_local_rotation_weight','gt_global_rotation_weight')}
            keep.update(known,reference=reference,natural_missing=torch.from_numpy(natural_missing)[None].cuda())
            rows.append({k:v.detach().cpu() for k,v in keep.items()})
            identities.append(dict(row,regime=regime,seed=seed,source_start_30fps=identity['source_start_30fps'],
                input_missing_fraction=float((batch['observation_meta'][...,0]==0).float().mean())))
            print('prepared',data.split,len(rows),row['group'],regime,row['sequence_id'],flush=True)
    bank={k:torch.cat([r[k] for r in rows]).cuda() for k in rows[0]}
    return bank,identities


def subset(bank,ids):return {k:v[ids] for k,v in bank.items()}


def fit_loss(prediction,batch,details):
    signal=prediction.new_ones(len(prediction))
    terms=base_losses(prediction,batch['motion'],batch,signal,gait_config=DEFAULT)
    p=forward_kinematics(prediction,batch['rest']);t=forward_kinematics(batch['motion'],batch['rest']).detach()
    missing=batch['natural_missing'];quality=batch['gt_position_weight']
    error=(p-t).square().sum(-1)
    hidden=(error*quality*missing).sum()/missing.sum().clamp_min(1)
    cp=batch['observation_meta'][...,0]*batch['observation_meta'][...,2]
    obs=((p-batch['trajectory'][...,:3]*2).square().sum(-1)*cp).sum()/cp.gt(0).sum().clamp_min(1)
    regularizer=details['rotation'].square().mean()+details['translation'].square().mean()
    loss=terms['total']+2*hidden+2*obs+.001*regularizer
    return loss,dict(base=terms['total'],hidden=hidden,observation=obs)


@torch.no_grad()
def metrics(prediction,batch):
    p=forward_kinematics(prediction,batch['rest']);t=forward_kinematics(batch['motion'],batch['rest'])
    error=(p-t).norm(dim=-1)*1000
    missing=batch['natural_missing'];input_missing=batch['observation_meta'][...,0]==0
    good=batch['gt_position_weight']>=1
    result={}
    for name,mask in dict(all=torch.ones_like(missing),occluded=missing,visible=~missing,
                         input_missing=input_missing,reliable_occluded=missing&good).items():
        n=int(mask.sum());result[name+'_count']=n
        result[name+'_mm']=float(error[mask].mean()) if n else None
    result['feet_mm']=float(error[:,:,(7,8,10,11)].mean())
    v=p.diff(dim=1)*30;tv=t.diff(dim=1)*30
    valid=derivative_valid(batch['gt_position_weight'])[:,:,(10,11)]>0
    stance=(tv[:,:,(10,11)].norm(dim=-1)<.2)&valid
    result['foot_velocity_rmse_m_s']=float(((v[:,:,(10,11)]-tv[:,:,(10,11)]).square().sum(-1)[valid]).mean().sqrt()) if valid.any() else None
    result['stance_speed_m_s']=float(v[:,:,(10,11)].norm(dim=-1)[stance].mean()) if stance.any() else None
    # Body-local left/right Z correlation checks alternation, not amplitude alone.
    local=lambda m,j:torch.einsum('btji,btkj->btki',rotation_from_6d(m[...,3:9]),j-j[:,:,:1])[:,:,(10,11),2]
    def corr(z):
        z=z-z.mean(1,keepdim=True);den=(z.square().sum(1).prod(-1)).sqrt()
        return ((z[:,:,0]*z[:,:,1]).sum(1)/den.clamp_min(1e-8)).mean()
    result['foot_alternation_correlation_error']=float((corr(local(prediction,p))-corr(local(batch['motion'],t))).abs())
    return result


def aggregate(rows):
    output={}
    for regime in ('clean','noisy'):
        selected=[r['metrics'] for r in rows if r['regime']==regime]
        output[regime]={k:float(np.mean([r[k] for r in selected if r[k] is not None]))
            for k in selected[0] if not k.endswith('_count') and any(r[k] is not None for r in selected)}
    return output


@torch.no_grad()
def evaluate(model,bank,identities,use_scene,tag):
    model.eval();rows=[];outputs=[]
    for i,identity in enumerate(identities):
        batch=subset(bank,slice(i,i+1));prediction,_=model(correction_payload(batch),use_scene=use_scene)
        rows.append(dict(identity,tag=tag,metrics=metrics(prediction,batch)))
        outputs.append(prediction.cpu())
    return dict(cases=rows,aggregate=aggregate(rows)),torch.cat(outputs)


def main():
    assert not (ROOT/'PROTOCOL.json').exists(),'Do not overwrite an existing trial; resume explicitly'
    audit=json.loads((ROOT/'DATA_AUDIT.json').read_text());assert audit['status']=='passed'
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.28)
    torch.manual_seed(SEED);np.random.seed(SEED)
    config=read_config(HERE/'native30_fresh_ego_config.json')
    source_hash=sha256(OLD)
    protocol=dict(status='preparing',created_at=datetime.datetime.now().astimezone().isoformat(),
        seed=SEED,steps_per_arm=UPDATES,batch_size=BATCH,lr=LR,precision='FP32',fps=30,frames=192,
        initialization='frozen legacy55k absolute-x0 prior + zero-initialized temporal SO3 EXO residual',
        legacy_checkpoint=str(OLD),legacy_checkpoint_sha256=source_hash,legacy_step=55000,
        old55k_official_TEST_overlap_recordings=12,scope='legacy-prior development only; not clean official benchmark',
        input_scope='ideal GT-visible-joint tracker simulation; known native shape; calibrated/synthetic cameras; observed surface only',
        reference='observed-position bone fitting, no hidden GT',
        prior='frozen old55k low-noise absolute-x0 projection, t99/66/33/0; no sparse control/no scene; observed-position root translation fit',
        clock_transfer='20Hz-trained frozen prior applied at30Hz; this is transfer, not identical old55k inference protocol',
        scene='one cached static and current dynamic nearest observed patch per initial joint within1.5m; no iterative CPU query or IK',
        arms=['no_scene','scene'],paired='same zero residual initialization, TRAIN minibatch indices, optimizer schedule and frozen initial predictions',
        train_windows_per_group=32,validation_windows_per_group=4,
        validation_selection='central legal EXO window per bundle, input-occlusion quantiles; selected before predictions',
        objectives='native30 base + balanced stance/swing, relative feet, .2/.4s stride and coordination; 2x occluded FK MSE +2x observed FK MSE +.001 residual penalty',
        gate=dict(min_relative_occluded_improvement=.05,min_scene_extra_relative_improvement=.02,max_visible_regression_mm=5.,
            max_foot_velocity_relative_regression=.10),
        budget='1000 updates per arm; no automatic extension; one resumable atomic last.pt per arm',
        max_training_seconds_per_arm=900,code_sha256=code_hashes(),data_config=config,
        runner_sha256=sha256(__file__),data_audit_sha256=sha256(ROOT/'DATA_AUDIT.json'))
    write_json(ROOT/'PROTOCOL.json',protocol)
    prior=OfflineSceneMI(256,(1,2,4),body_conditioning=True,contact_prediction=True).cuda()
    old=torch.load(OLD,map_location='cpu',weights_only=False,mmap=True)
    assert old['step']==55000;prior.load_state_dict(old['model'],strict=True)
    for param in prior.parameters():param.requires_grad_(False)
    del old
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),json.loads((HERE/config['mount_defaults']).read_text()),config['observation']['input_noise'])
    plans={};banks={};identities={}
    for split,count in (('train',32),('validation',4)):
        data=dataset(config,split);plan,screen=select_windows(data,count)
        write_json(ROOT/(split+'_PLAN.json'),dict(selected=plan,screened=screen,selection_before_prediction=True))
        banks[split],identities[split]=make_bank(data,builder,plan,prior,validation=split=='validation')
        plans[split]=plan
        del data
    del prior;torch.cuda.empty_cache()
    # Local compact replay inputs include labels separately; model interface only
    # reads initial/features/scene/rest. Generated tensors are never Git assets.
    save_checkpoint({s:{k:v.cpu() for k,v in b.items()} for s,b in banks.items()},ROOT/'replay_inputs.pt')
    write_json(ROOT/'IDENTITIES.json',identities)
    base={}
    for tag in ('reference','initial'):
        rows=[dict(identity,metrics=metrics(banks['validation'][tag][i:i+1],subset(banks['validation'],slice(i,i+1))))
              for i,identity in enumerate(identities['validation'])]
        base[tag]=dict(cases=rows,aggregate=aggregate(rows))
    write_json(ROOT/'BASELINE.json',base)
    torch.manual_seed(SEED)
    template=ExoResidual().cuda();initial_state=copy.deepcopy(template.state_dict());del template
    draws=np.random.default_rng(SEED).integers(0,len(identities['train']),size=(UPDATES,BATCH))
    write_json(ROOT/'BATCH_PLAN.json',dict(indices=draws.tolist(),seed=SEED))
    protocol.update(status='training',training_started_at=datetime.datetime.now().astimezone().isoformat(),
        actual_train_examples=len(identities['train']),actual_validation_examples=len(identities['validation']))
    write_json(ROOT/'PROTOCOL.json',protocol)
    reports={};predictions={};step_counts={}
    for arm in ('no_scene','scene'):
        folder=ROOT/arm;folder.mkdir(exist_ok=True)
        model=ExoResidual().cuda();model.load_state_dict(initial_state,strict=True)
        opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.01)
        began=time.monotonic();trace=[]
        for step,ids in enumerate(draws,1):
            torch.cuda.synchronize();tick=time.monotonic();model.train()
            batch=subset(banks['train'],ids);opt.zero_grad(set_to_none=True)
            pred,details=model(correction_payload(batch),use_scene=arm=='scene')
            loss,terms=fit_loss(pred,batch,details)
            assert torch.isfinite(loss),'Nonfinite loss'
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();torch.cuda.synchronize()
            row=dict(step=step,loss=float(loss.detach()),grad=float(grad),seconds=time.monotonic()-tick,
                sample_indices=ids.tolist(),**{k:float(v.detach()) for k,v in terms.items()})
            trace.append(row)
            with (folder/'training_log.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            if step%100==0 or step==1:
                print(arm,step,row['loss'],row['seconds'],flush=True)
                save_checkpoint(dict(step=step,model=model.state_dict(),optimizer=opt.state_dict(),
                    config=protocol,torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),
                    numpy_rng_state=np.random.get_state(),batch_plan_next_index=step,initialization='residual_only; frozen55k external dependency'),folder/'last.pt')
            if time.monotonic()-began>protocol['max_training_seconds_per_arm']:break
        step_counts[arm]=step
        # A bounded stop still gets its actual final atomic state.
        save_checkpoint(dict(step=step,model=model.state_dict(),optimizer=opt.state_dict(),config=protocol,
            torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),
            numpy_rng_state=np.random.get_state(),batch_plan_next_index=step),folder/'last.pt')
        report,output=evaluate(model,banks['validation'],identities['validation'],arm=='scene',arm)
        report.update(step=step,training_seconds=time.monotonic()-began,mean_update_seconds=float(np.mean([r['seconds'] for r in trace])),
            trainable_parameters=sum(p.numel() for p in model.parameters()))
        write_json(folder/'VALIDATION.json',report);reports[arm]=report;predictions[arm]=output
        if arm=='scene':
            ablation,out=evaluate(model,banks['validation'],identities['validation'],False,'scene_weights_input_OFF')
            write_json(folder/'SCENE_INPUT_OFF.json',ablation);reports['scene_input_off']=ablation;predictions['scene_input_off']=out
        del model,opt;torch.cuda.empty_cache()
    verdict={}
    for regime in ('clean','noisy'):
        initial=base['initial']['aggregate'][regime]
        reference=base['reference']['aggregate'][regime]
        on=reports['scene']['aggregate'][regime];off=reports['no_scene']['aggregate'][regime]
        checks=dict(occluded_improvement=on['occluded_mm']<=initial['occluded_mm']*.95,
            observation_reference_improvement=on['occluded_mm']<=reference['occluded_mm']*.95,
            extra_scene_improvement=on['occluded_mm']<=off['occluded_mm']*.98,
            visible_preserved=on['visible_mm']<=initial['visible_mm']+5,
            foot_velocity_preserved=on['foot_velocity_rmse_m_s']<=initial['foot_velocity_rmse_m_s']*1.10)
        verdict[regime]=dict(checks=checks,passed=all(checks.values()),initial=initial,
            observation_reference=reference,
            corrected_no_scene=off,corrected_scene=on,input_OFF=reports['scene_input_off']['aggregate'][regime])
    assert sha256(OLD)==source_hash,'Original55k checkpoint changed'
    assert code_hashes()==protocol['code_sha256'],'Sources changed during the pilot'
    paired=step_counts['scene']==step_counts['no_scene']==UPDATES
    status='passed_pilot_requires_expanded_validation' if paired and all(v['passed'] for v in verdict.values()) else 'not_passed_do_not_extend'
    save_checkpoint(dict(outputs=predictions,truth=banks['validation']['motion'].cpu(),rest=banks['validation']['rest'].cpu(),
        initial=banks['validation']['initial'].cpu(),reference=banks['validation']['reference'].cpu(),
        missing=banks['validation']['natural_missing'].cpu()),ROOT/'predictions.pt')
    write_json(ROOT/'RESULTS.json',dict(status=status,matched_full_budget=paired,step_counts=step_counts,verdict=verdict,
        old55k_unchanged=True,all_sources_unchanged=True,official_TEST_used=False,
        limitation='Small held-out VAL panel; legacy55k contains historical TEST overlap; ideal visible tracks; no real frontend; not publication acceptance'))
    protocol.update(status='completed',result_status=status,completed_at=datetime.datetime.now().astimezone().isoformat())
    write_json(ROOT/'PROTOCOL.json',protocol)
    print('COMPLETE',status,json.dumps(verdict),flush=True)


if __name__=='__main__':main()
