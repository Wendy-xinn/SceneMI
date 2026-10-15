"""Bounded paired continuation from clean10000; no automatic longer training."""
import copy
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config, make_model, window_options, sample_item, training_plan, code_hashes, write_json
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, training_motion
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.native30_losses import objective
from experiments.offline_camera_retrain_v1.native30_gait_supervision import DEFAULT, gait_terms
from experiments.offline_camera_retrain_v1.native30_gt_reliability import gt_weight
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d
from experiments.offline_camera_retrain_v1 import native30_validation as validation
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.train_native30 import StopAtUpdateBoundary

ROOT = Path(__file__).resolve().parent
UPDATES = 100
LR = 2e-5
MAX_SECONDS = 2400


def dataset(config, split):
    return Native30Data(HERE / config['train_manifest'], split, config['seed'], frames=config['frames'],
        memory_protocol=config['scene']['memory_protocol'], quality_config=config['gt_reliability'],
        exo_max_empty_frames=config['exo_max_empty_frames'], **window_options(config))


def motion_metrics(motion, truth, rest):
    p = torch.as_tensor(motion, dtype=torch.float32)[None]
    t = torch.as_tensor(truth, dtype=torch.float32)[None]
    r = torch.as_tensor(rest, dtype=torch.float32)[None]
    with torch.no_grad():
        pj, tj = forward_kinematics(p, r)[0], forward_kinematics(t, r)[0]
        def local(m, j):
            rot = rotation_from_6d(m[0, :, 3:9])
            return torch.einsum('tji,tkj->tki', rot, j-j[:, :1])[:, (10, 11)]
        pl, tl = local(p, pj), local(t, tj)
        amp = lambda x: ((x-x.mean(0)).square().sum(-1).mean(0)).sqrt()
        v, tv = pj.diff(dim=0)*30, tj.diff(dim=0)*30
        stance = tv[:, (10, 11)].norm(dim=-1) < .2
        return dict(mpjpe_mm=float((pj-tj).norm(dim=-1).mean()*1000),
            foot_world_error_mm=float((pj[:, (7,8,10,11)]-tj[:, (7,8,10,11)]).norm(dim=-1).mean()*1000),
            foot_velocity_vector_RMSE_m_s=float((v[:,(10,11)]-tv[:,(10,11)]).square().sum(-1).mean().sqrt()),
            body_local_foot_position_error_mm=float((pl-tl).norm(dim=-1).mean()*1000),
            stance_foot_speed_m_s=float(v[:, (10,11)].norm(dim=-1)[stance].mean()) if stance.any() else None,
            foot_articulation_RMS_ratio=(amp(pl)/amp(tl).clamp_min(1e-8)).tolist())


def choose_plan(val, config):
    fixed = [r for r in validation.monitoring_plan(val, config, 10000) if r['mode']=='ego' and r['panel']=='fixed']
    selected, screened = [], []
    for name, groups in (('trumans', ('trumans',)), ('egobody', ('camera_wearer','interactee')), ('rich', ('rich',))):
        pool=[]
        for group in groups:
            for member, (meta, _) in enumerate(val.members[group]):
                if 'ego' in meta.get('allowed_modes', []) and len(val.starts_for(group, member, 'ego')):
                    pool.append((meta['sequence_id'], group, member))
        pool.sort()
        eligible=[]
        for index in np.linspace(0, len(pool)-1, min(24, len(pool)), dtype=int):
            seq, group, member = pool[index]
            starts=val.starts_for(group, member, 'ego'); start=int(starts[len(starts)//2])
            sample, identity=val.sample(group, 'ego', member=member, start=start, yaw=0)
            m=torch.from_numpy(sample['motion'])[None];rest=torch.from_numpy(sample['rest'])[None]
            with torch.no_grad():
                j=forward_kinematics(m,rest)[0];rot=rotation_from_6d(m[0,:,3:9])
                local=torch.einsum('tji,tkj->tki',rot,j-j[:,:1])[:,(10,11)]
                amplitude=float(((local-local.mean(0)).square().sum(-1).mean()).sqrt())
                net=float((j[-1,0]-j[0,0]).norm())
                z=local[:,:,2].numpy(); corr=float(np.corrcoef(z.T)[0,1]) if np.min(z.std(0))>1e-6 else 1.
                stance=float(((j[:,(10,11)].diff(dim=0)*30).norm(dim=-1)<.2).float().mean())
            row=dict(dataset=name,group=group,member=member,start=start,sequence_id=seq,
                GT_root_net_m=net,GT_foot_body_local_RMS_m=amplitude,GT_left_right_Z_correlation=corr,GT_stance_fraction=stance)
            row['eligible']=bool(net>=.6 and amplitude>=.06 and corr<=-.1 and .1<=stance<=.9)
            screened.append(row)
            if row['eligible'] and seq not in {x['sequence_id'] for x in fixed}:eligible.append(row)
        if eligible:
            best=sorted(eligible,key=lambda x:(-x['GT_root_net_m'],x['sequence_id']))[0]
            key=json.dumps([best['group'],'ego',best['sequence_id'],best['start']],separators=(',',':'))
            seed=(config['validation']['seed']+int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little'))%(2**32)
            selected.append(dict(group=best['group'],mode='ego',member=best['member'],start=best['start'],joints=(15,),
                sequence_id=best['sequence_id'],dataset=name,panel='GT_locomotion',case_seed=seed))
    report=dict(selection='GT only, deterministic24 evenly-spaced eligible sequences/dataset; middle valid window; root net>=.6m, body-local feet RMS>=.06m, left/right sagittal correlation<=-.1, GT stance fraction0.1..0.9; largest GT net displacement per dataset; no prediction-based fallback.',
        fixed=fixed,locomotion=selected,screened=screened)
    write_json(ROOT/'VAL_PLAN.json',report)
    return fixed+selected


def summarize_results():
    reports={}
    for name in ('parent','control','gait_restore'):
        folder=ROOT/(name+'_validation')
        val=json.loads((folder/'validation.jsonl').read_text().splitlines()[-1])
        stages=json.loads((folder/'PROJECTION_REVIEW.json').read_text())['cases']
        fixed={regime:dict(mpjpe_mm=val['summary'][f'fixed/ego/{regime}/scene_on']['mpjpe_mm']['value'],
            stance_foot_speed_m_s=val['summary'][f'fixed/ego/{regime}/scene_on']['foot_speed_during_GT_stance_m_s']['value'],
            foot_world_error_mm=float(np.mean([r['stages']['after_physics']['foot_world_error_mm'] for r in stages if r['panel']=='fixed' and r['regime']==regime]))) for regime in ('clean','noisy')}
        locomotion=[r for r in stages if r['panel']=='GT_locomotion']
        reports[name]=dict(fixed=fixed,locomotion_conditions=len(locomotion),
            locomotion_amplitude_log_ratio_error=float(np.mean([abs(math.log(max(v,1e-8))) for r in locomotion for v in r['stages']['after_physics']['foot_articulation_RMS_ratio']])) if locomotion else None,
            locomotion_foot_velocity_RMSE_m_s=float(np.mean([r['stages']['after_physics']['foot_velocity_vector_RMSE_m_s'] for r in locomotion])) if locomotion else None)
    candidate=reports['gait_restore'];checks={}
    for other_name in ('parent','control'):
        other=reports[other_name]
        for regime in ('clean','noisy'):
            checks[f'{other_name}_{regime}_MPJPE_no_more_than2pct_worse']=candidate['fixed'][regime]['mpjpe_mm']<=other['fixed'][regime]['mpjpe_mm']*1.02
            checks[f'{other_name}_{regime}_foot_position_no_more_than5pct_worse']=candidate['fixed'][regime]['foot_world_error_mm']<=other['fixed'][regime]['foot_world_error_mm']*1.05
        checks[f'{other_name}_noisy_stance_no_more_than10pct_worse']=candidate['fixed']['noisy']['stance_foot_speed_m_s']<=other['fixed']['noisy']['stance_foot_speed_m_s']*1.10
        if candidate['locomotion_conditions']:
            checks[f'{other_name}_locomotion_amplitude_error_improves5pct']=candidate['locomotion_amplitude_log_ratio_error']<=other['locomotion_amplitude_log_ratio_error']*.95
            checks[f'{other_name}_locomotion_velocity_no_more_than5pct_worse']=candidate['locomotion_foot_velocity_RMSE_m_s']<=other['locomotion_foot_velocity_RMSE_m_s']*1.05
    checks['at_least_one_GT_locomotion_window']=candidate['locomotion_conditions']>=2
    original=HERE/'runs/native30_fresh_ego_oct14/training'
    old_val=json.loads((original/'validation.jsonl').read_text().splitlines()[-1])
    reproduced=reports['parent']['fixed']
    parent_error=max(abs(reproduced[r]['mpjpe_mm']-old_val['summary'][f'fixed/ego/{r}/scene_on']['mpjpe_mm']['value']) for r in ('clean','noisy'))
    assert parent_error<.01, ('Parent fixed VAL reproduction mismatch',parent_error)
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    unchanged=hashlib.sha256((original/'last.pt').read_bytes()).hexdigest()==protocol['parent_sha256']
    assert unchanged
    report=dict(observed_at=datetime.datetime.now().astimezone().isoformat(),status='passed_short_trial_gate' if all(checks.values()) else 'not_passed_do_not_extend',
        comparison=reports,checks=checks,parent_fixed_MPJPE_reproduction_max_difference_mm=parent_error,
        original_checkpoint_unchanged=unchanged,optimizer_steps_per_arm=UPDATES,
        limitation='Single paired100-update continuation, small fixed/GT-selected VAL panel and one sampling seed. A failure here is lack of evidence for longer training, not proof that every gait objective or longer budget fails.',
        action='No automatic further training or exo handoff.')
    write_json(ROOT/'RESULTS.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    labels=['parent10000','control+100','gait+100'];x=np.arange(3)
    for ax,key,title in ((axes[0],'mpjpe_mm','Fixed ego MPJPE (mm)'),(axes[1],'stance_foot_speed_m_s','GT-stance foot speed (m/s)')):
        for offset,regime,color in ((-.18,'clean','#246ed0'),(.18,'noisy','#bc187e')):
            ax.bar(x+offset,[reports[n]['fixed'][regime][key] for n in reports],width=.35,label=regime,color=color)
        ax.set_xticks(x);ax.set_xticklabels(labels);ax.set_title(title);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Paired bounded pilot / same starting Adam, RNG,100 batches, LR2e-5\nThree fixed official VAL windows; no automatic extension')
    fig.tight_layout();fig.savefig(ROOT/'PAIRED_RESULTS.png',dpi=140)
    print(json.dumps(report,ensure_ascii=False),flush=True)


def main():
    began=time.monotonic()
    preflight=json.loads((ROOT/'PREFLIGHT.json').read_text())
    assert preflight['status']=='passed' and preflight['code_sha256']==code_hashes()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.28)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    config=read_config(HERE/'native30_fresh_ego_config.json')
    checkpoint_path=HERE/'runs/native30_fresh_ego_oct14/training/last.pt'
    source=torch.load(checkpoint_path,map_location='cpu',weights_only=False,mmap=True)
    assert source['step']==10000
    val=dataset(config,'validation'); val_plan=choose_plan(val,config)
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),json.loads((HERE/config['mount_defaults']).read_text()),config['observation']['input_noise'])
    model=make_model(config).cuda()
    configs={name:copy.deepcopy(config) for name in ('control','gait_restore')}
    configs['gait_restore']['gait_supervision']=copy.deepcopy(DEFAULT)
    for c in configs.values():c['validation']['scene_ablation_interval_steps']=0
    protocol=dict(status='running',started_at=datetime.datetime.now().astimezone().isoformat(),pid=os.getpid(),
        parent_checkpoint=str(checkpoint_path.resolve()),parent_sha256=preflight['source_checkpoint_sha256'],parent_step=10000,
        max_updates_per_arm=UPDATES,batch_size=2,constant_lr=LR,optimizer_reset=False,sampler_rng_restored=True,
        max_wall_seconds_after_preflight=MAX_SECONDS,
        changed_objective_only=True,query_backend='unchanged CPU query; existing caches',
        original_schedule='Completed10000-stage cosine floor2e-5 continued constantly for both diagnostic arms; new provenance, not original strict resume.',
        code_sha256=preflight['code_sha256'],data_audit='PREFLIGHT.json',validation_plan='VAL_PLAN.json',
        gate=dict(fixed_MPJPE_both_regimes_max_regression_fraction=.02,fixed_foot_position_max_regression_fraction=.05,
            noisy_stance_speed_max_regression_fraction=.10,locomotion_articulation_log_ratio_error_min_improvement_fraction=.05,
            locomotion_foot_velocity_max_regression_fraction=.05,compare_against_both_parent_and_control=True,
            minimum_GT_locomotion_cases=1,note='Short-trial triage gate vs paired control, not paper acceptance. No automatic extension even if passed.'),
        arms={})
    write_json(ROOT/'PROTOCOL.json',protocol)
    for name,c in configs.items():write_json(ROOT/f'{name}_config.json',c)
    def deadline():
        if time.monotonic()-began>MAX_SECONDS:raise TimeoutError('Bounded pilot wall-time exhausted; no auto continuation')

    def evaluate(label,c,step):
        deadline();folder=ROOT/label;folder.mkdir(exist_ok=False)
        captured=[];original_forward=model.forward
        def traced(this,noisy,times,batch,**kwargs):
            deadline()
            want_details=kwargs.pop('return_details',False)
            result,details=original_forward(noisy,times,batch,return_details=True,**kwargs)
            if int(times[0])==0:
                raw=reconcile_motion(details['raw_motion'],batch['rest'])
                feedback=reconcile_motion(details['raw_motion'],batch['rest'],details['root_delta_m'],details['rotation_delta_rad'])
                ik=model.constrain_motion(feedback,times,batch)
                captured.append({k:v[0].detach().cpu().numpy().copy() for k,v in dict(reference=batch['reference_motion'],raw=raw,feedback=feedback,after_IK=ik,after_physics=result).items()})
            return (result,details) if want_details else result
        model.forward=types.MethodType(traced,model)
        original_plan=validation.monitoring_plan
        validation.monitoring_plan=lambda *args:copy.deepcopy(val_plan)
        try:validation.validate(model,val,builder,c,folder,step)
        finally:model.forward=original_forward;validation.monitoring_plan=original_plan
        stages=[];arrays={}
        with np.load(folder/'validation_latest.npz') as a:
            assert len(captured)==len(val_plan)*2
            for i,entry in enumerate(val_plan):
                for ri,regime in enumerate(('clean','noisy')):
                    poses=captured[i*2+ri]
                    for stage,motion in poses.items():arrays[f'{i}_{regime}_{stage}']=motion
                    stages.append(dict(case=i,panel=entry['panel'],dataset=entry['dataset'],regime=regime,
                        stages={k:motion_metrics(v,a[f'{i}_truth'],a[f'{i}_rest']) for k,v in poses.items()}))
        np.savez_compressed(folder/'projection_stages.npz',**arrays)
        write_json(folder/'PROJECTION_REVIEW.json',dict(placement='Last x0 only; other DDIM steps retain original projection. Not full no-projection ablation.',cases=stages))
        return folder

    model.load_state_dict(source['model'],strict=True)
    evaluate('parent_validation',configs['control'],10000)
    for name,c in configs.items():
        deadline();folder=ROOT/name;folder.mkdir(exist_ok=False)
        model.load_state_dict(source['model'],strict=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=config['optimizer']['weight_decay'])
        optimizer.load_state_dict(copy.deepcopy(source['optimizer']))
        for group in optimizer.param_groups:group['lr']=LR
        data=dataset(c,'train');data.load_sampler_state(copy.deepcopy(source['data_rng']))
        rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(source['condition_rng'])
        torch.set_rng_state(source['torch_rng']);torch.cuda.set_rng_state(source['cuda_rng']);np.random.set_state(source['numpy_rng'])
        plan=training_plan(data,c);counts=copy.deepcopy(source['counts'])
        def save(step):
            save_checkpoint(dict(step=10000+step,stage_step=step,planned_stage_steps=UPDATES,model=model.state_dict(),optimizer=optimizer.state_dict(),
                config=c,continuation_protocol=protocol,parent_checkpoint_step=10000,counts=counts,
                data_rng=data.sampler_state(),condition_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state()),folder/'last.pt')
        with StopAtUpdateBoundary() as stop,(folder/'trace.jsonl').open('w') as stream:
            for step in range(1,UPDATES+1):
                deadline();start=time.monotonic()
                items,identities=zip(*(sample_item(data,builder,c,plan,rng) for _ in range(c['batch_size'])))
                batch=collate_observations(items);known=model_inputs(c,batch)
                times=torch.randint(1000,(len(items),),device='cuda')
                low=torch.rand(len(items),device='cuda')<c['noise']['extra_low_noise_probability']
                times=torch.where(low,torch.randint(c['noise']['low_noise_steps'],times.shape,device='cuda'),times)
                signal=model.alphas[times];target=training_motion(c,batch,known)
                noisy=signal[:,None,None].sqrt()*target+(1-signal[:,None,None]).sqrt()*torch.randn_like(target)
                scene_on=rng.random()>=c['scene']['dropout_probability']
                model.train();optimizer.zero_grad(set_to_none=True)
                pred,details=model(noisy,times,known,use_scene=scene_on,return_details=True)
                loss,terms=objective(pred,batch['motion'],batch,signal,details,c,source['effective_config']['observation_coefficients'])
                if step==1:
                    joints=forward_kinematics(pred,batch['rest']);truth=forward_kinematics(batch['motion'],batch['rest'])
                    gait=gait_terms(joints,truth,gt_weight(batch,'gt_position_weight',pred),signal,DEFAULT)
                    delta=gait['total']-.075*terms['base']['foot_velocity_m2_s2']
                    baseline_loss=loss-delta if name=='gait_restore' else loss
                    gradient=torch.autograd.grad(baseline_loss,details['raw_motion'],retain_graph=True)[0]
                    added=torch.autograd.grad(delta,details['raw_motion'],retain_graph=True)[0]
                    legs=[3+6*j+k for j in (1,2,4,5,7,8,10,11) for k in range(6)]
                    ratio=float(added.norm()/gradient.norm().clamp_min(1e-12))
                    audit=dict(base_raw_motion_gradient_norm=float(gradient.norm()),gait_delta_raw_motion_gradient_norm=float(added.norm()),
                        gait_delta_leg_rotation_gradient_norm=float(added[...,legs].norm()),gradient_ratio=ratio,
                        terms={k:float(v.detach()) for k,v in gait.items()},timesteps=times.tolist())
                    write_json(folder/'TRAIN_GRADIENT_AUDIT.json',audit)
                    assert torch.isfinite(added).all() and float(added[...,legs].norm())>0 and ratio<5, audit
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite paired pilot objective')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['optimizer']['clip_norm'],error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize()
                for identity in identities:
                    key=f"{identity['group']}/{identity['mode']}";counts[key]=counts.get(key,0)+1
                identity_fields=('group','mode','sequence_id','start','yaw_augmentation_rad','source_start_30fps')
                row=dict(stage_step=step,step=10000+step,loss=float(loss.detach()),gradnorm=float(norm),lr=LR,
                    timesteps=times.tolist(),scene=bool(scene_on),identities=[{k:x[k] for k in identity_fields} for x in identities],
                    terms={group:{k:float(v.detach()) for k,v in values.items()} for group,values in terms.items()},
                    elapsed_s=time.monotonic()-start,support_cache_hits=data.support_cache.hits,support_cache_misses=data.support_cache.misses)
                stream.write(json.dumps(row)+'\n');stream.flush()
                if step%10==0:print(name,step,'loss',row['loss'],'seconds',row['elapsed_s'],flush=True)
                if step%25==0 or stop.requested:save(step)
                if stop.requested:raise InterruptedError('Stopped at completed optimizer boundary')
        protocol['arms'][name]=dict(status='trained',updates=UPDATES,checkpoint=str(folder/'last.pt'))
        write_json(ROOT/'PROTOCOL.json',protocol)
        evaluate(name+'_validation',c,10000+UPDATES)
        protocol['arms'][name]['status']='completed'
        write_json(ROOT/'PROTOCOL.json',protocol)
        del optimizer,data,batch,known,pred,details,loss,terms,counts
        gc.collect();torch.cuda.empty_cache()
    a=[json.loads(line) for line in (ROOT/'control/trace.jsonl').read_text().splitlines()]
    b=[json.loads(line) for line in (ROOT/'gait_restore/trace.jsonl').read_text().splitlines()]
    assert len(a)==len(b)==UPDATES
    for first,second in zip(a,b):
        assert all(first[k]==second[k] for k in ('step','timesteps','scene','identities','lr'))
    protocol.update(status='completed',completed_at=datetime.datetime.now().astimezone().isoformat(),elapsed_s=time.monotonic()-began,
        all_sampling_and_diffusion_draws_paired=True,no_automatic_long_training=True)
    write_json(ROOT/'PROTOCOL.json',protocol)
    print('PAIRED PROBE COMPLETED',protocol['elapsed_s'],flush=True)
    summarize_results()


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        p=ROOT/'PROTOCOL.json'
        record=json.loads(p.read_text()) if p.exists() else {}
        record.update(status='failed_or_bounded_stop',error=repr(exc),observed_at=datetime.datetime.now().astimezone().isoformat())
        write_json(p,record)
        raise
