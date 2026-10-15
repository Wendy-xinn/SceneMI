"""Bounded, zero-update, same-weight full-DDIM physics/scene diagnosis.

This evaluates the existing 10k model, not a newly trained V2 model. It cannot
select a final training architecture. All targets are read only by metrics.
"""
import copy
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import types

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import (
    code_hashes, make_model, sha256, window_options, write_json,
)
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, sample_motion
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d
from experiments.offline_camera_retrain_v1.unified_geometry import skin_points
from experiments.offline_camera_retrain_v1 import native30_validation as validation
from experiments.offline_camera_retrain_v1 import native30_physical_guidance as physics

ROOT = Path(__file__).resolve().parent
PARENT = HERE/'runs/native30_fresh_ego_oct14/training/last.pt'
EXPECTED = 'aaec44f9353a32acfca818aa8c5ed903e93aca78548b5d52f31cf506db35e835'
VARIANTS = dict(current=(True, .25, True), physics_off=(False, .25, True),
                late_four=(True, .90, True), late_two=(True, .98, True),
                scene_off=(False, .25, False))
MAX_SECONDS = 1800


def now():
    return datetime.datetime.now().astimezone().isoformat()


def extra_metrics(pred, batch):
    p = forward_kinematics(pred, batch['rest'])
    g = forward_kinematics(batch['motion'], batch['rest'])
    def local(m, j):
        r = rotation_from_6d(m[..., 3:9])
        return torch.einsum('btji,btkj->btki', r, j-j[:, :, :1])[:, :, (10, 11)]
    pl, gl = local(pred, p), local(batch['motion'], g)
    amp = lambda v: (v-v.mean(1, keepdim=True)).square().sum(-1).mean(1).sqrt()
    pa, ga = amp(pl), amp(gl)
    ratio = pa/ga.clamp_min(1e-8)
    pv, gv = p.diff(dim=1)*30, g.diff(dim=1)*30
    speed, true_speed = pv[:, :, (10, 11)].norm(dim=-1), gv[:, :, (10, 11)].norm(dim=-1)
    swing = true_speed >= .2
    def corr(v):
        z = v[..., 2]-v[..., 2].mean(1, keepdim=True)
        return (z[:, :, 0]*z[:, :, 1]).mean(1)/(z[:, :, 0].square().mean(1)*z[:, :, 1].square().mean(1)).sqrt().clamp_min(1e-8)
    return dict(foot_world_error_mm=float((p[:, :, (7,8,10,11)]-g[:, :, (7,8,10,11)]).norm(dim=-1).mean()*1000),
        body_local_foot_error_mm=float((pl-gl).norm(dim=-1).mean()*1000),
        articulation_RMS_ratio=ratio[0].tolist(),
        articulation_log_ratio_error=float(ratio.clamp_min(1e-8).log().abs().mean()),
        bilateral_sagittal_correlation_error=float((corr(pl)-corr(gl)).abs().mean()),
        foot_velocity_RMSE_m_s=float((pv[:, :, (10,11)]-gv[:, :, (10,11)]).square().sum(-1).mean().sqrt()),
        swing_velocity_RMSE_m_s=float((pv[:, :, (10,11)]-gv[:, :, (10,11)]).square().sum(-1)[swing].mean().sqrt()) if swing.any() else None,
        GT_body_local_foot_RMS_m=ga[0].tolist(),
        foot_acceleration_RMSE_m_s2=float(((pv.diff(dim=1)-gv.diff(dim=1))*30)[:, :, (10,11)].square().sum(-1).mean().sqrt()))


def metrics(pred, batch):
    conditioned = batch['observation_meta'][..., 0] > 0
    result = {k:v['value'] for k,v in validation.raw_metrics(pred, batch, conditioned).items()}
    result.update(extra_metrics(pred, batch))
    return result


def evaluate_surface(pred, evidence, batch):
    # Fixed GT-associated OBSERVED patches, only for evaluation, never inputs.
    # This is a shallow observed-patch proxy, not a scene-wide collision metric.
    vertices = skin_points(pred[0], batch['skin_assets'][0])
    delta = vertices-evidence['points']
    signed = (delta*evidence['normals']).sum(-1)
    feet = torch.isin(batch['skin_assets'][0]['labels'], signed.new_tensor([7,8,10,11], dtype=torch.long))
    mask = (evidence['contact_weight'] > 0) & feet[None] & (evidence['normals'][...,1] > .7)
    n = int(mask.sum())
    return dict(fixed_GT_observed_foot_patch_count=n,
        observed_patch_below_2cm_mean_mm=float((-signed[mask]-.02).clamp_min(0).mean()*1000) if n else None,
        fixed_patch_distance_mean_mm=float(delta.norm(dim=-1)[mask].mean()*1000) if n else None)


def main():
    began = time.monotonic()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seal = json.loads((HERE/'runs/native30_gait_v2_arch_review_oct15/prepared/effective_config.json').read_text())
    assert code_hashes() == seal['code_sha256'], 'Changed root source since engineering preflight'
    assert sha256(PARENT) == EXPECTED
    source = torch.load(PARENT, map_location='cpu', mmap=True, weights_only=False)
    assert source['step'] == 10000
    config = copy.deepcopy(source['effective_config']['config'])
    config.setdefault('runtime', {})['query_backend'] = 'cuda_exact_radius_v1'
    config['refinement']['joint_anchor_compact'] = False
    config['physical_guidance']['query_contact_regions_only'] = False
    write_json(ROOT/'config.json', config)
    protocol = dict(status='running',started_at=now(),pid=os.getpid(),optimizer_steps=0,
        checkpoint=str(PARENT.resolve()),checkpoint_sha256=EXPECTED,checkpoint_step=10000,
        scope='Same pretrained weights, whole DDIM trajectory schedule ablation, not separately trained arms; V2 learnability is not evaluated.',
        variants={k:dict(physical_enabled=v[0],minimum_alpha=v[1],use_scene=v[2]) for k,v in VARIANTS.items()},
        long_training_started=False,auto_continue=False,max_seconds=MAX_SECONDS,
        source_sha256=code_hashes(),runner_sha256=sha256(Path(__file__)),
        data_audit='Existing full TRAIN/VAL engineering seal; this run samples official VAL only, no TEST.',
        disk_free_before={str(p):shutil.disk_usage(p).free for p in (ROOT,Path('/mnt/e'))},
        timing_scope='Synchronized full DDIM20 including ViT, queries, IK, correction, and diagnostic scalar collection; excludes input build and posthoc GT metrics. One untimed warmup then case-rotated variant order.',
        surface_metric_scope='Fixed GT-associated observed patches only, evaluation after predictions; shallow proxy, not complete collision evaluation.')
    write_json(ROOT/'PROTOCOL.json', protocol)
    print('initializing official VAL data', flush=True)
    data = Native30Data(HERE/config['train_manifest'], 'validation', config['seed'], frames=config['frames'],
        memory_protocol=config['scene']['memory_protocol'],quality_config=config['gt_reliability'],
        exo_max_empty_frames=config['exo_max_empty_frames'], **window_options(config))
    previous = json.loads((HERE/'runs/native30_gait_restore_probe_oct15/VAL_PLAN.json').read_text())
    plan = previous['fixed']+previous['locomotion']
    additional = [r for r in validation.monitoring_plan(data,config,10000) if r['mode']=='ego' and r['panel']=='rotating']
    existing = {r['sequence_id'] for r in plan}
    for r in additional:
        if r['sequence_id'] not in existing:
            r['panel'] = 'additional_VAL';plan.append(r);existing.add(r['sequence_id'])
    write_json(ROOT/'VAL_PLAN.json',dict(plan=plan,selection='Predeclared previous fixed+GT-locomotion development panel and step10000 official rotating ego windows, deduplicated by sequence. Additional windows are not an untouched test set.',test_used=False))
    model = make_model(config).cuda()
    model.load_state_dict(source['model'],strict=True);model.eval()
    del source
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(),json.loads((HERE/config['mount_defaults']).read_text()),config['observation']['input_noise'])
    counter = {}; final = {}
    original_forward, original_constrain = model.forward, model.constrain_motion
    original_evidence, original_guide = physics.prepare_evidence, physics.guide_motion
    original_core = model._core_pass
    def counted_constrain(this, motion, times, batch):
        counter['constrain_calls'] = counter.get('constrain_calls',0)+1
        counter['eligible_IK_items'] = counter.get('eligible_IK_items',0)+int((this.alphas[times]>=.05).sum())
        return original_constrain(motion,times,batch)
    def counted_core(this,*args,**kwargs):
        counter['core_passes'] = counter.get('core_passes',0)+1
        return original_core(*args,**kwargs)
    def counted_evidence(*args,**kwargs):
        counter['physics_evidence_queries'] = counter.get('physics_evidence_queries',0)+1
        out = original_evidence(*args,**kwargs)
        counter['nonzero_contact_anchor_frames'] = counter.get('nonzero_contact_anchor_frames',0)+int((out['contact_weight']>0).sum())
        return out
    def counted_guide(motion,times,batch,details,c,constrain):
        result, diagnostic = original_guide(motion,times,batch,details,c,constrain)
        counter['accepted_physics_iterations'] = counter.get('accepted_physics_iterations',0)+float(diagnostic['physical_accepted_steps'].sum())
        counter['physical_energy_decrease'] = counter.get('physical_energy_decrease',0)+float((diagnostic['physical_energy_before']-diagnostic['physical_energy_after']).sum())
        if int(times[0]) == 0: final['before_physics'] = motion.detach()
        return result,diagnostic
    def traced(this,noisy,times,batch,**kwargs):
        want = kwargs.pop('return_details',False)
        result, details = original_forward(noisy,times,batch,return_details=True,**kwargs)
        if int(times[0]) == 0:
            final['raw'] = details['raw_motion'].detach()
            final.setdefault('before_physics',result.detach())
        return (result,details) if want else result
    model.forward=types.MethodType(traced,model)
    model.constrain_motion=types.MethodType(counted_constrain,model)
    model._core_pass=types.MethodType(counted_core,model)
    physics.prepare_evidence,physics.guide_motion=counted_evidence,counted_guide
    arrays = {}; rows = []; names=list(VARIANTS)
    try:
        with torch.no_grad(), (ROOT/'trace.jsonl').open('w') as stream:
            for case,entry in enumerate(plan):
                assert time.monotonic()-began<MAX_SECONDS
                sample, identity = data.sample(entry['group'],'ego',member=entry['member'],start=entry['start'],yaw=0)
                assert identity['sequence_id']==entry['sequence_id']
                for regime in ('clean','noisy'):
                    local_builder=builder
                    if regime=='noisy':
                        local_builder=Native30ObservationBuilder(builder.skin_cache,builder.mounts,dict(builder.noise_config,clean_probability=0.))
                    observed,scene,asset=local_builder.build(data,sample,identity,'ego',(15,),noise_rng=np.random.default_rng(entry['case_seed']) if regime=='noisy' else None)
                    batch=collate_observations([(observed,scene,asset,'ego')]);known=model_inputs(config,batch)
                    if case==0 and regime=='clean':
                        print('untimed scene-on/physics-on kernel warmup',flush=True)
                        sample_motion(model,known,config,steps=20,seed=entry['case_seed'])
                    outputs={}
                    order=names[case%len(names):]+names[:case%len(names)]
                    for name in order:
                        enabled,minimum,use_scene=VARIANTS[name]
                        model.guidance_enabled=enabled;model.guidance_config['minimum_alpha']=minimum
                        counter.clear();final.clear()
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
                        pred=sample_motion(model,known,config,steps=20,seed=entry['case_seed'],use_scene=use_scene)
                        torch.cuda.synchronize();seconds=time.monotonic()-start
                        assert torch.isfinite(pred).all()
                        assert counter['core_passes']==40
                        if not enabled: assert counter.get('physics_evidence_queries',0)==0
                        row=dict(case=case,panel=entry['panel'],dataset=entry['dataset'],regime=regime,variant=name,
                            seed=entry['case_seed'],sequence_id=entry['sequence_id'],DDIM_seconds=seconds,
                            peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,counters=copy.deepcopy(counter),metrics=metrics(pred,batch),
                            final_raw_metrics=metrics(reconcile_motion(final['raw'],batch['rest']),batch),
                            final_before_physics_metrics=metrics(final['before_physics'],batch))
                        outputs[name]=pred.clone()
                        arrays[f'{case}_{regime}_{name}']=pred[0].cpu().numpy()
                        arrays[f'{case}_{regime}_{name}_raw']=reconcile_motion(final['raw'],batch['rest'])[0].cpu().numpy()
                        rows.append(row)
                        print(case,regime,name,round(seconds,3),'MPJPE',round(row['metrics']['mpjpe_mm'],2),'accepted',counter.get('accepted_physics_iterations',0),flush=True)
                    # Build metric-only evidence AFTER every variant has finished.
                    evidence=original_evidence(batch['motion'],known,torch.ones(1,config['frames'],22,device='cuda'),model.guidance_config)
                    for row in rows[-len(names):]:
                        row['surface_metrics']=evaluate_surface(outputs[row['variant']],evidence,batch)
                        delta=forward_kinematics(outputs[row['variant']],batch['rest'])-forward_kinematics(outputs['physics_off'],batch['rest'])
                        row['joint_change_vs_physics_off_mm']=float(delta.norm(dim=-1).mean()*1000)
                        row['joint_change_max_vs_physics_off_mm']=float(delta.norm(dim=-1).max()*1000)
                        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                    arrays[f'{case}_{regime}_observations']=observed['trajectory']
                arrays[f'{case}_truth']=sample['motion'];arrays[f'{case}_rest']=sample['rest']
                np.savez_compressed(ROOT/'predictions.npz',**arrays)
                write_json(ROOT/'progress.json',dict(completed_cases=case+1,total_cases=len(plan),elapsed_s=time.monotonic()-began))
    finally:
        model.forward=original_forward;model.constrain_motion=original_constrain;model._core_pass=original_core
        physics.prepare_evidence,physics.guide_motion=original_evidence,original_guide
    assert sha256(PARENT)==EXPECTED
    aggregate={}
    for panel in ('all','fixed','GT_locomotion','additional_VAL'):
        aggregate[panel]={}
        for regime in ('clean','noisy'):
            aggregate[panel][regime]={}
            for name in names:
                group=[r for r in rows if r['variant']==name and r['regime']==regime and (panel=='all' or r['panel']==panel)]
                if not group:continue
                keys=[k for k,v in group[0]['metrics'].items() if isinstance(v,(int,float))]
                aggregate[panel][regime][name]=dict(cases=len(group),
                    metrics={k:float(np.mean([r['metrics'][k] for r in group if r['metrics'][k] is not None])) for k in keys},
                    DDIM_seconds_mean=float(np.mean([r['DDIM_seconds'] for r in group])),
                    physics_evidence_queries=sum(r['counters'].get('physics_evidence_queries',0) for r in group),
                    accepted_physics_iterations=sum(r['counters'].get('accepted_physics_iterations',0) for r in group),
                    change_vs_physics_off_mm=float(np.mean([r['joint_change_vs_physics_off_mm'] for r in group])))
    old=json.loads((PARENT.parent/'validation.jsonl').read_text().splitlines()[-1])
    errors={r:abs(aggregate['fixed'][r]['current']['metrics']['mpjpe_mm']-old['summary'][f'fixed/ego/{r}/scene_on']['mpjpe_mm']['value']) for r in ('clean','noisy')}
    assert max(errors.values())<.02, errors
    report=dict(status='completed_zero_update_inference_diagnosis',optimizer_steps=0,aggregate=aggregate,
        original_fixed_MPJPE_reproduction_difference_mm=errors,
        checkpoint_unchanged=True,raw_rows='trace.jsonl',plan='VAL_PLAN.json',
        limitations=['Same 10k weights trained with the original objective, no V2 training updates.',
            'Small development VAL panels, one diffusion seed per case, timings are diagnostic.',
            'Learned scene contribution: compare physics_off(scene ON) with scene_off; physics contribution: compare current/late with physics_off.',
            'Last raw stage still reflects earlier DDIM guidance; it is not a fully unconstrained sampler.'],
        no_automatic_training=True)
    write_json(ROOT/'RESULTS.json',report)
    protocol.update(status='completed',completed_at=now(),elapsed_s=time.monotonic()-began,
        disk_free_after={str(p):shutil.disk_usage(p).free for p in (ROOT,Path('/mnt/e'))})
    write_json(ROOT/'PROTOCOL.json',protocol)
    print('COMPLETED',protocol['elapsed_s'],flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        p=ROOT/'PROTOCOL.json'
        record=json.loads(p.read_text()) if p.exists() else {}
        record.update(status='failed',error=repr(exc),failed_at=now())
        write_json(p,record)
        raise
