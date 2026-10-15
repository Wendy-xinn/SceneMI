"""Isolated candidate: smooth solver increments, retain acceptance and scene.

Existing 10k inference only. Never smooth the generated motion/GT. The temporal
filter acts on bounded translation/axis-angle trial increments BEFORE the
existing energy/measurement-protected backtracking. Formal source unchanged.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from torch.nn import functional as F
from experiments.offline_camera_retrain_v1.prepare_native30 import HERE,make_model,write_json,sha256,window_options
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs,sample_motion
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1 import native30_physical_guidance as physics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'increment_smoothing';KERNEL=5
spec=importlib.util.spec_from_file_location('probe_metrics',ROOT/'run_probe.py')
metric_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric_module)


def smooth_increment(value,kernel=KERNEL):
    if kernel<1 or kernel%2!=1:raise ValueError('Positive odd kernel required')
    if kernel==1:return value
    b,t=value.shape[:2]
    channels=value.reshape(b,t,-1).transpose(1,2)
    return F.avg_pool1d(F.pad(channels,(kernel//2,kernel//2),mode='replicate'),kernel,stride=1).transpose(1,2).reshape_as(value)


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    # Meaningful small checks: constants unchanged; pulse energy/spread; batch/
    # joint independence; original norm bound after averaging; no in-place edit.
    constant=torch.ones(2,19,22,3);assert torch.equal(smooth_increment(constant),constant)
    pulse=torch.zeros_like(constant);pulse[0,9,3,1]=1;before=pulse.clone();filtered=smooth_increment(pulse)
    assert torch.equal(pulse,before) and int((filtered!=0).sum())==5
    assert torch.allclose(filtered.sum(),pulse.sum()) and filtered[:, :, 4:].abs().max()==0 and filtered[1].abs().max()==0
    assert smooth_increment(pulse,1) is pulse
    config=json.loads((ROOT/'config.json').read_text())
    parent=HERE/'runs/native30_fresh_ego_oct14/training/last.pt'
    expected=json.loads((ROOT/'PROTOCOL.json').read_text())['checkpoint_sha256'];assert sha256(parent)==expected
    source=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
    model=make_model(config).cuda();model.load_state_dict(source['model'],strict=True);model.eval();del source
    data=Native30Data(HERE/config['train_manifest'],'validation',config['seed'],frames=config['frames'],
        memory_protocol=config['scene']['memory_protocol'],quality_config=config['gt_reliability'],
        exo_max_empty_frames=config['exo_max_empty_frames'],**window_options(config))
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),json.loads((HERE/config['mount_defaults']).read_text()),config['observation']['input_noise'])
    plan=json.loads((ROOT/'VAL_PLAN.json').read_text())['plan']
    base_rows=[json.loads(line) for line in (ROOT/'trace.jsonl').read_text().splitlines()]
    original_clip=physics.norm_clip
    def clip_and_smooth(value,limit):
        return original_clip(smooth_increment(original_clip(value,limit)),limit)
    protocol=dict(status='running',optimizer_steps=0,kernel_frames=KERNEL,fps=30,
        algorithm='Replicate-padded temporal average of already bounded solver translation/axis-angle increments, then enforce original norm bound; existing energy, trusted observations, root-total limit and backtracking unchanged.',
        final_motion_postprocessing=False,GT_in_filter=False,formal_source_changed=False,
        checkpoint_sha256=expected,runner_sha256=sha256(Path(__file__)),metric_runner_sha256=sha256(Path(metric_module.__file__)),
        panel='Same9 development VAL windows and clean/forced-noisy seeds as schedule probe; candidate fixed to5 frames before any candidate predictions.',
        scope='Same-weight inference candidate; no independently trained smoothing model or paper claim.',
        synthetic_checks_passed=True)
    write_json(OUT/'PROTOCOL.json',protocol)
    rows=[];arrays={};start=time.monotonic()
    try:
        with torch.no_grad(),np.load(ROOT/'predictions.npz') as original_arrays,(OUT/'trace.jsonl').open('w') as stream:
            for case,entry in enumerate(plan):
                assert time.monotonic()-start<1500
                sample,identity=data.sample(entry['group'],'ego',member=entry['member'],start=entry['start'],yaw=0)
                for regime in ('clean','noisy'):
                    builder2=builder if regime=='clean' else Native30ObservationBuilder(builder.skin_cache,builder.mounts,dict(builder.noise_config,clean_probability=0.))
                    observed,scene,asset=builder2.build(data,sample,identity,'ego',(15,),noise_rng=np.random.default_rng(entry['case_seed']) if regime=='noisy' else None)
                    batch=collate_observations([(observed,scene,asset,'ego')]);known=model_inputs(config,batch)
                    if case==0 and regime=='clean':
                        # Reproduce the unmodified control before enabling adapter.
                        model.guidance_enabled=True;model.guidance_config['minimum_alpha']=.25
                        control=sample_motion(model,known,config,steps=20,seed=entry['case_seed'])
                        error=float((control[0].cpu()-torch.from_numpy(original_arrays[f'{case}_{regime}_current'])).abs().max())
                        assert error<1e-6, ('Unmodified control mismatch',error)
                        protocol['control_reproduction_max_feature_difference']=error
                        write_json(OUT/'PROTOCOL.json',protocol)
                    for name,minimum in (('current_smooth5',.25),('late_four_smooth5',.9)):
                        model.guidance_enabled=True;model.guidance_config['minimum_alpha']=minimum
                        physics.norm_clip=clip_and_smooth
                        accepted=[];original_guide=physics.guide_motion
                        def traced_guide(*args,**kwargs):
                            result,details=original_guide(*args,**kwargs)
                            accepted.append(dict(steps=float(details['physical_accepted_steps'].sum()),
                                energy_change=float((details['physical_energy_after']-details['physical_energy_before']).sum())))
                            assert float((details['physical_energy_after']-details['physical_energy_before']).max())<=1e-5
                            return result,details
                        physics.guide_motion=traced_guide
                        try:
                            torch.cuda.synchronize();tick=time.monotonic()
                            prediction=sample_motion(model,known,config,steps=20,seed=entry['case_seed'])
                            torch.cuda.synchronize();seconds=time.monotonic()-tick
                        finally:
                            physics.guide_motion=original_guide;physics.norm_clip=original_clip
                        assert torch.isfinite(prediction).all()
                        row=dict(case=case,panel=entry['panel'],dataset=entry['dataset'],regime=regime,variant=name,
                            DDIM_seconds=seconds,accepted_physics_iterations=sum(r['steps'] for r in accepted),
                            physical_energy_decrease=-sum(r['energy_change'] for r in accepted),
                            metrics=metric_module.metrics(prediction,batch))
                        rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                        arrays[f'{case}_{regime}_{name}']=prediction[0].cpu().numpy()
                        print(case,regime,name,'foot velocity',round(row['metrics']['foot_velocity_RMSE_m_s'],3),'MPJPE',round(row['metrics']['mpjpe_mm'],2),'accepted',row['accepted_physics_iterations'],flush=True)
                np.savez_compressed(OUT/'predictions.npz',**arrays)
    finally:physics.norm_clip=original_clip
    aggregate={}
    for panel in ('all','fixed','GT_locomotion','additional_VAL'):
        aggregate[panel]={}
        for regime in ('clean','noisy'):
            aggregate[panel][regime]={}
            for name in ('current','late_four','physics_off','current_smooth5','late_four_smooth5'):
                pool=rows if name.endswith('smooth5') else base_rows
                group=[r for r in pool if r['variant']==name and r['regime']==regime and (panel=='all' or r['panel']==panel)]
                metrics={k:float(np.mean([r['metrics'][k] for r in group if r['metrics'][k] is not None]))
                    for k,v in group[0]['metrics'].items() if isinstance(v,(float,int))}
                aggregate[panel][regime][name]=dict(metrics=metrics,cases=len(group),
                    accepted_physics_iterations=sum(r.get('accepted_physics_iterations',r.get('counters',{}).get('accepted_physics_iterations',0)) for r in group),
                    DDIM_seconds_mean=float(np.mean([r['DDIM_seconds'] for r in group])))
    assert sha256(parent)==expected
    write_json(OUT/'RESULTS.json',dict(status='completed_zero_update_candidate',optimizer_steps=0,aggregate=aggregate,
        limitations='Same existing10k weights and reused development VAL. Solver increments only. Lower velocity error may reflect smaller corrections; inspect accepted counts and pose/scene evidence jointly. No automatic promotion or training.'))
    protocol.update(status='completed',elapsed_s=time.monotonic()-start,checkpoint_unchanged=True)
    write_json(OUT/'PROTOCOL.json',protocol)


if __name__=='__main__':main()
