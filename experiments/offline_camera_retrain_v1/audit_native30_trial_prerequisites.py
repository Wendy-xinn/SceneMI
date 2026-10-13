"""Current noise/reliability integration and full-model backward, zero updates.

Uses historical review caches for interface checks only. Never marks formal
input_checks training_ready or replaces whole-corpus preparation.
"""
import hashlib
import json
import shutil
import time
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.prepare_native30 import read_config,make_model
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder,inference_payload,INPUT_KEYS
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1.native30_losses import objective
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas


def run():
    started=time.monotonic();before=shutil.disk_usage(HERE).free;config=read_config()
    suite=unittest.defaultTestLoader.loadTestsFromNames([
        'experiments.offline_camera_retrain_v1.test_native_gt_quality',
        'experiments.offline_camera_retrain_v1.test_native30',
        'experiments.offline_camera_retrain_v1.test_native30_protocol_v2'])
    tests=unittest.TextTestRunner(verbosity=1).run(suite)
    if not tests.wasSuccessful():raise AssertionError('Native30 reliability/noise regression failed')
    torch.set_num_threads(4);torch.manual_seed(20261013)
    mount_path=HERE/config['diagnostic_mount_defaults'];mounts=json.loads(mount_path.read_text())
    data=Native30Data(HERE/config['review_manifest'],'validation',diagnostic=True,frames=64,
        memory_protocol=config['scene']['memory_protocol'],quality_config=config['gt_reliability'])
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),mounts,config['observation']['input_noise'])
    checks=[];items=[]
    for mode in ('ego','exo'):
        sample,identity=data.sample('camera_wearer',mode,member=0,start=64,yaw=0)
        joints=(15,20,21) if mode=='ego' else None
        clean,scene,asset=builder.build(data,sample,identity,mode,joints)
        clean_builder=Native30ObservationBuilder(builder.skin_cache,mounts,dict(config['observation']['input_noise'],clean_probability=1.))
        clean_aug,_,_=clean_builder.build(data,sample,identity,mode,joints,noise_rng=np.random.default_rng(31))
        for key in ('trajectory','observation_meta','camera','camera_confidence'):
            np.testing.assert_array_equal(clean[key],clean_aug[key])
        noisy_builder=Native30ObservationBuilder(builder.skin_cache,mounts,dict(config['observation']['input_noise'],clean_probability=0.,gap_probability=0.))
        noisy,scene,asset=noisy_builder.build(data,sample,identity,mode,joints,noise_rng=np.random.default_rng(31))
        for key in ('motion','joints','rest','contact_target','contact_valid'):
            np.testing.assert_array_equal(clean[key],noisy[key])
        if mode=='ego':
            if np.array_equal(clean['trajectory'][:,20:22],noisy['trajectory'][:,20:22]):raise AssertionError('Selected wrists not perturbed')
        else:
            if noisy['observation_meta'][...,1].any():raise AssertionError('Invented exo rotation')
            np.testing.assert_array_equal(clean['camera'],noisy['camera'])
        checks.append(dict(mode=mode,sequence_id=identity['sequence_id'],clean_exact=True,GT_unchanged=True,
            selected_wrists_noisy=(mode=='ego'),exo_rotations_absent=(mode=='exo')))
        items.append((noisy,scene,asset,mode))
    batch=collate_observations(items);known=inference_payload(batch)
    if any(key.startswith('gt_') for key in known):raise AssertionError('GT reliability leaked into model conditions')
    model=make_model(config).cuda().train();alphas=cosine_alphas().cuda();t=torch.tensor([25,300],device='cuda');signal=alphas[t]
    state=signal[:,None,None].sqrt()*batch['motion']+(1-signal[:,None,None]).sqrt()*torch.randn_like(batch['motion'])
    torch.cuda.reset_peak_memory_stats();prediction,details=model(state,t,known,return_details=True)
    weights=config['observation']['maximum_coefficients']
    loss,terms=objective(prediction,batch['motion'],batch,signal,details,config,weights)
    if not torch.isfinite(loss):raise AssertionError('Nonfinite complete native30 objective')
    loss.backward()
    gradient={name:float(p.grad.detach().norm()) for name,p in model.named_parameters() if p.grad is not None}
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise AssertionError('Nonfinite complete native30 backward')
    for name in ('root_feedback.weight','rotation_feedback.weight','observation_encoder.0.weight'):
        if gradient.get(name,0)<=0:raise AssertionError(f'No gradient: {name}')
    from experiments.offline_camera_retrain_v1.native30_validation import validate
    model.zero_grad(set_to_none=True)
    data.members={k:v if k=='camera_wearer' else [] for k,v in data.members.items()}
    smoke_folder=HERE/'runs/exo_occlusion_review_oct12/validation_smoke';smoke_folder.mkdir(exist_ok=True)
    torch_before=torch.get_rng_state().clone();cuda_before=torch.cuda.get_rng_state().clone()
    sampler_before=data.sampler_state()
    validation_rows=validate(model,data,builder,config,smoke_folder,0)
    if not torch.equal(torch_before,torch.get_rng_state()) or not torch.equal(cuda_before,torch.cuda.get_rng_state()):
        raise AssertionError('Validation changed training RNG')
    if data.sampler_state()!=sampler_before:raise AssertionError('Validation changed data sampler')
    if not model.training or model._sampling_scene is not None:raise AssertionError('Validation left model mode/cache modified')
    files=['native30_input_noise.py','native30_gt_reliability.py','native30_observations.py','native30_losses.py',
        'native30_data.py','native30_config.json','prepare_native30.py','train_native30.py','test_native_gt_quality.py','native30_validation.py',Path(__file__).name]
    report=dict(protocol='native30-trial-prerequisite-interface-v1',optimizer_steps=0,training_ready=False,
        diagnostic_only=True,source_scope='old review caches; not formal whole corpus or official metrics',
        tests=dict(passed=tests.testsRun),real_input_checks=checks,GT_weights_excluded_from_input=True,
        full_backward=dict(batch=2,frames=64,model_parameters=sum(p.numel() for p in model.parameters()),
            finite=True,loss=float(loss.detach()),peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
            key_gradient_norms={name:gradient[name] for name in ('root_feedback.weight','rotation_feedback.weight','observation_encoder.0.weight')}),
        terms={group:{k:float(v.detach()) for k,v in values.items()} for group,values in terms.items()},
        validation_smoke=dict(cases=len(validation_rows),ddim_steps=config['sampling']['ddim_steps'],frames=64,
            training_rng_preserved=True,sampler_preserved=True,GT_metrics_unweighted=True,
            scope='Untrained historical review windows; not official validation results'),
        config=config,code_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in files},
        free_disk_before=before,free_disk_after=shutil.disk_usage(HERE).free,elapsed_s=time.monotonic()-started)
    (HERE/'runs/exo_occlusion_review_oct12/TRIAL_PREREQUISITES_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('optimizer_steps','training_ready','tests','full_backward','elapsed_s')}),flush=True)


if __name__=='__main__':
    run()
