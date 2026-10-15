"""Read-only step1000 audit; isolated inference switches, no optimizer updates."""
import copy
import datetime
import importlib.util
import json
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

from experiments.offline_camera_retrain_v1.prepare_native30 import (
    HERE, code_hashes, make_model, sha256, window_options, write_json,
)
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, sample_motion
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.native30_validation import raw_metrics
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent/'native30_gait_v2_ego55k_oct15'


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def decompose(prediction, batch):
    def local(motion):
        p = forward_kinematics(motion, batch['rest'])
        r = rotation_from_6d(motion[..., 3:9])
        return torch.einsum('btji,btkj->btki', r, p-p[:, :, :1])[:, :, (10, 11)]
    p, t = local(prediction), local(batch['motion'])
    delta = p-t
    bias = delta.mean(1, keepdim=True)
    return dict(body_local_bias_RMS_mm=float(bias.square().sum(-1).mean().sqrt()*1000),
        body_local_dynamic_RMS_mm=float((delta-bias).square().sum(-1).mean().sqrt()*1000),
        body_local_bias_per_foot_xyz_mm=(bias[0, 0]*1000).tolist())


def main():
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    checkpoint = PARENT/'training/last.pt'
    checkpoint_sha = sha256(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert state['step'] == 1000 and state['planned_steps'] == 55000
    assert code_hashes() == state['effective_config']['code_sha256']
    config = copy.deepcopy(state['effective_config']['config'])
    assert config == json.loads((PARENT/'config.json').read_text())
    def finite(values):
        return all(not v.is_floating_point() or bool(torch.isfinite(v).all())
                   for v in values.values() if torch.is_tensor(v))
    integrity = dict(step=state['step'], planned_steps=state['planned_steps'],
        checkpoint_sha256=checkpoint_sha, model_finite=finite(state['model']),
        optimizer_finite=all(finite(v) for v in state['optimizer']['state'].values()),
        optimizer_entries=len(state['optimizer']['state']),
        optimizer_parameter_steps=sorted({float(v['step']) for v in state['optimizer']['state'].values()}),
        rng_and_sampler_present=all(k in state for k in ('torch_rng','cuda_rng','numpy_rng','condition_rng','data_rng')),
        source_hashes_unchanged=True)
    model = make_model(config).cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    del state
    metric_source = HERE/config['run_control']['foot_metrics_source']
    spec = importlib.util.spec_from_file_location('audit_foot_metrics', metric_source)
    extra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extra)
    data = Native30Data(HERE/config['train_manifest'], 'validation', config['seed']+1,
        frames=config['frames'], memory_protocol=config['scene']['memory_protocol'],
        quality_config=config['gt_reliability'], exo_max_empty_frames=config['exo_max_empty_frames'],
        **window_options(config))
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(),
        json.loads((HERE/config['mount_defaults']).read_text()), config['observation']['input_noise'])
    plan = [r for r in json.loads((PARENT/'training/validation_plan.json').read_text())['plan']
            if r['panel'] == 'fixed']
    assert len(plan) == 3
    options = ('current', 'physics_off', 'proximal_and_physics_off', 'root_reference_low_confidence', 'scene_off')
    protocol = dict(status='running', started_at=now(), pid=os.getpid(), optimizer_updates=0,
        checkpoint_integrity=integrity, variants=options, cases=plan,
        scope='Three already-used fixed official VAL windows, clean only, identical DDIM seeds. Inference sensitivity, not retrained ablations or independent model selection.',
        disk_free_before={str(p):shutil.disk_usage(p).free for p in (ROOT, Path('/mnt/e'))})
    write_json(ROOT/'PROTOCOL.json', protocol)
    rows = []
    captures = {}
    original_forward = model.forward
    def captured(this, noisy, timestep, batch, **kwargs):
        want = kwargs.pop('return_details', False)
        result, details = original_forward(noisy, timestep, batch, return_details=True, **kwargs)
        if int(timestep[0]) == 0:
            captures['raw'] = reconcile_motion(details['raw_motion'], batch['rest']).detach()
        return (result, details) if want else result
    model.forward = types.MethodType(captured, model)
    gains = (model.observation_anchor_gain, model.joint_anchor_gain)
    with np.load(PARENT/'training/validation_latest.npz') as saved, torch.no_grad(), (ROOT/'trace.jsonl').open('w') as stream:
        assert int(saved['step']) == 1000
        for entry in plan:
            case = next(i for i, r in enumerate(json.loads((PARENT/'training/validation_plan.json').read_text())['plan']) if r == entry)
            sample, identity = data.sample(entry['group'], 'ego', member=entry['member'], start=entry['start'], yaw=0)
            observed, scene, asset = builder.build(data, sample, identity, 'ego', (15,))
            batch = collate_observations([(observed, scene, asset, 'ego')])
            known = model_inputs(config, batch)
            root_conf = known['reference_confidence'][0, :, 0, 1]
            ref_error = raw_metrics(known['reference_motion'], batch, batch['observation_meta'][..., 0]>0)
            confidence_diagnostic = dict(root_rotation_reference_confidence_mean=float(root_conf.mean()),
                observed_root_rotation_frames=int((batch['observation_meta'][0,:,0,1]>0).sum()),
                root_reference_orientation_error_deg=ref_error['root_orientation_deg']['value'])
            for variant in options:
                model.observation_anchor_gain, model.joint_anchor_gain = gains
                model.guidance_enabled = variant not in ('physics_off','proximal_and_physics_off')
                inputs = known
                if variant == 'proximal_and_physics_off':
                    model.observation_anchor_gain = model.joint_anchor_gain = 0.
                if variant == 'root_reference_low_confidence':
                    inputs = dict(known, reference_confidence=known['reference_confidence'].clone())
                    inputs['reference_confidence'][:,:,0,1] = .05
                tick = time.monotonic()
                pred = sample_motion(model, inputs, config, steps=20, seed=entry['case_seed'], use_scene=variant!='scene_off')
                torch.cuda.synchronize()
                result = {k:v['value'] for k,v in raw_metrics(pred,batch,batch['observation_meta'][...,0]>0).items()}
                result.update(extra.extra_metrics(pred,batch)); result.update(decompose(pred,batch))
                row = dict(case=case,dataset=entry['dataset'],sequence_id=entry['sequence_id'],variant=variant,
                    seconds=time.monotonic()-tick,metrics=result,reference_diagnostic=confidence_diagnostic,
                    last_raw_metrics=dict(extra.extra_metrics(captures['raw'],batch),**decompose(captures['raw'],batch)))
                if variant == 'current':
                    original = torch.from_numpy(saved[f'{case}_clean_prediction']).to(pred)
                    row['reproduction_max_feature_difference'] = float((pred[0]-original).abs().max())
                    assert row['reproduction_max_feature_difference'] < 1e-5
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();rows.append(row)
                print(case,variant,'MPJPE',round(result['mpjpe_mm'],2),'local feet',round(result['body_local_foot_error_mm'],2),'speed',round(result['foot_velocity_RMSE_m_s'],3),flush=True)
    assert code_hashes() == json.loads((PARENT/'prepared/effective_config.json').read_text())['code_sha256']
    assert sha256(checkpoint) == checkpoint_sha
    aggregate = {}
    for variant in options:
        group = [r for r in rows if r['variant']==variant]
        keys = [k for k,v in group[0]['metrics'].items() if isinstance(v,(float,int))]
        aggregate[variant] = {k:float(np.mean([r['metrics'][k] for r in group if r['metrics'][k] is not None])) for k in keys}
    protocol.update(status='completed',finished_at=now(),checkpoint_unchanged=True,
        aggregate=aggregate,raw_rows='trace.jsonl',disk_free_after={str(p):shutil.disk_usage(p).free for p in (ROOT,Path('/mnt/e'))})
    write_json(ROOT/'RESULTS.json',protocol)
    print('Completed read-only audit; checkpoint unchanged.',flush=True)


if __name__ == '__main__':
    main()
