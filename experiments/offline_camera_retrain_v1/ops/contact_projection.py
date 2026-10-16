"""Observed-support joint projection for frozen-motion CPU diagnostics.

No global floor, GT contact, body labels or whole-body head translation is used.
This is an inference/postprocessing candidate, not Newtonian gravity and not
evidence that the denoiser learned physical support.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.native30_observation_ik import position_proximal
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.unified_geometry import skin_points

BASE = Path(__file__).resolve().parents[1]
DEFAULT_OUT = BASE / 'runs/native30_live_monitor_oct16/support_ablation/postprocess_projection'
DEFAULTS = dict(rounds=3, ik_iterations=6, foot_confidence=.5, foot_uncertainty_m=.03,
    target_step_limit_m=.03, max_support_height_m=.25, max_penetration_m=.05,
    tangent_limit_m=.08, min_normal_y=.9, max_foot_speed_m_s=.5,
    root_tilt_limit_deg=60., torso_tilt_limit_deg=75., clearance_m=.003,
    damping_m2=.0025, angle_limit_rad=.35)


def _config(config):
    settings = dict(DEFAULTS, **(config or {}))
    if set(settings) != set(DEFAULTS):
        raise ValueError('Unknown projection setting')
    for key, value in settings.items():
        if isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError('Invalid projection setting: ' + key)
    if not isinstance(settings['rounds'], int) or not 0 <= settings['rounds'] <= 3:
        raise ValueError('Projection rounds must be an integer from 0 to 3')
    if not isinstance(settings['ik_iterations'], int) or not 1 <= settings['ik_iterations'] <= 6:
        raise ValueError('IK iterations must be an integer from 1 to 6')
    if not 0 <= settings['foot_confidence'] <= .5:
        raise ValueError('Pseudo-foot confidence must be at most 0.5')
    if settings['target_step_limit_m'] > .03:
        raise ValueError('Each target update must be at most 3 cm')
    return settings


def _support_evidence(motion, known, cfg):
    """Freeze only predicted slow soles and local observed upward patches."""
    positions = forward_kinematics(motion, known['rest'])
    rotations = global_rotations(motion)
    root_up = rotations[:, :, 0, :, 1]
    torso = positions[:, :, 12] - positions[:, :, 0]
    torso_up = torso[..., 1] / torso.norm(dim=-1).clamp_min(1e-8)
    upright = ((root_up[..., 1] > np.cos(np.deg2rad(cfg['root_tilt_limit_deg'])))
               & (torso_up > np.cos(np.deg2rad(cfg['torso_tilt_limit_deg']))))
    all_evidence = []
    for b, (asset, scene) in enumerate(zip(known['skin_assets'], known['surface_windows'])):
        if known['mode'][b] != 'ego':
            all_evidence.append(None)
            continue
        labels = torch.as_tensor(asset['labels'], device=motion.device)
        pools = [torch.isin(labels, labels.new_tensor(pair)).nonzero().flatten()
                 for pair in ((7, 10), (8, 11))]
        if any(len(ids) == 0 for ids in pools):
            all_evidence.append(None)
            continue
        vertices = skin_points(motion[b], asset)
        ids = torch.stack([pool[vertices[:, pool, 1].argmin(-1)] for pool in pools], -1)
        frame = torch.arange(motion.shape[1], device=motion.device)[:, None]
        soles = vertices[frame, ids]
        rotation = rotations[b, frame, labels[ids]]
        if motion.device.type == 'cuda' and hasattr(scene, 'query_wide_motion'):
            raw = scene.query_wide_motion(soles.detach().cpu().numpy(), rotation.detach().cpu().numpy(), motion.device)
        else:
            raw = getattr(scene, 'query_wide', scene.query)(soles.detach().cpu().numpy(), rotation.detach().cpu().numpy())
        q = {key: torch.as_tensor(value, device=motion.device)[:, :, 2] for key, value in raw.items()}
        delta = soles[:, :, None] - q['points']
        signed = (delta * q['normals']).sum(-1)
        tangent = (delta - signed[..., None] * q['normals']).norm(dim=-1)
        valid = (q['valid'] & q['normal_reliable'] & (q['owners'] == 0)
                 & (q['normals'][..., 1] > cfg['min_normal_y'])
                 & (tangent < cfg['tangent_limit_m'])
                 & (signed < cfg['max_support_height_m']) & (signed > -cfg['max_penetration_m']))
        scores, nearest = torch.where(valid, delta.norm(dim=-1), torch.inf).min(-1)
        take = nearest[..., None, None].expand(*nearest.shape, 1, 3)
        points = q['points'].gather(2, take).squeeze(2)
        normals = q['normals'].gather(2, take).squeeze(2)
        velocity = torch.zeros_like(soles)
        if len(soles) > 1:
            velocity[1:] = soles.diff(dim=0) * 30
            velocity[0] = velocity[1]
        slow = velocity.norm(dim=-1) < cfg['max_foot_speed_m_s']
        compatible = torch.isfinite(scores) & slow & upright[b, :, None]
        persistent = torch.zeros_like(compatible)
        if len(soles) >= 3:
            height_stable = (points[2:, :, 1] - points[:-2, :, 1]).abs() < .05
            persistent[1:-1] = compatible[:-2] & compatible[1:-1] & compatible[2:] & height_stable
        # Never overwrite a measured foot with a scene-derived pseudo target.
        measured = known['observation_meta'][b, :, (10, 11), 0] > 0
        persistent &= ~measured
        all_evidence.append(dict(ids=ids, points=points.detach(), normals=normals.detach(),
                                  valid=persistent.detach(), initial_soles=soles.detach()))
    return all_evidence


def observed_contact_projection(motion, known_inference_batch, config=None, return_diagnostics=False):
    """Jointly fit real observations and conservative scene-derived feet.

    Root translation and root rotation stay fixed. Each of at most three rounds
    moves a foot-joint target by at most 3 cm along a reliable local normal and
    uses the existing six-iteration observation IK with pseudo confidence <=.5.
    Three consecutive low-speed predicted sole frames are required. Original
    head metadata and uncertainties are copied unchanged into the same solver.
    """
    cfg = _config(config)
    if motion.ndim != 3 or motion.shape[-1] != 201:
        raise ValueError('Expected motion [B,T,201]')
    forbidden = {'motion', 'joints', 'contact_target', 'contact_valid', 'gt_position_weight', 'support_points'}
    if forbidden.intersection(known_inference_batch):
        raise ValueError('Use the inference-only model_inputs payload')
    known = known_inference_batch
    diag = dict(ground_truth_read=False, root_fixed=True, projection_rounds=0,
                support_foot_frames=0, max_pseudo_confidence=cfg['foot_confidence'],
                target_step_limit_m=cfg['target_step_limit_m'], support_signed_height_before_mm=None,
                support_signed_height_after_mm=None)
    if cfg['rounds'] == 0 or cfg['foot_confidence'] == 0:
        return (motion, diag) if return_diagnostics else motion
    with torch.no_grad():
        evidence = _support_evidence(motion, known, cfg)
        diag['support_foot_frames'] = sum(int(e['valid'].sum()) for e in evidence if e is not None)
        if not diag['support_foot_frames']:
            return (motion, diag) if return_diagnostics else motion
        def signed_heights(value):
            values = []
            for b, e in enumerate(evidence):
                if e is None or not bool(e['valid'].any()):
                    continue
                skin = skin_points(value[b], known['skin_assets'][b])
                frame = torch.arange(value.shape[1], device=value.device)[:, None]
                soles = skin[frame, e['ids']]
                signed = ((soles - e['points']) * e['normals']).sum(-1)
                values.append(signed[e['valid']])
            return torch.cat(values)
        diag['support_signed_height_before_mm'] = float(signed_heights(motion).mean() * 1000)
        current = motion
        for _ in range(cfg['rounds']):
            target = known['trajectory'].clone()
            meta = known['observation_meta'].clone()
            uncertainty = known['observation_uncertainty'].clone()
            positions = forward_kinematics(current, known['rest'])
            for b, e in enumerate(evidence):
                if e is None:
                    continue
                skin = skin_points(current[b], known['skin_assets'][b])
                frame = torch.arange(current.shape[1], device=current.device)[:, None]
                soles = skin[frame, e['ids']]
                signed = ((soles - e['points']) * e['normals']).sum(-1)
                shift = (cfg['clearance_m'] - signed).clamp(-cfg['target_step_limit_m'], cfg['target_step_limit_m'])
                delta = shift[..., None] * e['normals']
                for foot, joint in enumerate((10, 11)):
                    active = e['valid'][:, foot]
                    target[b, active, joint, :3] = (positions[b, active, joint] + delta[active, foot]) / 2
                    meta[b, active, joint] = 0
                    meta[b, active, joint, 0] = 1
                    meta[b, active, joint, 2] = cfg['foot_confidence']
                    uncertainty[b, active, joint, 0] = cfg['foot_uncertainty_m']
            constrained = dict(known, trajectory=target, observation_meta=meta, observation_uncertainty=uncertainty)
            current = position_proximal(current, constrained, current.new_ones(len(current)), 1.,
                iterations=cfg['ik_iterations'], damping_m2=cfg['damping_m2'],
                angle_limit_rad=cfg['angle_limit_rad'], compact=True)
            diag['projection_rounds'] += 1
        if not torch.equal(current[..., :3], motion[..., :3]) or not torch.allclose(current[..., 3:9], motion[..., 3:9], atol=2e-6, rtol=0):
            raise RuntimeError('Support projection altered the fixed root')
        # Remove normalization roundoff in root rotation, then rebuild joint
        # channels so the root is also byte-identical in exported motion.
        current = torch.cat((motion[..., :9], current[..., 9:]), -1)
        current = torch.cat((current[..., :135], (forward_kinematics(current, known['rest']) / 2).flatten(2)), -1)
        diag['support_signed_height_after_mm'] = float(signed_heights(current).mean() * 1000)
        diag['joint_displacement_max_mm'] = float((forward_kinematics(current, known['rest'])
                                                    - forward_kinematics(motion, known['rest'])).norm(dim=-1).max() * 1000)
    return (current, diag) if return_diagnostics else current


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def run_cpu_review(out, *, case_ids=(0, 1)):
    """Reuse registered baseline exports; no model, GPU, or training update."""
    from experiments.offline_camera_retrain_v1.prepare_native30 import window_options
    from experiments.offline_camera_retrain_v1.native30_data import Native30Data
    from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
    from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs
    from experiments.offline_camera_retrain_v1.native30_surface_cache import CanonicalSurfaceCache
    from experiments.offline_camera_retrain_v1 import native30_surface_cache as surfaces, native30_memory as memory
    from .motion_ablation import load_module, FULL, surface_metrics
    from .support_metrics import support_metrics

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    root = DEFAULT_OUT.parent
    source_plan = json.loads((root / 'PLAN.json').read_text())
    source_rows = [json.loads(line) for line in (root / 'rows.jsonl').read_text().splitlines()]
    selected = [r for r in source_rows if r['case'] in case_ids and r['arm'] == 'baseline']
    if len(selected) != len(case_ids) * 2:
        raise ValueError('Wait for both baseline regimes for each requested case')
    out.mkdir(parents=True, exist_ok=True)
    sources = {str(Path(__file__)): _sha(__file__), str(root / 'PLAN.json'): _sha(root / 'PLAN.json')}
    sources.update({r['artifact']: _sha(r['artifact']) for r in selected})
    plan = dict(protocol='observed-contact-projection-postprocess-v1', created_at=dt.datetime.now().astimezone().isoformat(),
        checkpoint_step=source_plan['checkpoint_step'], cases=list(case_ids), regimes=['clean', 'noisy'],
        settings=DEFAULTS, source_sha256=sources, evaluation_only=True, cpu_only=True,
        interpretation='Frozen baseline motions receive postprocessing; not DDIM sampling or training. GT masks used only by evaluators.',
        outputs='One output NPZ and before/after metrics per source baseline export; shared scene and GT-associated evaluation mask.')
    if (out / 'PLAN.json').exists():
        raise ValueError('Output already exists; keep the first registered experiment')
    _write(out / 'PLAN.json', plan)
    io = load_module('projection_read_crc', BASE / 'runs/native30_final_review_oct16/io_recovery.py')
    surfaces.np = io.NumpyProxy(surfaces.np, out / 'IO_READS.jsonl')
    memory.np = io.NumpyProxy(memory.np, out / 'IO_READS.jsonl')
    effective = json.loads((BASE / 'runs/native30_gait_v2_ego55k_oct15/prepared/effective_config.json').read_text())
    cfg = effective['config']
    data = Native30Data(BASE / cfg['train_manifest'], 'validation', cfg['seed'] + 1, frames=cfg['frames'],
        memory_protocol=cfg['scene']['memory_protocol'], quality_config=cfg['gt_reliability'],
        exo_max_empty_frames=cfg['exo_max_empty_frames'], **window_options(cfg))
    data.runtime_cache_root = out / 'native30_runtime_cache_v2'
    original_cache_path = data.scene_memory._cache_path
    data.scene_memory._cache_path = lambda *args: (p if (p := original_cache_path(*args)) is not None and p.exists() else None)
    data.surface_cache = CanonicalSurfaceCache(data, limit=1)
    data.scene_memory.cached_streams = 1
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(device='cpu'),
        json.loads((BASE / cfg['mount_defaults']).read_text()), cfg['observation']['input_noise'])
    noisy = Native30ObservationBuilder(builder.skin_cache, builder.mounts, dict(builder.noise_config, clean_probability=0.))
    rows = []
    for row in selected:
        entry = next(e for e in source_plan['entries'] if e['case'] == row['case'])
        members = [i for i, (meta, _) in enumerate(data.members[entry['group']]) if meta['path'] == entry['path']]
        if len(members) != 1:
            raise ValueError('Ambiguous recording identity')
        sample, identity = data.sample(entry['group'], 'ego', member=members[0], start=entry['start'], yaw=0)
        observed, scene, asset = (builder if row['regime'] == 'clean' else noisy).build(data, sample, identity, 'ego', (15,),
            noise_rng=None if row['regime'] == 'clean' else np.random.default_rng(entry['case_seed']))
        batch = collate_observations([(observed, scene, asset, 'ego')], device='cpu')
        known = model_inputs(cfg, batch)
        with np.load(row['artifact'], allow_pickle=False) as saved:
            prediction = saved['prediction'].copy()
            if not (np.array_equal(saved['truth'], sample['motion']) and np.array_equal(saved['rest'], sample['rest'])
                    and np.array_equal(saved['observations'], observed['trajectory'])
                    and np.array_equal(saved['observation_meta'], observed['observation_meta'])):
                raise ValueError('Rebuilt body/input differs from frozen baseline')
        started = time.monotonic()
        fixed, diagnostics = observed_contact_projection(torch.from_numpy(prediction)[None], known, DEFAULTS, True)
        fixed = fixed[0].numpy()
        metrics = {}
        for name, value in [('before', prediction), ('after', fixed)]:
            metrics[name] = support_metrics(value, sample['motion'], sample['rest'], observed['trajectory'], observed['observation_meta'], asset)
            metrics[name].update(surface_metrics(value, sample['motion'], asset, scene))
        artifact = out / f"case{row['case']:02d}_{row['regime']}.npz"
        np.savez_compressed(artifact, prediction=fixed, baseline=prediction, truth=sample['motion'], rest=sample['rest'],
            observations=observed['trajectory'], observation_meta=observed['observation_meta'], identity_json=np.asarray(json.dumps(identity)),
            step=np.asarray(source_plan['checkpoint_step']), arm=np.asarray('observed_contact_projection'),
            baseline_artifact_sha256=np.asarray(row['artifact_sha256']))
        result = dict(case=row['case'], sequence_id=entry['sequence_id'], regime=row['regime'],
            artifact=str(artifact), artifact_sha256=_sha(artifact), source_artifact=row['artifact'],
            cpu_seconds=time.monotonic()-started, diagnostics=diagnostics, metrics=metrics,
            inference_keys=sorted(known), cuda_initialized=torch.cuda.is_initialized())
        rows.append(result)
        with (out / 'rows.jsonl').open('a') as stream:
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
        _write(out / 'STATUS.json', dict(status='running', completed=len(rows), planned=len(selected), last_case=entry['sequence_id']))
    keys = ['mpjpe_mm', 'observed_head_position_error_mm', 'foot_joint_horizontal_speed_gt_low_m_s',
            'foot_skin_anchor_low_y_abs_error_gt_low_mm', 'support_calibrated_absolute_error_mm', 'support_float_over_50mm_fraction']
    lines = ['# 观测支撑关节投影：CPU 后处理诊断', '',
        '同一15100步baseline输出，仅增加最多3轮观测关节IK。不是DDIM内约束或训练结果。GT只用于事后指标。', '',
        '|窗口|regime|支撑脚帧|MPJPE前→后 mm|头对输入前→后 mm|GT低速期水平脚速前→后 m/s|', '|---|---|---:|---:|---:|---:|']
    for row in rows:
        before, after = row['metrics']['before'], row['metrics']['after']
        pairs = [f'{before[k]:.3f} → {after[k]:.3f}' if before.get(k) is not None and after.get(k) is not None else '无有效样本' for k in keys[:3]]
        lines.append(f"|{row['sequence_id']}|{row['regime']}|{row['diagnostics']['support_foot_frames']}|" + '|'.join(pairs) + '|')
    lines += ['', '局限：预测低速不是接触真值；未覆盖楼梯、跳跃和全面的非直立反例。固定root不能补偿错误root高度，最多3轮3cm目标步长只检查局部修复能力。']
    (out / 'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    _write(out / 'SUMMARY.json', dict(status='complete', rows=len(rows), keys=keys, cuda_initialized=torch.cuda.is_initialized(),
        source_sha256_after={path: _sha(path) for path in sources}, source_unchanged=all(_sha(path)==digest for path,digest in sources.items())))
    _write(out / 'STATUS.json', dict(status='complete', completed=len(rows), planned=len(selected)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--cases', type=int, nargs='+', default=[0, 1])
    args = parser.parse_args()
    run_cpu_review(args.out, case_ids=tuple(args.cases))


if __name__ == '__main__':
    main()
