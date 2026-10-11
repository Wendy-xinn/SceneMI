"""Executable leakage, geometry, missingness and warm-start checks."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import HERE, PROJECTS
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
from experiments.offline_camera_retrain_v1.unified_geometry import SkinAnchorCache, skin_points, ObservedSurfaceWindow
from experiments.offline_camera_retrain_v1.unified_observations import (
    ObservationBuilder, collate_observations, inference_payload, observation_derivatives)
from experiments.offline_camera_retrain_v1.unified_model import UnifiedSceneMI
from experiments.offline_camera_retrain_v1.export_egobody_exo import camera_rays, cast_body


def dataset(config, split, seed=2026101141):
    return NativeBodyData(split, seed=seed, **{k: config[k] for k in (
        'skeleton_profile', 'rich_source', 'trumans_scene_manifest',
        'trumans_window_protocol', 'temporal_scene_manifest')},
        contact_root=config['rich_contact_root'], window_sampling='duration')


def run_checks(cp, output):
    torch.set_num_threads(4); torch.manual_seed(2026101141)
    c = cp['config']; data = dataset(c, 'validation')
    cache = SkinAnchorCache(); builder = ObservationBuilder(cache)
    samples = {}
    report = dict(status='running', seed=2026101141, checks={})
    for group in ('camera_wearer', 'trumans'):
        sample, identity = data.sample(128, group, sequence_index=0, start_index=0)
        asset = cache.get(identity['native_body'])
        test_motion = torch.from_numpy(sample['motion'][[0, 32, 96]]).cuda().requires_grad_(True)
        sparse = skin_points(test_motion, asset)
        native, _, _ = decode_native_mesh(test_motion.detach().cpu(), identity['native_body'])
        error = float(np.abs(sparse.detach().cpu().numpy() - native[:, asset['vertex_ids']]).max())
        assert error < 2e-5, (group, error)
        sparse.square().mean().backward(); assert torch.isfinite(test_motion.grad).all()
        assert len(set(asset['vertex_ids'])) == 67
        report['checks'][f'{group}_native_skin_max_error_m'] = error
        samples[group] = (sample, identity)
    sample, identity = samples['camera_wearer']
    prepared, scene, asset = builder.build(data, sample, identity, 'ego')
    batch = collate_observations([(prepared, scene, asset, 'ego')]); known = inference_payload(batch)
    model = UnifiedSceneMI(c['latent_dim'], tuple(c['dim_mults']), body_conditioning=True, contact_prediction=True).cuda().eval()
    model.load_state_dict(cp['model']); model.surface_enabled = False
    original = OfflineSceneMI(c['latent_dim'], tuple(c['dim_mults']), body_conditioning=True, contact_prediction=True).cuda().eval()
    original.load_state_dict(cp['model'])
    noisy = torch.randn(1, 128, 201, device='cuda'); t = torch.tensor([55], device='cuda')
    mask = fixed_control_mask(1, 128, 'head', 'cuda')
    with torch.no_grad():
        old = original(noisy, t, known, control_mask=mask)
        new = model(noisy, t, known, control_mask=mask)
    error = float((old - new).abs().max())
    relative = float((old - new).square().mean().sqrt() / old.square().mean().sqrt())
    # A wider FP32 GEMM changes reduction order despite zero new columns.
    assert error < .001 and relative < .0001, (error, relative)
    report['checks']['legacy_mapping_max_error'] = error
    report['checks']['legacy_mapping_relative_rms_error'] = relative
    del original
    model.surface_enabled = True
    with torch.no_grad():
        reference = model(noisy, t, known, control_mask=mask)
        poisoned_labels = dict(batch)
        for key in ('motion', 'joints', 'contact_target'):
            poisoned_labels[key] = torch.full_like(batch[key], float('nan'))
        poison = model(noisy, t, inference_payload(poisoned_labels), control_mask=mask)
        assert torch.equal(reference, poison)
        hidden = dict(known); trajectory = known['trajectory'].clone()
        trajectory[..., :3] = torch.where(known['observation_meta'][..., 0:1] > 0, trajectory[..., :3], float('nan'))
        trajectory[..., 3:] = torch.where(known['observation_meta'][..., 1:2] > 0, trajectory[..., 3:], float('nan'))
        hidden['trajectory'] = trajectory
        hidden_output = model(noisy, t, hidden, control_mask=mask)
        assert torch.equal(reference, hidden_output)
        empty = dict(known, surface_windows=[ObservedSurfaceWindow(np.empty((0, 3)))])
        assert torch.isfinite(model(noisy, t, empty, control_mask=mask)).all()
    report['checks'].update(label_poison_exact_equal=True, hidden_nan_exact_equal=True,
                            empty_scene_finite=True, query_pose_source='generated proposal; no body labels')
    gap_meta = known['observation_meta'].clone(); gap_meta[:, 40:45, 15, :4] = 0
    *_, derivative_masks = observation_derivatives(known['trajectory'], gap_meta)
    assert not derivative_masks[0, 40:46, 15, 0].any()
    assert not derivative_masks[0, 40:47, 15, 1].any()
    report['checks']['temporal_gaps_not_bridged'] = True
    # Near-plane geometry in the scan must not hide all surfaces behind it.
    vertices = np.array([[-10, -10, .05], [10, -10, .05], [0, 10, .05],
                         [-10, -10, 2], [10, -10, 2], [0, 10, 2]], np.float32)
    depth, _ = cast_body(vertices, np.array([[0, 1, 2], [3, 4, 5]], np.uint32),
                         np.array([[0, 0, 0, 0, 0, 1]], np.float32))
    assert abs(float(depth[0]) - 2) < 1e-6
    foreground, _ = cast_body(vertices[3:] * np.array([1, 1, .5], np.float32),
                              np.array([[0, 1, 2]], np.uint32), np.array([[0, 0, 0, 0, 0, 1]], np.float32))
    assert foreground[0] < depth[0]
    report['checks'].update(near_clip_second_hit=True, foreground_occlusion=True)
    calibration = json.loads((PROJECTS / 'egobody/kinect_cam_params/kinect_master/Color.json').read_text())
    rays, pixels = camera_rays(calibration, np.eye(4))
    selected = np.linspace(0, len(rays) - 1, 500, dtype=int)
    projected, _ = cv2.projectPoints(rays[selected, 3:].astype(np.float64), np.zeros(3), np.zeros(3),
                                     np.asarray(calibration['camera_mtx']), np.asarray(calibration['k']))
    error = float(np.abs(projected[:, 0] - pixels[selected]).max()); assert error < .001, error
    report['checks']['distorted_reprojection_max_error_px'] = error
    # Exported observed skeleton must be the same native target, not a second
    # model's approximate joint annotation.
    exo_folder = builder.exo_root / 'validation' / identity['sequence_id']
    if exo_folder.joinpath('metadata.json').exists():
        exo_sample, exo_scene, exo_asset = builder.build(data, sample, identity, 'exo', joints=(15, 20, 21))
        exo = collate_observations([(exo_sample, exo_scene, exo_asset, 'exo')])
        available = exo['observation_meta'][..., 0] > 0
        error = float((exo['trajectory'][..., :3] * 2 - exo['joints'] * 2).norm(dim=-1)[available].max()) if available.any() else 0.
        assert error < 1e-4, error
        assert not exo['observation_meta'][..., 1].any()
        assert torch.isfinite(model(noisy, t, inference_payload(exo), control_mask=torch.ones_like(mask))).all()
        report['checks'].update(exo_native_observed_joint_max_error_m=error,
                                exo_has_no_rotation=True, exo_forward_finite=True)
    else:
        raise RuntimeError('Validation exo pilot must exist before training')
    # Validated split scope is recording, not person/role/window.
    from experiments.offline_camera_retrain_v1.data import SOURCE
    records = {split: {r['recording'] for r in map(json.loads, (SOURCE / f'egobody_sequences/{split}.jsonl').read_text().splitlines())}
               for split in ('train', 'validation', 'test')}
    assert not (records['train'] & records['validation'] or records['train'] & records['test'] or records['validation'] & records['test'])
    report['checks']['recording_split_disjoint'] = {key: len(value) for key, value in records.items()}
    model.train(); model.surface_bps_weight = .1
    prediction = model(noisy, t, known, control_mask=mask)
    loss = prediction.square().mean() + model.contact_logits(prediction, known).square().mean()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    assert torch.isfinite(norm)
    report['checks']['model_backward_finite'] = True
    report['status'] = 'passed'; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=HERE / 'runs/ego_exo_surface_continue_oct11/input_checks.json')
    args = parser.parse_args()
    cp = torch.load(HERE / 'runs/native_dynamic_scene20_contact_55k_oct07/last.pt', map_location='cpu', weights_only=False)
    run_checks(cp, args.output)
