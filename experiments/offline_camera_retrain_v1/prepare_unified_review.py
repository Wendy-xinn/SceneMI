"""Export every registered comparison and paired inference scene ablations."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.audit_unified_inputs import dataset
from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
from experiments.offline_camera_retrain_v1.scene_model import ddim_sample
from experiments.offline_camera_retrain_v1.unified_geometry import SkinAnchorCache
from experiments.offline_camera_retrain_v1.unified_model import UnifiedSceneMI
from experiments.offline_camera_retrain_v1.unified_observations import (
    ObservationBuilder, collate_observations, inference_payload)
from experiments.offline_camera_retrain_v1.unified_metrics import score


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def prepare(folder):
    torch.set_num_threads(4)
    evaluation = folder / 'evaluation'
    protocol = json.loads((evaluation / 'protocol.json').read_text())
    assert protocol['status'] == 'completed'
    assert file_hash(folder / 'last.pt') == protocol['checkpoint_sha256']
    state = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
    config = state['config']
    model = UnifiedSceneMI(config['latent_dim'], tuple(config['dim_mults']),
                           body_conditioning=True, contact_prediction=True).cuda().eval()
    model.load_state_dict(state['model'])
    model.surface_bps_weight = config['surface_bps_weight']
    del state
    data = dataset(config, 'validation', seed=protocol['seed'])
    builder = ObservationBuilder(SkinAnchorCache())
    mounts = json.loads((folder / 'TRAIN_camera_mounts.json').read_text())
    output = folder / 'visual_review'
    output.mkdir(exist_ok=True)
    groups = defaultdict(dict)
    for row in json.loads((evaluation / 'rows.json').read_text()):
        number = protocol['selected'].index(row['window'])
        groups[(number, row['mode'])][row['model']] = row
    manifest = dict(status='exporting', checkpoint_step=protocol['checkpoint_step'],
                    checkpoint_sha256=protocol['checkpoint_sha256'], seed=protocol['seed'],
                    ddim_steps=protocol['ddim_steps'], cases=[],
                    selection='All 18 mode/windows from the preregistered development panel; no selection by score.',
                    input_scene='Own-view offline observed static union and current observed dynamic points only.',
                    ablations={
                        'no_scene': 'use_scene=False: global ViT, legacy BPS and generated-surface branches disabled',
                        'global_off': 'Zero ViT output embedding; retain legacy BPS and generated-surface branches',
                        'attention_off': 'Zero surface_projection output only; retain generated body BPS and global ViT; contact gating is then disconnected from the denoiser'},
                    training='Finished 3000 steps; no new optimization; awaiting user visual review')
    ablation_rows = []
    for (number, mode), paired in sorted(groups.items()):
        row = paired['unified']; w = row['window']; identity = row['identity']
        sample, actual = data.sample(128, w['group'], sequence_index=w['sequence_index'], start_index=w['start_index'])
        assert actual == identity
        observed, scene, asset = builder.build(data, sample, identity, mode,
                                               joints=(15,) if mode == 'ego' else (15, 20, 21))
        batch = collate_observations([(observed, scene, asset, mode)])
        known = inference_payload(batch)
        mask = batch['observation_meta'][..., :2].amax(-1) > 0
        motions = {}; metrics = {}
        for label in ('original55k', 'unified'):
            with np.load(evaluation / f'{number:03d}_{mode}_{label}.npz') as saved:
                assert np.array_equal(saved['observation_meta'], observed['observation_meta'])
                assert np.array_equal(saved['trajectory'], observed['trajectory'])
                motions[label] = saved['motion'].copy()
            metrics[label] = paired[label]['metrics']
        if not manifest['cases']:
            calls = [0]
            def count_forward(module, inputs, result):
                calls[0] += 1
            handle = model.core.register_forward_hook(count_forward)
            try:
                repeated = ddim_sample(model, known, 128, steps=protocol['ddim_steps'],
                                       seed=protocol['seed'], control_mask=mask)
            finally:
                handle.remove()
            difference = float(np.max(np.abs(repeated[0].cpu().numpy() - motions['unified'])))
            assert difference < 1e-4, difference
            assert calls[0] == 2 * protocol['ddim_steps'], calls
            manifest['cached_generation_check'] = dict(max_motion_error=difference,
                                                       core_forwards=calls[0], ddim_steps=protocol['ddim_steps'])
        visible = None
        if mode == 'exo':
            frozen = builder.exo[identity['sequence_id']]
            start = int(np.searchsorted(frozen['source_frame_ids'], identity['source_start_30fps']))
            visible = np.asarray(frozen['position_available'][start:start + 128])
        for variant in ('no_scene', 'global_off', 'attention_off'):
            cached = output / f'{number:03d}_{mode}_{variant}_motion.npz'
            if cached.exists():
                with np.load(cached) as saved:
                    assert str(saved['checkpoint_sha256']) == protocol['checkpoint_sha256']
                    assert int(saved['seed']) == protocol['seed']
                    motion = torch.from_numpy(saved['motion']).cuda()[None]
            else:
                handle = None
                if variant == 'global_off':
                    handle = model.core.scene_embedding.register_forward_hook(lambda module, inputs, result: torch.zeros_like(result))
                elif variant == 'attention_off':
                    handle = model.surface_projection.register_forward_hook(lambda module, inputs, result: torch.zeros_like(result))
                try:
                    motion = ddim_sample(model, known, 128, steps=protocol['ddim_steps'],
                                         seed=protocol['seed'], control_mask=mask,
                                         use_scene=variant != 'no_scene')
                finally:
                    if handle is not None:
                        handle.remove()
                np.savez_compressed(cached, motion=motion[0].cpu().numpy(),
                                    checkpoint_sha256=protocol['checkpoint_sha256'], seed=protocol['seed'])
            values, joints = score(motion, batch, visible, mounts[w['group']])
            values['output_change_from_full_mm'] = float(np.linalg.norm(
                joints - np.load(evaluation / f'{number:03d}_{mode}_unified.npz')['joints'], axis=-1).mean() * 1000)
            metrics[variant] = values
            ablation_rows.append(dict(number=number, mode=mode, group=w['group'], variant=variant, metrics=values))
            if variant == 'no_scene':
                motions[variant] = motion[0].cpu().numpy()
        arrays = dict(camera=observed['camera'], observation_meta=observed['observation_meta'],
                      input_positions=np.where(observed['observation_meta'][..., :1] > 0,
                                               observed['trajectory'][..., :3] * 2, 0.),
                      static_points=scene.points[::max(1, int(np.ceil(len(scene.points) / 90000)))],
                      anchor_ids=asset['vertex_ids'])
        dynamics = scene.dynamic or [np.empty((0, 3), np.float32) for _ in range(128)]
        arrays['dynamic_offsets'] = np.r_[0, np.cumsum([len(x) for x in dynamics])]
        arrays['dynamic_points'] = np.concatenate(dynamics, axis=0)
        motions['gt'] = observed['motion']
        decode_checks = {}
        for label, motion in motions.items():
            vertices, faces, joints = decode_native_mesh(motion, identity['native_body'],
                                                         data.model(identity['native_body']['model'], identity['native_body']['gender']))
            arrays[label + '_vertices'] = vertices.astype(np.float32)
            arrays[label + '_joints'] = joints.astype(np.float32)
            arrays['faces'] = faces.astype(np.uint32)
            if label in ('original55k', 'unified'):
                expected = np.load(evaluation / f'{number:03d}_{mode}_{label}.npz')['joints']
            elif label == 'gt':
                expected = observed['joints'] * 2
            else:
                from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
                expected = forward_kinematics(torch.from_numpy(motion).cuda()[None], batch['rest'])[0].cpu().numpy()
            error = float(np.max(np.abs(joints - expected)))
            assert error < 2e-5, (number, mode, label, error)
            decode_checks[label] = error
        filename = f'{number:03d}_{mode}.npz'
        np.savez_compressed(output / filename, **arrays)
        manifest['cases'].append(dict(number=number, mode=mode, group=w['group'], file=filename,
                                      identity=identity, metrics=metrics, mesh_joint_max_errors_m=decode_checks))
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (output / 'ablation_rows.json').write_text(json.dumps(ablation_rows, indent=2) + '\n')
        print('REVIEW', number, mode, {k: round(v['world_unaligned_mpjpe_mm'], 2) for k, v in metrics.items()}, flush=True)
    manifest['status'] = 'completed'
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    grouped = defaultdict(list)
    for case in manifest['cases']:
        for label, values in case['metrics'].items():
            grouped[(case['mode'], label)].append(values)
    summary = {}
    for (mode, label), values in grouped.items():
        summary[f'{mode}/{label}'] = {key: float(np.mean([x[key] for x in values if x.get(key) is not None]))
                                    if any(x.get(key) is not None for x in values) else None for key in values[0]}
        summary[f'{mode}/{label}']['windows'] = len(values)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('COMPLETED', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, default=HERE / 'runs/ego_exo_surface_continue_oct11')
    prepare(parser.parse_args().folder)
