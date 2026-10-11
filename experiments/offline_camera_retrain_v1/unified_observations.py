"""Frozen, label-free ego/exo inference payloads and physical-time derivatives."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import matrix_to_axis_angle

from experiments.offline_camera_retrain_v1.data import HERE, anchor_rotation, rotation6d
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D, rotation_from_6d
from experiments.offline_camera_retrain_v1.unified_geometry import (
    OfflineSceneCache, ObservedSurfaceWindow, occupancy_from_points)

FPS = 20.
INPUT_KEYS = ('camera', 'trajectory', 'observation_meta', 'occupancy', 'bps',
              'bps_valid', 'rest', 'body_type', 'body_scale')


def inference_payload(batch):
    """No motion/joints/contact/GT initial pose can enter the model interface."""
    output = {key: batch[key] for key in INPUT_KEYS}
    for key in ('surface_windows', 'skin_assets', 'mode'):
        output[key] = batch[key]
    return output


def observation_derivatives(trajectory, meta):
    """Backward differences; unavailable stencils never bridge a gap.

    Position is transported as m/2, derivatives are m/s, m/s² and rad/s.
    The task is offline; backward differencing is merely a stencil choice.
    """
    pm = (meta[..., 0] > 0) & torch.isfinite(trajectory[..., :3]).all(-1)
    rm = (meta[..., 1] > 0) & torch.isfinite(trajectory[..., 3:]).all(-1)
    pos = torch.where(pm[..., None], trajectory[..., :3] * 2, 0.)
    rot6 = torch.where(rm[..., None], trajectory[..., 3:],
                       trajectory.new_tensor(IDENTITY_6D))
    rot = rotation_from_6d(rot6 - trajectory.new_tensor(IDENTITY_6D))
    velocity, acceleration, angular = [torch.zeros_like(pos) for _ in range(3)]
    vm, am, wm = [torch.zeros_like(pm) for _ in range(3)]
    vm[:, 1:] = pm[:, 1:] & pm[:, :-1]
    velocity[:, 1:] = (pos[:, 1:] - pos[:, :-1]) * FPS
    am[:, 2:] = vm[:, 2:] & vm[:, 1:-1]
    acceleration[:, 2:] = (velocity[:, 2:] - velocity[:, 1:-1]) * FPS
    wm[:, 1:] = rm[:, 1:] & rm[:, :-1]
    angular[:, 1:] = matrix_to_axis_angle(rot[:, 1:] @ rot[:, :-1].transpose(-1, -2)) * FPS
    values = [torch.where(m[..., None], v, 0.) for v, m in
              zip((velocity, acceleration, angular), (vm, am, wm))]
    return (*values, torch.stack((vm, am, wm), -1))


class ObservationBuilder:
    def __init__(self, skin_cache, exo_root=None):
        self.skin_cache = skin_cache
        self.exo_root = Path(exo_root or HERE / 'data/egobody_exo_offline_v2')
        self.scenes = OfflineSceneCache()
        self.exo = {}

    def build(self, data, sample, identity, mode='ego', joints=(15,), noise_rng=None):
        """Simulation happens here; model receives the whitelisted output only.

        Native shape is explicitly known calibration in this controlled input
        experiment. Visibility/confidence are independent of missing labels.
        """
        sample = {key: value.copy() for key, value in sample.items()}
        frames = len(sample['camera'])
        if mode == 'exo':
            if identity['group'] not in ('camera_wearer', 'interactee'):
                raise ValueError('First exo experiment is EgoBody only')
            name = identity['sequence_id']; folder = self.exo_root / data.split / name
            if name not in self.exo:
                metadata = json.loads((folder / 'metadata.json').read_text())
                if metadata['split'] != data.split or metadata['sequence_id'] != name:
                    raise ValueError('Exo split/identity mismatch')
                self.exo[name] = {key: np.load(folder / f'{key}.npy', mmap_mode='r') for key in
                                 ('source_frame_ids', 'visible_joints_world', 'position_available',
                                  'position_confidence', 'static_visible_points_world', 'sensor_to_world')}
            obs = self.exo[name]; ids = obs['source_frame_ids']
            start = int(np.searchsorted(ids, identity['source_start_30fps']))
            expected = identity['source_start_30fps'] + np.arange(frames) * 1.5
            if not np.allclose(ids[start:start + frames], expected):
                raise ValueError('Exo/native timestamp mismatch')
            sensor = obs['sensor_to_world']; origin = sensor[:3, 3]
            anchor = anchor_rotation(np.broadcast_to(sensor[:3, :3], (frames, 3, 3)))
            # Labels are re-anchored here, never consulted by the forward pass.
            pose, trans, rest, world_joints, _ = data._native(name, identity['group'])
            first = data.prepared[name].get('source_first_frame', 0)
            ix = int(round((identity['source_start_30fps'] - first) / 1.5))
            local = Rotation.from_rotvec(pose[ix:ix + frames].reshape(-1, 3)).as_matrix().reshape(frames, 22, 3, 3)
            local[:, 0] = anchor[None] @ local[:, 0]
            target_joints = (world_joints[ix:ix + frames] - origin) @ anchor.T
            transl = ((trans[ix:ix + frames] + rest[0] - origin) @ anchor.T - rest[0]) / 2
            sample['motion'] = np.concatenate((transl, (rotation6d(local) - np.array(IDENTITY_6D)).reshape(frames, 132), target_joints.reshape(frames, 66) / 2), -1).astype(np.float32)
            sample['joints'] = (target_joints / 2).astype(np.float32)
            camera_rotation = anchor @ sensor[:3, :3]
            sample['camera'] = np.broadcast_to(np.r_[np.zeros(3), camera_rotation[:, 0], camera_rotation[:, 1]], (frames, 9)).astype(np.float32).copy()
            available = np.asarray(obs['position_available'][start:start + frames], bool)
            confidence = np.asarray(obs['position_confidence'][start:start + frames])
            trajectory = np.zeros((frames, 22, 9), np.float32)
            trajectory[..., :3] = ((np.asarray(obs['visible_joints_world'][start:start + frames]) - origin) @ anchor.T) / 2
            meta = np.zeros((frames, 22, 5), np.float32)
            meta[..., 0] = available; meta[..., 2] = confidence; meta[..., 4] = 1
            points = (np.asarray(obs['static_visible_points_world']) - origin) @ anchor.T
            scene = ObservedSurfaceWindow(points)
            sample['occupancy'] = occupancy_from_points(points)
            sample['bps'] = np.zeros((frames, 67, 3), np.float32)
            sample['bps_valid'] = np.zeros((frames, 67), bool)
        elif mode == 'ego':
            trajectory = sample['trajectory'].copy()
            meta = np.ones((frames, 22, 5), np.float32); meta[:, 15, 4] = 0
            scene = self.scenes.ego(identity, frames)
            sample['occupancy'] = occupancy_from_points(scene.points)
        else:
            raise ValueError(mode)
        selected = np.zeros(22, bool); selected[list(joints)] = True
        meta[..., :4] *= selected[None, :, None]
        # Mild coherent estimator perturbations plus true missing intervals.
        # Their masks are frozen before the model; no hidden GT is transported.
        if noise_rng is not None:
            if noise_rng.random() < .5:
                phase = np.linspace(0, 1, frames)[:, None, None]
                drift = noise_rng.normal(0, .01, (1, 22, 3)) + phase * noise_rng.normal(0, .02, (1, 22, 3))
                trajectory[..., :3] += drift / 2
                meta[..., 2] *= np.exp(-np.linalg.norm(drift, axis=-1) / .05)
            if noise_rng.random() < .3:
                joint = int(noise_rng.choice(joints)); first = int(noise_rng.integers(max(1, frames - 12)))
                meta[first:first + 12, joint, :4] = 0
        trajectory[..., :3] = np.where(meta[..., 0:1] > 0, trajectory[..., :3], 0)
        trajectory[..., 3:] = np.where(meta[..., 1:2] > 0, trajectory[..., 3:], 0)
        sample.update(trajectory=trajectory, observation_meta=meta)
        return sample, scene, self.skin_cache.get(identity['native_body'])


def collate_observations(items, device='cuda'):
    from experiments.offline_camera_retrain_v1.data import collate
    samples, scenes, assets, modes = zip(*items)
    batch = {key: value.to(device) for key, value in collate(samples).items()}
    batch.update(surface_windows=list(scenes), skin_assets=list(assets), mode=list(modes))
    return batch
