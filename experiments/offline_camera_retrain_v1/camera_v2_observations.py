"""Explicit camera/target relation, intrinsics and supplied mount calibration."""
import json
from pathlib import Path

import numpy as np
from smplx.lbs import blend_shapes
import torch

from experiments.offline_camera_retrain_v1.data import PROJECTS
from experiments.offline_camera_retrain_v1.unified_observations import (
    ObservationBuilder, inference_payload)

CAMERA_KEYS = ('camera_available', 'camera_confidence', 'camera_intrinsics',
               'camera_intrinsics_valid', 'camera_context', 'mount_rotation',
               'mount_translation_m', 'mount_valid', 'mount_uncertainty')


def inference_payload_v2(batch):
    output = inference_payload(batch)
    output.update({key: batch[key] for key in CAMERA_KEYS})
    return output


def validate_mount(calibration):
    r = np.asarray(calibration['rotation'], float)
    t = np.asarray(calibration['translation_m'], float)
    if r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ValueError('Mount must have finite SO(3) rotation and metre translation')
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1., atol=1e-5):
        raise ValueError('Mount rotation is not in SO(3)')
    for key in ('position_p90_m', 'rotation_p90_rad'):
        if not np.isfinite(calibration[key]) or calibration[key] < 0:
            raise ValueError('Invalid mount uncertainty')


def master_intrinsics():
    calibration = json.loads((PROJECTS / 'egobody/kinect_cam_params/kinect_master/Color.json').read_text())
    k = np.asarray(calibration['camera_mtx'])
    return np.array([k[0, 0] / 1920, k[1, 1] / 1080,
                     k[0, 2] / 1920, k[1, 2] / 1080], np.float32)


def load_frozen_exo_v2(folder, body_profile, *, start=0, frames=128, skin_cache=None, device='cuda'):
    """No body motion labels: existing observed arrays + known body profile."""
    from experiments.offline_camera_retrain_v1.frozen_exo_inputs import load_frozen_exo
    result = load_frozen_exo(folder, body_profile, start=start, frames=frames,
                             joints=tuple(range(22)), skin_cache=skin_cache, device=device)
    arrays = dict(camera_available=np.ones((frames, 2), np.float32),
                  camera_confidence=np.ones((frames, 2), np.float32),
                  camera_intrinsics=np.broadcast_to(master_intrinsics(), (frames, 4)).copy(),
                  camera_intrinsics_valid=np.ones(frames, np.float32),
                  camera_context=np.array([0., 1., 1., 0., 0., 1.], np.float32),
                  mount_rotation=np.eye(3, dtype=np.float32),
                  mount_translation_m=np.zeros(3, np.float32),
                  mount_valid=np.array(0., np.float32), mount_uncertainty=np.zeros(2, np.float32))
    result.update({key: torch.from_numpy(value)[None].to(device) for key, value in arrays.items()})
    return result


class CameraObservationBuilder(ObservationBuilder):
    def __init__(self, skin_cache, mounts, exo_root=None):
        super().__init__(skin_cache, exo_root)
        for calibration in mounts.values():
            validate_mount(calibration)
            if calibration.get('fitting_split') != 'train':
                raise ValueError('Dataset default mounts must be frozen TRAIN calibrations')
        self.mounts = mounts
        self.intrinsics_cache = {}
        self.height_cache = {}

    def build(self, data, sample, identity, mode='ego', joints=None, noise_rng=None,
              mount_override=None):
        joints = tuple(range(22)) if mode == 'exo' and joints is None else tuple(joints or (15,))
        prepared, scene, asset = super().build(data, sample, identity, mode, joints, noise_rng)
        frames = len(prepared['camera'])
        if mode == 'exo':
            key = 'master_intrinsics'
            if key not in self.intrinsics_cache:
                self.intrinsics_cache[key] = master_intrinsics()
            intrinsics = np.broadcast_to(self.intrinsics_cache[key], (frames, 4)).copy()
            context = [0., 1., 1., 0., 0., 1.]
            calibration = None  # External camera has no target head mount.
            camera_available = np.ones((frames, 2), np.float32)
            camera_confidence = np.ones((frames, 2), np.float32)
        else:
            folder = Path(identity['scene_bundle'])
            if str(folder) not in self.intrinsics_cache:
                ids = np.load(folder / 'source_frame_ids.npy', mmap_mode='r')
                path = folder / 'camera_intrinsics_128x96.npy'
                if path.exists():
                    values = np.load(path, mmap_mode='r')
                elif identity['group'] == 'trumans':
                    # This exporter uses render_scene's declared fixed FOV,
                    # rather than a saved per-frame intrinsics array.
                    import inspect
                    from experiments.offline_camera_retrain_v1.scene_visibility_v2 import render_scene
                    signature = inspect.signature(render_scene)
                    fx = 64 / np.tan(np.deg2rad(signature.parameters['hfov'].default / 2))
                    fy = 48 / np.tan(np.deg2rad(signature.parameters['vfov'].default / 2))
                    values = np.broadcast_to([fx, fy, 64, 48], (len(ids), 4))
                else:
                    raise FileNotFoundError(path)
                self.intrinsics_cache[str(folder)] = (ids, values)
            ids, values = self.intrinsics_cache[str(folder)]
            expected = identity['source_start_30fps'] + np.arange(frames) * 1.5
            ix = np.searchsorted(ids, expected)
            if np.any(ix >= len(ids)) or not np.allclose(ids[ix], expected, atol=1e-6):
                raise ValueError('Camera intrinsics timestamp mismatch')
            intrinsics = np.asarray(values[ix], np.float32) / np.array([128, 96, 128, 96], np.float32)
            real = identity['group'] == 'camera_wearer'
            context = [1., 0., float(real), float(not real), 1., -1.]
            calibration = mount_override or self.mounts[identity['group']]
            validate_mount(calibration)
            # This is the SAME sensor stream as the self head-camera slot.
            # Keep its perturbations/dropout/confidence coupled; never let a
            # second, clean copy bypass the observation augmentation.
            prepared['camera'] = prepared['trajectory'][:, 15].copy()
            camera_available = prepared['observation_meta'][:, 15, :2].copy()
            camera_confidence = prepared['observation_meta'][:, 15, 2:4].copy()
        prepared.update(
            camera_available=camera_available,
            camera_confidence=camera_confidence,
            camera_intrinsics=intrinsics.astype(np.float32),
            camera_intrinsics_valid=np.ones(frames, np.float32),
            camera_context=np.asarray(context, np.float32),
            mount_rotation=np.asarray(calibration['rotation'] if calibration else np.eye(3), np.float32),
            mount_translation_m=np.asarray(calibration['translation_m'] if calibration else np.zeros(3), np.float32),
            mount_valid=np.array(float(calibration is not None), np.float32),
            mount_uncertainty=np.array([calibration['position_p90_m'], calibration['rotation_p90_rad']]
                                       if calibration else [0., 0.], np.float32))
        if not np.isfinite(intrinsics).all() or np.any(intrinsics[:, :2] <= 0):
            raise ValueError('Invalid camera intrinsics')
        return prepared, scene, asset

    def body_height(self, info):
        """Declared native shape height, not estimated from a posed head track."""
        key = (info['model'], info['gender'], tuple(info['betas']), info['scale'])
        if key not in self.height_cache:
            self.skin_cache.get(info)
            model = self.skin_cache.models[key[:2]]
            beta = torch.tensor(info['betas'], dtype=torch.float32)[None]
            with torch.no_grad():
                vertices = model.v_template[None] + blend_shapes(beta, model.shapedirs[:, :, :beta.shape[-1]])
            self.height_cache[key] = float((vertices[0, :, 1].max() - vertices[0, :, 1].min()) * info['scale'])
        return self.height_cache[key]
