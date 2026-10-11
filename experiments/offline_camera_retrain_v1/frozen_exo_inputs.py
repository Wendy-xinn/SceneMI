"""Deployable frozen EXO input loader: no body labels or ego assets required."""
from pathlib import Path
import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import anchor_rotation
from experiments.offline_camera_retrain_v1.unified_geometry import (
    SkinAnchorCache, ObservedSurfaceWindow, occupancy_from_points)


def load_frozen_exo(folder, body_profile, *, start=0, frames=128,
                    joints=(15, 20, 21), skin_cache=None, device='cuda'):
    """Calibration profile supplies model/gender/shape/scale, never body pose.

    The current controlled experiment uses known native shape. A real frontend
    can populate the same observation arrays/profile with its own estimates.
    Scene and trajectory must already share the calibrated metric world frame.
    """
    folder = Path(folder)
    cache = skin_cache or SkinAnchorCache(device=device)
    asset = cache.get(body_profile)
    ids = np.load(folder / 'source_frame_ids.npy', mmap_mode='r')
    if start < 0 or frames not in (64, 128, 192) or start + frames > len(ids):
        raise ValueError('Invalid frozen observation interval')
    if not np.allclose(np.diff(ids[start:start + frames]), 1.5):
        raise ValueError('Require exact20 frame clock')
    transform = np.load(folder / 'sensor_to_world.npy')
    if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4) or np.linalg.det(transform[:3, :3]) < .999:
        raise ValueError('Invalid sensor calibration rotation')
    origin = transform[:3, 3]
    anchor = anchor_rotation(np.broadcast_to(transform[:3, :3], (frames, 3, 3)))
    available = np.asarray(np.load(folder / 'position_available.npy', mmap_mode='r')[start:start + frames], bool)
    confidence = np.asarray(np.load(folder / 'position_confidence.npy', mmap_mode='r')[start:start + frames])
    observed = np.asarray(np.load(folder / 'visible_joints_world.npy', mmap_mode='r')[start:start + frames])
    if not np.isfinite(observed[available]).all() or not np.isfinite(confidence).all():
        raise ValueError('Nonfinite available observation')
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError('Confidence must be in [0,1]')
    selected = np.zeros(22, bool); selected[list(joints)] = True
    available = available & selected[None]
    trajectory = np.zeros((frames, 22, 9), np.float32)
    trajectory[..., :3] = np.where(available[..., None], ((observed - origin) @ anchor.T) / 2, 0)
    meta = np.zeros((frames, 22, 5), np.float32)
    meta[..., 0] = available; meta[..., 2] = np.where(available, confidence, 0); meta[..., 4] = 1
    points = (np.asarray(np.load(folder / 'static_visible_points_world.npy')) - origin) @ anchor.T
    scene = ObservedSurfaceWindow(points)
    r = anchor @ transform[:3, :3]
    arrays = dict(camera=np.broadcast_to(np.r_[np.zeros(3), r[:, 0], r[:, 1]], (frames, 9)).astype(np.float32).copy(),
                  trajectory=trajectory, observation_meta=meta, occupancy=occupancy_from_points(points),
                  bps=np.zeros((frames, 67, 3), np.float32), bps_valid=np.zeros((frames, 67), bool),
                  rest=(asset['joints'][:22] * asset['scale']).detach().cpu().numpy(),
                  body_type=np.eye(2, dtype=np.float32)[int(body_profile['model'] == 'smplx')],
                  body_scale=np.array([body_profile['scale']], np.float32))
    result = {key: torch.from_numpy(np.ascontiguousarray(value))[None].to(device) for key, value in arrays.items()}
    result.update(surface_windows=[scene], skin_assets=[asset], mode=['exo'])
    return result
