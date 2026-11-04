"""Declared synthetic frontend estimates, distinct from clean AMASS labels.

The pilot simulates estimated native pose/shape parameters. This is a richer
input than the historical partial-joint experiment; real mesh-only exports
need a separately validated pose fit before using this motion initializer.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from scenemi.data.amass_dataset import six_columns
from scenemi.data.exo_corruption import correlated
from scenemi.geometry.body import PARENTS, encode_body
from scenemi.geometry.shape import numpy_profile
from scenemi.observations import INPUT_KEYS

INITIALIZATION_KEYS = ('initial_motion', 'initial_rest', 'initial_reliability',
                       'initial_root_reliability', 'estimate_available', 'joint_visible')
CONDITIONS = ('identity', 'tracking', 'occlusion', 'gap')


def rotations_from_motion(motion):
    six = motion[:, 3:135].reshape(-1, 22, 6).astype(np.float64)+[1, 0, 0, 0, 1, 0]
    first = six[..., :3]/np.linalg.norm(six[..., :3], axis=-1, keepdims=True)
    second = six[..., 3:]-(first*six[..., 3:]).sum(-1, keepdims=True)*first
    second /= np.linalg.norm(second, axis=-1, keepdims=True)
    return np.stack((first, second, np.cross(first, second)), -1)


def fill_estimate_gaps(rotations, root, available):
    """Interpolate estimated neighbors only; hold nearest estimate at ends."""
    frames = len(root)
    valid = np.flatnonzero(available)
    if len(valid)<2:
        raise ValueError('Initialization needs at least two tracked frames')
    query = np.clip(np.arange(frames), valid[0], valid[-1])
    filled_root = np.stack([np.interp(query, valid, root[valid, i]) for i in range(3)], -1)
    filled = np.stack([Slerp(valid, Rotation.from_matrix(rotations[valid, j]))(query).as_matrix()
                       for j in range(22)], 1)
    # Preserve provided matrices exactly; hidden payloads never enter Slerp.
    filled[available] = rotations[available]
    filled_root[available] = root[available]
    return filled, filled_root


def repair_example(data, rng, policy, *, case=None, condition=None):
    options = dict(temporal_dropout=None)
    if case is not None:
        options.update(member=case['member'], start=case['start'], reflected=False, yaw=0.)
    sample, identity = data.sample(rng, 'exo', **options)
    if case is not None and identity['variant_sha256'] != case['variant_sha256']:
        raise ValueError('Validation source identity changed')
    if condition is None:
        condition = str(rng.choice(CONDITIONS, p=policy['probabilities']))
    if condition not in CONDITIONS:
        raise ValueError('Unknown synthetic frontend condition')
    frames = len(sample['motion'])
    clean = condition == 'identity'
    # This is an explicit synthetic estimator, not an oracle inference path.
    # Perfect estimates occur only in the declared identity condition.
    beta_sigma = 0. if clean else policy['beta_sigma']
    estimated_betas = sample['target_betas']+rng.normal(0, beta_sigma, 16)
    rest, _, _ = numpy_profile(data.basis, estimated_betas)
    local = rotations_from_motion(sample['motion'])
    root = sample['motion'][:, :3].astype(np.float64)*2+sample['target_rest'][0]
    pose_sigma = 0. if clean else float(rng.uniform(*policy['pose_sigma_rad']))
    root_sigma = 0. if clean else float(rng.uniform(*policy['root_sigma_m']))
    errors = correlated(rng, frames, (22, 3), pose_sigma, policy['pose_rho'])
    root_error = correlated(rng, frames, (3,), root_sigma, policy['root_rho'])
    # Unobservable constant horizontal offsets are removed by the common anchor.
    root_error -= root_error[:1].copy()
    reliability = np.full((frames, 22), 1-min(pose_sigma/.12, 1.))
    root_reliability = np.full(frames, 1-min(root_sigma/.06, 1.))
    visible = np.ones((frames, 22), bool)
    available = np.ones(frames, bool)
    interval = None
    if condition in ('occlusion', 'gap'):
        width = int(rng.integers(policy['gap_frames'][0], policy['gap_frames'][1]+1))
        placement = str(rng.choice(['prefix', 'interior', 'suffix']))
        start = 0 if placement=='prefix' else frames-width if placement=='suffix' else int(rng.integers(1, frames-width))
        interval = dict(start=start, end=start+width, placement=placement)
        if condition == 'occlusion':
            groups = ((16, 18, 20), (17, 19, 21), (1, 4, 7, 10), (2, 5, 8, 11))
            joints = groups[int(rng.integers(len(groups)))]
            hidden_sigma = float(rng.uniform(*policy['occluded_pose_sigma_rad']))
            extra = correlated(rng, frames, (len(joints), 3), hidden_sigma, policy['pose_rho'])
            for k, j in enumerate(joints):
                errors[start:start+width, j] += extra[start:start+width, k]
                reliability[start:start+width, j] = 0.
                visible[start:start+width, j] = False
            interval['local_rotation_joints'] = list(joints)
        else:
            available[start:start+width] = False
            visible[start:start+width] = False
            reliability[start:start+width] = 0.
            root_reliability[start:start+width] = 0.
    estimated = Rotation.from_rotvec(errors.reshape(-1, 3)).as_matrix().reshape(frames, 22, 3, 3)@local
    estimated_root = root+root_error
    if not available.all():
        # Poison missing estimates before interpolation to prove no hidden fit
        # survives into the initializer or the allowed observation payload.
        estimated[~available] = np.nan
        estimated_root[~available] = np.nan
        estimated, estimated_root = fill_estimate_gaps(estimated, estimated_root, available)
    initial = encode_body(estimated, estimated_root, rest)
    anchor_frame = int(np.flatnonzero(available)[0])
    origin = estimated_root[anchor_frame].copy()
    origin[1] = 0.
    initial[:, :3] -= origin/2
    initial[:, 135:] -= np.tile(origin/2, 22)
    sample['motion'][:, :3] -= origin/2
    sample['motion'][:, 135:] -= np.tile(origin/2, 22)
    global_rotations = [estimated[:, 0]]
    global_reliability = [reliability[:, 0]]
    for j in range(1, 22):
        global_rotations.append(global_rotations[PARENTS[j]]@estimated[:, j])
        global_reliability.append(np.minimum(global_reliability[PARENTS[j]], reliability[:, j]))
    confidence = np.minimum(np.stack(global_reliability, 1), root_reliability[:, None])
    confidence = np.clip(confidence, .05, 1.)*available[:, None]
    sample['trajectory'][:] = 0.
    sample['trajectory'][..., :3] = initial[:, 135:].reshape(frames, 22, 3)*available[:, None, None]
    sample['trajectory'][..., 3:] = six_columns(np.stack(global_rotations, 1))*available[:, None, None]
    sample['observation_meta'][..., :4] = 0.
    sample['observation_meta'][..., 0] = available[:, None]
    sample['observation_meta'][..., 1] = available[:, None]
    sample['observation_meta'][..., 2] = confidence
    sample['observation_meta'][..., 3] = confidence
    sample['observation_uncertainty'][...] = 0.
    sample.update(initial_motion=initial, initial_rest=rest.astype(np.float32),
        initial_reliability=reliability.astype(np.float32),
        initial_root_reliability=root_reliability.astype(np.float32),
        estimate_available=available.astype(np.float32), joint_visible=visible.astype(np.float32))
    identity.update(frontend_condition=condition, frontend_estimate='synthetic_native_pose_and_betas',
        pose_sigma_rad=pose_sigma, root_sigma_m=root_sigma, beta_sigma=beta_sigma,
        estimated_anchor_shift_m=origin.tolist(), estimate_gap_or_occlusion=interval,
        reliability_from_realized_target_error=False, missing_estimates_interpolated_from_available_only=True)
    return sample, identity


def repair_payload(batch):
    return {key:batch[key] for key in INPUT_KEYS+INITIALIZATION_KEYS}
