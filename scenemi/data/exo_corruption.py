"""Reproducible synthetic tracking errors, distinct from clean motion labels."""
import numpy as np


def correlated(rng, frames, shape, sigma, rho):
    value = rng.normal(size=(frames, *shape))
    value[0] *= sigma
    innovation = sigma*np.sqrt(1-rho*rho)
    for frame in range(1, frames):
        value[frame] = rho*value[frame-1]+innovation*value[frame]
    return value


def corrupt_exo(sample, rng, policy, *, force=False):
    """Modify only observed positions/meta/uncertainty; units are meters.

    Confidence is drawn from the declared noise level, never from realized
    target error. The shared root error starts at zero, then varies smoothly;
    arbitrary constant global offsets are not identifiable without an anchor.
    """
    if sample['camera_context'][0] != 0:
        raise ValueError('EXO corruption must not affect EGO cameras')
    noisy = force or rng.random() >= policy['clean_probability']
    if not noisy:
        return dict(kind='clean_masked', local_sigma_m=0., root_drift_sigma_m=0.)
    frames = len(sample['trajectory'])
    local = float(rng.uniform(*policy['joint_sigma_m']))
    drift = float(rng.uniform(*policy['root_drift_sigma_m']))
    joint_error = correlated(rng, frames, (22, 3), local, policy['joint_rho'])
    root = correlated(rng, frames, (1, 3), drift, policy['root_rho'])
    root -= root[:1].copy()
    root *= np.linspace(0, 1, frames)[:, None, None]
    present = sample['observation_meta'][..., 0] > 0
    positions = sample['trajectory'][..., :3]
    positions[:] = np.where(present[..., None], positions+(joint_error+root)/2, 0.)
    declared = float(np.hypot(local, drift))
    sample['observation_meta'][..., 2] = present*np.clip(1/(1+declared/.05), .2, 1.)
    sample['observation_uncertainty'][..., 0] = present*declared
    return dict(kind='correlated_joint_and_world_root', local_sigma_m=local,
                root_drift_sigma_m=drift, confidence_uses_realized_error=False,
                synthetic_camera_depth_axis=False)
