"""Synthetic tracking outages on the existing 30 Hz observation clock.

Only observation streams are removed. Motion/shape supervision and timestamps
are preserved. Missing intervals never change the sequence length or join two
unrelated recordings.
"""
import math

import numpy as np

PROTOCOL = 'temporal-observation-gaps-v1'


def validate_policy(policy, frames=128):
    expected = {'protocol', 'probabilities', 'gap_frames', 'placements', 'minimum_observed_frames'}
    if not isinstance(policy, dict) or set(policy) != expected or policy['protocol'] != PROTOCOL:
        raise ValueError('Expected an explicit temporal observation gap policy')
    probabilities = policy['probabilities']
    if (not isinstance(probabilities, dict) or set(probabilities) != {'ego', 'exo'}
            or any(isinstance(p, bool) or not isinstance(p, (int, float))
                   or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())):
        raise ValueError('Gap probabilities must be finite numbers in [0,1] for EGO and EXO')
    limits, minimum = policy['gap_frames'], policy['minimum_observed_frames']
    if (not isinstance(limits, (list, tuple)) or len(limits) != 2
            or any(type(x) is not int for x in limits)
            or type(minimum) is not int or not 1 <= minimum < frames
            or not 1 <= limits[0] <= limits[1] <= frames-minimum):
        raise ValueError('Gap lengths must leave the configured minimum observed frames')
    if policy['placements'] != ['prefix', 'interior', 'suffix'] or limits[1] > frames-2:
        raise ValueError('Use explicit prefix/interior/suffix outages with valid context')
    return policy


def sample_valid_frames(rng, mode, frames, policy=None):
    """Draw one outage or retain the clean window; all three placements occur."""
    valid = np.ones(frames, dtype=bool)
    if policy is None:
        return valid, dict(protocol='none', frame_ranges=[], observed_frames=frames)
    validate_policy(policy, frames)
    if rng.random() >= policy['probabilities'][mode]:
        return valid, dict(protocol=PROTOCOL, frame_ranges=[], observed_frames=frames)
    width = int(rng.integers(policy['gap_frames'][0], policy['gap_frames'][1]+1))
    placement = policy['placements'][int(rng.integers(3))]
    if placement == 'prefix':
        start = 0
    elif placement == 'suffix':
        start = frames-width
    else:
        start = int(rng.integers(1, frames-width))
    valid[start:start+width] = False
    return valid, dict(protocol=PROTOCOL, placement=placement,
                       frame_ranges=[[start, start+width-1]], observed_frames=int(valid.sum()))


def mask_observations(sample, valid):
    """Zero unavailable payloads as well as masks; leave all labels untouched."""
    hidden = ~np.asarray(valid, dtype=bool)
    for key in ('camera', 'camera_available', 'camera_confidence',
                'camera_intrinsics', 'camera_intrinsics_valid',
                'trajectory', 'observation_uncertainty'):
        sample[key][hidden] = 0
    sample['observation_meta'][hidden, :, :4] = 0
