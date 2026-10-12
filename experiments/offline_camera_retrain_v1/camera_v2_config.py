"""Executable sampling/loss/optimizer configuration for the first three tasks."""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.train_unified_continue import sampling_plan
from experiments.offline_camera_retrain_v1.camera_v2_model import NEW_PREFIXES
from experiments.offline_camera_retrain_v1.unified_losses import track_losses, SCALES

CONFIG_PATH = HERE / 'configs/ego_exo_camera_v2.json'


def read_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text())
    if config['version'] != 'ego_exo_camera_v2' or config['frames'] != 128 or config['fps'] != 20:
        raise ValueError('This stage uses fixed128 at20Hz')
    if config['scene']['architecture'] != 'unchanged_offline_surface_v1':
        raise ValueError('Scene changes are outside the current stage')
    if not np.isclose(sum(config['mode_probabilities'].values()), 1):
        raise ValueError('Mode probabilities must sum to one')
    if any(p <= 0 for p in config['mode_probabilities'].values()):
        raise ValueError('Invalid mode probability')
    if config['supervision']['scales'] != SCALES:
        raise ValueError('Configured physical scales differ from the implemented loss scales')
    return config


def effective_sampling_plan(data, config):
    old = sampling_plan(data, config['frames'])
    ego = old['group_probabilities']
    counts = old['valid_frames']
    denominator = counts['camera_wearer'] + counts['interactee']
    exo = {group: counts[group] / denominator for group in ('camera_wearer', 'interactee')}
    branches = {f'{group}/ego': config['mode_probabilities']['ego'] * p for group, p in ego.items()}
    branches.update({f'{group}/exo': config['mode_probabilities']['exo'] * p for group, p in exo.items()})
    assert np.isclose(sum(branches.values()), 1)
    return dict(branch_probabilities=branches, valid_frames=counts,
                ego_domain_probabilities=old['domain_probabilities'],
                weighting='ego domains sqrt covered duration; roles by covered duration; exo EgoBody only',
                exo_joint_policy='all22 subject to frozen first-hit availability/confidence',
                frames=128, mode_probabilities=config['mode_probabilities'])


def sample_item(data, builder, config, plan, rng):
    keys = list(plan['branch_probabilities'])
    branch = str(rng.choice(keys, p=list(plan['branch_probabilities'].values())))
    group, mode = branch.split('/')
    sample, identity = data.sample(config['frames'], group)
    if mode == 'exo':
        joints = tuple(range(22))
    else:
        joints = (15, 20, 21) if rng.random() < config['ego_head_wrists_probability'] else (15,)
    observed, scene, asset = builder.build(data, sample, identity, mode, joints=joints, noise_rng=rng)
    return (observed, scene, asset, mode), identity


def observation_losses(prediction, truth, batch, signal_weight):
    """Use supplied per-example mounts; no dataset/held-out-body fitting here."""
    safe = dict(batch)
    safe['observation_meta'] = batch['observation_meta'].clone()
    calibrations = {}
    for b in range(len(prediction)):
        valid = bool(batch['mount_valid'][b] > 0)
        if not valid:
            sensor = safe['observation_meta'][b, :, 15, 4] < .5
            safe['observation_meta'][b, sensor, 15, :4] = 0
        calibrations[str(b)] = dict(
            rotation=batch['mount_rotation'][b].detach().cpu().tolist() if valid else np.eye(3).tolist(),
            translation_m=batch['mount_translation_m'][b].detach().cpu().tolist() if valid else [0., 0., 0.],
            position_p90_m=float(batch['mount_uncertainty'][b, 0]) if valid else 0.,
            rotation_p90_rad=float(batch['mount_uncertainty'][b, 1]) if valid else 0.)
    return track_losses(prediction, truth, safe, signal_weight, calibrations,
                        [str(b) for b in range(len(prediction))])


def optimizer_groups(model, config):
    new, existing = [], []
    for name, parameter in model.named_parameters():
        if config['scene']['freeze_weights'] and name.startswith((
                'core.scene_embedding.', 'core.cond_process.', 'contact_head.', 'anchor_embedding.',
                'point_embedding.', 'surface_attention.', 'surface_projection.', 'surface_contact.')):
            parameter.requires_grad_(False)
        if parameter.requires_grad:
            (new if name.startswith(NEW_PREFIXES) else existing).append(parameter)
    return [dict(params=existing, lr=config['optimizer']['existing_lr'], name='existing'),
            dict(params=new, lr=config['optimizer']['new_lr'], name='camera_observation')]


def balanced_weights(rows, targets, caps):
    """TRAIN-only equal-mode average of within-mode median gradient ratios."""
    weights = {}
    for term, target in targets.items():
        modes = {row['branch'].split('/')[-1] for row in rows}
        medians = []
        for mode in sorted(modes):
            ratios = [row['terms'][term]['gradient_norm'] / max(row['base_gradient_norm'], 1e-12)
                      for row in rows if row['branch'].endswith('/' + mode)
                      and row['terms'][term]['gradient_norm'] > 1e-12]
            if ratios:
                medians.append(float(np.median(ratios)))
        if not medians:
            raise ValueError(f'No valid TRAIN gradient calibration for {term}')
        weights[term] = min(caps[term], target / float(np.mean(medians)))
    result = copy.deepcopy(rows)
    for row in result:
        for term, entry in row['terms'].items():
            entry['coefficient'] = weights[term]
            entry['weighted_gradient_ratio'] = weights[term] * entry['gradient_norm'] / max(row['base_gradient_norm'], 1e-12)
        for bucket in row['body_buckets'].values():
            for term, entry in bucket['terms'].items():
                entry['coefficient'] = weights[term]
                entry['weighted_gradient_ratio'] = weights[term] * entry['gradient_norm'] / max(row['base_gradient_norm'], 1e-12)
    return weights, result
