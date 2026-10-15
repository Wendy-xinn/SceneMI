"""Isolated Native30 candidates; importing this module changes no live code.

The two switches under config['candidate'] default to False.  Hook installation
is explicit and affects only the calling process, preserving frozen root files.
"""
import math

import torch

from experiments.offline_camera_retrain_v1.native30_gt_reliability import gt_weight
from experiments.offline_camera_retrain_v1.native30_observations import sanitize_observations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d


DEFAULT = dict(root_reference_cap_enabled=False, root_rotation_confidence_cap=.05,
               body_local_foot_enabled=False, body_local_foot_coefficient=1.,
               raw_body_local_foot_weight=.5)


def candidate_options(config):
    supplied = config.get('candidate', {})
    unknown = set(supplied) - set(DEFAULT)
    if unknown:
        raise ValueError('Unknown isolated candidate options: ' + ', '.join(sorted(unknown)))
    options = dict(DEFAULT, **supplied)
    for name in ('root_reference_cap_enabled', 'body_local_foot_enabled'):
        if not isinstance(options[name], bool):
            raise ValueError(name + ' must be boolean')
    for name in ('root_rotation_confidence_cap', 'body_local_foot_coefficient',
                 'raw_body_local_foot_weight'):
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid candidate scalar: ' + name)
    if options['root_rotation_confidence_cap'] > 1 or options['raw_body_local_foot_weight'] > 1:
        raise ValueError('Candidate confidence and raw weight must be at most one')
    return options


def conservative_root_reference(known, inferred_rotation_confidence=.05):
    """Cap guessed pelvis heading confidence; retain directly evidenced roots.

    This changes confidence only, preserving the complete motion estimate.
    A direct root rotation or observable torso fit retains its original value.
    No body/contact target is accessed.
    """
    if not math.isfinite(inferred_rotation_confidence) or not 0 <= inferred_rotation_confidence <= 1:
        raise ValueError('Expected finite confidence in [0,1]')
    _, meta = sanitize_observations(known['trajectory'], known['observation_meta'])
    direct = (meta[:, :, 0, 1] > 0) & (meta[:, :, 0, 3] > 0)
    torso = known['torso_anchor_confidence'] > 0
    confidence = known['reference_confidence'].clone()
    current = confidence[:, :, 0, 1]
    confidence[:, :, 0, 1] = torch.where(direct | torso, current,
        current.clamp_max(inferred_rotation_confidence))
    return dict(known, reference_confidence=confidence)


def body_local_foot_position(prediction, truth, batch, signal):
    """Per-foot root-frame position error in m^2, gated by alpha_bar.

    Each foot keeps its mean pose and exact timing.  GT is detached.  Soft GT
    weights attenuate using the full denominator, matching the base pose loss.
    Root orientation reliability is required because it defines these axes.
    """
    if prediction.shape != truth.shape or prediction.ndim != 3 or prediction.shape[-1] != 201:
        raise ValueError('Expected matching [B,T,201] native motion')
    if signal.numel() != len(prediction):
        raise ValueError('Expected one alpha_bar value per sequence')

    def local(motion):
        joints = forward_kinematics(motion, batch['rest'])
        root = rotation_from_6d(motion[..., 3:9])
        relative = joints[:, :, (10, 11)] - joints[:, :, :1]
        return (root.transpose(-1, -2)[:, :, None] @ relative[..., None]).squeeze(-1)

    predicted = local(prediction)
    target = local(truth.detach()).detach()
    quality = gt_weight(batch, 'gt_position_weight', prediction).detach()
    rotation_quality = gt_weight(batch, 'gt_global_rotation_weight', prediction).detach()
    mask = torch.minimum(quality[:, :, (10, 11)], quality[:, :, :1])
    mask = torch.minimum(mask, rotation_quality[:, :, :1])
    error = (predicted - target).square().sum(-1) * mask
    return (error.flatten(1).mean(1) * signal.detach().reshape(-1)).mean()


def objective_with_body_local(original_objective, prediction, truth, batch, signal,
                              details, config, observation_weights, *,
                              coefficient=1., raw_weight=.5):
    """Preserve existing objective and add independently logged final/raw terms."""
    total, terms = original_objective(prediction, truth, batch, signal, details,
                                      config, observation_weights)
    if coefficient == 0:
        return total, terms
    final = body_local_foot_position(prediction, truth, batch, signal)
    raw = (body_local_foot_position(details['raw_motion'], truth, batch, signal)
           if raw_weight else prediction.sum() * 0)
    extra = coefficient * (final + raw_weight * raw)
    base = dict(terms['base'])
    base.update(candidate_body_local_foot_m2=final,
                candidate_raw_body_local_foot_m2=raw,
                candidate_body_local_foot_total=extra,
                total=base['total'] + extra)
    return total + extra, dict(terms, base=base)


def _original(function):
    # Unwrap our own prior installation, including after module reload.
    while '_native30_final_review_original' in getattr(function, '__dict__', {}):
        function = function.__dict__['_native30_final_review_original']
    return function


def install_hooks(config):
    """Install candidate switches in trainer and validation, without stacking.

    Reinstalling with all switches False restores the original function objects.
    Options are captured at installation; use the same config for preparation,
    training, checkpoints and evaluation to retain experiment provenance.
    """
    from experiments.offline_camera_retrain_v1 import train_native30 as trainer
    from experiments.offline_camera_retrain_v1 import native30_validation as validation

    options = candidate_options(config)
    for module in (trainer, validation):
        original = _original(module.model_inputs)
        if options['root_reference_cap_enabled']:
            def inputs(active_config, batch, _original=original):
                known = _original(active_config, batch)
                # A supplied frontend root is a different evidence source;
                # this candidate changes only the built-in track initializer.
                if 'reference_confidence' in known and 'reference_motion' not in batch:
                    known = conservative_root_reference(known, options['root_rotation_confidence_cap'])
                return known
            inputs._native30_final_review_original = original
            module.model_inputs = inputs
        else:
            module.model_inputs = original

    original_objective = _original(trainer.objective)
    if options['body_local_foot_enabled']:
        def objective(prediction, truth, batch, signal, details, active_config, observation_weights):
            return objective_with_body_local(original_objective, prediction, truth, batch,
                signal, details, active_config, observation_weights,
                coefficient=options['body_local_foot_coefficient'],
                raw_weight=options['raw_body_local_foot_weight'])
        objective._native30_final_review_original = original_objective
        trainer.objective = objective
    else:
        trainer.objective = original_objective
    return options
