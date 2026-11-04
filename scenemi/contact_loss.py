"""Source-reliability weights for future scene contact objectives.

Ground-truth skin/support evidence is a loss label, never an inference input.
The running prior and its sealed losses remain unchanged. Swing motion keeps
all motion supervision; attraction and planting alone receive these weights.
"""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from scenemi.config import ROOT
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


DEFAULT_POLICY = ROOT / 'configs/scene_contact_reliability.yaml'


def load_policy(path=DEFAULT_POLICY):
    policy = yaml.safe_load(Path(path).read_text())
    source, loss = policy['source'], policy['loss']
    if (policy['version'] != 'source-contact-reliability-v1'
            or policy['source_evidence_is_model_input'] is not False
            or policy['prediction_changes_source_weights'] is not False
            or policy['fps'] != 30
            or not 0.5 < source['minimum_stance_probability'] <= 1
            or not 0 <= source['full_weight_gap_m'] < source['zero_weight_gap_m']
            or not 0 <= source['full_weight_speed_mps'] < source['zero_weight_speed_mps']
            or not 0 < source['minimum_supported_fraction'] <= 1
            or loss['reduction'] != 'nominal_stance_count'
            or loss['temporal_weight'] != 'minimum_of_both_endpoints'
            or loss['unknown_surface_weight'] != 0
            or loss['preserve_swing_motion_supervision'] is not True
            or min(loss['distance_scale_m'], loss['sliding_scale_mps']) <= 0
            or min(loss['distance_tolerance_m'], loss['sliding_tolerance_mps']) < 0):
        raise ValueError('Invalid detached, per-foot source-contact policy')
    return policy


def contact_reliability(probability, sole_gap, supported_fraction, speed, frame_valid, policy):
    """Independent weights for each frame/foot, without any predicted quantity."""
    probability, sole_gap, supported_fraction, speed = (
        np.asarray(x, np.float32) for x in (probability, sole_gap, supported_fraction, speed))
    shape = probability.shape
    frame_valid = np.asarray(frame_valid, bool)
    if (len(shape) != 2 or shape[1] != 2 or frame_valid.shape != shape[:1]
            or any(x.shape != shape for x in (sole_gap, supported_fraction, speed))):
        raise ValueError('Contact evidence clocks/left-right dimensions differ')
    for value in (probability, supported_fraction):
        finite = value[np.isfinite(value)]
        if ((finite < 0) | (finite > 1)).any():
            raise ValueError('Source probabilities/fractions must be in [0,1]')
    if (speed[np.isfinite(speed)] < 0).any():
        raise ValueError('Foot speed must be nonnegative')
    s = policy['source']
    stance = np.isfinite(probability) & (probability >= s['minimum_stance_probability']) & frame_valid[:, None]
    finite = np.isfinite(sole_gap) & np.isfinite(speed) & np.isfinite(supported_fraction)
    gap_weight = np.clip((s['zero_weight_gap_m']-np.abs(sole_gap)) /
                         (s['zero_weight_gap_m']-s['full_weight_gap_m']), 0, 1)
    speed_weight = np.clip((s['zero_weight_speed_mps']-speed) /
                           (s['zero_weight_speed_mps']-s['full_weight_speed_mps']), 0, 1)
    supported = supported_fraction >= s['minimum_supported_fraction']
    weight = np.where(stance & finite & supported, probability*gap_weight*speed_weight, 0)
    return dict(weight=weight.astype(np.float32), stance=stance,
                source_geometry_valid=finite, source_support_agrees=supported)


def _weights(weight, like):
    weight = torch.as_tensor(weight, device=like.device, dtype=like.dtype).detach()
    if not bool(torch.isfinite(weight).all()) or bool(((weight < 0) | (weight > 1)).any()):
        raise ValueError('Contact loss weights must be finite, detached and in [0,1]')
    return weight


def contact_terms(signed_gap, relative_velocity, *, weight, stance, distance_valid,
                  velocity_valid, signal, policy):
    """Observed support-plane gap and surface-relative sliding objectives.

    The caller must explicitly mark valid observed support queries. No nearest
    riser, absent point neighbourhood or unobserved surface may count as support.
    Nominal denominators keep uncertain batches weaker instead of re-amplifying
    the few remaining trusted labels to full strength.
    """
    weight = _weights(weight, signed_gap)
    stance = torch.as_tensor(stance, device=signed_gap.device, dtype=torch.bool).detach()
    dv = torch.as_tensor(distance_valid, device=signed_gap.device, dtype=torch.bool).detach()
    vv = torch.as_tensor(velocity_valid, device=signed_gap.device, dtype=torch.bool).detach()
    shape = signed_gap.shape
    if (len(shape) != 3 or shape[-1] != 2 or weight.shape != shape or stance.shape != shape
            or dv.shape != shape or vv.shape != (shape[0], shape[1]-1, 2)
            or relative_velocity.shape != (shape[0], shape[1]-1, 2, 3)
            or signal.shape != (shape[0],)):
        raise ValueError('Contact losses require B,T,left/right and interval velocity clocks')
    if not bool(torch.isfinite(signed_gap[dv]).all()) or not bool(torch.isfinite(relative_velocity[vv]).all()):
        raise FloatingPointError('Nonfinite prediction on a valid observed support query')
    settings = policy['loss']
    safe_gap = torch.where(dv, signed_gap, torch.zeros_like(signed_gap))
    excess = (safe_gap.abs()-settings['distance_tolerance_m']).clamp_min(0)/settings['distance_scale_m']
    distance = F.smooth_l1_loss(excess, torch.zeros_like(excess), reduction='none')
    effective = weight*stance*dv
    per_distance = (distance*effective).sum((1,2))/stance.sum((1,2)).clamp_min(1)
    interval_stance = stance[:,1:] & stance[:,:-1]
    interval_weight = torch.minimum(weight[:,1:],weight[:,:-1])*interval_stance*vv
    safe_velocity = torch.where(vv[...,None],relative_velocity,torch.zeros_like(relative_velocity))
    slip = ((safe_velocity.norm(dim=-1)-settings['sliding_tolerance_mps']).clamp_min(0) /
            settings['sliding_scale_mps']).square()
    per_slip = (slip*interval_weight).sum((1,2))/interval_stance.sum((1,2)).clamp_min(1)
    return dict(scene_contact_distance=(per_distance*signal.square()).mean(),
                scene_contact_sliding=(per_slip*signal.square()).mean())


def reliable_planting(prediction, target, batch, signal, details, weight):
    """Versioned replacement for the prior's kinematic planting term only."""
    joints = forward_kinematics(prediction, details['rest'])
    truth = forward_kinematics(target, batch['target_rest']).detach()
    velocity, true_velocity = torch.diff(joints, dim=1)*30, torch.diff(truth, dim=1)*30
    weight = _weights(weight, prediction)
    if weight.shape != (*prediction.shape[:2],2):
        raise ValueError('Expected source contact weights per batch/frame/foot')
    weight = torch.minimum(weight[:,1:],weight[:,:-1])
    feet = (10,11)
    excess = (velocity[:,:,feet].norm(dim=-1)-true_velocity[:,:,feet].norm(dim=-1)-.02).clamp_min(0).square()
    nominal = true_velocity[:,:,feet].norm(dim=-1) < .2
    per = (excess*nominal*weight).flatten(1).sum(1)/nominal.flatten(1).sum(1).clamp_min(1)
    return (per*signal.square()).mean()


def reliable_native_motion_loss(prediction, target, batch, signal, *, details, config,
                               contact_weight, scene_contact=None, policy=None,
                               geometry_coefficients=None):
    """Opt-in scene-stage objective; no change to the sealed base function.

    Foot velocity and all pose reconstruction supervision remain active on
    swing frames. Source quality labels never enter the model's forward pass.
    Geometry terms require explicit future-stage coefficients and observations.
    """
    from scenemi.scene_model import native_motion_loss
    terms = native_motion_loss(prediction,target,batch,signal,details=details,config=config)
    old = terms['planting']
    terms['planting'] = reliable_planting(prediction,target,batch,signal,details,contact_weight)
    terms['total'] = terms['total']+config['loss']['coefficients']['planting']*(terms['planting']-old)
    if scene_contact is not None:
        if policy is None or geometry_coefficients is None or set(geometry_coefficients) != {'scene_contact_distance','scene_contact_sliding'}:
            raise ValueError('Explicit observed geometry policy and coefficients are required')
        if any(not np.isfinite(v) or v < 0 for v in geometry_coefficients.values()):
            raise ValueError('Contact coefficients must be finite and nonnegative')
        extra = contact_terms(weight=contact_weight,signal=signal,policy=policy,**scene_contact)
        terms.update(extra)
        terms['total'] = terms['total']+sum(geometry_coefficients[k]*v for k,v in extra.items())
    return terms
