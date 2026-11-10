"""Native30-style motion objectives with separate predicted and target bodies."""
import torch
from torch.nn import functional as F

from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d, IDENTITY_6D
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations


def gated_mean(value, signal):
    return (value.flatten(1).mean(1) * signal.square()).mean()


def motion_loss(prediction, target, batch, signal, *, details, config):
    # Deliberately separate rest skeletons. Root channels use one common
    # neutral pelvis reference in both cases; neither target FK nor labels
    # are ever computed with the predicted shape.
    joints = forward_kinematics(prediction, details['rest'])
    truth = forward_kinematics(target, batch['target_rest']).detach()
    rotations, true_rotations = global_rotations(prediction), global_rotations(target).detach()
    weights = prediction.new_ones(201)
    weights[:3] = 2
    raw = details['raw_motion']
    terms = dict(x0=((prediction-target).square()*weights).mean(),
                 raw_x0=((raw-target).square()*weights).mean(),
                 rotation=(rotations-true_rotations).square().mean(),
                 position=(joints-truth).square().mean(),
                 head_position=(joints[:, :, 15]-truth[:, :, 15]).square().mean())
    velocity, true_velocity = torch.diff(joints, dim=1)*30, torch.diff(truth, dim=1)*30
    terms['velocity'] = gated_mean((velocity-true_velocity).square(), signal)
    terms['acceleration'] = gated_mean((torch.diff(velocity, dim=1)*30-torch.diff(true_velocity, dim=1)*30).square(), signal)
    chordal = (rotations-true_rotations).square().sum((-1, -2))*.5
    terms.update(root_orientation=chordal[:, :, 0].mean(), head_orientation=chordal[:, :, 15].mean(),
                 neck_orientation=chordal[:, :, 12].mean())
    relative = rotations[:, :, 0].transpose(-1, -2) @ rotations[:, :, 15]
    true_relative = true_rotations[:, :, 0].transpose(-1, -2) @ true_rotations[:, :, 15]
    terms['head_root_coordination'] = (relative-true_relative).square().mean()
    terms['turning'] = prediction.new_zeros(())
    for lag in (1, 6, 12):
        p = rotations[:, lag:, (0, 15)] @ rotations[:, :-lag, (0, 15)].transpose(-1, -2)
        t = true_rotations[:, lag:, (0, 15)] @ true_rotations[:, :-lag, (0, 15)].transpose(-1, -2)
        terms['turning'] = terms['turning'] + gated_mean((p-t).square(), signal)*(30/lag)**2/3
    feet = (10, 11)
    terms['foot_velocity'] = gated_mean((velocity[:, :, feet]-true_velocity[:, :, feet]).square().sum(-1), signal)
    excess = (velocity[:, :, feet].norm(dim=-1)-true_velocity[:, :, feet].norm(dim=-1)-.02).clamp_min(0).square()
    stance = true_velocity[:, :, feet].norm(dim=-1) < .2
    per = (excess*stance).flatten(1).sum(1)/stance.flatten(1).sum(1).clamp_min(1)
    terms['planting'] = (per*signal.square()).mean()
    raw_fk = forward_kinematics(raw, details['rest'])
    terms['raw_consistency'] = (raw[..., 135:].reshape(*raw.shape[:2], 22, 3)*2-raw_fk).square().mean()
    terms['shape_betas'] = F.mse_loss(details['predicted_betas'], batch['target_betas'])
    terms['shape_rest'] = F.mse_loss(details['rest'], batch['target_rest'])
    observations = details['observations']
    ego = observations['camera_context'][:, 0] > 0
    ca, cc = observations['camera_available'], observations['camera_confidence']
    predicted_camera = joints[:, :, 15] + (rotations[:, :, 15] @ details['head_camera_translation'][:, None, :, None])[..., 0]
    pm = ego[:, None] * ca[..., 0] * cc[..., 0]
    rm = ego[:, None] * ca[..., 1] * cc[..., 1]
    terms['camera_position'] = ((predicted_camera-observations['camera'][..., :3]*2).square().mean(-1)*pm).sum()/pm.sum().clamp_min(1)
    identity = prediction.new_tensor(IDENTITY_6D)
    camera_six = torch.where(ca[..., 1:] > 0, observations['camera'][..., 3:], identity)
    camera_rotation = rotation_from_6d(camera_six-identity)
    terms['camera_rotation'] = ((rotations[:, :, 15]-camera_rotation).square().mean((-1, -2))*rm).sum()/rm.sum().clamp_min(1)
    meta = observations['observation_meta']
    mask = meta[..., 0]*meta[..., 2]
    observed = observations['trajectory'][..., :3]*2
    error = F.smooth_l1_loss((joints-observed)/.02, torch.zeros_like(joints), reduction='none').mean(-1)
    terms['observed_position'] = (error*mask*signal[:, None, None].square()).sum()/mask.sum().clamp_min(1)
    coefficients = config['loss']['coefficients']
    if set(coefficients) != set(terms):
        raise ValueError('Loss configuration does not name exactly the implemented terms')
    terms['total'] = sum(coefficients[key]*value for key, value in terms.items())
    return terms
