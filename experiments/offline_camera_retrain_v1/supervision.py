"""SceneMI-style motion supervision for residual 6D SMPL rotations."""
import torch
from torch.nn import functional as F
from experiments.offline_camera_retrain_v1.coordination import coordination_losses

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
           16, 17, 18, 19)
HEAD = 15
FEET = (10, 11)
IDENTITY_6D = (1., 0., 0., 0., 1., 0.)


def rotation_from_6d(features):
    """Convert identity-centred residual 6D features to rotation matrices."""
    identity = features.new_tensor(IDENTITY_6D)
    features = features + identity
    first = F.normalize(features[..., :3], dim=-1, eps=1e-6)
    second_raw = features[..., 3:] - (first * features[..., 3:]).sum(-1, keepdim=True) * first
    second = F.normalize(second_raw, dim=-1, eps=1e-6)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def forward_kinematics(motion, rest):
    """Return 22 global joints in the camera-first anchored frame, meters."""
    if motion.ndim != 3 or motion.shape[-1] != 201 or rest.shape != (len(motion), 22, 3):
        raise ValueError('Expected motion [B,T,201] and rest [B,22,3]')
    local = rotation_from_6d(motion[..., 3:135].reshape(*motion.shape[:2], 22, 6))
    rotations = [local[:, :, 0]]
    joints = [motion[..., :3] * 2 + rest[:, None, 0]]
    for index in range(1, 22):
        parent = PARENTS[index]
        parent_rotation = rotations[parent]
        offset = rest[:, index] - rest[:, parent]
        world_offset = (parent_rotation @ offset[:, None, :, None]).squeeze(-1)
        joints.append(joints[parent] + world_offset)
        rotations.append(parent_rotation @ local[:, :, index])
    return torch.stack(joints, dim=2)


def gait_losses(joints, target, signal_weight=None):
    """Balanced stance/swing supervision, in meters and seconds (20 Hz).

    Targets are clean-motion FK, not a second skeleton's joint annotations.
    This is training-only supervision: no contact or support truth is fed to
    the network or used to modify a sampled motion. Signal weighting limits
    phase-specific penalties on intrinsically ambiguous high-noise examples.
    """
    feet = joints[:, :, FEET]
    target_feet = target[:, :, FEET]
    delta = torch.diff(feet, dim=1)
    target_delta = torch.diff(target_feet, dim=1)
    stance = target_delta.norm(dim=-1) < .01
    weight = (joints.new_ones(len(joints)) if signal_weight is None
              else signal_weight.detach().to(joints).reshape(len(joints)))

    def reduce(value, mask=None):
        if mask is None:
            per_example = value.flatten(1).mean(1)
        else:
            per_example = ((value * mask).flatten(1).sum(1)
                           / mask.flatten(1).sum(1).clamp_min(1))
        return (per_example * weight).mean()

    # Match measured velocity, rather than declaring that every slow joint
    # must have exactly zero velocity (toe roll and fitted-pose noise exist).
    velocity_error = ((delta - target_delta) * 20).square().sum(-1)
    stance_velocity = reduce(velocity_error, stance)
    swing_velocity = reduce(velocity_error, ~stance)
    height = (feet[:, 1:, :, 1] - target_feet[:, 1:, :, 1]).square()
    support_height = reduce(height, stance)
    swing_height = reduce(height, ~stance)
    relative = feet - joints[:, :, :1]
    target_relative = target_feet - target[:, :, :1]
    relative_position = reduce((relative - target_relative).square().sum(-1))
    # Multi-frame displacement gives the model a stronger gait signal than
    # tiny one-frame differences, without inventing a fixed stride/phase.
    stride = joints.new_zeros(())
    for lag in (4, 8):
        error = ((relative[:, lag:] - relative[:, :-lag])
                 - (target_relative[:, lag:] - target_relative[:, :-lag]))
        stride = stride + reduce(error.square().sum(-1)) / 2
    return dict(gait_stance_velocity_m2_s2=stance_velocity,
                gait_swing_velocity_m2_s2=swing_velocity,
                gait_support_height_m2=support_height,
                gait_swing_height_m2=swing_height,
                gait_relative_position_m2=relative_position,
                gait_stride_displacement_m2=stride)


def phase_free_support_losses(joints, target):
    """Support targets invariant to exchanging left and right gait phase.

    The minimum foot height is an annotation-derived support proxy, not a
    floor/SDF query. Frames with neither target foot slow are excluded from
    the speed penalty, so genuine flight is not forced onto the ground.
    """
    feet, true_feet = joints[:, :, FEET], target[:, :, FEET]
    support = true_feet[..., 1].min(-1).values
    height = (feet[..., 1].min(-1).values - support).square().mean()
    penetration = (support[..., None] - feet[..., 1]).clamp_min(0).square().mean()
    speed = torch.diff(feet, dim=1).norm(dim=-1).min(-1).values * 20
    true_speed = torch.diff(true_feet, dim=1).norm(dim=-1).min(-1).values * 20
    support_mask = true_speed < .2
    # A hinge permits quiet stance/foot roll, and does not penalize generated
    # support being more stationary than a noisy fitted reference joint.
    excess = (speed - true_speed.detach()).clamp_min(0).square()
    planting = (excess * support_mask).sum() / support_mask.sum().clamp_min(1)
    return dict(support_envelope_m2=height, support_penetration_m2=penetration,
                phase_free_planting_m2_s2=planting)


def supervised_losses(prediction, truth, batch, *, profile='baseline', signal_weight=None):
    if prediction.shape != truth.shape:
        raise ValueError('Prediction and truth dimensions differ')
    # x0 feature objective from SceneMI; trajectory channels have higher weight.
    weights = torch.ones((1, 1, 201), device=prediction.device)
    weights[..., :3] = 2.
    simple = ((prediction - truth).square() * weights).mean()
    predicted_rotations = rotation_from_6d(
        prediction[..., 3:135].float().reshape(*prediction.shape[:2], 22, 6))
    true_rotations = rotation_from_6d(
        truth[..., 3:135].float().reshape(*truth.shape[:2], 22, 6))
    rotation_matrix = (predicted_rotations - true_rotations).square().mean()
    fk_joints = forward_kinematics(prediction.float(), batch['rest'].float())
    true_joints = batch['joints'].float() * 2
    joint_pos = (fk_joints - true_joints).square().mean()
    predicted_velocity = fk_joints[:, 1:] - fk_joints[:, :-1]
    true_velocity = true_joints[:, 1:] - true_joints[:, :-1]
    joint_vel = (predicted_velocity - true_velocity).square().mean()
    predicted_acceleration = predicted_velocity[:, 1:] - predicted_velocity[:, :-1]
    true_acceleration = true_velocity[:, 1:] - true_velocity[:, :-1]
    joint_acceleration = (predicted_acceleration - true_acceleration).square().mean()
    head_position = (fk_joints[:, :, HEAD] - true_joints[:, :, HEAD]).square().mean()
    true_foot_velocity = true_velocity[:, :, FEET]
    predicted_foot_velocity = predicted_velocity[:, :, FEET]
    stance = true_foot_velocity.norm(dim=-1) < .01
    # The mask is derived only from training truth. At inference the reported
    # sliding metric uses the same held-out truth solely for evaluation.
    if stance.any():
        foot_stillness = predicted_foot_velocity.square().sum(-1)[stance].mean()
        foot_sliding = predicted_foot_velocity.norm(dim=-1)[stance].mean()
    else:
        foot_stillness = predicted_foot_velocity.square().mean() * 0.
        foot_sliding = predicted_foot_velocity.norm(dim=-1).mean() * 0.
    direct_joints = prediction[..., 135:].reshape(*prediction.shape[:2], 22, 3) * 2
    consistency = (direct_joints - fk_joints).square().mean()
    total = (simple + .2 * rotation_matrix + 2. * joint_pos + 10. * joint_vel
             + 2. * joint_acceleration + head_position + 10. * foot_stillness
             + .5 * consistency)
    extra = {}
    if profile in ('gait_v1', 'gait_v2', 'coordination_v1', 'orientation_v1', 'orientation_v2', 'orientation_v3'):
        with torch.no_grad():
            target_fk = forward_kinematics(truth.float(), batch['rest'].float())
        extra = gait_losses(fk_joints, target_fk, signal_weight)
        gait_total = (.10 * extra['gait_stance_velocity_m2_s2']
                      + .05 * extra['gait_swing_velocity_m2_s2']
                      + 2. * extra['gait_support_height_m2']
                      + .5 * extra['gait_swing_height_m2']
                      + extra['gait_relative_position_m2']
                      + 2. * extra['gait_stride_displacement_m2'])
        extra['gait_total'] = gait_total
        total = total + gait_total
        if profile in ('gait_v2', 'coordination_v1', 'orientation_v1', 'orientation_v2', 'orientation_v3'):
            extra.update(phase_free_support_losses(fk_joints, target_fk))
            physical = (4. * extra['support_envelope_m2']
                        + 2. * extra['support_penetration_m2']
                        + .10 * extra['phase_free_planting_m2_s2'])
            # The old GT-phase-specific term conflicts with other valid
            # phases at high noise. gait_losses already teaches stance/swing
            # with clean-signal weighting; use phase-free planting here.
            total = total - 10. * foot_stillness + physical
            extra['phase_free_support_total'] = physical
        if profile in ('coordination_v1', 'orientation_v1', 'orientation_v2', 'orientation_v3'):
            extra.update(coordination_losses(fk_joints, target_fk, signal_weight))
            coordination = (.05 * extra['coordination_relation_mse']
                            + 5. * extra['coordination_amplitude_m2'])
            total = total + coordination
            extra['coordination_total'] = coordination
        if profile in ('orientation_v1', 'orientation_v2', 'orientation_v3'):
            from experiments.offline_camera_retrain_v1.orientation_supervision import orientation_losses
            extra.update(orientation_losses(prediction, truth, signal_weight, accumulated=profile in ('orientation_v2','orientation_v3'), normalize_duration=profile=='orientation_v3'))
            total = total + extra['orientation_total']
    elif profile != 'baseline':
        raise ValueError(f'Unknown loss profile: {profile}')
    return dict(total=total, **extra, x0=simple, rotation_matrix_mse=rotation_matrix,
                fk_joint_mse=joint_pos, fk_velocity_mse=joint_vel,
                fk_acceleration_mse=joint_acceleration,
                head_position_mse=head_position, foot_stillness_mse=foot_stillness,
                branch_consistency_mse=consistency,
                fk_mpjpe_m=(fk_joints - true_joints).norm(dim=-1).mean(),
                head_error_m=(fk_joints[:, :, HEAD] - true_joints[:, :, HEAD]).norm(dim=-1).mean(),
                foot_sliding_m_per_frame=foot_sliding)
