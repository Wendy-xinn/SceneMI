"""SceneMI-style motion supervision for residual 6D SMPL rotations."""
import torch
from torch.nn import functional as F

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


def supervised_losses(prediction, truth, batch):
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
    return dict(total=total, x0=simple, rotation_matrix_mse=rotation_matrix,
                fk_joint_mse=joint_pos, fk_velocity_mse=joint_vel,
                fk_acceleration_mse=joint_acceleration,
                head_position_mse=head_position, foot_stillness_mse=foot_stillness,
                branch_consistency_mse=consistency,
                fk_mpjpe_m=(fk_joints - true_joints).norm(dim=-1).mean(),
                head_error_m=(fk_joints[:, :, HEAD] - true_joints[:, :, HEAD]).norm(dim=-1).mean(),
                foot_sliding_m_per_frame=foot_sliding)
