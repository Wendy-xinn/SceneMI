"""Weak TRAIN-only contact supervision from observed patches and GT skin.

These are auxiliary targets, never inference evidence. Unobserved surfaces,
unreliable normals and unresolved dynamic velocities produce UNKNOWN. Existing
official labels take priority. This prototype does not alter the old trainer.
"""
import numpy as np
import torch

from experiments.offline_camera_retrain_v1.unified_geometry import skin_points
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations


def support_targets(vertices, labels, query, *, fps=30., positive_m=.05,
                    negative_m=.10, max_relative_speed=.20, weak_weight=.25,
                    official_target=None, official_valid=None):
    vertices = np.asarray(vertices); labels = np.asarray(labels)
    frames, anchors = vertices.shape[:2]
    if labels.shape != (anchors,) or query['points'].shape[:2] != (frames, anchors):
        raise ValueError('Anchor labels and scene query must share the GT clock')
    point = query['points'][:, :, 2, 0]
    distance = np.linalg.norm(vertices - point, axis=-1)
    known = query['valid'][:, :, 2, 0] & query['normal_reliable'][:, :, 2, 0]
    owner = query['owners'][:, :, 2, 0]
    velocity = query['velocities'][:, :, 2, 0]
    velocity_valid = query['velocity_valid'][:, :, 2, 0] | (owner == 0)
    body_velocity = np.zeros_like(vertices)
    body_velocity[1:] = np.diff(vertices, axis=0) * fps
    relative_speed = np.linalg.norm(body_velocity - velocity, axis=-1)
    temporal = known & velocity_valid
    temporal[0] = False
    temporal[1:] &= known[:-1] & (owner[1:] == owner[:-1])
    positive = temporal & (distance <= positive_m) & (relative_speed <= max_relative_speed)
    # A nearby wall is not a ground support surface for the feet.
    feet = np.isin(labels, (7, 8, 10, 11))[None]
    upward = query['normals'][:, :, 2, 0, 1] > .7
    positive &= ~feet | upward
    # Demand three consecutive reliable support observations. Frames near a
    # transition remain unknown instead of being labelled non-contact.
    persistent = np.zeros_like(positive)
    if frames >= 3:
        runs = positive[:-2] & positive[1:-1] & positive[2:]
        persistent[:-2] |= runs; persistent[1:-1] |= runs; persistent[2:] |= runs
    target = np.zeros((frames, 22), np.float32)
    valid = np.zeros((frames, 22), bool)
    weight = np.zeros((frames, 22), np.float32)
    # Anatomical contact candidates; do not invent contact on arbitrary torso
    # anchors. Labels 0/1/2 capture some sitting support, 20/21 wrist surfaces.
    for joint in (0, 1, 2, 7, 8, 10, 11, 20, 21):
        ids = labels == joint
        if not ids.any(): continue
        pos = persistent[:, ids].any(-1)
        # All sampled regional anchors must see a reliable, distant surface
        # before declaring non-contact; an unknown anchor prevents a negative.
        neg = (known[:, ids] & (distance[:, ids] >= negative_m)).all(-1)
        valid[:, joint] = pos | neg; target[:, joint] = pos
        weight[:, joint] = weak_weight * valid[:, joint]
    if (official_target is None) != (official_valid is None):
        raise ValueError('Official targets and validity must be supplied together')
    if official_target is not None:
        mask = np.asarray(official_valid, bool)
        target[mask] = np.asarray(official_target)[mask]
        valid[mask] = True; weight[mask] = 1.
    return dict(target=target, valid=valid, weight=weight,
                weak_positive_anchor_count=int(persistent.sum()),
                protocol='observed-patch-temporally-stable-weak-contact-v1')


@torch.no_grad()
def training_support_targets(sample, scene, asset):
    """Caller must keep this result outside model_inputs/inference_payload."""
    cpu_asset = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in asset.items()}
    motion = torch.as_tensor(sample['motion'])
    vertices = skin_points(motion, cpu_asset)
    rotations = global_rotations(motion[None])[0, :, cpu_asset['labels']]
    query = scene.query(vertices.numpy(), rotations.numpy())
    result = support_targets(vertices.numpy(), cpu_asset['labels'], query,
                             official_target=sample['contact_target'], official_valid=sample['contact_valid'])
    if 'gt_position_weight' in sample:
        uncertain = np.asarray(sample['gt_position_weight']) < .75
        weak = result['weight'] < 1
        result['valid'][uncertain & weak] = False
        result['weight'][uncertain & weak] = 0
    return result


def region_contact_loss(predicted_distance, target_distance, labels, observed_reliable,
                        contact_target, contact_valid, label_weight=None,
                        *, evidence_limit_m=.05, slack_m=.02):
    """Calibrate sparse-marker contact against GT instead of pulling all to 0.

    Distances use the same observed surface correspondences. Select one
    supported GT marker for each contacting region (training target only).
    Unrepresented contacts/unknown surface patches have no contact force.
    Perfect GT has zero loss despite marker/surface sampling offsets.
    """
    if predicted_distance.shape != target_distance.shape or observed_reliable.shape != target_distance.shape:
        raise ValueError('Scene correspondences must share the body anchor layout')
    labels = torch.as_tensor(labels, device=predicted_distance.device)
    target_distance = target_distance.detach()
    label_weight = torch.ones_like(contact_target) if label_weight is None else label_weight
    numerator = predicted_distance.sum() * 0; denominator = predicted_distance.new_zeros(())
    for joint in range(22):
        ids = (labels == joint).nonzero().flatten()
        if not len(ids): continue
        reliable = observed_reliable[..., ids]
        distances = target_distance[..., ids]
        chosen = torch.where(reliable, distances, torch.inf).argmin(-1, keepdim=True)
        gt = distances.gather(-1, chosen)[..., 0]
        pred = predicted_distance[..., ids].gather(-1, chosen)[..., 0]
        active = reliable.any(-1) & (gt <= evidence_limit_m) & contact_valid[..., joint] & (contact_target[..., joint] > .5)
        weights = active.to(pred) * label_weight[..., joint]
        numerator = numerator + ((pred - gt - slack_m).clamp_min(0).square() * weights).sum()
        denominator = denominator + weights.sum()
    return numerator / denominator.clamp_min(1)
