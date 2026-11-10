"""Additional, dimensioned supervision for camera-only generalization.

The ordinary EGO objective stays in pipeline.native_motion_loss. No labels in
this module are passed to the predictor, sampler, or camera anchoring module.
"""
import torch
from torch.nn import functional as F
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

LOWER = (1, 2, 4, 5, 7, 8, 10, 11)
PARENTS = (0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def geodesic_squared(a, b):
    """SO(3) squared radians, with finite derivatives at identical rotations."""
    d = a @ b.transpose(-1, -2)
    skew = torch.stack((d[..., 2, 1]-d[..., 1, 2], d[..., 0, 2]-d[..., 2, 0],
                        d[..., 1, 0]-d[..., 0, 1]), -1)
    sine = (skew.square().sum(-1) + 1e-12).sqrt()/2
    cosine = ((d.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1)
    return torch.atan2(sine, cosine).square()


def bone_lengths(rest):
    return (rest[:, 1:]-rest[:, list(PARENTS[1:])]).norm(dim=-1)


def skeleton_height(rest):
    """Head-to-mean-foot rest height; explicitly a skeleton, not mesh, height."""
    return rest[:, 15, 1]-rest[:, [10, 11], 1].mean(-1)


def shape_objective(details, batch, c):
    pred, truth = details['rest'], batch['target_rest']
    zero = pred.sum()*0
    human = (batch['body_format'] == 1) | (batch['body_format'] == 2)
    amass = batch['body_format'] == 0
    terms = dict(shape_rest=F.mse_loss(pred, truth),
        native_shape_betas=F.mse_loss(details['predicted_native_betas'][human], batch['target_native_betas'][human]) if human.any() else zero,
        native_template=F.cross_entropy(details['template_logits'][human], batch['target_template'][human]) if human.any() else zero,
        amass_shape_betas=F.mse_loss(details['predicted_betas'][amass], batch['target_betas'][amass]) if amass.any() else zero,
        bone_length=F.smooth_l1_loss(bone_lengths(pred)/c['bone_scale_m'], bone_lengths(truth)/c['bone_scale_m']),
        skeleton_height=F.smooth_l1_loss(skeleton_height(pred)/c['height_scale_m'], skeleton_height(truth)/c['height_scale_m']))
    terms['total'] = sum(c['weights'][k]*v for k, v in terms.items())
    return terms


def focused_objective(prediction, batch, details, alpha, timestep, policy, root_weight):
    """Return an increment, replacing only the gated part of root chordal loss.

    Position is divided by 0.5m, velocity by 1m/s; orientation is radians².
    GT velocity matching penalizes amplitude collapse, not motion itself.
    """
    target = batch['motion']
    p = forward_kinematics(prediction, details['rest'])
    g = forward_kinematics(target, batch['target_rest']).detach()
    r, s = global_rotations(prediction), global_rotations(target).detach()
    gate = (timestep <= policy['max_timestep']).to(alpha)*alpha
    mean = lambda x: (x.flatten(1).mean(1)*gate).mean()
    root_geo = mean(geodesic_squared(r[:, :, 0], s[:, :, 0]))
    root_old = mean((r[:, :, 0]-s[:, :, 0]).square().sum((-1, -2))*.5)
    # Existing root->head supervision remains. These two links constrain how
    # that relative rotation is distributed through the complete FK chain.
    links = []
    for a, b in ((0, 9), (9, 15)):
        links.append(geodesic_squared(r[:, :, a].transpose(-1, -2) @ r[:, :, b],
                                     s[:, :, a].transpose(-1, -2) @ s[:, :, b]))
    pr = p[:, :, LOWER]-p[:, :, :1]
    gr = g[:, :, LOWER]-g[:, :, :1]
    pl = rotation_from_6d(prediction[..., 3:135].reshape(*prediction.shape[:2], 22, 6))
    gl = rotation_from_6d(target[..., 3:135].reshape(*target.shape[:2], 22, 6)).detach()
    terms = dict(root_geodesic_rad2=root_geo,
        root_replacement=root_weight*(policy['root_strength']*root_geo-root_old),
        torso_chain_rad2=mean(torch.stack(links, -1)),
        lower_position_normalized=mean(((pr-gr)/policy['position_scale_m']).square()),
        lower_local_rad2=mean(geodesic_squared(pl[:, :, LOWER], gl[:, :, LOWER])),
        lower_velocity_normalized=mean(((torch.diff(pr, dim=1)-torch.diff(gr, dim=1))*30/policy['velocity_scale_m_s']).square()),
        focused_gate_mean=gate.mean())
    terms['focused_increment'] = terms['root_replacement'] + sum(
        terms[k]*v for k, v in policy['weights'].items())
    return terms
