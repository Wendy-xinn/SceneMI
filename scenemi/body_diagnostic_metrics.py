"""Pose/shape/temporal diagnostics; private targets are evaluation-only."""
import torch
from scenemi.metrics_batch import metric_tensors
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d

JOINT_NAMES = ('pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2',
    'left_ankle','right_ankle','spine3','left_foot','right_foot','neck','left_collar',
    'right_collar','head','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist')
LOWER = (1,2,4,5,7,8,10,11)
UPPER = (3,6,9,12,13,14,15,16,17,18,19,20,21)


def rotation_error_deg(prediction, target):
    delta = prediction.double() @ target.double().transpose(-1,-2)
    skew = torch.stack((delta[...,2,1]-delta[...,1,2], delta[...,0,2]-delta[...,2,0],
        delta[...,1,0]-delta[...,0,1]), -1)
    sine = skew.norm(dim=-1)/2
    cosine = (delta.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
    return torch.atan2(sine, cosine.clamp(-1,1))*180/torch.pi


def motion_from_rotations(rotations, root, rest):
    identity = rotations.new_tensor([1,0,0,0,1,0])
    six = torch.cat((rotations[..., :,0], rotations[..., :,1]), -1)-identity
    prefix = torch.cat(((root-rest[:,None,0])/2, six.flatten(2)), -1)
    initial = torch.cat((prefix, prefix.new_zeros(*prefix.shape[:2],66)), -1)
    return torch.cat((prefix, forward_kinematics(initial, rest).flatten(2)/2), -1)


def rest_readout(motion, old_rest, new_rest):
    root = motion[...,:3]*2+old_rest[:,None,0]
    rotations = rotation_from_6d(motion[...,3:135].reshape(*motion.shape[:2],22,6))
    return motion_from_rotations(rotations,root,new_rest)


def standardized_points(motion, rest, *, remove_root_orientation=False):
    rotations = rotation_from_6d(motion[...,3:135].reshape(*motion.shape[:2],22,6)).clone()
    if remove_root_orientation:
        rotations[:,:,0] = torch.eye(3, device=motion.device, dtype=motion.dtype)
    neutral = rest-rest[:,:1]
    root = motion.new_zeros(*motion.shape[:2],3)
    encoded = motion_from_rotations(rotations,root,neutral)
    return forward_kinematics(encoded,neutral)


def diagnostics(prediction, motion, rest, target, target_motion, target_rest, standard_rest, *, fps=30):
    """Contact partitions are explicitly kinematic support proxies, not HSC."""
    values = metric_tensors(prediction,target,fps=fps)
    p, g = prediction.double(), target.double()
    pr, gr = p-p[:,:,:1], g-g[:,:,:1]
    error = (p-g).norm(dim=-1)*1000
    relative = (pr-gr).norm(dim=-1)*1000
    mean = lambda x:x.flatten(1).mean(1)
    pl = rotation_from_6d(motion[...,3:135].reshape(*motion.shape[:2],22,6))
    gl = rotation_from_6d(target_motion[...,3:135].reshape(*motion.shape[:2],22,6))
    angles = rotation_error_deg(pl,gl)
    sp = standardized_points(motion,standard_rest)
    sg = standardized_points(target_motion,standard_rest)
    standard_error = (sp-sg).norm(dim=-1)*1000
    lp = standardized_points(motion,standard_rest,remove_root_orientation=True)
    lg = standardized_points(target_motion,standard_rest,remove_root_orientation=True)
    amplitude = lambda x:mean((x-x.mean(1,keepdim=True)).square().sum(-1)).sqrt()*1000
    pred_amp, gt_amp = amplitude(pr[:,:,1:]), amplitude(gr[:,:,1:])
    bone = lambda x:x[:,1:]-x[:,list((-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19))[1:]]
    values.update(root_orientation_deg=angles[:,:,0].mean(1), local_joint_rotation_deg=mean(angles[:,:,1:]),
        upper_body_mpjpe_mm=mean(error[:,:,UPPER]), lower_body_mpjpe_mm=mean(error[:,:,LOWER]),
        upper_body_rootrelative_mm=mean(relative[:,:,UPPER]), lower_body_rootrelative_mm=mean(relative[:,:,LOWER]),
        standard_bone_rootrelative_mm=mean(standard_error),
        standard_bone_rootaligned_mm=mean((lp-lg).norm(dim=-1))*1000,
        rootrelative_mpjve_mm_s=mean((torch.diff(pr,dim=1)-torch.diff(gr,dim=1)).norm(dim=-1))*fps*1000,
        rootrelative_pred_jitter_100m_s3=mean(torch.diff(pr,n=3,dim=1).norm(dim=-1))*fps**3/100,
        rootrelative_gt_jitter_100m_s3=mean(torch.diff(gr,n=3,dim=1).norm(dim=-1))*fps**3/100,
        pred_jitter_100m_s3=values['jerk_m_s3']/100, gt_jitter_100m_s3=values['gt_jerk_m_s3']/100,
        pred_rootrelative_amplitude_mm=pred_amp, gt_rootrelative_amplitude_mm=gt_amp,
        amplitude_ratio=pred_amp/gt_amp.clamp_min(1e-9),
        pred_local_angular_speed_deg_s=mean(rotation_error_deg(pl[:,1:,1:],pl[:,:-1,1:]))*fps,
        gt_local_angular_speed_deg_s=mean(rotation_error_deg(gl[:,1:,1:],gl[:,:-1,1:]))*fps,
        bone_length_mae_mm=(bone(rest).norm(dim=-1)-bone(target_rest).norm(dim=-1)).abs().mean(1)*1000)
    # A frame has support if any GT ankle/toe is low and slow. These are not
    # annotations of sitting, hand/wall contact or general scene interaction.
    feet = g[:,:,[7,8,10,11]]
    speed = (feet[:,1:]-feet[:,:-1]).norm(dim=-1)*fps
    low = feet[:,1:,:,1] <= feet.new_tensor([.08,.08,.04,.04])
    support = (low & (speed <= .15)).any(-1)
    output = []
    scalars = {k:v.cpu().tolist() for k,v in values.items()}
    vectors = dict(joint_mpjpe_mm=error.mean(1),joint_rootrelative_mm=relative.mean(1),
        joint_standard_bone_mm=standard_error.mean(1),joint_rotation_deg=angles.mean(1))
    vectors = {k:v.cpu().tolist() for k,v in vectors.items()}
    for i in range(len(p)):
        row = {k:v[i] for k,v in scalars.items()}
        for name, mask in (('gt_foot_support_proxy',support[i]),('gt_no_foot_support_proxy',~support[i])):
            row[name+'_frames'] = int(mask.sum())
            row[name+'_mpjpe_mm'] = float(error[i,1:][mask].mean()) if mask.any() else None
        output.append(dict(metrics=row,**{k:v[i] for k,v in vectors.items()}))
    return output
