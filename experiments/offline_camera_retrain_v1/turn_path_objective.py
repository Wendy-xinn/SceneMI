"""Training-only signed turn path and root/leg rate guard; GT never conditions inference."""
import torch
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d, forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
LEGS=(1,2,4,5,7,8)

def turn_path_losses(prediction,truth,rest,teacher_prediction,signal_weight):
    pred=global_rotations(prediction);target=global_rotations(truth).detach()
    weight=signal_weight.detach().reshape(-1).square()
    def reduce(x):return (x.flatten(1).mean(1)*weight).mean()
    pf=pred[:,:,0,:,2][...,[0,2]];tf=target[:,:,0,:,2][...,[0,2]]
    def increments(f):
        a,b=f[:,:-1],f[:,1:]
        return torch.atan2(a[...,1]*b[...,0]-a[...,0]*b[...,1],(a*b).sum(-1)+1e-6)
    valid=(tf[:,:-1].norm(dim=-1)>.3)&(tf[:,1:].norm(dim=-1)>.3)
    # Principal signed difference is integrated over time: rotations agreeing
    # at the endpoint can still have opposite winding throughout the path.
    diff=increments(pf)-increments(tf);diff=torch.atan2(diff.sin(),diff.cos())*valid
    path=reduce(diff.cumsum(1).square())
    local=rotation_from_6d(prediction[...,3:135].reshape(*prediction.shape[:2],22,6))
    true_local=rotation_from_6d(truth[...,3:135].reshape(*truth.shape[:2],22,6)).detach()
    joints=(0,*LEGS)
    pp=local[:,1:,joints]@local[:,:-1,joints].transpose(-1,-2)
    tt=true_local[:,1:,joints]@true_local[:,:-1,joints].transpose(-1,-2)
    # sqrt chordal = 2*sin(angle/2), stable derivative with epsilon. Guard
    # target rate + 3deg/frame, not total turning angle or a universal joint limit.
    speed=(.5*(pp-torch.eye(3,device=prediction.device)).square().sum((-1,-2))+1e-8).sqrt()
    true_speed=(.5*(tt-torch.eye(3,device=prediction.device)).square().sum((-1,-2))+1e-8).sqrt()
    excess=reduce((speed-true_speed-.05236).clamp_min(0).square())
    relative=reduce(.5*(pp-tt).square().sum((-1,-2)))
    pelvis=reduce(.5*(pred[:,:,0]-target[:,:,0]).square().sum((-1,-2)))
    legs=reduce(.5*(pred[:,:,LEGS]-target[:,:,LEGS]).square().sum((-1,-2)))
    feet=forward_kinematics(prediction,rest)[:,:,(7,8,10,11)]
    with torch.no_grad():old_feet=forward_kinematics(teacher_prediction,rest)[:,:,(7,8,10,11)]
    foot_position=reduce((feet-old_feet).square().sum(-1))
    foot_velocity=reduce(((torch.diff(feet,dim=1)-torch.diff(old_feet,dim=1))*20).square().sum(-1))
    total=.2*path+.25*pelvis+.1*legs+2*relative+5*excess+2*foot_position+.1*foot_velocity
    return dict(turn_path=path,root_leg_rate_excess=excess,root_leg_relative_rotation=relative,global_pelvis=pelvis,global_legs=legs,teacher_foot_position=foot_position,teacher_foot_velocity=foot_velocity,turn_total=total)
