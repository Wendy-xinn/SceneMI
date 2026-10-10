"""Training-only turn/landing targets, without a failed-candidate teacher."""
import torch
from experiments.offline_camera_retrain_v1.turn_path_objective import turn_path_losses
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

def replay_losses(prediction,truth,rest,pred_soles,true_soles,teacher_prediction,signal):
    weight=signal.detach().reshape(-1).square()
    def mean(x,mask=None):
        if mask is None:v=x.flatten(1).mean(1)
        else:v=(x*mask).flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1)
        return (v*weight).mean()
    turn=turn_path_losses(prediction,truth,rest,teacher_prediction,signal)
    # Do not freeze old world foot positions while asking the pelvis to take
    # another winding. Preserve speed amplitude, supervise landing with GT.
    turn_total=turn['turn_total']-2*turn['teacher_foot_position']-.1*turn['teacher_foot_velocity']
    pr=global_rotations(prediction);tr=global_rotations(truth).detach()
    head=mean(.5*(pr[:,:,15]-tr[:,:,15]).square().sum((-1,-2)))
    neck=mean(.5*(pr[:,:,12]-tr[:,:,12]).square().sum((-1,-2)))
    ht=true_soles.detach()[...,1];hp=pred_soles[...,1]
    tv=torch.diff(true_soles.detach(),dim=1)*20;pv=torch.diff(pred_soles,dim=1)*20
    ground=ht.min(-1).values;stance=(tv.norm(dim=-1)<.2)&(ht[:,1:]<=ground[:,1:,None]+.03)
    height=mean(((hp[:,1:]-ht[:,1:])/.05).square(),stance)
    plant=mean(((pv-tv)/.2).square().sum(-1),stance)
    swing=mean(((pv-tv)/.4).square().sum(-1),~stance)
    envelope=mean(((hp.min(-1).values-ground)/.05).square())
    # Multi-frame planted displacement catches a foot that drifts throughout
    # a turn, instead of merely matching an averaged per-frame speed.
    anchor=prediction.new_zeros(())
    for lag in (4,8):
        stable=stance.unfold(1,lag,1).all(-1)
        dp=pred_soles[:,lag:]-pred_soles[:,:-lag]
        dt=true_soles[:,lag:].detach()-true_soles[:,:-lag].detach()
        anchor=anchor+mean(((dp-dt)/.05).square().sum(-1),stable)/2
    old=forward_kinematics(teacher_prediction.detach(),rest)[:,:,(10,11)]
    feet=forward_kinematics(prediction,rest)[:,:,(10,11)]
    speed_keep=mean(((torch.diff(feet,dim=1).norm(dim=-1)-torch.diff(old,dim=1).norm(dim=-1))*20/.2).square())
    from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d
    def leg_steps(m):
        r=rotation_from_6d(m[...,3:135].reshape(*m.shape[:2],22,6))[:,:,(1,2,4,5,7,8)]
        return r[:,1:]@r[:,:-1].transpose(-1,-2)
    accel=mean(.5*((torch.diff(leg_steps(prediction),dim=1)-torch.diff(leg_steps(truth).detach(),dim=1))/.035).square().sum((-1,-2)))
    total=turn_total+.5*head+.25*neck+.15*envelope+.1*height+.12*plant+.02*swing+.08*anchor+.01*speed_keep+.01*accel
    return dict(replay_total=total,turn_without_foot_lock=turn_total,global_head=head,global_neck=neck,native_envelope=envelope,stance_height=height,stance_velocity=plant,swing_velocity=swing,planted_displacement=anchor,teacher_speed_keep=speed_keep,leg_accel=accel)
