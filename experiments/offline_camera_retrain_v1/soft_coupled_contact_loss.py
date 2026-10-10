"""Soft observed rotation and labeled native-joint contact geometry proxies.

No inference projection; no scene or GT contact labels become model inputs.
Regional joint targets are geometry proxies, not contact mesh vertices or SDFs.
"""
import torch
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d,IDENTITY_6D,forward_kinematics

def chordal(a,b):return .5*(a-b).square().sum((-1,-2))

def soft_coupled_losses(prediction,truth,batch,control_mask,signal_weight):
 pred=global_rotations(prediction);target=global_rotations(truth).detach()
 meta=batch['observation_meta'][:,:,15]
 active=control_mask[:,:,15].to(prediction)*meta[:,:,1]*(meta[:,:,4]==1)
 confidence=meta[:,:,3].detach().clamp(0,1)
 # Availability normalizes the number of frames; confidence scales loss rather
 # than cancelling out in the denominator. Observed head is known at any tau.
 observed=rotation_from_6d(batch['trajectory'][:,:,15,3:].detach()-prediction.new_tensor(IDENTITY_6D))
 observed_loss=(chordal(pred[:,:,15],observed)*active*confidence).sum()/active.sum().clamp_min(1)
 weight=signal_weight.detach().to(prediction).reshape(-1).square()
 relative=[];relative_rate=[]
 for parent in (0,9,12):
  rp=pred[:,:,parent].transpose(-1,-2)@pred[:,:,15]
  rt=target[:,:,parent].transpose(-1,-2)@target[:,:,15]
  relative.append((chordal(rp,rt).mean(1)*weight).mean())
  lag=4
  dp=rp[:,lag:]@rp[:,:-lag].transpose(-1,-2)
  dt=rt[:,lag:]@rt[:,:-lag].transpose(-1,-2)
  relative_rate.append((chordal(dp,dt).mean(1)*weight).mean()*(20/lag)**2)
 coupling=torch.stack(relative).mean();rate=torch.stack(relative_rate).mean()
 total=.25*observed_loss+.25*coupling+.01*rate
 return dict(soft_observed_head_orientation=observed_loss,head_body_relative_orientation=coupling,head_body_relative_turn_rate=rate,soft_coupled_total=total)

def contact_geometry_proxy_losses(prediction,truth,batch,signal_weight):
 pred=forward_kinematics(prediction,batch['rest'])
 target=forward_kinematics(truth,batch['rest']).detach()
 label=batch['contact_target'].detach()>0.5
 valid=batch['contact_valid'].detach().bool()
 mask=(label&valid).to(prediction)
 weight=signal_weight.detach().to(prediction).reshape(-1,1,1).square()
 def masked(value,available):
  return (value*available*weight).sum()/available.sum().clamp_min(1)
 position=masked((pred-target).square().sum(-1),mask)
 # Contact can slide; match measured reference displacement, do not impose zero.
 adjacent=mask[:,1:]*mask[:,:-1]
 velocity=masked(((pred[:,1:]-pred[:,:-1]-target[:,1:]+target[:,:-1])*20).square().sum(-1),adjacent)
 foot_mask=torch.stack((mask[:,:,7].maximum(mask[:,:,10]),mask[:,:,8].maximum(mask[:,:,11])),dim=-1)
 pf=pred[:,:,(10,11),1];tf=target[:,:,(10,11),1]
 # A labeled ankle/foot establishes only a reference joint support height.
 # This is NOT a floor plane; stairs and rolling feet retain per-foot targets.
 height=masked((pf-tf).square(),foot_mask)
 below=masked((tf-pf-.025).clamp_min(0).square(),foot_mask)
 total=2*position+.02*velocity+4*height+8*below
 return dict(labeled_contact_joint_position_m2=position,labeled_contact_joint_velocity_m2_s2=velocity,labeled_foot_height_m2=height,labeled_foot_below_reference_m2=below,contact_geometry_proxy_total=total,contact_geometry_positive_entries=mask.sum())
