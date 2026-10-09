"""Native global orientation and signed turning supervision; never an inference input."""
import torch
from experiments.offline_camera_retrain_v1.supervision import PARENTS,rotation_from_6d

def global_rotations(motion):
 local=rotation_from_6d(motion[...,3:135].reshape(*motion.shape[:2],22,6))
 result=[local[:,:,0]]
 for j in range(1,22):result.append(result[PARENTS[j]]@local[:,:,j])
 return torch.stack(result,dim=2)

def orientation_losses(prediction,truth,signal_weight=None,accumulated=False, normalize_duration=False, gate_turn_noise=False, supervise_neck=False):
 pred=global_rotations(prediction);target=global_rotations(truth).detach()
 weight=prediction.new_ones(len(prediction)) if signal_weight is None else signal_weight.detach().to(prediction).reshape(-1)
 def reduce(x, temporal=False):
  effective=weight.square() if temporal and gate_turn_noise else weight
  return (x.flatten(1).mean(1)*effective).mean()
 # Squared Frobenius / 2 = 2(1-cos(theta)); stable at zero and differentiable.
 def chordal(a,b):return (a-b).square().sum((-1,-2))*.5
 head=reduce(chordal(pred[:,:,15],target[:,:,15]))
 pelvis=reduce(chordal(pred[:,:,0],target[:,:,0]))
 legs=reduce(chordal(pred[:,:,(1,2,4,5,7,8)],target[:,:,(1,2,4,5,7,8)]))
 turning=prediction.new_zeros(())
 # Relative global rotations preserve left/right sign and avoid Euler wrap.
 lags=[lag for lag in (1,4,8) if prediction.shape[1]>lag]
 for lag in lags:
  rp=pred[:,lag:,(0,15)]@pred[:,:-lag,(0,15)].transpose(-1,-2)
  rt=target[:,lag:,(0,15)]@target[:,:-lag,(0,15)].transpose(-1,-2)
  turning=turning+reduce(chordal(rp,rt), temporal=True)*(20/lag)**2/max(len(lags),1)
 total=.5*head+.25*pelvis+.1*legs+(.1 if accumulated else .02)*turning
 extra={}
 if supervise_neck:
  # GT global neck-parent orientation teaches the torso chain, rather than
  # forcing an uncertain input camera rotation into the local head joint.
  neck=reduce(chordal(pred[:,:,12],target[:,:,12]), temporal=True)
  total=total+.25*neck
  extra["global_neck_parent_orientation_loss"]=neck
 if accumulated:
  # Horizontal forward-vector increments distinguish a long left turn from
  # a short right turn to the same final rotation. Never unwrap Euler angles.
  pf=pred[:,:,(0,15),: ,2][...,[0,2]];tf=target[:,:,(0,15),:,2][...,[0,2]]
  def increments(f):
   a,b=f[:,:-1],f[:,1:]
   return torch.atan2(a[...,1]*b[...,0]-a[...,0]*b[...,1],(a*b).sum(-1)+1e-6)
  valid=(tf[:,:-1].norm(dim=-1)>.3)&(tf[:,1:].norm(dim=-1)>.3)
  error=(increments(pf)-increments(tf))*valid
  heading=reduce(error.cumsum(dim=1).square(), temporal=True)
  if normalize_duration:
   # Normalize mean elapsed-time squared; exactly preserve the 128-frame
   # reference objective, without overweighting the first few increments.
   elapsed=torch.arange(1,prediction.shape[1],device=prediction.device,dtype=prediction.dtype)/20
   reference=torch.arange(1,128,device=prediction.device,dtype=prediction.dtype)/20
   heading=heading*reference.square().mean()/elapsed.square().mean().clamp_min(1e-6)
  total=total+.2*heading
  extra['accumulated_heading_loss']=heading
 return dict(**extra,global_head_orientation_loss=head,global_pelvis_orientation_loss=pelvis,global_leg_orientation_loss=legs,signed_turn_rate_loss=turning,orientation_total=total)
