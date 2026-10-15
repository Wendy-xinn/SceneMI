"""Train/inference-identical temporal residuals and observation-branch protection.

Reference SO(3) smoothing uses observations only. Targets remain untouched.
Short residual filtering and cosine permission ramps suppress visibility flicker
without averaging final joints or relaxing fixed native bone lengths.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
PILOT=Path(__file__).resolve().parent.parent/'legacy55k_exo_residual_oct15'
sys.path.insert(0,str(PILOT))
import torch
from torch.nn import functional as F
from pytorch3d.transforms import matrix_to_axis_angle,axis_angle_to_matrix
from run_reference_probe import ReferenceResidual
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d,IDENTITY_6D,PARENTS,forward_kinematics
from experiments.offline_camera_retrain_v1.native30_gt_reliability import derivative_valid


def temporal_mean(values,radius=2,sigma=.8):
    clock=torch.arange(values.shape[1],device=values.device)
    weights=values.new_tensor([torch.exp(torch.tensor(-.5*(k/sigma)**2)) for k in range(-radius,radius+1)])
    weights=weights/weights.sum()
    return sum(w*values[:,(clock+k).clamp(0,values.shape[1]-1)] for w,k in zip(weights,range(-radius,radius+1)))


@torch.no_grad()
def smooth_reference(motion,rest,radius=2,sigma=.8):
    rotation=rotation_from_6d(motion[...,3:135].reshape(*motion.shape[:2],22,6))
    clock=torch.arange(motion.shape[1],device=motion.device)
    weights=motion.new_tensor([torch.exp(torch.tensor(-.5*(k/sigma)**2)) for k in range(-radius,radius+1)])
    weights=weights/weights.sum()
    tangent=sum(w*matrix_to_axis_angle(rotation.transpose(-1,-2)@rotation[:,(clock+k).clamp(0,len(clock)-1)])
                for w,k in zip(weights,range(-radius,radius+1)))
    filtered=rotation@axis_angle_to_matrix(tangent)
    six=(torch.cat((filtered[..., :,0],filtered[..., :,1]),-1)-motion.new_tensor(IDENTITY_6D)).flatten(2)
    root=temporal_mean(motion[...,:3],radius,sigma)
    return reconcile_motion(torch.cat((root,six,motion[...,135:]),-1),rest)


def branch_permissions(available):
    descendants=torch.zeros(22,22,device=available.device)
    for child in range(1,22):
        parent=int(PARENTS[child])
        while parent>=0:
            descendants[parent,child]=1
            if parent==0:break
            parent=int(PARENTS[parent])
    rotation=torch.einsum('btj,kj->btk',available.float(),descendants)==0
    translation=~available.any(-1,keepdim=True)
    return torch.cat((translation,rotation),-1)


def tapered_permissions(allowed,ramp_frames=4):
    # Distance to a forbidden frame, measured symmetrically in offline context.
    # First newly allowed frame is zero; full permission requires4 interior
    # frames on both sides. Replicate endpoints avoids artificial window fades.
    x=allowed.float().transpose(1,2);distance=torch.zeros_like(x)
    for radius in range(1,ramp_frames+1):
        full=F.avg_pool1d(F.pad(x,(radius,radius),mode='replicate'),2*radius+1,stride=1)>.99999
        distance=distance+full.to(x)
    return (.5-.5*torch.cos(torch.pi*distance/ramp_frames)).transpose(1,2)


class SmoothResidual(ReferenceResidual):
    def forward(self,known,use_scene=True,use_prior=True):
        raw,details=super().forward(known,use_scene=use_scene,use_prior=use_prior)
        available=known['features'][...,267:289]>0
        permission=tapered_permissions(branch_permissions(available))
        translation=temporal_mean(details['translation'],3,1.)*permission[...,:1]
        rotation=temporal_mean(details['rotation'],3,1.)*permission[...,1:,None]
        pred=reconcile_motion(known['initial'],known['rest'],translation,rotation)
        return pred,dict(translation=translation,rotation=rotation,raw_motion=raw,permission=permission)


def alternation_loss(prediction,truth,rest,quality):
    """Dedicated left/right foot relation in body coordinates; GT training only.

    Correlation error is windowed and stabilized at3cm. Velocity/stride losses
    still enforce motion amplitude and timing; this is not a low-speed prior.
    """
    def local(m):
        j=forward_kinematics(m,rest)
        return torch.einsum('btji,btkj->btki',rotation_from_6d(m[...,3:9]),j-j[:,:,:1])[:,:,(10,11),2]
    p=local(prediction);t=local(truth).detach();width=min(48,len(p[0]))
    def relation(z):
        w=z.unfold(1,width,max(1,width//2));w=w-w.mean(-1,keepdim=True)
        scale=(w.square().mean(-1)+.03**2).sqrt()
        return (w[:,:,0]*w[:,:,1]).mean(-1)/(scale[:,:,0]*scale[:,:,1])
    q=torch.minimum(quality[:,:,(10,11)].amin(-1),quality[:,:,0])
    valid=q.unfold(1,width,max(1,width//2)).amin(-1)>=1
    return ((relation(p)-relation(t)).square()*valid).sum()/valid.sum().clamp_min(1)
