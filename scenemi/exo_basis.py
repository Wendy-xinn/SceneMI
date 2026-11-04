"""Repair by choosing explicit temporal motion proposals around an estimate.

All proposals use estimated motion only. The frozen diffusion prior supplies
features for learned mixture weights; no arbitrary absolute body is substituted.
"""
import torch
from torch import nn
from torch.nn import functional as F
from pytorch3d.transforms import matrix_to_axis_angle

from scenemi.exo_repair import InitializedExoPrior
from scenemi.observations import public_observations, body_condition
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d


@torch.no_grad()
def continuation_proposal(initial, available):
    """Estimated velocity extrapolation / cubic interpolation for one outage."""
    result=initial.clone();frames=initial.shape[1]
    for b in range(len(initial)):
        indices=torch.where(available[b]==0)[0]
        if not len(indices):continue
        start,end=int(indices[0]),int(indices[-1])+1
        x=initial[b,:,:135]
        left,right=start-1,end
        if left<0:
            # A source outage can touch an already-missing prefix, so ``right``
            # may equal the frame count.  Clamp both anchors before indexing;
            # the final frame then provides a zero/constant continuation.
            anchor=min(max(right,0),frames-1)
            after=min(frames-1,anchor+6)
            velocity=(x[after]-x[anchor])/max(after-anchor,1)
            query=torch.arange(start,end,device=x.device,dtype=x.dtype)-right
            result[b,start:end,:135]=x[anchor]+query[:,None]*velocity
        elif right>=frames:
            before=max(0,left-6)
            velocity=(x[left]-x[before])/max(left-before,1)
            query=torch.arange(start,end,device=x.device,dtype=x.dtype)-left
            result[b,start:end,:135]=x[left]+query[:,None]*velocity
        else:
            before,after=max(0,left-6),min(frames-1,right+6)
            v0=(x[left]-x[before])/max(left-before,1)
            v1=(x[after]-x[right])/max(after-right,1)
            span=right-left
            u=(torch.arange(start,end,device=x.device,dtype=x.dtype)-left)/span
            result[b,start:end,:135]=((2*u**3-3*u**2+1)[:,None]*x[left]
                +(u**3-2*u**2+u)[:,None]*span*v0+(-2*u**3+3*u**2)[:,None]*x[right]
                +(u**3-u**2)[:,None]*span*v1)
    return result


class TemporalBasisExoPrior(InitializedExoPrior):
    widths=(3,5,9,17)

    def __init__(self,base,**kwargs):
        super().__init__(base,**kwargs)
        self.readout=nn.Conv1d(self.readout.in_channels,23*6,1)
        nn.init.zeros_(self.readout.weight)
        with torch.no_grad():
            # A declared untrained mixture is evaluated at step zero. It is
            # never silently credited as a learned effect.
            self.readout.bias.copy_(self.readout.bias.new_tensor([0.,0.,0.,1.,1.,-1.]).repeat(23))
        self._proposal_cache=None

    @torch.no_grad()
    def proposals(self,payload):
        initial,rest=payload['initial_motion'],payload['initial_rest']
        cache=self._proposal_cache
        if cache is not None and cache[0] is initial and cache[1] is payload['estimate_available']:
            return cache[2],cache[3]
        values=initial[:,:,:135].transpose(1,2)
        motions=[initial]
        for width in self.widths:
            # Triangular kernels have no learned or label-derived directions.
            axis=torch.arange(width,device=initial.device,dtype=initial.dtype)
            kernel=(width//2+1-(axis-width//2).abs());kernel/=kernel.sum()
            smooth=F.conv1d(F.pad(values,(width//2,width//2),mode='replicate'),
                kernel.reshape(1,1,width).expand(135,1,width),groups=135).transpose(1,2)
            proposal=initial.clone();proposal[:,:,:135]=smooth
            motions.append(reconcile_motion(proposal,rest))
        motions.append(reconcile_motion(continuation_proposal(initial,payload['estimate_available']),rest))
        rotations=rotation_from_6d(initial[:,:,3:135].reshape(len(initial),128,22,6))
        roots=torch.stack([(motion[:,:,:3]-initial[:,:,:3])*2 for motion in motions],2)
        angles=torch.stack([matrix_to_axis_angle(rotation_from_6d(
            motion[:,:,3:135].reshape(len(initial),128,22,6))@rotations.transpose(-1,-2))
            for motion in motions],3)
        roots[:,:,0]=0.;angles[:,:,:,0]=0.
        self._proposal_cache=(initial,payload['estimate_available'],roots,angles)
        return roots,angles

    def forward(self,noisy,timestep,payload,*,return_details=False):
        if (payload['camera_context'][:,0]>0).any():
            return super().forward(noisy,timestep,payload,return_details=return_details)
        initial,rest=payload['initial_motion'],payload['initial_rest']
        observations=public_observations(payload)
        with torch.no_grad():
            conditioned=body_condition(observations,dict(rest=rest,head_camera_translation=rest.new_zeros(len(rest),3)))
            empty=(noisy.new_zeros(len(noisy),self.base.core.latent_dim),None,None,None)
            prior=self.base._core_pass(noisy,timestep,conditioned,empty,None,False,True)
        reliability=payload['initial_reliability'].clamp(0,1)
        root_reliability=payload['initial_root_reliability'].clamp(0,1)
        signal=self.alphas[timestep].sqrt()[:,None,None].expand(-1,128,1)
        features=torch.cat((prior.clamp(-5,5),initial,noisy,reliability,root_reliability[...,None],
            payload['estimate_available'][...,None],payload['joint_visible'],signal),-1)
        weights=self.readout(self.temporal(features.transpose(1,2))).transpose(1,2).reshape(len(noisy),128,23,6).softmax(-1)
        roots,angles=self.proposals(payload)
        root=(weights[:,:,0,:,None]*roots).sum(2)*(1-root_reliability[...,None])
        rotation=(weights[:,:,1:,:,None]*angles).sum(3)*(1-reliability[...,None])
        root=root*(self.root_limit_m/root.norm(dim=-1,keepdim=True).clamp_min(self.root_limit_m)).clamp_max(1)
        rotation=rotation*(self.rotation_limit_rad/rotation.norm(dim=-1,keepdim=True).clamp_min(self.rotation_limit_rad)).clamp_max(1)
        moved=reconcile_motion(initial,rest,translation_m=root,rotation_axis_angle=rotation)
        fixed=(root_reliability>=1)&(reliability>=1).all(-1)
        prediction=torch.where(fixed[...,None],initial,moved)
        details=dict(rest=rest,root_delta_m=root,rotation_delta_rad=rotation,frozen_prior_x0=prior,
            fixed_frames=fixed,proposal_weights=weights)
        return (prediction,details) if return_details else prediction
