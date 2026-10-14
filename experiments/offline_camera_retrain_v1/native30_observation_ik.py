"""Damped observation proximal step on articulated pose inside diffusion x0.

Only position measurements enter this step. Root pose is handled separately;
local bone lengths remain fixed, and unobserved/end-joint rotations remain free
for the learned scene-conditioned correction. No label/contact target is read.
"""
import torch
from experiments.offline_camera_retrain_v1.supervision import PARENTS,forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.native30_observations import sanitize_observations
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion


def position_proximal(motion,batch,signal,gain,iterations=6,damping_m2=.0025,angle_limit_rad=.35,compact=False):
    if gain==0 or float(signal.max())<.05:return motion
    trajectory,meta=sanitize_observations(batch['trajectory'],batch['observation_meta'])
    uncertainty=torch.nan_to_num(batch['observation_uncertainty'][...,0],nan=1.,posinf=1.).clamp_min(0)
    weight=meta[...,0]*meta[...,2]*.05**2/(.05**2+uncertainty.square())
    ancestry=motion.new_zeros(22,21)
    for end in range(22):
        node=int(PARENTS[end])
        while node>0:
            ancestry[end,node-1]=1;node=int(PARENTS[node])
    strength=(weight[...,None]*ancestry).amax(-2)
    active = (strength > 0).any((0, 1)).nonzero().flatten() if compact else torch.arange(21, device=motion.device)
    if not len(active): return motion
    columns = (active[:, None] * 3 + torch.arange(3, device=motion.device)).flatten()
    target=trajectory[...,:3]*2
    identity=torch.eye(len(columns),device=motion.device,dtype=motion.dtype)
    parent_indices=torch.as_tensor(PARENTS[1:],device=motion.device)
    for _ in range(iterations):
        positions=forward_kinematics(motion,batch['rest']);rotation=global_rotations(motion)
        lever=(positions[:,:,:,None]-positions[:,:,None,1:]).detach()
        x,y,z=lever.unbind(-1);zero=torch.zeros_like(x)
        # columns are world-space angular axes; d p / d theta = axis cross lever
        jac=torch.stack((zero,z,-y,-z,zero,x,y,-x,zero),-1).reshape(*lever.shape[:-1],3,3)
        jac=jac*ancestry[None,None,:,:,None,None]
        jac=jac.transpose(-3,-2).reshape(*motion.shape[:2],66,63)
        jac=jac[..., columns]
        w=weight.repeat_interleave(3,-1).sqrt()
        j=jac*w[...,None];error=((target-positions)*weight.sqrt()[...,None]).flatten(2)
        matrix=j.transpose(-1,-2)@j+damping_m2*identity
        solved=torch.linalg.solve(matrix,(j.transpose(-1,-2)@error[...,None])).squeeze(-1).reshape(*motion.shape[:2],len(active),3)
        delta=motion.new_zeros(*motion.shape[:2],21,3).index_copy(-2,active,solved)
        delta=delta*torch.clamp(angle_limit_rad/delta.norm(dim=-1,keepdim=True).clamp_min(1e-8),max=1)
        delta=delta*strength[...,None]*gain*signal[:,None,None,None]
        # Express each update in its current parent frame. Keeping this frame
        # and the Jacobian detached avoids second-order IK curvature in training.
        local=(rotation[:,:,parent_indices].detach().transpose(-1,-2)@delta[...,None]).squeeze(-1)
        update=torch.cat((local.new_zeros(*motion.shape[:2],1,3),local),2)
        motion=reconcile_motion(motion,batch['rest'],rotation_axis_angle=update)
    return motion
