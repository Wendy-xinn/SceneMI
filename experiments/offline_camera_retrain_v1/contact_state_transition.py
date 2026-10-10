"""Native SO(3) transition plus lower-limb IK from inferred/observed stance.

No future body/contact truth. Contact is a joint-speed proxy; this is not a
mesh/scene collision solver. Head/pelvis future constraints remain soft model
inputs. FK bone lengths are preserved; knees receive X-axis IK deltas only.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,rotation_from_6d,IDENTITY_6D
from experiments.offline_camera_retrain_v1.state_anchored_forecast import anchor_forecast

LEG_JOINTS=(1,2,4,5,7,8)


def transition_pose(motion,history,rest,*,frames=16):
    output=anchor_forecast(motion,history,rest,mode='translation')
    h=history.shape[1];n=min(frames,motion.shape[1]-h)
    pred=rotation_from_6d(output[:,h:h+n,3:135].reshape(len(motion),n,22,6))
    last=rotation_from_6d(history[:,-1,3:135].reshape(len(motion),22,6))
    correction=last@pred[:,0].transpose(-1,-2)
    rv=Rotation.from_matrix(correction.detach().cpu().numpy().reshape(-1,3,3).copy()).as_rotvec().reshape(len(motion),22,3)
    taper=.5*(1+np.cos(np.pi*np.arange(n)/(n-1)))
    rotations=Rotation.from_rotvec((rv[:,None]*taper[None,:,None,None]).reshape(-1,3)).as_matrix().reshape(len(motion),n,22,3,3)
    local=torch.as_tensor(rotations,device=motion.device,dtype=motion.dtype)@pred
    output[:,h:h+n,3:135]=(torch.cat((local[..., :,0],local[..., :,1]),-1)-motion.new_tensor(IDENTITY_6D)).flatten(2)
    output[:,h:,135:]=(forward_kinematics(output[:,h:],rest)/2).flatten(2)
    return output


def so3_exp(vector):
    theta=vector.norm(dim=-1)
    x,y,z=vector.unbind(-1);zero=torch.zeros_like(x)
    skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*vector.shape[:-1],3,3)
    first=torch.sinc(theta/torch.pi)[...,None,None]
    second=(.5*torch.sinc(theta/(2*torch.pi)).square())[...,None,None]
    return torch.eye(3,device=vector.device,dtype=vector.dtype)+first*skew+second*(skew@skew)


def contact_targets(base,history,rest,*,frames):
    h=history.shape[1];n=min(frames,base.shape[1]-h)
    feet=forward_kinematics(base[:,h:h+n],rest)[:,:,10:12]
    past=forward_kinematics(history,rest)[:,-4:,10:12]
    previous=torch.cat((past[:,-1:],feet[:,:-1]),1)
    speed=(feet-previous).norm(dim=-1)
    # Interior speed at first frame comes from planned feet, not the pose jump.
    speed[:,0]=(feet[:,1]-feet[:,0]).norm(dim=-1)
    support=past[:,-1,:,1].min(-1).values
    low=(feet[...,1]-support[:,None,None]).abs()<.10
    planted=(speed<.015)&low
    past_planted=torch.diff(past,dim=1).norm(dim=-1).mean(1)<.01
    target=feet.clone()
    for b in range(len(feet)):
        for foot in range(2):
            t=0
            while t<n:
                if not planted[b,t,foot]:t+=1;continue
                end=t+1
                while end<n and planted[b,end,foot]:end+=1
                if end-t<2:planted[b,t:end,foot]=False
                else:
                    anchor=past[b,-1,foot] if t==0 and past_planted[b,foot] else feet[b,t:end,foot].median(0).values
                    target[b,t:end,foot]=anchor
                t=end
    return target,planted


def contact_transition(motion,history,rest,*,frames=16,iterations=40):
    # Call safely from an inference-mode evaluator; optimization is inference IK.
    with torch.inference_mode(False),torch.enable_grad():
        source=motion.detach().clone();observed=history.detach().clone();body_rest=rest.detach().clone()
        base=anchor_forecast(source,observed,body_rest,mode='translation').detach()
        smooth=transition_pose(source,observed,body_rest,frames=frames).detach()
        h=observed.shape[1];n=min(frames,source.shape[1]-h)
        initial=smooth[:,h:h+n].clone();local=rotation_from_6d(initial[...,3:135].reshape(len(source),n,22,6)).detach()
        target,contact=contact_targets(base,observed,body_rest,frames=n)
        target=target.detach();contact=contact.detach()
        raw=torch.zeros(len(source),n,6,3,device=source.device,requires_grad=True)
        axis_mask=torch.ones_like(raw);axis_mask[:,:,2:4,1:]=0
        # Never alter the imposed first pose; no abrupt IK jump at either endpoint.
        endpoint=raw.new_ones(1,n,1,1);endpoint[:,0]=0;endpoint[:,-1]=0
        optimizer=torch.optim.Adam([raw],lr=.035)
        def decode():
            delta=.8*torch.tanh(raw/.8)*axis_mask*endpoint
            revised=[]
            for j in range(22):
                revised.append(so3_exp(delta[:,:,LEG_JOINTS.index(j)])@local[:,:,j] if j in LEG_JOINTS else local[:,:,j])
            rotations=torch.stack(revised,2)
            features=(torch.cat((rotations[..., :,0],rotations[..., :,1]),-1)-source.new_tensor(IDENTITY_6D)).flatten(2)
            native=torch.cat((initial[...,:3],features,initial[...,135:]),-1)
            return native,delta
        initial_error=None
        for step in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            native,delta=decode();joints=forward_kinematics(native,body_rest)
            error=(joints[:,:,10:12]-target).square().sum(-1)
            foot=(error*contact).sum()/contact.sum().clamp_min(1)
            if initial_error is None:initial_error=float(foot.detach())
            prior=delta.square().mean()
            smoothness=torch.diff(delta,dim=1).square().mean()
            acceleration=torch.diff(delta,n=2,dim=1).square().mean()
            loss=100*foot+.03*prior+.2*smoothness+.2*acceleration
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite contact IK')
            loss.backward();optimizer.step()
        with torch.no_grad():
            native,delta=decode();smooth[:,h:h+n]=native
            smooth[:,h:,135:]=(forward_kinematics(smooth[:,h:],body_rest)/2).flatten(2)
            final_feet=forward_kinematics(native,body_rest)[:,:,10:12]
            final_error=float((((final_feet-target).square().sum(-1))*contact).sum()/contact.sum().clamp_min(1))
        return smooth.detach(),dict(contact_entries=int(contact.sum()),initial_foot_target_mse=initial_error,final_foot_target_mse=final_error,iterations=iterations,future_gt_consumed=False)
