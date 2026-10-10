"""Measured-state feet trajectories + soft head IK on the native body.

Foot targets retain generated future foot displacement/phase and anchor their
first state to measured feet. They do not freeze guessed contacts. Upper-body
head fitting cannot alter root/legs. No future body/contact GT is consumed.
"""
import torch
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,rotation_from_6d,IDENTITY_6D
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.state_anchored_forecast import anchor_forecast
from experiments.offline_camera_retrain_v1.contact_state_transition import transition_pose,so3_exp

JOINTS=(1,2,4,5,7,8,3,6,9,12,15)


def refine_from_state(motion,history,batch,*,pose_frames=32,iterations=60):
    with torch.inference_mode(False),torch.enable_grad():
        source=motion.detach().clone();past=history.detach().clone();rest=batch['rest'].detach().clone()
        h=past.shape[1];b=len(source);n=source.shape[1]-h
        base=anchor_forecast(source,past,rest,mode='translation').detach()
        initial=(transition_pose(source,past,rest,frames=pose_frames) if pose_frames else base).detach()
        local=rotation_from_6d(initial[:,h:,3:135].reshape(b,n,22,6)).detach()
        base_feet=forward_kinematics(base[:,h:],rest)[:,:,10:12].detach()
        observed=forward_kinematics(past,rest)[:,-4:,10:12].detach()
        foot_velocity=torch.diff(observed,dim=1).median(1).values
        desired=observed[:,-1]+foot_velocity
        feet_target=desired[:,None]+base_feet-base_feet[:,:1]
        head_track=batch['trajectory'][:,h:,15].detach().clone()
        meta=batch['observation_meta'][:,h:,15].detach().clone()
        head_position=head_track[...,:3]*2
        head_rotation=rotation_from_6d(head_track[...,3:]-head_track.new_tensor(IDENTITY_6D)).detach()
        position_weight=meta[...,0]*meta[...,2].square()
        rotation_weight=meta[...,1]*meta[...,3].square()
        raw=torch.zeros(b,n,len(JOINTS),3,device=source.device,requires_grad=True)
        axis_mask=torch.ones_like(raw);axis_mask[:,:,2:4,1:]=0
        limits=raw.new_tensor([.8]*6+[.35]*4+[.5])[None,None,:,None]
        ramp=raw.new_ones(1,n,1,1)
        if pose_frames:
            # First pose is the carried actual state, not an IK discontinuity.
            ramp[:,0]=0
        optimizer=torch.optim.Adam([raw],lr=.03)
        def decode():
            delta=limits*torch.tanh(raw/limits)*axis_mask*ramp
            corrected=[so3_exp(delta[:,:,JOINTS.index(j)])@local[:,:,j] if j in JOINTS else local[:,:,j] for j in range(22)]
            rotations=torch.stack(corrected,2)
            features=(torch.cat((rotations[..., :,0],rotations[..., :,1]),-1)-source.new_tensor(IDENTITY_6D)).flatten(2)
            native=torch.cat((initial[:,h:,:3],features,initial[:,h:,135:]),-1)
            return native,delta
        for step in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            native,delta=decode();joints=forward_kinematics(native,rest)
            foot_error=(joints[:,:,10:12]-feet_target).square().sum(-1).mean()
            foot_velocity_error=(torch.diff(joints[:,:,10:12],dim=1)-torch.diff(feet_target,dim=1)).square().sum(-1).mean()
            head_error=(joints[:,:,15]-head_position).square().sum(-1)
            head_matrix_error=(global_rotations(native)[:,:,15]-head_rotation).square().sum((-1,-2))*.5
            position=(head_error*position_weight).mean()
            orientation=(head_matrix_error*rotation_weight).mean()
            prior=delta.square().mean()
            velocity=torch.diff(delta,dim=1).square().mean()
            acceleration=torch.diff(delta,n=2,dim=1).square().mean()
            loss=100*foot_error+2000*foot_velocity_error+30*position+.15*orientation+.03*prior+.3*velocity+.5*acceleration
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite state-relative IK')
            loss.backward();torch.nn.utils.clip_grad_norm_([raw],10.,error_if_nonfinite=True);optimizer.step()
        with torch.no_grad():
            native,delta=decode();output=initial.clone();output[:,h:]=native
            joints=forward_kinematics(output[:,h:],rest)
            output[:,h:,135:]=(joints/2).flatten(2)
            audit=dict(iterations=iterations,pose_frames=pose_frames,future_body_or_contact_gt_consumed=False,
                       foot_target_error_cm=float((joints[:,:,10:12]-feet_target).norm(dim=-1).mean()*100),
                       foot_velocity_target_error_cm_frame=float((torch.diff(joints[:,:,10:12],dim=1)-torch.diff(feet_target,dim=1)).norm(dim=-1).mean()*100),
                       head_observation_error_cm=float((joints[:,:,15]-head_position).norm(dim=-1).mean()*100),
                       observed_head_position_mean_confidence=float(meta[...,2].mean()),max_delta_degrees=float(delta.norm(dim=-1).max()*180/torch.pi))
        assert torch.equal(output[:,:h],source[:,:h])
        return output.detach(),audit
