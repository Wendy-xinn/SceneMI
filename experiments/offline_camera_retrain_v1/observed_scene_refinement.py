"""Frozen-generator short-block geometry correction; no future body/contact GT.

Uses observed local half-spaces, not a complete scene SDF. Soft contact applies
only to supported static upward surfaces; moving contact needs object velocity.
"""
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d,forward_kinematics,IDENTITY_6D
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.contact_state_transition import so3_exp
JOINTS=(1,2,4,5,7,8,3,6,9,12,15)

def geometry_metrics(motion,skin,scene):
    with torch.no_grad():
        points=skin(motion);d,n,owner,valid,p=scene.query(points);depth=torch.relu(-d)*valid
        count=int(valid.sum());frames=valid.any(1)
        return {'observed_halfspace_mean_depth_cm':float(depth.sum()/valid.sum().clamp_min(1)*100),'observed_halfspace_p95_depth_cm':float(torch.quantile(depth[valid],.95)*100) if count else None,'observed_halfspace_max_depth_cm':float(depth.max()*100),'observed_halfspace_over1cm_fraction':float(((depth>.01)&valid).sum()/valid.sum().clamp_min(1)),'observed_halfspace_query_coverage':float(valid.float().mean()),'observed_halfspace_frames_with_over1cm':int((depth>.01).any(1).sum()),'mesh_sample_count':len(skin.indices),'not_closed_mesh_collision':True}

def refine_observed_scene(motion,history,batch,scene,skin,*,contact=True,iterations=80):
    with torch.inference_mode(False),torch.enable_grad():
        source=motion.detach().clone();past=history.detach().clone();rest=batch['rest'].detach().clone();h=past.shape[1];n=source.shape[1]-h
        if len(source)!=1 or n!=len(scene.frames):raise ValueError('Short native B1 and scene aligned frame count required')
        native=source[0,h:].clone();local=rotation_from_6d(native[:,3:135].reshape(n,22,6)).detach();base_points=skin(native).detach();base_soles=skin.soles(base_points)
        d,normal,owner,valid,p=scene.query(base_soles)
        # Soft geometric support evidence; source foot phase is uncertain, not GT.
        velocities=torch.cat((base_soles[1:]-base_soles[:-1],base_soles[-1:]-base_soles[-2:-1]),0)
        contact_weight=(torch.exp(-(d/.04).square())*torch.exp(-(velocities.norm(dim=-1)/.025).square())*valid*(owner==0)*(normal[...,1]>.7)).detach()
        contact_pairs=(contact_weight[1:]*contact_weight[:-1]).sqrt().detach()
        knots=np.unique(np.r_[np.arange(0,n,4),n-1]);basis=torch.tensor(CubicSpline(knots,np.eye(len(knots)),axis=0,bc_type='natural')(np.arange(n)),device=source.device,dtype=source.dtype)
        raw=torch.zeros(len(knots),len(JOINTS)*3+4,device=source.device,requires_grad=True)
        first=torch.ones_like(raw);first[0]=0
        limits=torch.tensor([.35]*18+[.15]*15+[.06]*3+[.3],device=source.device)
        axis=torch.ones(n,len(JOINTS),3,device=source.device);axis[:,2:4,1:]=0
        target_head=batch['trajectory'][0,h:h+n,15].detach().clone();meta=batch['observation_meta'][0,h:h+n,15].detach().clone()
        target_r=rotation_from_6d(target_head[:,3:]-source.new_tensor(IDENTITY_6D));optimizer=torch.optim.Adam([raw],lr=.03);history_velocity=torch.diff(forward_kinematics(past,rest)[0,-4:,0],dim=0).median(0).values
        def decode():
            changes=limits*torch.tanh((basis@(raw*first))/limits)
            pose=changes[:,:len(JOINTS)*3].reshape(n,len(JOINTS),3)*axis
            rotations=[so3_exp(pose[:,JOINTS.index(j)])@local[:,j] if j in JOINTS else local[:,j] for j in range(22)]
            yaw=source.new_zeros(n,3);yaw[:,1]=changes[:,-1];rotations[0]=so3_exp(yaw)@rotations[0]
            rotations=torch.stack(rotations,1);features=(torch.cat((rotations[..., :,0],rotations[..., :,1]),-1)-source.new_tensor(IDENTITY_6D)).flatten(1)
            value=torch.cat((native[:,:3]+changes[:,-4:-1]/2,features,native[:,135:]),-1)
            return value,changes
        for step in range(iterations):
            optimizer.zero_grad(set_to_none=True);value,changes=decode();points=skin(value);signed,norm,owners,available,near=scene.query(points)
            penetration=(torch.relu(.002-signed).square()*available).sum()/available.sum().clamp_min(1)
            soles=skin.soles(points);foot_prior=(soles-base_soles).square().sum(-1).mean();velocity_prior=(torch.diff(soles,dim=0)-torch.diff(base_soles,dim=0)).square().sum(-1).mean()
            # Static supporting surface: relative surface velocity is zero.
            slip=(torch.diff(soles,dim=0).square().sum(-1)*contact_pairs).mean()
            sd,sn,so,sv,sp=scene.query(soles);height=((sd-.003).square()*contact_weight*sv*(so==0)*(sn[...,1]>.7)).mean()
            fk=forward_kinematics(value[None],rest)[0];position=((fk[:,15]-target_head[:,:3]*2).square().sum(-1)*meta[:,0]*meta[:,2].square()).mean();head=((global_rotations(value[None])[0,:,15]-target_r).square().sum((-1,-2))*meta[:,1]*meta[:,3].square()).mean()
            pose_prior=changes[:,:33].square().mean();root_prior=changes[:,33:36].square().mean();yaw_prior=changes[:,-1].square().mean()
            smooth=torch.diff(changes,dim=0).square().mean()+5*torch.diff(changes,n=2,dim=0).square().mean()
            boundary=(fk[1,0]-fk[0,0]-history_velocity).square().sum()
            # Broad native X hinge / local neck bounds, additional to trust region.
            r=rotation_from_6d(value[:,3:135].reshape(n,22,6));knee=torch.atan2(r[:,[4,5],2,1],r[:,[4,5],1,1]);knee_limit=(torch.relu(-.15-knee).square()+torch.relu(knee-2.7).square()).mean()
            trace=r[:,[12,15]].diagonal(dim1=-2,dim2=-1).sum(-1)
            neck_cost=torch.relu((3-trace)-2*(1-torch.cos(value.new_tensor(.95)))).square().mean()
            loss=2000*penetration+20*foot_prior+500*velocity_prior+30*position+.08*head+.15*pose_prior+10*root_prior+.2*yaw_prior+3*smooth+20*boundary+2*knee_limit+.2*neck_cost
            if contact:loss=loss+20000*slip+100*height
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite observed scene refinement')
            loss.backward();torch.nn.utils.clip_grad_norm_([raw],10.,error_if_nonfinite=True);optimizer.step()
        with torch.no_grad():
            value,changes=decode();output=source.clone();output[0,h:]=value;output[:,h:,135:]=(forward_kinematics(output[:,h:],rest)/2).flatten(2)
            assert torch.equal(output[:,:h],source[:,:h]);assert torch.allclose(output[0,h],source[0,h],atol=1e-6,rtol=0)
            audit={'iterations':iterations,'contact':contact,'contact_weight_mean':float(contact_weight.mean()),'contact_weight_over_half_fraction':float((contact_weight>.5).float().mean()),'root_correction_max_cm':float(changes[:,33:36].norm(dim=-1).max()*100),'root_yaw_correction_max_deg':float(changes[:,-1].abs().max()*180/torch.pi),'future_body_or_contact_GT_read':False,'dynamic_contact_velocity_unavailable_excluded':True,'scene_semantics':'observed oriented local halfspace only; no global inside/outside guarantee','first_future_pose_fixed':True}
        return output.detach(),audit
