"""GT-free yaw-path refinement with exact two-bone ankle/foot preservation.

Experimental kinematics, not contact physics. Generated feet and head rotation
are references; camera-relative turning is soft and never initializes body GT.
"""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,rotation_from_6d,IDENTITY_6D
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

def align(a,b):
    a=a/np.linalg.norm(a);b=b/np.linalg.norm(b);v=np.cross(a,b);c=np.clip(a@b,-1,1)
    if c<-.99999:raise ValueError('Antiparallel swing outside small correction trust region')
    x,y,z=v;k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+k+k@k/(1+c)

def coordinate_turn(motion,camera,rest,*,mode='smooth',camera_weight=.25):
    # Accept only explicit observations, generated state, and body template.
    device=motion.device;source=motion.detach().cpu().float()[None];rest=rest.detach().cpu().float()[None]
    local=rotation_from_6d(source[:,:,3:135].reshape(1,-1,22,6))[0].numpy().astype(float)
    world=global_rotations(source)[0].numpy().astype(float);joints=forward_kinematics(source,rest)[0].numpy().astype(float);body=rest[0].numpy().astype(float)
    yaw=np.unwrap(np.arctan2(world[:,0,0,2],world[:,0,2,2]));smooth=gaussian_filter1d(yaw,3,mode='nearest')
    reference=smooth
    if mode=='camera_soft':
        cam=camera.detach().cpu().numpy();forward=np.cross(cam[:,3:6],cam[:,6:9]);cyaw=np.unwrap(np.arctan2(forward[:,0],forward[:,2]));relative=yaw[0]+cyaw-cyaw[0]
        reference=(1-camera_weight)*smooth+camera_weight*relative
    elif mode=='identity':reference=yaw
    elif mode!='smooth':raise ValueError(mode)
    target=np.clip(reference-yaw,-.5,.5)
    grid=np.linspace(-.5,.5,321);rot=Rotation.from_rotvec(np.stack((np.zeros_like(grid),grid,np.zeros_like(grid)),1)).as_matrix()
    feasible=np.ones((len(yaw),len(grid)),bool)
    for hip,knee,ankle in [(1,4,7),(2,5,8)]:
        length1=np.linalg.norm(body[knee]-body[hip]);length2=np.linalg.norm(body[ankle]-body[knee])
        hips=joints[:,0,None]+np.einsum('gij,tjk,k->tgi',rot,world[:,0],body[hip]-body[0])
        distance=np.linalg.norm(joints[:,ankle,None]-hips,axis=-1)
        feasible&=(distance<=length1+length2+1e-6)&(distance>=abs(length1-length2)-1e-6)
    # Generated-head trust budget, not a GT/head hard projection.
    head_offset=joints[:,15]-joints[:,0]
    moved_head=joints[:,0,None]+np.einsum('gij,tj->tgi',rot,head_offset)
    feasible&=(np.linalg.norm(moved_head-joints[:,15,None],axis=-1)<=.02)&(abs(grid)[None]<=np.deg2rad(15))
    feasible[:,160]=True # generated original pose is the guaranteed fallback
    # Preserve own first generated pose, not an observed/GT initial body.
    feasible[0]=False;feasible[0,160]=True
    unary=(grid[None]-target[:,None])**2+.1*grid[None]**2;unary[~feasible]=np.inf
    transition=10*(grid[:,None]-grid[None,:])**2;cost=unary[0];parents=[]
    for t in range(1,len(yaw)):
        table=cost[:,None]+transition;parent=table.argmin(0);parents.append(parent);cost=table[parent,np.arange(len(grid))]+unary[t]
    indices=[int(cost.argmin())]
    for parent in reversed(parents):indices.append(int(parent[indices[-1]]))
    indices=np.array(indices[::-1]);changes=grid[indices];new=local.copy();new[:,0]=rot[indices]@world[:,0]
    for t in range(len(yaw)):
        for hip,knee,ankle in [(1,4,7),(2,5,8)]:
            h=joints[t,0]+new[t,0]@(body[hip]-body[0]);foot=joints[t,ankle];direction=foot-h;distance=np.linalg.norm(direction);unit=direction/distance
            l1=np.linalg.norm(body[knee]-body[hip]);l2=np.linalg.norm(body[ankle]-body[knee]);a=(l1*l1-l2*l2+distance*distance)/(2*distance)
            perpendicular=joints[t,knee]-h;perpendicular-=unit*(perpendicular@unit)
            if np.linalg.norm(perpendicular)<1e-8:perpendicular=np.cross(world[t,hip,:,0],unit)
            perpendicular/=np.linalg.norm(perpendicular);bend=h+a*unit+np.sqrt(max(l1*l1-a*a,0))*perpendicular
            upper=align(joints[t,knee]-joints[t,hip],bend-h)@world[t,hip]
            lower=align(joints[t,ankle]-joints[t,knee],foot-bend)@world[t,knee]
            new[t,hip]=new[t,0].T@upper;new[t,knee]=upper.T@lower;new[t,ankle]=lower.T@world[t,ankle]
        # Preserve generated head global orientation; position is only audited.
        parent=(rot[indices[t]]@world[t,12]);new[t,15]=parent.T@world[t,15]
    output=source.clone();features=np.concatenate((new[..., :,0],new[..., :,1]),-1)-np.array(IDENTITY_6D)
    output[0,:,3:135]=torch.tensor(features.reshape(len(yaw),132),dtype=output.dtype);output[:,:,135:]=(forward_kinematics(output,rest)/2).flatten(2)
    final=forward_kinematics(output,rest)[0].numpy();error=np.max(np.linalg.norm(final[:,[7,8,10,11]]-joints[:,[7,8,10,11]],axis=-1))
    assert error<2e-5,error
    correction=Rotation.from_matrix((new@local.transpose(0,1,3,2)).reshape(-1,3,3)).magnitude().reshape(len(yaw),22)
    audit={'mode':mode,'camera_weight':camera_weight if mode=='camera_soft' else 0,'foot_world_max_difference_cm':float(error*100),'root_yaw_max_correction_deg':float(np.rad2deg(abs(changes).max())),
        'leg_local_max_correction_deg':float(np.rad2deg(correction[:,[1,2,4,5,7,8]].max())),'generated_head_position_max_change_cm':float(np.linalg.norm(final[:,15]-joints[:,15],axis=-1).max()*100),
        'future_body_contact_GT_read':False,'initial_pose_reference':'own generated first pose','not_physical_collision_certificate':True}
    return output[0].to(device),audit
