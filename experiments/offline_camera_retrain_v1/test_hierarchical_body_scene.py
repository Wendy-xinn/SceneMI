import numpy as np
import torch
from experiments.offline_camera_retrain_v1.body_local_scene import BodySceneWindow
from experiments.offline_camera_retrain_v1.hierarchical_body_scene import BodySurfaceAttention,HierarchicalBodySceneMI
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.known_input_generation import camera_inputs_only,generate_without_body_initialization
from experiments.offline_camera_retrain_v1.hard_head_coordination import project_head_pose,_yaw_path
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

def main():
    torch.set_num_threads(2);torch.manual_seed(616)
    # Future surfaces cannot affect current tokens; moving old locations vanish.
    scene=BodySceneWindow([[0,0,0],[9,0,0]],[0,1],[0,1],[[[.1,0,0]],[[5,0,0]]])
    p=np.zeros((2,22,3));p[1]=.1
    tokens=scene.neighbourhoods(p)
    altered=BodySceneWindow([[0,0,0],[1,0,0]],[0,1],[0,1],scene.dynamic)
    assert np.array_equal(tokens[0],altered.neighbourhoods(p)[0])
    assert not tokens[1,...,5].any()
    # Large-radius tokens cover an actual larger surface extent.
    cloud=np.column_stack((np.linspace(.01,1.9,300),np.zeros((300,2))))
    varied=BodySceneWindow(cloud,np.zeros(300),[0],[np.empty((0,3))]).neighbourhoods(np.zeros((1,22,3)))
    assert (varied[0,0,2,:,0]*2).max()>(varied[0,0,0,:,0]*.25).max()+1
    attention=BodySurfaceAttention();queries=torch.randn(1,2,22,21)
    x=torch.tensor(tokens[None]);frame,contact,weights,active=attention(queries,x)
    with torch.no_grad():attention.part_output.weight.normal_();attention.contact_output.weight.normal_()
    a=attention(queries,x)[0];changed=x.clone();changed[...,0]+=.1
    b=attention(queries,changed)[0];assert (a-b).abs().max()>.001
    empty=attention(queries,torch.zeros_like(x));assert torch.isfinite(empty[2]).all() and empty[0].count_nonzero()==0 and empty[1].count_nonzero()==0
    attention.zero_grad();sum(y.square().mean() for y in attention(queries,x)[:2]).backward();assert attention.attention.in_proj_weight.grad.abs().sum()>0
    base=OfflineSceneMI(64,(1,2),body_conditioning=True,contact_prediction=True).eval()
    with torch.no_grad():
        for parameter in base.parameters():
            if parameter.ndim>=2 and not parameter.count_nonzero():parameter.normal_(0,.01)
    model=HierarchicalBodySceneMI(64,(1,2),body_conditioning=True,contact_prediction=True).eval()
    missing,extra=model.load_state_dict(base.state_dict(),strict=False);assert not extra and all(k.startswith('body_surface_attention.') for k in missing)
    camera=torch.zeros(1,128,9);camera[...,3]=1;camera[...,7]=1
    s=BodySceneWindow([[0,0,0]],[0],np.arange(128),[np.empty((0,3))]*128)
    batch=camera_inputs_only(dict(camera=camera,occupancy=torch.zeros(1,24,48,48),bps=torch.zeros(1,128,67,3),bps_valid=torch.ones(1,128,67,dtype=torch.bool),rest=torch.randn(1,22,3)*.05,body_type=torch.tensor([[1.,0.]]),body_scale=torch.ones(1,1),body_scene_query=[s]))
    noise=torch.randn(1,128,201);t=torch.tensor([50])
    with torch.no_grad():assert torch.equal(base(noise,t,batch),model(noise,t,batch))
    poisoned=dict(batch,motion=torch.randn_like(noise)*100,trajectory=torch.randn(1,128,22,9)*100,contact_target=torch.ones(1,128,22))
    assert torch.equal(generate_without_body_initialization(model,batch,seed=33,steps=2),generate_without_body_initialization(model,poisoned,seed=33,steps=2))
    target_position=torch.randn(1,128,3)*.1;target_rotation=torch.eye(3).expand(1,128,3,3);valid=torch.ones(1,128,dtype=torch.bool)
    projected=project_head_pose(noise,batch['rest'],target_position,target_rotation,valid,valid)
    assert (forward_kinematics(projected,batch['rest'])[:,:,15]-target_position).abs().max()<1e-5
    assert (global_rotations(projected)[:,:,15]-target_rotation).abs().max()<1e-5
    masked=project_head_pose(noise,batch['rest'],target_position,target_rotation,~valid,~valid)
    assert torch.equal(masked[...,:135],noise[...,:135])
    variable=noise.clone().requires_grad_();project_head_pose(variable,batch['rest'],target_position,target_rotation,valid,valid).square().mean().backward();assert torch.isfinite(variable.grad).all()
    from scipy.spatial.transform import Rotation
    yaw=np.linspace(0,-220,128)
    rotations=Rotation.from_euler('y',yaw,degrees=True).as_matrix()@Rotation.from_euler('x',90,degrees=True).as_matrix()
    recovered=_yaw_path(torch.tensor(rotations[None]));assert np.max(abs(recovered.numpy()[0]-np.deg2rad(yaw)))<1e-6
    print('PASS timestamp/ghost masking, real attention scene sensitivity/gradients, all-unknown stability, zero-init original equivalence, GT poison invariance, exact differentiable masked head pose')
if __name__=='__main__':main()
