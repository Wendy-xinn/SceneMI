import numpy as np
import torch
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.turn_manifold_sampling import sample_turn_manifold
from experiments.offline_camera_retrain_v1.hard_head_coordination import _yaw_path
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

class ToyPrior(torch.nn.Module):
    def forward(self,noisy,timestep,batch,**kwargs):return noisy*.05

def main():
    torch.set_num_threads(2);torch.manual_seed(88);t=64
    camera=torch.zeros(1,t,9);camera[...,3]=1;camera[...,7]=1
    batch=dict(camera=camera,rest=torch.randn(1,22,3)*.05)
    reference=torch.zeros(1,t,201);position=torch.randn(1,t,3)*.1
    rotation=torch.tensor(Rotation.from_euler('y',np.linspace(0,-220,t),degrees=True).as_matrix()[None],dtype=torch.float32)
    model=ToyPrior().train();output,audit=sample_turn_manifold(model,batch,position,rotation,reference,seed=117,steps=4)
    assert model.training and audit['projected_steps']==2
    assert (forward_kinematics(output,batch['rest'])[:,:,15]-position).abs().max()<1e-5
    rotations=global_rotations(output)
    assert (rotations[:,:,15]-rotation).abs().max()<1e-5
    assert (_yaw_path(rotations[:,:,0])-_yaw_path(rotation)).abs().max()<1e-5
    poison=dict(batch,motion=torch.randn(1,t,201)*100,trajectory=torch.randn(1,t,22,9)*100,contact_target=torch.ones(1,t,22))
    changed,_=sample_turn_manifold(model,poison,position,rotation,reference,seed=117,steps=4)
    assert torch.equal(changed,output)
    print('PASS all-noise sampling, complete continuous turn/head constraints, GT-field isolation, training-mode restoration')
if __name__=='__main__':main()
