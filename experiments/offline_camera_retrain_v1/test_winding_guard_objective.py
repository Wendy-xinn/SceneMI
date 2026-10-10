import math
import torch
from experiments.offline_camera_retrain_v1.winding_guard_objective import winding_guard_losses
def motion(angle):
    m=torch.zeros(1,len(angle),201);r=torch.zeros(len(angle),3,3)
    r[:,0,0]=r[:,2,2]=angle.cos();r[:,0,2]=angle.sin();r[:,2,0]=-angle.sin();r[:,1,1]=1
    m[:,:,3:9]=torch.cat((r[:,:,0],r[:,:,1]),-1)-torch.tensor([1.,0,0,0,1,0]);return m
def main():
    truth=motion(torch.linspace(0,-math.pi,128));same=truth.clone().requires_grad_()
    o=winding_guard_losses(same,truth,torch.zeros(1));assert o['winding_guard_total']<1e-7;o['winding_guard_total'].backward();assert torch.isfinite(same.grad).all()
    bad=motion(torch.linspace(0,math.pi,128)).requires_grad_()
    assert torch.allclose(bad[0,-1],truth[0,-1],atol=1e-6)
    o=winding_guard_losses(bad,truth,torch.zeros(1));assert o['head_winding_path']>1;o['winding_guard_total'].backward();assert torch.isfinite(bad.grad).all()
    angle=torch.zeros(128);angle[64]=.4;o=winding_guard_losses(motion(angle),motion(torch.zeros(128)),torch.ones(1));assert o['rotation_excess_tail']>0
    print('PASS same endpoints opposite winding at high noise, identical finite gradients, isolated rate spike')
if __name__=='__main__':main()
