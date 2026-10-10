import torch,numpy as np
from experiments.offline_camera_retrain_v1.body_local_scene_model import BodyLocalSceneMI
from experiments.offline_camera_retrain_v1.body_local_scene import BodySceneWindow
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.known_input_generation import camera_inputs_only,generate_without_body_initialization

def main():
 torch.set_num_threads(2);torch.manual_seed(1133);baseline=OfflineSceneMI(64,(1,2),body_conditioning=True,contact_prediction=True).eval()
 with torch.no_grad():
  for name,p in baseline.named_parameters():
   if p.ndim>=2 and not p.count_nonzero():p.normal_(0,.01) # stand in for trained, nonzero output layers
 local=BodyLocalSceneMI(64,(1,2),body_conditioning=True,contact_prediction=True).eval();missing,extra=local.load_state_dict(baseline.state_dict(),strict=False);assert not extra and all(k.startswith('body_local_encoder.') for k in missing)
 scene=BodySceneWindow(np.array([[0.,0.,0.]]),np.array([0.]),np.arange(128),[np.empty((0,3))]*128)
 camera=torch.zeros(1,128,9);camera[:,:,3]=1;camera[:,:,7]=1
 batch=dict(camera=camera,occupancy=torch.zeros(1,24,48,48),bps=torch.zeros(1,128,67,3),bps_valid=torch.ones(1,128,67,dtype=torch.bool),rest=torch.zeros(1,22,3),body_type=torch.tensor([[1.,0.]]),body_scale=torch.ones(1,1),body_scene_query=[scene]);batch=camera_inputs_only(batch);noise=torch.randn(1,128,201);t=torch.tensor([50])
 with torch.no_grad():a=baseline(noise,t,batch);b=local(noise,t,batch)
 assert torch.equal(a,b),(a-b).abs().max()
 local(noise,t,batch).square().mean().backward();assert local.body_local_encoder[-1].weight.grad.abs().sum()>0
 poisoned=dict(batch,motion=torch.randn(1,128,201)*100,joints=torch.randn(1,128,22,3)*100,trajectory=torch.randn(1,128,22,9)*100,contact_target=torch.ones(1,128,22))
 a=generate_without_body_initialization(local,batch,seed=33,steps=2);b=generate_without_body_initialization(local,poisoned,seed=33,steps=2);assert torch.equal(a,b)
 print('PASS zero adapter exactly original prediction; local adapter receives gradients; GT body/contact poisoning leaves sampling identical')
if __name__=='__main__':main()
