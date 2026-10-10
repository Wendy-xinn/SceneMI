import torch
from experiments.offline_camera_retrain_v1.turn_path_objective import turn_path_losses
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D
B,T=2,128
truth=torch.zeros(B,T,201);rest=torch.randn(B,22,3)*.1
same=truth.clone().requires_grad_();loss=turn_path_losses(same,truth,rest,truth,torch.ones(B));assert loss['turn_total'].abs()<1e-6;loss['turn_total'].backward();assert torch.isfinite(same.grad).all()
angle=torch.linspace(0,2,T);ca,sa=angle.cos(),angle.sin();truth[:,:,3:9]=torch.stack([ca,torch.zeros(T),-sa,torch.zeros(T),torch.ones(T),torch.zeros(T)],-1)-torch.tensor(IDENTITY_6D)
bad=truth.clone();bad[:,:,3:9]=torch.stack([ca,torch.zeros(T),sa,torch.zeros(T),torch.ones(T),torch.zeros(T)],-1)-torch.tensor(IDENTITY_6D);bad.requires_grad_();loss=turn_path_losses(bad,truth,rest,truth,torch.ones(B));assert loss['turn_path']>1;loss['turn_total'].backward();assert torch.isfinite(bad.grad).all()
zero=turn_path_losses(bad,truth,rest,truth,torch.zeros(B));assert zero['turn_total']==0
print('PASS: identical/finite gradient; reverse winding detected; noise gating')

impulse=torch.zeros(B,T,201);impulse[:,64,3:9]=torch.tensor([.6967067,0.,-.7173561,0.,1.,0.])-torch.tensor(IDENTITY_6D)
result=turn_path_losses(impulse,torch.zeros_like(impulse),rest,torch.zeros_like(impulse),torch.ones(B))
assert result['root_leg_rate_excess']>0
print('PASS: isolated root angular spike triggers rate guard')
