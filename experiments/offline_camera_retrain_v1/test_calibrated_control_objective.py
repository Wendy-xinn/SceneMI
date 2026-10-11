import torch
from experiments.offline_camera_retrain_v1.calibrated_control_objective import calibrated_control_losses,angular_balanced_control_losses

def main():
 truth=torch.zeros(1,32,201);rest=torch.zeros(1,22,3);signal=torch.ones(1)
 same=truth.clone().requires_grad_();a=calibrated_control_losses(same,truth,rest,signal);assert a['calibrated_control_total']<1e-7;a['calibrated_control_total'].backward();assert torch.isfinite(same.grad).all()
 pred=truth.clone();pred[:,:,:3]+=.025;pred.requires_grad_();a=calibrated_control_losses(pred,truth,rest,signal);assert torch.allclose(a['calibrated_head_position'],torch.tensor(1.5))
 half=calibrated_control_losses(pred,truth,rest,signal,torch.tensor([.5]));assert torch.allclose(half['calibrated_control_total'],a['calibrated_control_total']/2)
 none=calibrated_control_losses(pred,truth,rest,signal,torch.tensor([0.]));assert none['calibrated_control_total']==0
 a['calibrated_control_total'].backward();assert torch.isfinite(pred.grad).all()
 rot=truth.clone();rot[:,:,93:99]=torch.tensor([-1.,1.,0.,-1.,-1.,0.]);rot.requires_grad_()
 old=calibrated_control_losses(rot,truth,rest,torch.zeros(1));new=angular_balanced_control_losses(rot,truth,rest,torch.zeros(1));assert old['calibrated_head_body_relative']==0 and new['calibrated_head_body_relative']>1
 new['calibrated_control_total'].backward();assert torch.isfinite(rot.grad).all() and rot.grad.abs().sum()>0
 half=angular_balanced_control_losses(rot,truth,rest,torch.zeros(1),torch.tensor([.5]));assert torch.allclose(half['calibrated_control_total'],new['calibrated_control_total']/2)
 print('PASS declared 5cm units, identical finite gradients, confidence scales without denominator cancellation')
if __name__=='__main__':main()
