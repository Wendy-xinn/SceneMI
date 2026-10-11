import torch
from experiments.offline_camera_retrain_v1.absolute_root_objective import absolute_root_loss
from experiments.offline_camera_retrain_v1.calibrated_control_objective import calibrated_control_losses

def main():
    torch.manual_seed(41);truth=torch.zeros(1,32,201);pred=(torch.randn_like(truth)*.15).requires_grad_();rest=torch.zeros(1,22,3)
    relative=calibrated_control_losses(pred,truth,rest,torch.ones(1))['calibrated_head_body_relative']
    g=torch.autograd.grad(relative,pred,retain_graph=True)[0];assert g[:,:,3:9].abs().max()<1e-6
    absolute=absolute_root_loss(pred,truth,torch.zeros(1));h=torch.autograd.grad(absolute,pred)[0]
    assert torch.isfinite(h).all() and h[:,:,3:9].abs().sum()>1e-3
    same=truth.clone().requires_grad_();loss=absolute_root_loss(same,truth,torch.zeros(1));loss.backward();assert loss<1e-6 and torch.isfinite(same.grad).all()
    print('PASS relative head-body loss has zero common-root gradient; absolute pelvis objective has nonzero finite root gradient at high noise')
if __name__=='__main__':main()
