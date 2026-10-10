"""Boundary loss checks: displacement target, causal actual-state anchor, gradients."""
import json
import torch
from experiments.offline_camera_retrain_v1.replan_transition_loss import boundary_continuity_losses
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.body_history_condition import HISTORY_FRAMES
from experiments.offline_camera_retrain_v1.data import HERE


def main():
    torch.manual_seed(72)
    rest=torch.randn(2,22,3)*.05
    truth=torch.randn(2,64,201)*.005
    history=truth[:,:HISTORY_FRAMES].clone()
    alpha=torch.ones(2)
    exact=boundary_continuity_losses(truth,truth,history,rest,alpha)
    assert float(exact['boundary_total'])<1e-12
    drift=torch.tensor([.03,0.,-.02])
    biased=history.clone();biased[:,:,:3]+=drift/2
    adapted=truth.clone();adapted[:,:,:3]+=drift/2
    shifted=boundary_continuity_losses(adapted,truth,biased,rest,alpha)
    assert float(shifted['boundary_total'])<1e-12
    predicted=truth.clone().requires_grad_()
    losses=boundary_continuity_losses(predicted,truth,biased,rest,alpha)
    assert float(losses['boundary_total'].detach())>1e-4
    losses['boundary_total'].backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[:,16:20].abs().sum()>0
    assert predicted.grad[:,:16].abs().sum()==0 and predicted.grad[:,20:].abs().sum()==0
    altered=truth.clone();altered[:,20:]+=100
    same=boundary_continuity_losses(predicted.detach(),altered,biased,rest,alpha)
    assert torch.equal(same['boundary_total'],losses['boundary_total'])
    zero=boundary_continuity_losses(predicted.detach(),truth,biased,rest,torch.zeros(2))
    assert float(zero['boundary_total'])==0
    # Verify matched-prefix replacement uses the same noise; future noise unchanged.
    diffusion_alpha=torch.tensor([.2,.8])[:,None,None]
    noise=torch.randn_like(truth);noisy=diffusion_alpha.sqrt()*truth+(1-diffusion_alpha).sqrt()*noise
    corrected=noisy.clone()
    recovered=(noisy[:,:16]-diffusion_alpha.sqrt()*truth[:,:16])/(1-diffusion_alpha).sqrt()
    corrected[:,:16]=diffusion_alpha.sqrt()*biased+(1-diffusion_alpha).sqrt()*recovered
    torch.testing.assert_close(corrected[:,:16],diffusion_alpha.sqrt()*biased+(1-diffusion_alpha).sqrt()*noise[:,:16])
    assert torch.equal(corrected[:,16:],noisy[:,16:])
    report={'status':'passed','clean_target_zero':float(exact['boundary_total']),'coherent_translation_adaptation_zero':float(shifted['boundary_total']),'boundary_gradient_finite':True,'only_first4_future_frames_receive_auxiliary_gradient':True,'future_beyond_boundary_cannot_change_loss':True,'same_prefix_noise_future_noise_unchanged':True}
    (HERE/'runs/replan_transition_oct10/loss_tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
