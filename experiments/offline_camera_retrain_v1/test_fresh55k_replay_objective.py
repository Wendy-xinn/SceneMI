"""Verify the new planted displacement target is differentiable and active."""
import json
from pathlib import Path
import torch
from experiments.offline_camera_retrain_v1.sole_support_objective import SoleCache
from experiments.offline_camera_retrain_v1.fresh55k_replay_objective import replay_losses
def main():
    torch.set_num_threads(4)
    identity=json.loads((Path(__file__).parent/'runs/state_relative_spline_oct10/demo/manifest.json').read_text())['cases'][0]['identity']
    cache=SoleCache();skin=cache.get(identity['native_body']);rest=skin.joints[:22][None]*skin.scale
    truth=torch.zeros(1,64,201,device='cuda');soles=cache.soles(truth,[identity]);pred=truth.clone().requires_grad_()
    loss=replay_losses(pred,truth,rest,cache.soles(pred,[identity]),soles,truth,torch.ones(1,device='cuda'))
    assert loss['replay_total']<1e-6;loss['replay_total'].backward();assert torch.isfinite(pred.grad).all()
    pred=truth.clone();pred[:,:,0]=torch.linspace(0,.15,64,device='cuda')
    loss=replay_losses(pred,truth,rest,cache.soles(pred,[identity]),soles,truth,torch.ones(1,device='cuda'))
    assert loss['planted_displacement']>0 and loss['stance_velocity']>0
    print('PASS identical zero/finite gradient, multi-frame planted drift detection')
if __name__=='__main__':main()
