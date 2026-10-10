import json
from pathlib import Path
import torch
from experiments.offline_camera_retrain_v1.sole_support_objective import SoleCache,sole_support_losses
from experiments.offline_camera_retrain_v1.native_surface_points import NativeSurfacePoints

def main():
    torch.set_num_threads(4);case=json.loads((Path(__file__).parent/'runs/state_relative_spline_oct10/demo/manifest.json').read_text())['cases'][0];body=case['identity']['native_body'];cache=SoleCache();full=NativeSurfacePoints(body)
    truth=torch.zeros(1,64,201,device='cuda');target=cache.soles(truth,[case['identity']]);reference=full.soles(full(truth[0]));assert torch.max(abs(target[0]-reference))<1e-5
    same=truth.clone().requires_grad_();soles=cache.soles(same,[case['identity']]);loss=sole_support_losses(same,truth,soles,target,truth,torch.ones(1,device='cuda'));assert loss['sole_total']<1e-7;loss['sole_total'].backward();assert torch.isfinite(same.grad).all()
    lifted=truth.clone();lifted[:,:,1]+=.04;lifted.requires_grad_();loss=sole_support_losses(lifted,truth,cache.soles(lifted,[case['identity']]),target,truth,torch.ones(1,device='cuda'));assert loss['sole_floating_15mm_band']>0 and loss['sole_total']>0;loss['sole_total'].backward();assert torch.isfinite(lifted.grad).all() and lifted.grad[:,:,1].sum()>0
    moving=truth.clone();moving[:,:,0]=torch.linspace(0,.1,64,device='cuda');moving_soles=cache.soles(moving,[case['identity']]);loss=sole_support_losses(moving,truth,moving_soles,target,truth,torch.ones(1,device='cuda'));assert loss['sole_plant_velocity']>0
    flight=truth.clone();flight[:,:,1]=torch.sin(torch.linspace(0,3.14159,64,device='cuda'))*.1;flight_soles=cache.soles(flight,[case['identity']]);loss=sole_support_losses(flight,flight,flight_soles,flight_soles,flight,torch.ones(1,device='cuda'));assert loss['sole_total']<1e-7
    print('PASS native sole subset, identical finite gradients, lift correction gradient, planted sliding, legitimate flight')
if __name__=='__main__':main()
