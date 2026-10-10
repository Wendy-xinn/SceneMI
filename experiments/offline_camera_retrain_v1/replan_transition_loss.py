"""Short transition objective from measured past, without a hard future head.

Synthetic execution errors keep GT future targets in the ordinary objective.
Only this four-frame auxiliary objective uses GT relative displacement added
onto the actually observed final state. It is a training target, not input.
"""
import torch
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.body_history_condition import HISTORY_FRAMES


def boundary_continuity_losses(prediction, truth, history, rest, signal_weight):
    h=HISTORY_FRAMES; n=4
    actual=forward_kinematics(history,rest).detach()
    gt=forward_kinematics(truth[:,h-1:h+n],rest).detach()
    future=forward_kinematics(prediction[:,h:h+n],rest)
    target=actual[:,-1:]+gt[:,1:]-gt[:,:1]
    decay=future.new_tensor([1.,.75,.5,.25])[None,:,None,None]
    displacement=((future-target).square()*decay).mean((1,2,3))
    root=((future[:,:,0]-target[:,:,0]).square()*decay[:,:,0]).mean((1,2))
    # Include the cross-boundary velocity excluded by future-only losses.
    joined=torch.cat((actual[:,-1:],future),1)
    velocity=((joined[:,1:]-joined[:,:-1]-(gt[:,1:]-gt[:,:-1])).square()*decay).mean((1,2,3))
    weight=signal_weight.square()
    terms={'boundary_fk_mse':(weight*displacement).mean(),
           'boundary_root_mse':(weight*root).mean(),
           'boundary_velocity_mse':(weight*velocity).mean()}
    terms['boundary_total']=10*terms['boundary_fk_mse']+20*terms['boundary_root_mse']+20*terms['boundary_velocity_mse']
    return terms
