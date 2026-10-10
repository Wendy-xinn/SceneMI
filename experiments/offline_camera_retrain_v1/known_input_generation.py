"""Original55k generation from known camera/scene and random noise only.

Body templates are configured parameters, not initial poses. No executed/GT
body prefix, target contact, or body-track slots are forwarded to the model.
"""
import torch
from experiments.offline_camera_retrain_v1.scene_model import ddim_sample
from experiments.offline_camera_retrain_v1.control import fixed_control_mask

def camera_inputs_only(observations):
    allowed=('camera','occupancy','bps','bps_valid','rest','body_type','body_scale')
    inputs={k:observations[k] for k in allowed if k in observations}
    camera=inputs['camera'];b,t=camera.shape[:2]
    trajectory=camera.new_zeros(b,t,22,9);trajectory[:,:,15]=camera
    inputs['trajectory']=trajectory
    return inputs

@torch.no_grad()
def generate_without_body_initialization(model,observations,*,seed=777,steps=20,use_scene=True):
    inputs=camera_inputs_only(observations);b,t=inputs['camera'].shape[:2]
    # Keeps the official55k camera semantics; no anatomical GT head conversion.
    return ddim_sample(model,inputs,t,seed=seed,steps=steps,
                       control_mask=fixed_control_mask(b,t,'head',inputs['camera'].device),use_scene=use_scene)
