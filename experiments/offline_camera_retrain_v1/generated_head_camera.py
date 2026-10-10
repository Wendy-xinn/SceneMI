"""Display-only camera reconstruction from native generated body and fixed mount."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

def head_world_track(motion,native_joints,anchor_rotation,origin):
    rotations=global_rotations(torch.as_tensor(motion,dtype=torch.float32)[None])[0,:,15].numpy()
    return native_joints[:,15]@anchor_rotation+origin,anchor_rotation.T@rotations

def apply_camera_mount(head_position,head_rotation,mount_rotation,mount_translation):
    return (head_position+np.einsum('tij,j->ti',head_rotation,mount_translation),head_rotation@mount_rotation)

def diagnostic_fixed_mount(reference_head_position,reference_head_rotation,camera_position,camera_rotation):
    # Explicit GT/reference first-frame calibration ONLY for display diagnostics.
    # No per-frame fitting, time alignment, pose/root reset or generation edits.
    r=reference_head_rotation[0];return r.T@camera_rotation[0],r.T@(camera_position[0]-reference_head_position[0])

def camera_errors(position,rotation,reference_position,reference_rotation):
    distance=np.linalg.norm(position-reference_position,axis=-1)*100
    degrees=np.rad2deg(Rotation.from_matrix((reference_rotation.transpose(0,2,1)@rotation).copy()).magnitude())
    return dict(position_mean_cm=float(distance.mean()),position_p95_cm=float(np.quantile(distance,.95)),position_max_cm=float(distance.max()),orientation_mean_deg=float(degrees.mean()),orientation_p95_deg=float(np.quantile(degrees,.95)),orientation_max_deg=float(degrees.max()))

def frustum_lines(position,rotation,depth=.12,hfov=66.56,vfov=40.49):
    x=depth*np.tan(np.deg2rad(hfov/2));y=depth*np.tan(np.deg2rad(vfov/2))
    corners=np.array([[-x,-y,depth],[x,-y,depth],[x,y,depth],[-x,y,depth]])@rotation.T+position
    return np.concatenate((np.stack((np.tile(position,(4,1)),corners),1),np.stack((corners,np.roll(corners,-1,axis=0)),1))).astype(np.float32)
