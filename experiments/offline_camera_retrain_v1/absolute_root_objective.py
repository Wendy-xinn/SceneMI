"""Training-only absolute native pelvis SO(3), no GT inference state."""
import torch
from torch.nn import functional as F
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.calibrated_control_objective import rotation_residual,angular_balanced_control_losses
ROOT_WEIGHT=.20 # TRAIN norm calibration: four times .05 probe
ROOT_TOLERANCE_DEG=15.
def absolute_root_loss(pred,truth,signal):
    p=global_rotations(pred)[:,:,0];t=global_rotations(truth.detach())[:,:,0].detach()
    residual=rotation_residual(p,t,ROOT_TOLERANCE_DEG)
    per_frame=F.smooth_l1_loss(residual,torch.zeros_like(residual),reduction='none')
    weight=.25+.75*signal.detach().reshape(-1)
    return (per_frame.mean(1)*weight).mean()
def root_facing_control_losses(pred,truth,rest,signal,confidence=None):
    losses=angular_balanced_control_losses(pred,truth,rest,signal,confidence)
    root=absolute_root_loss(pred,truth,signal)
    losses['absolute_pelvis_orientation']=root
    losses['weighted_absolute_pelvis_orientation']=ROOT_WEIGHT*root
    losses['calibrated_control_total']=losses['calibrated_control_total']+ROOT_WEIGHT*root
    return losses
