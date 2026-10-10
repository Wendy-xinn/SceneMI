"""Training-only absolute head and head/body losses in declared tolerances.

GT is a supervision label, never initial state or inference pose projection.
The current synthetic camera/head mount is learned implicitly as in original55k.
"""
import math
import torch
from torch.nn import functional as F
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

CONTROL_WEIGHT_SCALE=.1 # selected using TRAIN shared condition-encoder gradient audit
POSITION_TOLERANCE_M=.05
HEAD_TOLERANCE_DEG=5.
BODY_RELATIVE_TOLERANCE_DEG=15.
def rotation_residual(a,b,tolerance_deg):
    # sqrt-chordal = 2 sin(theta/2); finite gradient at identical rotations.
    distance=(.5*(a-b).square().sum((-1,-2))+1e-8).sqrt()-1e-4
    return distance/(2*math.sin(math.radians(tolerance_deg)/2))
def calibrated_control_losses(pred,truth,rest,signal,confidence=None):
    confidence=pred.new_ones(len(pred)) if confidence is None else confidence.detach().to(pred).clamp(0,1)
    weight=(.25+.75*signal.detach().reshape(-1))*confidence
    def reduce(x,w):return (x.flatten(1).mean(1)*w).mean()
    j=forward_kinematics(pred,rest);gtj=forward_kinematics(truth.detach(),rest).detach()
    p=global_rotations(pred);t=global_rotations(truth).detach()
    pos=reduce(F.smooth_l1_loss((j[:,:,15]-gtj[:,:,15])/POSITION_TOLERANCE_M,torch.zeros_like(j[:,:,15]),reduction='none').sum(-1),weight)
    angle=rotation_residual(p[:,:,15],t[:,:,15],HEAD_TOLERANCE_DEG)
    orient=reduce(F.smooth_l1_loss(angle,torch.zeros_like(angle),reduction='none'),weight)
    relative=[]
    for joint in (0,9,12):
        rp=p[:,:,joint].transpose(-1,-2)@p[:,:,15];rt=t[:,:,joint].transpose(-1,-2)@t[:,:,15]
        residual=rotation_residual(rp,rt,BODY_RELATIVE_TOLERANCE_DEG)
        relative.append(reduce(F.smooth_l1_loss(residual,torch.zeros_like(residual),reduction='none'),signal.detach().reshape(-1)*confidence))
    coupling=torch.stack(relative).mean()
    return dict(calibrated_head_position=pos,calibrated_head_orientation=orient,calibrated_head_body_relative=coupling,calibrated_control_total=CONTROL_WEIGHT_SCALE*(.20*pos+.02*orient+.10*coupling))
