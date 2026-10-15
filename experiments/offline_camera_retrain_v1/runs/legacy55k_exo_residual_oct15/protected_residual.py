"""Conservative inference guard: only move fully unobserved descendant branches.

This can repair a hidden foot/wrist branch without moving any available joint
position. It deliberately cannot solve hidden joints between visible anchors;
those need a coupled constraint solver rather than independent rotations.
"""
import torch
from experiments.offline_camera_retrain_v1.supervision import PARENTS
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion


def protected_residual(initial,details,available,rest):
    descendants=torch.zeros(22,22,device=initial.device,dtype=initial.dtype)
    for child in range(1,22):
        parent=int(PARENTS[child])
        while parent>=0:
            descendants[parent,child]=1
            if parent==0:break
            parent=int(PARENTS[parent])
    observed_descendants=torch.einsum('btj,kj->btk',available.to(initial),descendants)>0
    rotation=details['rotation']*(~observed_descendants)[...,None]
    translation=details['translation']*(~available.any(-1))[...,None]
    return reconcile_motion(initial,rest,translation,rotation)
