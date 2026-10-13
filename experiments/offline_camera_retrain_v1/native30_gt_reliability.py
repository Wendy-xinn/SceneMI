"""Training-only soft reliability for isolated source-fit candidates.

No target editing, smoothing, window removal or inference-side GT masks.
General high speed/large rotation and axis-angle wrapping are not downweighted.
"""
import numpy as np
import torch

from experiments.offline_camera_retrain_v1.audit_native_gt_quality import PARENTS,THRESHOLDS

DEFAULT_GT_RELIABILITY=dict(protocol='native30-isolated-candidate-soft-v1',enabled=True,
    candidate_weight=.25,rotation_step_deg=60.,isolated_return_deg=20.,
    isolated_joint_excursion_m=.08,isolated_neighbor_displacement_m=.04,
    derivative_policy='skip_stencils_touching_candidates',target_editing=False)


def validate_gt_reliability(config):
    if config['protocol']!=DEFAULT_GT_RELIABILITY['protocol'] or config['derivative_policy']!=DEFAULT_GT_RELIABILITY['derivative_policy'] or config['target_editing']:
        raise ValueError('Unknown GT reliability policy or target editing requested')
    if not 0<config['candidate_weight']<=1:raise ValueError('Candidate weight must be in (0,1]')
    for key in THRESHOLDS:
        if key in config and (not np.isfinite(config[key]) or config[key]<=0):raise ValueError('Invalid GT screen threshold')
    return config


def quality_weights(motion,rest,config):
    """Use a source halo before slicing, so random window edges do not hide events."""
    validate_gt_reliability(config);n=len(motion);position=np.ones((n,22),np.float32);local=position.copy()
    if not config['enabled'] or n<3:return dict(gt_position_weight=position,gt_local_rotation_weight=local,gt_global_rotation_weight=local.copy())
    six=motion[:,3:135].reshape(n,22,6)+np.array([1,0,0,0,1,0])
    r=np.stack((six[...,:3],six[...,3:],np.cross(six[...,:3],six[...,3:])),axis=-1)
    distance=lambda a,b:np.degrees(np.arccos(np.clip((np.sum(a*b,axis=(-2,-1))-1)/2,-1,1)))
    step=distance(r[1:],r[:-1]);back=distance(r[2:],r[:-2])
    rotation=(step[:-1]>config['rotation_step_deg'])&(step[1:]>config['rotation_step_deg'])&(back<config['isolated_return_deg'])
    j=motion[:,135:].reshape(n,22,3)*2
    left=np.linalg.norm(j[1:-1]-j[:-2],axis=-1);right=np.linalg.norm(j[1:-1]-j[2:],axis=-1)
    neighbors=np.linalg.norm(j[2:]-j[:-2],axis=-1)
    pulse=(np.minimum(left,right)>config['isolated_joint_excursion_m'])&(neighbors<config['isolated_neighbor_displacement_m'])
    local[1:-1][rotation]=config['candidate_weight'];position[1:-1][pulse]=config['candidate_weight']
    global_weight=local.copy()
    # Local rotations affect descendant orientation/positions, not their own
    # joint position; a wrist/toe leaf rotation never invalidates the whole body.
    for k in range(1,22):
        parent=PARENTS[k]
        global_weight[:,k]=np.minimum(global_weight[:,k],global_weight[:,parent])
        position[:,k]=np.minimum(position[:,k],np.minimum(position[:,parent],global_weight[:,parent]))
    return dict(gt_position_weight=position,gt_local_rotation_weight=local,gt_global_rotation_weight=global_weight)


def gt_weight(batch,key,reference):
    value=batch.get(key)
    return reference.new_ones(reference.shape[:2]+(22,)) if value is None else value.to(reference)


def derivative_valid(weight,lag=1):
    """Every frame in the stencil must be reliable, including interior frames."""
    valid=weight[:,lag:]>=1
    for offset in range(lag):valid=valid&(weight[:,offset:weight.shape[1]-lag+offset]>=1)
    return valid.to(weight)


def soft_mean(value,weight,gate=None):
    """Keep the full denominator: soft weights attenuate rather than cancel."""
    while weight.ndim<value.ndim:weight=weight[...,None]
    per=(value*weight).flatten(1).mean(1)
    if gate is not None:per=per*gate.detach().reshape(-1)
    return per.mean()
