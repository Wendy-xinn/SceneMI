"""Training-only gait candidate: legacy foot targets and explicit bilateral motion.

Height/envelope targets are fitted GT feet, never an inferred scene floor.
Every temporal relation follows the target (including hopping/stationary motion),
not a prescribed left/right gait phase. No smoothing or target editing is used.
"""
import copy
import math
import torch
from experiments.offline_camera_retrain_v1.native30_gt_reliability import derivative_valid
from experiments.offline_camera_retrain_v1.native30_gait_supervision import (
    DEFAULT as V1_DEFAULT, gait_terms as v1_terms,
)

PROTOCOL = 'native30-gait-bilateral-v2'
DEFAULT = dict(enabled=True, protocol=PROTOCOL, fps=30, stance_speed_m_s=.2,
    stride_lags_s=[.2, .4], coordination_window_s=1.6, raw_gait_weight=.5,
    signal_gate='alpha_bar', coefficients=dict(V1_DEFAULT['coefficients'],
        support_height=2., swing_height=.5, support_envelope=4., support_penetration=2.,
        phase_free_planting=.1, bilateral_stride=1., bilateral_amplitude=5., bilateral_relation=.05))


def validate_config(options):
    if options.get('enabled') is not True or options.get('protocol') != PROTOCOL or options.get('fps') != 30:
        raise ValueError('Unknown bilateral gait supervision')
    if options.get('stride_lags_s') != [.2, .4] or options.get('coordination_window_s') != 1.6:
        raise ValueError('Gait physical-time windows changed')
    if options.get('signal_gate') != 'alpha_bar':
        raise ValueError('V2 requires an explicit alpha_bar gate, including support proxies')
    for key in ('stance_speed_m_s', 'raw_gait_weight'):
        value=options.get(key)
        if not isinstance(value,(float,int)) or not math.isfinite(value) or value<0:
            raise ValueError('Invalid gait scalar: '+key)
    if options['stance_speed_m_s']<=0 or options['raw_gait_weight']>1:
        raise ValueError('Invalid stance threshold or raw gait weight')
    values=options.get('coefficients',{})
    if set(values)!=set(DEFAULT['coefficients']) or any(not math.isfinite(v) or v<0 for v in values.values()):
        raise ValueError('Invalid bilateral gait coefficients')
    return options


def gait_terms(joints,target,quality,signal,options,*,root_rotations=None,target_root_rotations=None):
    validate_config(options)
    old=copy.deepcopy(V1_DEFAULT)
    old['stance_speed_m_s']=options['stance_speed_m_s']
    old['coefficients']={k:options['coefficients'][k] for k in old['coefficients']}
    terms=v1_terms(joints,target,quality,signal,old)
    target=target.detach();gate=signal.detach().reshape(len(joints))
    def reduce(value,mask):
        mask=mask.to(value)
        per=(value*mask).flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1)
        return (per*gate).mean()
    feet=joints[:,:,(10,11)];true_feet=target[:,:,(10,11)]
    q=quality[:,:,(10,11)]
    velocity=feet.diff(dim=1)*options['fps'];true_velocity=true_feet.diff(dim=1)*options['fps']
    reliable=derivative_valid(q);stance=true_velocity.norm(dim=-1)<options['stance_speed_m_s']
    height=(feet[:,1:,:,1]-true_feet[:,1:,:,1]).square()
    terms['support_height']=reduce(height,reliable*stance)
    terms['swing_height']=reduce(height,reliable*~stance)
    envelope=true_feet[...,1].min(-1).values
    point_valid=(q>=1).all(-1)
    terms['support_envelope']=reduce((feet[...,1].min(-1).values-envelope).square(),point_valid)
    terms['support_penetration']=reduce((envelope[...,None]-feet[...,1]).clamp_min(0).square(),point_valid[...,None].expand_as(q))
    speed=velocity.norm(dim=-1).min(-1).values
    true_speed=true_velocity.norm(dim=-1).min(-1).values
    support=(true_speed<options['stance_speed_m_s'])&reliable.bool().all(-1)
    terms['phase_free_planting']=reduce((speed-true_speed).clamp_min(0).square(),support)

    if (root_rotations is None)!=(target_root_rotations is None):
        raise ValueError('Supply both predicted and target root rotations')
    relative=feet-joints[:,:,:1];true_relative=true_feet-target[:,:,:1]
    if root_rotations is not None:
        shape=(*joints.shape[:2],3,3)
        if root_rotations.shape!=shape or target_root_rotations.shape!=shape:
            raise ValueError('Root rotations must have shape [B,T,3,3]')
        relative=(root_rotations.transpose(-1,-2)[:,:,None]@relative[...,None]).squeeze(-1)
        true_relative=(target_root_rotations.detach().transpose(-1,-2)[:,:,None]@true_relative[...,None]).squeeze(-1)
    # Both feet and root must be reliable; skip all stencils touching a pulse.
    pair_quality=torch.minimum(q.amin(-1),quality[:,:,0])
    difference=relative[:,:,0]-relative[:,:,1]
    true_difference=true_relative[:,:,0]-true_relative[:,:,1]
    stride=joints.sum()*0
    lags=[round(s*options['fps']) for s in options['stride_lags_s']]
    for lag in lags:
        if joints.shape[1]<=lag:continue
        error=(difference[:,lag:]-difference[:,:-lag])-(true_difference[:,lag:]-true_difference[:,:-lag])
        stride+=reduce(error.square().sum(-1),derivative_valid(pair_quality,lag))/len(lags)
    terms['bilateral_stride']=stride
    width=min(round(options['coordination_window_s']*options['fps']),joints.shape[1])
    step=max(1,width//2)
    def windows(x):
        w=x.unfold(1,width,step).movedim(-1,2)
        return w-w.mean(2,keepdim=True)
    pred,truth=windows(relative),windows(true_relative)
    valid=pair_quality.unfold(1,width,step).amin(-1)>=1
    pred_diff=pred[:,:,:,0]-pred[:,:,:,1]
    true_diff=truth[:,:,:,0]-truth[:,:,:,1]
    floor=.03**2
    rms=lambda x:(x.square().sum(-1).mean(2)+floor).sqrt()
    terms['bilateral_amplitude']=reduce((rms(pred_diff)-rms(true_diff)).square(),valid)
    def relation(x):
        covariance=(x[:,:,:,0]*x[:,:,:,1]).sum(-1).mean(2)
        scale=(x.square().sum(-1).mean(2)+floor).sqrt()
        return covariance/(scale[:,:,0]*scale[:,:,1])
    terms['bilateral_relation']=reduce((relation(pred)-relation(truth)).square(),valid)
    terms['total']=sum(options['coefficients'][k]*terms[k] for k in options['coefficients'])
    return terms
