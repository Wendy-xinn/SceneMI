"""Explicit observation-error profiles; never perturb body/contact targets."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation

from experiments.offline_camera_retrain_v1.data import rotation6d

DEFAULT_INPUT_NOISE = dict(
    protocol='native30-smooth-observation-error-v1',clean_probability=.5,
    ego_head=dict(offset_sigma_m=.01,endpoint_sigma_m=.02,rotation_offset_deg=2.,rotation_endpoint_deg=3.),
    ego_wrists=dict(enabled=True,offset_sigma_m=.005,endpoint_sigma_m=.01,rotation_offset_deg=2.,rotation_endpoint_deg=3.),
    exo=dict(offset_sigma_m=.01,endpoint_sigma_m=.02,shared_offset_sigma_m=.005,shared_endpoint_sigma_m=.01),
    gap_probability=.3,gap_seconds=.6,ego_gap_joints=[15],wrist_extra_gap=False)


def validate_input_noise(config):
    if config['protocol']!=DEFAULT_INPUT_NOISE['protocol']:raise ValueError('Unknown observation noise protocol')
    if not 0<=config['clean_probability']<=1 or not 0<=config['gap_probability']<=1:
        raise ValueError('Noise probabilities must be in [0,1]')
    if not 0<config['gap_seconds']<=6.4:raise ValueError('Invalid observation gap duration')
    if config['ego_gap_joints']!=[15] or config['wrist_extra_gap']:
        raise ValueError('This profile adds ego gaps to head only')
    for key in ('ego_head','ego_wrists','exo'):
        for name,value in config[key].items():
            if name!='enabled' and (not np.isfinite(value) or value<0):
                raise ValueError(f'Invalid noise amplitude: {key}/{name}')
    return config


def apply_input_noise(trajectory,meta,cam,ca,cc,uncertainty,mode,joints,rng,config):
    """50% whole observations clean by default; otherwise coherent SE(3) errors.

    The ego camera and anatomical head are linked by inverse_head afterwards.
    Wrist errors only affect selected wrists; exo rotations are never invented.
    """
    validate_input_noise(config)
    if rng.random()<config['clean_probability']:return False
    frames=len(cam);clock=np.linspace(0,1,frames)[:,None]
    def drift(profile,shared=False):
        prefix='shared_' if shared else ''
        return rng.normal(0,profile[prefix+'offset_sigma_m'],(1,3))+clock*rng.normal(0,profile[prefix+'endpoint_sigma_m'],(1,3))
    def position(values,confidence,uncertainty_values,profile,extra=None):
        error=drift(profile)
        if extra is not None:error+=extra
        values[:,:3]+=error/2
        confidence*=np.exp(-np.linalg.norm(error,axis=-1)/.05)
        uncertainty_values[:]=np.maximum(uncertainty_values,np.linalg.norm(error,axis=-1))
    def orientation(values,confidence,uncertainty_values,profile):
        error=rng.normal(0,np.deg2rad(profile['rotation_offset_deg']),(1,3))+clock*rng.normal(0,np.deg2rad(profile['rotation_endpoint_deg']),(1,3))
        six=values[:,3:];r=np.stack((six[:,:3],six[:,3:],np.cross(six[:,:3],six[:,3:])),axis=-1)
        values[:,3:]=rotation6d(Rotation.from_rotvec(error).as_matrix()@r)
        confidence*=np.exp(-np.linalg.norm(error,axis=-1)/np.deg2rad(10))
        uncertainty_values[:]=np.maximum(uncertainty_values,np.linalg.norm(error,axis=-1))
    if mode=='ego':
        profile=config['ego_head']
        position(cam,cc[:,0],uncertainty[:,15,0],profile)
        orientation(cam,cc[:,1],uncertainty[:,15,1],profile)
        if config['ego_wrists']['enabled']:
            for joint in (20,21):
                if joint not in joints:continue
                profile=config['ego_wrists']
                position(trajectory[:,joint],meta[:,joint,2],uncertainty[:,joint,0],profile)
                orientation(trajectory[:,joint],meta[:,joint,3],uncertainty[:,joint,1],profile)
        eligible=tuple(j for j in config['ego_gap_joints'] if j in joints)
    else:
        shared=drift(config['exo'],shared=True)
        for joint in joints:position(trajectory[:,joint],meta[:,joint,2],uncertainty[:,joint,0],config['exo'],shared)
        eligible=tuple(joints)
    if eligible and rng.random()<config['gap_probability']:
        joint=int(rng.choice(eligible));length=min(frames,int(round(config['gap_seconds']*30)))
        first=int(rng.integers(max(1,frames-length+1)));meta[first:first+length,joint,:4]=0
        if mode=='ego' and joint==15:ca[first:first+length]=0;cc[first:first+length]=0
    return True


def default_input_noise():return copy.deepcopy(DEFAULT_INPUT_NOISE)
