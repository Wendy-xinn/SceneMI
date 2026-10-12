"""Anatomical observed joints, optical-camera geometry, strict confidence masks."""
from pathlib import Path
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import matrix_to_axis_angle

from experiments.offline_camera_retrain_v1.data import rotation6d
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D, rotation_from_6d
from experiments.offline_camera_retrain_v1.native30_geometry import Native30SurfaceWindow
from experiments.offline_camera_retrain_v1.native30_memory import ObservedStaticMemory, OFFLINE_MEMORY_PROTOCOL, frozen_memory_entries
from experiments.offline_camera_retrain_v1.unified_geometry import occupancy_from_points
from experiments.offline_camera_retrain_v1.camera_v2_observations import validate_mount, CAMERA_KEYS

FPS = 30.
INPUT_KEYS = ('camera','trajectory','observation_meta','observation_uncertainty','occupancy',
              'bps','bps_valid','rest','body_type','body_scale') + CAMERA_KEYS


def inference_payload(batch):
    """Whitelist excludes motion, contact truth, full mesh and target GT pose."""
    return {k:batch[k] for k in INPUT_KEYS+('surface_windows','skin_assets','mode')}


def sanitize_observations(trajectory, meta):
    meta=torch.nan_to_num(meta).clone(); meta[...,2:4]=meta[...,2:4].clamp(0,1)
    pm=(meta[...,0]>0)&(meta[...,2]>0)&torch.isfinite(trajectory[...,:3]).all(-1)
    raw_rotation=torch.nan_to_num(trajectory[...,3:])
    nondegenerate=(raw_rotation[...,:3].norm(dim=-1)>1e-6)&(torch.cross(raw_rotation[...,:3],raw_rotation[...,3:],dim=-1).norm(dim=-1)>1e-6)
    rm=(meta[...,1]>0)&(meta[...,3]>0)&torch.isfinite(trajectory[...,3:]).all(-1)&nondegenerate
    meta[...,0]=pm;meta[...,1]=rm
    meta[...,2]*=pm;meta[...,3]*=rm
    safe=torch.cat((torch.where(pm[...,None],trajectory[...,:3],0.),
                    torch.where(rm[...,None],trajectory[...,3:],0.)), -1)
    return safe,meta


def observation_derivatives(trajectory, meta):
    trajectory,meta=sanitize_observations(trajectory,meta)
    pm,rm=meta[...,0]>0,meta[...,1]>0
    pos=trajectory[...,:3]*2
    six=torch.where(rm[...,None],trajectory[...,3:],trajectory.new_tensor(IDENTITY_6D))
    rot=rotation_from_6d(six-trajectory.new_tensor(IDENTITY_6D))
    v,a,w=[torch.zeros_like(pos) for _ in range(3)]
    vm,am,wm=[torch.zeros_like(pm) for _ in range(3)]
    vm[:,1:]=pm[:,1:]&pm[:,:-1]; v[:,1:]=torch.diff(pos,dim=1)*FPS
    am[:,2:]=vm[:,2:]&vm[:,1:-1]; a[:,2:]=torch.diff(v[:,1:],dim=1)*FPS
    wm[:,1:]=rm[:,1:]&rm[:,:-1]
    w[:,1:]=matrix_to_axis_angle(rot[:,1:]@rot[:,:-1].transpose(-1,-2))*FPS
    return (*(torch.where(m[...,None],x,0.) for x,m in zip((v,a,w),(vm,am,wm))),
            torch.stack((vm,am,wm),-1))


def inverse_head(camera, available, confidence, mount):
    """Sensor m/2 -> anatomical m/2; never fill missing rotation from GT."""
    validate_mount(mount); camera=np.asarray(camera).copy()
    available=np.asarray(available,bool).copy(); confidence=np.clip(np.asarray(confidence),0,1).copy()
    available &= np.c_[np.isfinite(camera[:,:3]).all(-1),np.isfinite(camera[:,3:]).all(-1)] & (confidence>0)
    six=np.nan_to_num(camera[:,3:]);cross=np.cross(six[:,:3],six[:,3:])
    available[:,1] &= (np.linalg.norm(six[:,:3],axis=-1)>1e-6)&(np.linalg.norm(cross,axis=-1)>1e-6)
    safe=np.where(np.repeat(available,[3,6],axis=1),camera,0.)
    six=np.where(available[:,1,None],safe[:,3:],IDENTITY_6D)
    b1=six[:,:3]/np.linalg.norm(six[:,:3],axis=-1,keepdims=True)
    b2=six[:,3:]-(b1*six[:,3:]).sum(-1,keepdims=True)*b1
    b2=b2/np.linalg.norm(b2,axis=-1,keepdims=True)
    r=np.stack((b1,b2,np.cross(b1,b2)), -1)
    rh=r@np.asarray(mount['rotation']).T; offset=np.asarray(mount['translation_m'])
    hp=safe[:,:3]*2-np.einsum('tij,j->ti',rh,offset)
    if np.linalg.norm(offset)>0:
        available[:,0] &= available[:,1]
        confidence[:,0] = np.minimum(confidence[:,0],confidence[:,1])
    result=np.concatenate((hp/2,rotation6d(rh)), -1).astype(np.float32)
    result=np.where(np.repeat(available,[3,6],axis=1),result,0.)
    return result,available.astype(np.float32),(confidence*available).astype(np.float32)


def perturb_input_tracks(trajectory, meta, cam, ca, cc, mode, joints, noise_rng):
    """Historical position/gap profile; ego wrists stay unchanged by request.

    Mutates observation arrays only. Head inversion follows the same perturbed
    camera stream in the builder. Native visibility/confidence is preserved for
    ego wrists; a natural absence is never filled by this augmentation.
    """
    frames=len(cam)
    drift=noise_rng.normal(0,.01,(1,22,3))+np.linspace(0,1,frames)[:,None,None]*noise_rng.normal(0,.02,(1,22,3))
    noise_joints=tuple(joints)
    if mode=='ego':
        drift[:,np.arange(22)!=15]=0
        noise_joints=(15,) if 15 in joints else ()
    trajectory[...,:3]+=drift/2
    meta[...,2]*=np.exp(-np.linalg.norm(drift,axis=-1)/.05)
    if mode=='ego':
        cam[:,:3]+=drift[:,15]/2
        cc[:,0]*=np.exp(-np.linalg.norm(drift[:,15],axis=-1)/.05)
    if noise_rng.random()<.3 and noise_joints:
        joint=int(noise_rng.choice(noise_joints));first=int(noise_rng.integers(max(1,frames-18)))
        meta[first:first+18,joint,:4]=0
        if mode=='ego' and joint==15:
            ca[first:first+18]=0;cc[first:first+18]=0


class Native30ObservationBuilder:
    def __init__(self, skin_cache, mounts):
        self.skin_cache=skin_cache;self.mounts=mounts
        for calibration in mounts.values():
            validate_mount(calibration)
            if calibration.get('fitting_split')!='train':
                raise ValueError('Approximate real mounts must be frozen on TRAIN')

    def build(self,data,sample,identity,mode='ego',joints=None,noise_rng=None):
        if mode not in identity.get('allowed_modes',['ego','exo']):
            raise ValueError('Observer mode unavailable in this source bundle')
        prepared={k:v.copy() for k,v in sample.items()}; frames=len(prepared['camera'])
        folder=identity['path'];sl=slice(identity['start'],identity['start']+frames)
        origin=np.asarray(identity['origin']);anchor=np.asarray(identity['anchor'])
        meta=np.zeros((frames,22,5),np.float32);meta[...,4]=1
        uncertainty=np.zeros((frames,22,2),np.float32)
        trajectory=np.zeros((frames,22,9),np.float32)
        cam=prepared['camera'].copy(); ca=prepared.get('camera_available',np.ones((frames,2),np.float32)).copy()
        cc=prepared.get('camera_confidence',ca).copy()
        if mode=='exo':
            trajectory[...,:3]=((np.asarray(data.load(folder,'exo_trajectory_world')[sl])-origin)@anchor.T)/2
            meta[...,0]=data.load(folder,'exo_position_available')[sl]
            meta[...,2]=data.load(folder,'exo_position_confidence')[sl]
            intr=np.broadcast_to(data.load(folder,'exo_intrinsics'),(frames,4)).copy()
            context=np.array([0,1,1,0,0,1],np.float32);mount=None
            points=(np.asarray(data.load(folder,'exo_frame_points_world')[sl])-origin)@anchor.T
            owners=np.asarray(data.load(folder,'exo_frame_owner')[sl]).copy()
            observing_cameras=cam[:,:3]*2
            joints=tuple(range(22)) if joints is None else joints
        elif mode=='ego':
            if 'motion' in prepared and 'joints' in prepared:
                # Dataset simulator may supply trusted wrists; label-free
                # camera-only inference uses the fallback head conversion.
                trajectory[...,:3]=prepared['joints']
                trajectory[...,3:]=rotation6d(global_rotations(torch.from_numpy(prepared['motion'][None]))[0].numpy())
                meta[...,:4]=1
            elif joints is not None and set(joints)-{15}:
                raise ValueError('Camera-only loader cannot invent reliable wrists')
            intr=np.asarray(data.load(folder,'camera_intrinsics_128x96')[sl],np.float32)/[128,96,128,96]
            real=identity['group']=='camera_wearer';context=np.array([1,0,int(real),int(not real),1,-1],np.float32)
            mount=identity['actual_virtual_mount'] or self.mounts['camera_wearer']
            points=(np.asarray(data.load(folder,'visible_frame_points_scenemi_yup')[sl])-origin)@anchor.T
            owners=np.asarray(data.load(folder,'visible_frame_owner')[sl]).copy()
            observing_cameras=cam[:,:3]*2
            joints=(15,) if joints is None else joints
        else:
            raise ValueError(mode)
        selected=np.zeros(22,bool);selected[list(joints)]=True;meta[...,:4]*=selected[None,:,None]
        if noise_rng is not None:
            # One perturbed sensor stream: head conversion follows sensor noise/dropout.
            perturb_input_tracks(trajectory,meta,cam,ca,cc,mode,joints,noise_rng)
        if mode=='ego':
            head,ha,hc=inverse_head(cam,ca,cc,mount)
            trajectory[:,15]=head;meta[:,15,:2]=ha;meta[:,15,2:4]=hc
            uncertainty[:,15]=[mount['position_p90_m'],mount['rotation_p90_rad']]
        trajectory,meta=(v.numpy() for v in sanitize_observations(torch.from_numpy(trajectory),torch.from_numpy(meta)))
        # No clean optical motion bypass: self-camera encoder consumes mount/K
        # only; stored optical pose is the same noisy/masked sensor for geometry.
        cam=np.where(np.repeat((ca>0)&(cc>0),[3,6],axis=1),cam,0.).astype(np.float32)
        kwargs={}
        if (Path(folder)/'object_positions.npy').exists():
            kwargs=dict(object_positions=(np.asarray(data.load(folder,'object_positions')[sl])-origin)@anchor.T,
                        object_rotations=anchor[None,None]@np.asarray(data.load(folder,'object_rotations')[sl]),
                        object_valid=np.asarray(data.load(folder,'object_valid')[sl]))
        if not hasattr(data,'scene_memory'):
            # Standalone offline loaders retain the supplied static observations.
            data.scene_memory=ObservedStaticMemory(data.load,[identity])
        scene=Native30SurfaceWindow(points,owners,observing_cameras,
            static_memory=data.scene_memory.window(identity,mode,frames),**kwargs)
        prepared.update(trajectory=trajectory,observation_meta=meta,observation_uncertainty=uncertainty,
            camera=cam,camera_available=ca,camera_confidence=cc,camera_intrinsics=np.asarray(intr,np.float32),
            camera_intrinsics_valid=np.ones(frames,np.float32),camera_context=context,
            mount_rotation=np.asarray(mount['rotation'] if mount else np.eye(3),np.float32),
            mount_translation_m=np.asarray(mount['translation_m'] if mount else np.zeros(3),np.float32),
            mount_valid=np.array(float(mount is not None),np.float32),
            mount_uncertainty=np.asarray([mount['position_p90_m'],mount['rotation_p90_rad']] if mount else [0,0],np.float32),
            occupancy=occupancy_from_points(scene.points),bps=np.zeros((frames,67,3),np.float32),
            bps_valid=np.zeros((frames,67),bool))
        return prepared,scene,self.skin_cache.get(identity['native_body'])


def load_frozen_exo(folder,*,start=0,frames=192,skin_cache,device='cuda',memory_manifest=None,
                    memory_protocol=OFFLINE_MEMORY_PROTOCOL):
    """Deployment-style master input: never open body-motion/contact labels."""
    from experiments.offline_camera_retrain_v1.data import anchor_rotation,collate
    folder=Path(folder);meta=json.loads((folder/'metadata.json').read_text())
    if meta['fps']!=30 or meta['group'] not in ('camera_wearer','interactee'):
        raise ValueError('Expected a raw30 EgoBody external observer bundle')
    if start<0 or start+frames>meta['frames']:
        raise ValueError('Frozen observation window uncovered')
    allowed={'exo_sensor_to_world','exo_intrinsics','exo_trajectory_world','exo_position_available',
             'exo_position_confidence','exo_frame_points_world','exo_frame_owner','object_positions',
             'object_rotations','object_valid','rest','source_frame_ids','observation_valid'}
    class ObservationsOnly:
        def load(self,path,key):
            if key not in allowed:
                raise ValueError(f'Body truth is forbidden in frozen input loader: {key}')
            return np.load(Path(path)/f'{key}.npy',mmap_mode='r')
    source=ObservationsOnly();transform=source.load(folder,'exo_sensor_to_world')
    source.scene_memory=ObservedStaticMemory(source.load,frozen_memory_entries(folder,meta,memory_manifest),protocol=memory_protocol)
    camera_r=np.broadcast_to(transform[:3,:3],(frames,3,3));anchor=np.asarray(anchor_rotation(camera_r),np.float64)
    camera=np.concatenate((np.zeros((frames,3)),rotation6d(anchor[None]@camera_r)), -1).astype(np.float32)
    sample=dict(camera=camera,rest=np.asarray(source.load(folder,'rest')).copy(),
                body_type=np.eye(2,dtype=np.float32)[int(meta['native_body']['model']=='smplx')],
                body_scale=np.array([meta['native_body']['scale']],np.float32))
    identity=dict(meta,path=str(folder),start=start,origin=transform[:3,3].tolist(),anchor=anchor.tolist())
    builder=Native30ObservationBuilder(skin_cache,{})
    prepared,scene,asset=builder.build(source,sample,identity,'exo')
    batch={k:v.to(device) for k,v in collate([prepared]).items()}
    batch.update(surface_windows=[scene],skin_assets=[asset],mode=['exo'])
    return inference_payload(batch)


def load_frozen_ego(folder,*,mounts,start=0,frames=192,skin_cache,device='cuda',memory_manifest=None,
                    memory_protocol=OFFLINE_MEMORY_PROTOCOL):
    """Camera-only self input; source body-motion and contact files are forbidden."""
    from experiments.offline_camera_retrain_v1.data import anchor_rotation,collate
    folder=Path(folder);meta=json.loads((folder/'metadata.json').read_text())
    if meta['fps']!=30 or start<0 or start+frames>meta['frames']:
        raise ValueError('Uncovered raw30 self observation window')
    allowed={'camera_position_scenemi_yup','camera_rotation_scenemi_yup','camera_intrinsics_128x96',
             'visible_frame_points_scenemi_yup','visible_frame_owner','observation_valid','rest',
             'object_positions','object_rotations','object_valid','source_frame_ids'}
    class ObservationsOnly:
        def load(self,path,key):
            if key not in allowed:raise ValueError(f'Body truth forbidden: {key}')
            return np.load(Path(path)/f'{key}.npy',mmap_mode='r')
    source=ObservationsOnly();sl=slice(start,start+frames)
    source.scene_memory=ObservedStaticMemory(source.load,frozen_memory_entries(folder,meta,memory_manifest),protocol=memory_protocol)
    p=np.asarray(source.load(folder,'camera_position_scenemi_yup')[sl]);r=np.asarray(source.load(folder,'camera_rotation_scenemi_yup')[sl])
    anchor=np.asarray(anchor_rotation(r),np.float64);origin=p[0]
    camera=np.concatenate(((p-origin)@anchor.T/2,rotation6d(anchor[None]@r)), -1).astype(np.float32)
    valid=np.asarray(source.load(folder,'observation_valid')[sl],np.float32)
    sample=dict(camera=camera,rest=np.asarray(source.load(folder,'rest')).copy(),
        body_type=np.eye(2,dtype=np.float32)[int(meta['native_body']['model']=='smplx')],
        body_scale=np.array([meta['native_body']['scale']],np.float32),
        camera_available=np.repeat(valid[:,None],2,1),camera_confidence=np.repeat(valid[:,None],2,1))
    identity=dict(meta,path=str(folder),start=start,origin=origin.tolist(),anchor=anchor.tolist())
    prepared,scene,asset=Native30ObservationBuilder(skin_cache,mounts).build(source,sample,identity,'ego')
    batch={k:v.to(device) for k,v in collate([prepared]).items()}
    batch.update(surface_windows=[scene],skin_assets=[asset],mode=['ego'])
    return inference_payload(batch)
