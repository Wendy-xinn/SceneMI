"""External observers and first-hit inputs for all three native30 corpora.

RICH/EgoBody use released calibrated cameras. TRUMANS uses an explicitly
synthetic fixed observer with known K, never an unverified release-camera K.
"""
import hashlib
import json
import pickle
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from experiments.offline_camera_retrain_v1.data import PROJECTS
from experiments.offline_camera_retrain_v1.export_egobody_exo import cast_body

EXO_PROTOCOL = 'three-corpus-calibrated-or-explicit-synthetic-firsthit-v1'


def look_at_cv(position, target):
    z=np.asarray(target)-position;z/=np.linalg.norm(z)
    x=np.cross(z,[0.,1.,0.]);x/=np.linalg.norm(x)
    return np.stack((x,np.cross(z,x),z),-1)


def external_camera(row, transform, params, digest, rest=None, source_ids=None):
    scale,wr,wt=transform
    if row['group'] in ('camera_wearer','interactee'):
        path=PROJECTS/'egobody/kinect_cam_params/kinect_master/Color.json'
        calibration=json.loads(path.read_text());digest.update(path.read_bytes())
        return dict(kind='released_calibrated',view_id='kinect_master',position=np.asarray(wt),
            rotation=np.asarray(wr),K=np.asarray(calibration['camera_mtx']),
            distortion=np.asarray(calibration['k']),source_size=[1920,1080],size=[192,108],
            calibration=str(path),near=.1,far=12.,benchmark_camera_verified=True)
    if row['group']=='rich':
        folder=PROJECTS/'RICH/extracted/scan_calibration'/row['scene']/'calibration'
        paths=sorted(folder.glob('*.xml'))
        if not paths:raise FileNotFoundError(f'No RICH cameras: {folder}')
        # Identity-only selection, independent of target pose/visibility/errors.
        index=int(hashlib.sha256(row['sequence_id'].encode()).hexdigest()[:8],16)%len(paths)
        path=paths[index];digest.update(path.read_bytes());root=ET.parse(path).getroot()
        matrix=np.fromstring(root.find('./CameraMatrix/data').text,sep=' ').reshape(3,4)
        r=matrix[:,:3];centre=-r.T@matrix[:,3]
        return dict(kind='released_calibrated',view_id=f'rich_{path.stem}',
            selection='SHA256(sequence_id) modulo released calibrated views; no pose selection',
            available_views=[p.stem for p in paths],position=scale*wr@centre+wt,rotation=wr@r.T,
            K=np.fromstring(root.find('./Intrinsics/data').text,sep=' ').reshape(3,3),
            distortion=np.fromstring(root.find('./Distortion/data').text,sep=' '),
            source_size=[4112,3008],size=[256,192],calibration=str(path),near=.1,far=20.,
            benchmark_camera_verified=True)
    if row['group']!='trumans':raise ValueError(row['group'])
    paths=[PROJECTS/'TRUMANS'/folder/(row['sequence_id']+'_smplx_results.pkl')
           for folder in ('smplx_result','smplx_result_in_cam')]
    fits=[]
    for path in paths:
        raw=path.read_bytes();digest.update(raw);fits.append(pickle.loads(raw))
    world,camera=fits
    for key in ('body_pose','left_hand_pose','right_hand_pose','jaw_pose','expression','betas'):
        if not np.array_equal(world[key],camera[key]):raise ValueError('TRUMANS paired local body parameters differ')
    q=np.asarray(source_ids,int);rw=Rotation.from_rotvec(np.asarray(world['global_orient'])[q]).as_matrix()
    rc=Rotation.from_rotvec(np.asarray(camera['global_orient'])[q]).as_matrix();r=rc@rw.transpose(0,2,1)
    pivot=np.asarray(rest)[0]
    t=np.asarray(camera['transl'])[q]+pivot-np.einsum('tij,tj->ti',r,np.asarray(world['transl'])[q]+pivot)
    centres=-np.einsum('tji,tj->ti',r,t)
    released=np.broadcast_to(np.eye(4),(len(q),4,4)).copy()
    released[:,:3,:3]=r.transpose(0,2,1);released[:,:3,3]=centres
    # Use a released camera location inside the room, with an explicit synthetic
    # optical orientation/K. A radius around the actor can land behind a wall.
    position=centres[0].copy();target=np.median(params['transl'],0).copy();target[1]=1.
    w,h=192,108;f=w/(2*np.tan(np.deg2rad(66.56/2)))
    return dict(kind='synthetic_training_observer',view_id='synthetic_at_first_release_camera',
        selection='fixed first released camera centre; synthetic look-at recording centre; no visibility selection',
        position=position,rotation=look_at_cv(position,target),
        K=np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]]),distortion=np.zeros(8),
        source_size=[w,h],size=[w,h],calibration=None,near=.1,far=12.,
        benchmark_camera_verified=False,
        release_camera_status='paired extrinsics recovered with native rest pivot; K and optical axes unverified',
        release_camera_sources=[str(p) for p in paths],_released_camera_reference=released.astype(np.float32))


def serial_camera(camera):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in camera.items()}


def camera_rays(camera):
    w,h=camera['size'];sw,sh=camera['source_size']
    x,y=np.meshgrid((np.arange(w)+.5)*sw/w,(np.arange(h)+.5)*sh/h)
    pixels=np.stack((x,y),-1).reshape(-1,2)
    xy=cv2.undistortPointsIter(pixels[:,None],camera['K'],camera['distortion'],None,None,
        (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,1e-10))[:,0]
    directions=np.c_[xy,np.ones(len(xy))]@camera['rotation'].T
    return np.c_[np.broadcast_to(camera['position'],directions.shape),directions].astype(np.float32)


def first_hit(camera,rays,static_depth,bodies,face_labels,objects,t,valid=True):
    near,far=camera['near'],camera['far']
    depth=np.full(len(rays),np.inf,np.float32);owner=np.full(len(rays),-1,np.int32)
    def merge(candidate,identity):
        take=np.isfinite(candidate)&(candidate>=near)&(candidate<=far)&(candidate<depth)
        depth[take]=candidate[take];owner[take]=identity
    merge(static_depth,0)
    for scene,rot,pos,identity,ov in objects:
        if not ov[t]:continue
        local=rays.copy();local[:,:3]=(rays[:,:3]-pos[t])@rot[t];local[:,3:]=rays[:,3:]@rot[t]
        local[:,:3]+=near*local[:,3:]
        merge(scene.cast_rays(o3d.core.Tensor(local))['t_hit'].numpy()+near,identity)
    parts=np.full(len(rays),-1,np.int32)
    for i,(vertices,faces) in enumerate(bodies):
        # cast_body uses near=.1; all external cameras share this near plane.
        d,primitive=cast_body(vertices,faces,rays)
        hits=np.isfinite(d)&(d>=near)&(d<=far)
        if i==0:parts[hits]=face_labels[primitive[hits]]
        merge(d,100+i)
    reference=np.bincount(parts[parts>=0],minlength=22)
    visible=np.bincount(parts[owner==100],minlength=22)
    blockers={name:np.bincount(parts[(parts>=0)&mask],minlength=22)
        for name,mask in [('person',owner>100),('static',owner==0),('rigid',(owner>0)&(owner<100))]}
    if not np.array_equal(reference,visible+sum(blockers.values())):
        raise AssertionError('External occlusion count partition failed')
    fraction=visible/np.maximum(reference,1)
    available=(visible>=2)&(fraction>=.15)&bool(valid)
    confidence=np.where(available,fraction*np.minimum(visible/12.,1),0).astype(np.float32)
    points=np.zeros((len(rays),3),np.float32);mask=owner>=0
    points[mask]=rays[mask,:3]+depth[mask,None]*rays[mask,3:]
    return points,owner,parts,depth,available,confidence,reference,visible,blockers


def external_transforms(load,folder,start,frames):
    matrix=np.asarray(load(folder,'exo_sensor_to_world'))
    if matrix.ndim==2:matrix=np.broadcast_to(matrix,(frames,4,4))
    else:matrix=matrix[start:start+frames]
    if matrix.shape!=(frames,4,4):raise ValueError('External camera clock mismatch')
    return matrix


def external_intrinsics(load,folder,start,frames):
    k=np.asarray(load(folder,'exo_intrinsics'))
    return np.broadcast_to(k,(frames,4)).copy() if k.ndim==1 else k[start:start+frames].copy()
