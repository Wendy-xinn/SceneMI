"""Diagnostic exo first-hit audit and display bundles; never training exports.

Released calibrated cameras: EgoBody master and RICH multicamera XML.
TRUMANS uses explicitly synthetic static observers. Joint positions are ideal
GT-derived tracker simulations, with geometry-derived reliability, not RGB
frontend predictions. Keep current availability and a high-confidence review
filter separate, so the review cannot silently change training conditions.
"""
import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh

from experiments.offline_camera_retrain_v1.native30_export import (
    inventory, raw_scene, rows, sampled_environment, forward_chunk, load_objects)
from experiments.offline_camera_retrain_v1.export_egobody_exo import cast_body, regional_faces
from experiments.offline_camera_retrain_v1.data import HERE, PROJECTS, SOURCE

RUN = HERE / 'runs/exo_occlusion_review_oct12'
CACHE = HERE / 'data/exo_occlusion_review_oct12'
PROTOCOL = 'diagnostic-exo-firsthit-owner-regions-native30-v1'
JOINTS = ['pelvis','left_hip','right_hip','spine1','left_knee','right_knee',
          'spine2','left_ankle','right_ankle','spine3','left_foot','right_foot',
          'neck','left_collar','right_collar','head','left_shoulder',
          'right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def source_scene(row):
    if row['group'] == 'trumans':
        clip = next(r for r in rows(PROJECTS/'TRUMANS/processed/scene_expert_v1/clips.jsonl') if r['clip_name']==row['sequence_id'])
        return PROJECTS/'TRUMANS'/clip['scene']['mesh'], clip
    if row['group'] == 'rich':
        variant = 'chair' if 'chair' in row['sequence_id'] else 'yoga'
        filename = f'scan_{variant}_scene_camcoord.ply' if row['scene']=='LectureHall' else 'scan_camcoord.ply'
        return PROJECTS/'RICH/extracted/scan_calibration'/row['scene']/filename, None
    source = next(r for r in rows(SOURCE/'sequences.jsonl') if r['sequence_id']==f"{row['recording']}/{row['group']}")
    return Path(source['scene_mesh']), None


def look_at_cv(position, target):
    z = np.asarray(target)-position; z /= np.linalg.norm(z)
    x = np.cross(z, [0.,1.,0.]); x /= np.linalg.norm(x)
    # Optical coordinates are x-right/y-down/z-forward, proper SO(3).
    return np.stack((x, np.cross(z,x), z), -1)


def camera_spec(row, transform, joints, camera_id=0, azimuth=0., rich_image_size=(4112,3008)):
    scale, wr, wt = transform
    if row['group'] in ('camera_wearer','interactee'):
        path = PROJECTS/'egobody/kinect_cam_params/kinect_master/Color.json'
        obj = json.loads(path.read_text())
        return dict(kind='released_calibrated', name='EgoBody Kinect master',
                    position=np.asarray(wt), rotation=np.asarray(wr),
                    K=np.asarray(obj['camera_mtx']), distortion=np.asarray(obj['k']),
                    source_size=[1920,1080], size=[192,108], calibration=str(path),
                    image_size_source='official Kinect master RGB dimensions', near=.1, far=12.)
    if row['group']=='rich':
        path = PROJECTS/'RICH/extracted/scan_calibration'/row['scene']/'calibration'/f'{camera_id:03d}.xml'
        root=ET.parse(path).getroot()
        matrix=np.fromstring(root.find('./CameraMatrix/data').text,sep=' ').reshape(3,4)
        k=np.fromstring(root.find('./Intrinsics/data').text,sep=' ').reshape(3,3)
        distortion=np.fromstring(root.find('./Distortion/data').text,sep=' ')
        r=matrix[:,:3]; centre=-r.T@matrix[:,3]
        return dict(kind='released_calibrated', name=f'RICH cam_{camera_id:02d}',
                    position=scale*wr@centre+wt, rotation=wr@r.T, K=k, distortion=distortion,
                    source_size=list(rich_image_size), size=[256,192], calibration=str(path),
                    image_size_source='RICH toolkit sample image dimensions (verified separately)', near=.1,far=20.)
    centre=np.median(joints[:,0],0); target=centre.copy();target[1]=1.
    position=centre+np.array([3.2*np.sin(azimuth),0.,3.2*np.cos(azimuth)])
    position[1]=1.65
    w,h=192,108; f=w/(2*np.tan(np.deg2rad(66.56/2)))
    return dict(kind='synthetic_diagnostic',name=f'TRUMANS synthetic fixed observer yaw {azimuth:.4f}',
                position=position,rotation=look_at_cv(position,target),
                K=np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]]),distortion=np.zeros(8),
                source_size=[w,h],size=[w,h],calibration=None,
                image_size_source='explicit diagnostic camera configuration',near=.1,far=12.)


def rays_for(camera):
    w,h=camera['size']; sw,sh=camera['source_size']
    x,y=np.meshgrid((np.arange(w)+.5)*sw/w,(np.arange(h)+.5)*sh/h)
    pixels=np.stack((x,y),-1).reshape(-1,2)
    xy=cv2.undistortPointsIter(pixels[:,None],camera['K'],camera['distortion'],None,None,
                             (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,1e-10))[:,0]
    direction=np.c_[xy,np.ones(len(xy))]@camera['rotation'].T
    return np.c_[np.broadcast_to(camera['position'],direction.shape),direction].astype(np.float32)


def merge_hits(depth,owner,candidate,identity,near,far):
    valid=np.isfinite(candidate)&(candidate>=near)&(candidate<=far)
    take=valid&(candidate<depth)
    depth[take]=candidate[take];owner[take]=identity


def regional_observations(parts,owner,valid=True):
    reference=np.bincount(parts[parts>=0],minlength=22)
    visible=np.bincount(parts[owner==100],minlength=22)
    person=np.bincount(parts[(parts>=0)&(owner>100)],minlength=22)
    static=np.bincount(parts[(parts>=0)&(owner==0)],minlength=22)
    rigid=np.bincount(parts[(parts>=0)&(owner>0)&(owner<100)],minlength=22)
    fraction=visible/np.maximum(reference,1)
    available=(visible>=2)&(fraction>=.15)&bool(valid)
    confidence=np.where(available,fraction*np.minimum(visible/12.,1),0).astype(np.float32)
    return dict(reference=reference,visible=visible,person=person,static=static,rigid=rigid,
                fraction=fraction.astype(np.float32),available=available,confidence=confidence,
                high=available&(confidence>=.7))


def first_hit(camera,rays,static,bodies,face_labels,objects,t,static_depth=None):
    near,far=camera['near'],camera['far']
    if static_depth is None:
        clipped=rays.copy();clipped[:,:3]+=near*clipped[:,3:]
        static_depth=static.cast_rays(o3d.core.Tensor(clipped))['t_hit'].numpy()+near
    depth=np.full(len(rays),np.inf,np.float32);owner=np.full(len(rays),-1,np.int32)
    merge_hits(depth,owner,static_depth,0,near,far)
    for scene,rot,pos,identity,valid in objects:
        if not valid[t]:continue
        local=rays.copy();local[:,:3]=(rays[:,:3]-pos[t])@rot[t];local[:,3:]=rays[:,3:]@rot[t]
        local[:,:3]+=near*local[:,3:]
        d=scene.cast_rays(o3d.core.Tensor(local))['t_hit'].numpy()+near
        merge_hits(depth,owner,d,identity,near,far)
    parts=np.full(len(rays),-1,np.int32)
    for i,(vertices,faces) in enumerate(bodies):
        d,primitive=cast_body(vertices,faces,rays)
        hits=np.isfinite(d)&(d>=near)&(d<=far)
        if i==0:parts[hits]=face_labels[primitive[hits]]
        merge_hits(depth,owner,d,100+i,near,far)
    return depth,owner,parts,static_depth


def project(joints,camera):
    local=(joints-camera['position'])@camera['rotation']
    uv=cv2.projectPoints(local.reshape(-1,3),np.zeros(3),np.zeros(3),camera['K'],camera['distortion'])[0][:,0].reshape(*joints.shape[:-1],2)
    uv*=np.array(camera['size'])/camera['source_size']
    inside=(local[...,2]>=camera['near'])&(local[...,2]<=camera['far'])&(uv[...,0]>=0)&(uv[...,0]<camera['size'][0])&(uv[...,1]>=0)&(uv[...,1]<camera['size'][1])
    return uv.astype(np.float32),inside


def load_raw(row,q):
    digest=hashlib.sha256();raw=raw_scene(row,q,digest)
    people,transform,static=raw[:3]
    body=[forward_chunk(m,p,0,len(q),transform) for m,p in people]
    return raw,body,digest


def serial_camera(camera):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in camera.items()}


def scout():
    torch.set_num_threads(1);RUN.mkdir(parents=True,exist_ok=True)
    sources=inventory('train',split_protocol='legacy-internal-scene-v1')+inventory('validation',split_protocol='legacy-internal-scene-v1')
    names=[('recording_20210907_S04_S03_01_camera_wearer_2001-2901','validation'),
           ('recording_20210907_S04_S03_02_interactee_1601-2681','validation'),
           ('2023-02-13@18-03-43','train'),('2023-02-13@22-31-47','validation'),
           ('LectureHall_003_wipingchairs1','validation'),('ParkingLot1_002_overfence2','train')]
    result=[]
    for name,split in names:
        row=next(r for r in sources if r['sequence_id']==name and r['split']==split)
        if row['group'] in ('camera_wearer','interactee'):
            q=np.arange(row['raw_first'],row['raw_last_exclusive'],30,dtype=float)
        elif row['group']=='trumans':
            q=np.array([816,846,876,906,936,966],float) if '18-03' in name else np.array([70,100,130,160,190,1020],float)
        else:q=np.linspace(row['raw_first']+20,row['raw_last_exclusive']-21,8).astype(int).astype(float)
        raw,body,digest=load_raw(row,q);people,transform,static=raw[:3];objects,valid=raw[-2:]
        label=regional_faces(people[0][0]);cameras=[]
        if row['group']=='rich':
            ids=sorted(int(p.stem) for p in (PROJECTS/'RICH/extracted/scan_calibration'/row['scene']/'calibration').glob('*.xml'))
            cameras=[camera_spec(row,transform,body[0][1],i) for i in ids]
        elif row['group']=='trumans':cameras=[camera_spec(row,transform,body[0][1],azimuth=a) for a in np.arange(8)*np.pi/4]
        else:cameras=[camera_spec(row,transform,body[0][1])]
        for camera in cameras:
            rays=rays_for(camera);sd=None;frames=[]
            for t in range(len(q)):
                depth,owner,parts,sd=first_hit(camera,rays,static,[(b[0][t],p[0].faces) for b,p in zip(body,people)],label,objects,t,sd)
                obs=regional_observations(parts,owner,valid[t])
                frames.append(dict(source_frame=int(q[t]),high=int(obs['high'].sum()),available=int(obs['available'].sum()),target_pixels=int(obs['reference'].sum()),
                                   person=int(obs['person'].sum()),static=int(obs['static'].sum()),rigid=int(obs['rigid'].sum()),valid=bool(valid[t])))
            item=dict(sequence_id=name,split=split,camera=serial_camera(camera),frames=frames)
            result.append(item);print(name,camera['name'],[(f['source_frame'],f['high'],f['person'],f['static'],f['rigid']) for f in frames],flush=True)
    (RUN/'scout.json').write_text(json.dumps(result,indent=2))


def export_case(spec):
    row=next(r for r in inventory(spec['split'],split_protocol='legacy-internal-scene-v1') if r['sequence_id']==spec['sequence_id'])
    q=np.arange(spec['start'],spec['start']+spec.get('frames',192),dtype=float)
    if q[0]<row['raw_first'] or q[-1]>=row['raw_last_exclusive']:raise ValueError('Review crosses scoped source boundary')
    folder=CACHE/spec['id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metadata.json').exists():
        meta=json.loads((folder/'metadata.json').read_text())
        if meta['export_sha256']!=sha(__file__) or meta['spec']!=spec:raise ValueError('Stale review bundle; choose a new output ID')
        for name,digest in meta['output_sha256'].items():
            if sha(folder/name)!=digest:raise ValueError(f'Corrupt review {name}')
        return meta
    started=time.monotonic();raw,body,digest=load_raw(row,q)
    people,transform,static=raw[:3];objects,valid=raw[-2:];n=len(q)
    camera=camera_spec(row,transform,body[0][1],spec.get('camera_id',0),spec.get('azimuth',0.))
    rays=rays_for(camera);w,h=camera['size'];label=regional_faces(people[0][0]);sd=None
    arrays={k:np.zeros((n,22),dtype) for k,dtype in [('reference',np.int32),('visible',np.int32),('person',np.int32),('static',np.int32),('rigid',np.int32),('fraction',np.float32),('available',bool),('confidence',np.float32),('high',bool)]}
    arrays.update(source_frame_ids=q.astype(int),observation_valid=valid,gt_joints_reference=body[0][1].astype(np.float32),
                  owner=np.zeros((n,h,w),np.int16),target_parts=np.zeros((n,h,w),np.int8),depth=np.zeros((n,h,w),np.float32),
                  frame_points=np.zeros((n,512,3),np.float32),frame_owner=np.full((n,512),-1,np.int32))
    for i,(b,p) in enumerate(zip(body,people)):
        np.save(folder/f'person_{i}_vertices.npy',b[0].astype(np.float32));np.save(folder/f'person_{i}_faces.npy',p[0].faces.astype(np.int32))
    for t in range(n):
        depth,owner,parts,sd=first_hit(camera,rays,static,[(b[0][t],p[0].faces) for b,p in zip(body,people)],label,objects,t,sd)
        obs=regional_observations(parts,owner,valid[t])
        for k,v in obs.items():arrays[k][t]=v
        arrays['owner'][t]=owner.reshape(h,w);arrays['target_parts'][t]=parts.reshape(h,w);arrays['depth'][t]=depth.reshape(h,w)
        hp=np.zeros((len(rays),3),np.float32);mask=owner>=0;hp[mask]=rays[mask,:3]+depth[mask,None]*rays[mask,3:]
        if valid[t]:arrays['frame_points'][t],arrays['frame_owner'][t]=sampled_environment(hp,owner)
        if t%32==0:print(spec['id'],t,n,flush=True)
    # Both condition payloads hide unavailable positions; GT stays reference-only.
    for mask_key,prefix in [('available','current'),('high','high')]:
        mask=arrays[mask_key]
        arrays[f'{prefix}_input_positions']=np.where(mask[...,None],body[0][1],0).astype(np.float32)
        arrays[f'{prefix}_input_confidence']=np.where(mask,arrays['confidence'],0).astype(np.float32)
    arrays['joint_pixels'],arrays['joint_in_fov']=project(body[0][1],camera)
    scan,clip=source_scene(row);mesh=trimesh.load(scan,force='mesh',process=False)
    reference,_=trimesh.sample.sample_surface(mesh,40000,seed=20261012)
    if row['group'] not in ('camera_wearer','interactee'):reference=reference*transform[0]@transform[1].T+transform[2]
    arrays['complete_scan_reference']=reference.astype(np.float32)
    object_meta=[]
    if clip:
        for i,(name,path,r,p,source,ov,_) in enumerate(load_objects(PROJECTS/'TRUMANS',clip,q)):
            obj=trimesh.load(path,force='mesh',process=False)
            for k,a in [('vertices',np.asarray(obj.vertices,np.float32)),('faces',np.asarray(obj.faces,np.int32)),('rotations',r.astype(np.float32)),('positions',p.astype(np.float32)),('valid',ov)]:np.save(folder/f'object_{i}_{k}.npy',a)
            object_meta.append(dict(name=name,source=source,path=str(path),sha256=sha(path),owner=i+1))
    for name,array in arrays.items():np.save(folder/(name+'.npy'),array)
    high=arrays['high'];ref=arrays['reference'];vis=arrays['visible']
    denominator=max(int(ref.sum()),1)
    stats=dict(mean_available_joints=float(arrays['available'].sum(1).mean()),mean_high_joints=float(high.sum(1).mean()),
               min_high_joints=int(high.sum(1).min()),max_high_joints=int(high.sum(1).max()),
               external_person_fraction=float(arrays['person'].sum()/denominator),
               external_static_fraction=float(arrays['static'].sum()/denominator),
               external_rigid_fraction=float(arrays['rigid'].sum()/denominator),
               person_occluded_frames=int((arrays['person'].sum(1)>0).sum()),
               static_occluded_frames=int((arrays['static'].sum(1)>0).sum()),
               rigid_occluded_frames=int((arrays['rigid'].sum(1)>0).sum()),
               high_with_partial_external_occlusion=int((high&(vis<ref)).sum()),
               high_out_of_joint_fov=int((high&~arrays['joint_in_fov']).sum()),
               invalid_frames=int((~valid).sum()),
               target_skin_in_scene_points=int((arrays['frame_owner']==100).sum()))
    blocked=arrays['person']+arrays['static']+arrays['rigid']
    critical=int(np.argmax(blocked.sum(1)))
    joints_report=[dict(joint=JOINTS[j],available_frames=int(arrays['available'][:,j].sum()),high_frames=int(high[:,j].sum()),
                        person_blocked_pixels=int(arrays['person'][:,j].sum()),static_blocked_pixels=int(arrays['static'][:,j].sum()),rigid_blocked_pixels=int(arrays['rigid'][:,j].sum())) for j in range(22)]
    # The optical similarity for RICH scales translation/depth but not orientation.
    uv=arrays['joint_pixels'];local=(body[0][1]-camera['position'])@camera['rotation'];back=(local@camera['rotation'].T+camera['position'])
    checks=dict(world_camera_roundtrip_max_m=float(np.abs(back-body[0][1]).max()),
                current_masked_positions_zero=bool(np.all(arrays['current_input_positions'][~arrays['available']]==0)),
                high_masked_positions_zero=bool(np.all(arrays['high_input_positions'][~high]==0)),
                count_partition_exact=bool(np.array_equal(ref,vis+blocked)),
                target_skin_excluded=stats['target_skin_in_scene_points']==0,
                optical_rotation_error=float(np.abs(camera['rotation'].T@camera['rotation']-np.eye(3)).max()),
                current_input_gt_max_m=float(np.abs(arrays['current_input_positions'][arrays['available']]-body[0][1][arrays['available']]).max()) if arrays['available'].any() else 0.)
    if not all(checks[k] for k in ['current_masked_positions_zero','high_masked_positions_zero','count_partition_exact','target_skin_excluded']):raise AssertionError(checks)
    meta=dict(protocol=PROTOCOL,diagnostic_only=True,training_steps=0,spec=spec,path=str(folder),
              sequence_id=row['sequence_id'],dataset=row['dataset'],group=row['group'],split=row['split'],fps=30,frames=n,
              camera=serial_camera(camera),people=len(people),objects=object_meta,source_mesh=str(scan),source_sha256=digest.hexdigest(),
              native_body=raw[8],input_kind='ideal_GT_joint_tracker_simulation_geometry_confidence',
              current_policy='visible>=2 pixels AND visible/target-alone>=0.15; confidence=fraction*min(visible/12,1)',
              high_policy='diagnostic filter: current availability AND confidence>=0.7; training configuration unchanged',
              critical_frame=critical,statistics=stats,per_joint=joints_report,checks=checks,
              export_sha256=sha(__file__),elapsed_s=time.monotonic()-started)
    meta['output_sha256']={p.name:sha(p) for p in sorted(folder.glob('*.npy'))}
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scout',action='store_true');parser.add_argument('--selection',type=Path)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.scout:scout();return
    specs=json.loads((args.selection or RUN/'selection.json').read_text())
    RUN.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    results=[export_case(s) for s in specs]
    (RUN/'manifest.json').write_text(json.dumps(dict(protocol=PROTOCOL,cases=results,training_steps=0),indent=2))
    print('EXO REVIEW COMPLETE',len(results),flush=True)


if __name__=='__main__':main()
