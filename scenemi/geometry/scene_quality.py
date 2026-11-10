"""GRAIL geometry diagnostics on preselected paired model predictions.

Full source geometry is used only for scoring. Model inputs stay observed and
causal. Stance/seated masks are proxies derived once from target motion and
support, shared by every prediction; they are not contact ground truth.
"""
import gc
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from scenemi.config import ROOT
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.grail_contact import ContactBody, geometry, DEFAULT_CONFIG
from scenemi.training import now
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d
from experiments.offline_camera_retrain_v1.scene_visibility_v2 import static_ray_scene
from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array


def supported_stance(distances, centres, *, fps=30):
    speed=np.linalg.norm(np.diff(centres,axis=0,prepend=centres[:1]),axis=-1)*fps
    mask=(distances<=.03)&(speed<=.2)
    mask[0]=False
    # Require three neighbouring frames, avoiding a single grazing point.
    padded=np.pad(mask,((1,1),(0,0)),constant_values=False)
    return padded[:-2]&padded[1:-1]&padded[2:]


def masked_mean(values, mask):
    return float(np.mean(values[mask])) if mask.any() else None


def decode(body, motion, rest, identity):
    motion=np.asarray(motion,np.float32);n=len(motion)
    matrix=rotation_from_6d(torch.from_numpy(motion[:,3:135].copy()).reshape(n,22,6)).numpy()
    angles=Rotation.from_matrix(matrix.reshape(-1,3,3)).as_rotvec().reshape(n,66).astype(np.float32)
    params={key:np.zeros((n,dim),np.float32) for key,dim in (
        ('betas',10),('expression',10),('jaw_pose',3),('leye_pose',3),('reye_pose',3),
        ('left_hand_pose',45),('right_hand_pose',45))}
    params.update(global_orient=angles[:,:3],body_pose=angles[:,3:],
        transl=motion[:,:3]*2+rest[:1]-body.joints[:1].numpy())
    anchor=np.asarray(identity['anchor'],np.float32);origin=np.asarray(identity['origin'],np.float32)
    vertices,joints=body.forward(params,anchor.T,origin)
    expected=forward_kinematics(torch.from_numpy(motion[None].copy()),
                              torch.from_numpy(rest[None].copy()))[0].numpy()@anchor+origin
    error=float(np.abs(joints-expected).max())
    if error>5e-5:raise ValueError('Prediction/scene coordinate mismatch: '+str(error))
    return vertices,joints,params,error


def evaluate_geometry(out):
    import open3d as o3d
    import trimesh

    out=Path(out);result=json.loads((out/'RESULT.json').read_text())
    if result['state']!='complete':raise ValueError('Finish paired predictions before geometry scoring')
    policy=json.loads(DEFAULT_CONFIG.read_text());body=ContactBody(policy);rows=[]
    atomic_json(out/'GEOMETRY_STATUS.json',dict(state='running',completed=0,total=len(result['geometry_arrays']),updated_at_utc=now()))
    try:
        for saved in result['geometry_arrays']:
            path=ROOT/saved['file']
            if sha256(path)!=saved['sha256']:raise ValueError('Paired prediction array changed')
            identity=saved['identity'];folder=Path(identity['path']);meta=dict(json.loads((folder/'metadata.json').read_text()),path=str(folder))
            if meta['native_body']['template']!='grail_g1_baked_v1':raise ValueError('Geometry audit requires fixed G1')
            v,f,_,scene,obj=geometry(meta)
            watertight=bool(trimesh.Trimesh(vertices=v,faces=f,process=True).is_watertight)
            triangles=np.asarray(v)[np.asarray(f)];normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
            horizontal=(np.abs(normal[:,1])/np.maximum(np.linalg.norm(normal,axis=-1),1e-12)>=.9)&(triangles.mean(1)[:,1]>.2)
            seat_scene=static_ray_scene(v,np.asarray(f)[horizontal]) if horizontal.any() else None
            measured={};validation=[];joint_sets={}
            with np.load(path,allow_pickle=False) as arrays:
                for name in ('target','baseline','scene'):
                    rest=arrays['target_rest' if name=='target' else 'rest']
                    vertices,joints,params,error=decode(body,arrays[name+'_motion'],rest,identity)
                    joint_sets[name]=joints
                    if not rows:validation.append(dict(condition=name,full_lbs=body.validate_full(params,
                        np.asarray(identity['anchor'],np.float32).T,np.asarray(identity['origin'],np.float32))))
                    distance=[];centres=[];ground=[];volume=[]
                    for ids in body.feet:
                        points=vertices[:,ids].copy();tensor=o3d.core.Tensor(points)
                        d=scene.compute_distance(tensor,nthreads=2).numpy()
                        distance.append(np.quantile(d,.1,axis=-1));centres.append(points.mean(1))
                        ground.append((points[...,1]<-.02).mean(-1))
                        if watertight:
                            inside=obj.compute_occupancy(tensor,nthreads=2).numpy()>.5
                            deep=obj.compute_distance(tensor,nthreads=2).numpy()>.02
                            volume.append((inside&deep).mean(-1))
                    if seat_scene is not None:
                        d=seat_scene.compute_distance(o3d.core.Tensor(vertices[:,body.seat].copy()),nthreads=2).numpy()
                        seat=np.quantile(d,.1,axis=-1)
                    else:seat=None
                    measured[name]=dict(distance=np.stack(distance,-1),centres=np.stack(centres,1),
                        ground=np.stack(ground,-1),volume=np.stack(volume,-1) if volume else None,
                        seat=seat,fk_error_m=error)
            target=measured['target'];stance=supported_stance(target['distance'],target['centres'])
            gt=joint_sets['target'];thigh=gt[:,[4,5]]-gt[:,[1,2]]
            seated=(np.abs(thigh[...,1])/np.maximum(np.linalg.norm(thigh,axis=-1),1e-8)<.4).all(-1)
            seated &= np.abs(np.diff(gt[:,0,1],prepend=gt[0,0,1]))*30<.1
            seated &= target['seat']<.03 if target['seat'] is not None else False
            if identity['category']!='sitting':seated[:]=False
            values={}
            for name,m in measured.items():
                # Use take to keep NumPy's advanced-index axes in a fixed order.
                centres=np.take(m['centres'],[0,2],axis=-1)
                velocity=np.diff(centres,axis=0,prepend=centres[:1])*30
                values[name]=dict(stance_support_q10_distance_mm=masked_mean(m['distance']*1000,stance),
                    stance_floating_over_5cm_fraction=masked_mean((m['distance']>.05).astype(float),stance),
                    stance_foot_horizontal_speed_mm_s=masked_mean(np.linalg.norm(velocity,axis=-1)*1000,stance),
                    ground_penetrating_foot_vertex_fraction=float(m['ground'].mean()),
                    closed_object_deep_foot_vertex_fraction=float(m['volume'].mean()) if m['volume'] is not None else None,
                    seated_support_q10_distance_mm=masked_mean(m['seat']*1000,seated) if m['seat'] is not None else None,
                    fk_max_error_m=m['fk_error_m'])
            rows.append(dict(case_id=saved['case_id'],identity=identity,stance_foot_frames=int(stance.sum()),
                seated_frames=int(seated.sum()),object_watertight=watertight,metrics=values,
                metadata_sha256=sha256(folder/'metadata.json'),prediction_sha256=saved['sha256'],
                full_lbs_validation=validation))
            atomic_json(out/'GEOMETRY_STATUS.json',dict(state='running',completed=len(rows),total=len(result['geometry_arrays']),updated_at_utc=now()))
            del scene,obj;gc.collect()
        report=dict(state='complete',rows=rows,source_sha256=sha256(Path(__file__)),
            predictions_protocol_sha256=sha256(out/'PROTOCOL.json'),test_used=False,
            method='Two seeded independent families per GRAIL action; fixed target-derived support proxies; all 3 conditions scored against the same full geometry.',
            limitations=['Diagnostic subset, not an independently selected test or all-validation contact result.',
                'Full hidden scene geometry is used exclusively for scoring, never inference.',
                'Stance/seated masks are target-derived proxies, not validated contact labels.',
                'No signed object-volume claim for open meshes; report ground overlap separately.',
                'Nearest-surface distances do not certify full sole or seat contact area.'],completed_at_utc=now())
        atomic_json(out/'GEOMETRY_RESULT.json',report)
        atomic_json(out/'GEOMETRY_STATUS.json',dict(state='complete',completed=len(rows),updated_at_utc=now()))
        return dict(state='complete',geometry_cases=len(rows))
    except BaseException as error:
        atomic_json(out/'GEOMETRY_STATUS.json',dict(state='failed',completed=len(rows),error=repr(error),updated_at_utc=now()));raise
