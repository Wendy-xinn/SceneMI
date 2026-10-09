"""Export time-resolved TRUMANS scene input and a mesh inspection bundle.

Chair assets use object_flag/object_mat (including shape variant), not raw
unscaled Object_all replacements. Other tracked objects use their Euler tracks.
Every query is a source 30 Hz timestamp; motion output is 20 Hz, no frame clamp.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
import smplx
import trimesh
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.scene_visibility_v2 import (render_scene, sample_scene_hits, wearer_faces,
                                 interpolate_transforms, static_ray_scene)

HERE=Path(__file__).resolve().parent
PROJECTS=HERE.parents[2]
sys.path.insert(0,str(PROJECTS/'diffusion-motion-inbetweening'))
from sample.virtual_head_camera import virtual_head_camera
from sample.render_trumans_gt_smpl import slerp_axis_angles, linear_resample, HEAD_CHAIN


def load_objects(root, row, query):
    path=row['object_tracks']['path']
    if not path:
        raise FileNotFoundError(f"{row['clip_name']}: no source object tracks; do not fabricate a chair")
    tracks=np.load(root/path,allow_pickle=True).item()
    flags=np.load(root/'object_flag.npy',mmap_mode='r')
    matrices=np.load(root/'object_mat.npy',mmap_mode='r')
    names=np.load(root/'object_list.npy')
    lo=row['global_start']; hi=row['global_end_exclusive']
    objects=[]; replaced=set()
    # Validate both neighbors of fractional frame queries. A missing track is
    # not silently held forward or borrowed from another sequence.
    active=np.flatnonzero((flags[lo:hi]>=0).any(axis=0))
    for idx in active:
        name=str(names[idx]); slots=np.asarray(flags[lo:hi,idx],int)
        if np.any(slots<0):
            raise ValueError(f'{name}: intermittent chair track requires explicit presence mask')
        transforms=np.asarray(matrices[np.arange(lo,hi),slots])
        finite=np.isfinite(transforms).all(axis=(1,2))
        rotations=transforms[:,:3,:3].copy();positions=transforms[:,:3,3].copy()
        if not np.allclose(np.linalg.det(rotations[finite]),1.,atol=1e-4):raise ValueError(f'{name}: improper object rotation')
        if not np.allclose(rotations[finite].transpose(0,2,1)@rotations[finite],np.eye(3),atol=1e-4):raise ValueError(f'{name}: non-rigid object_mat')
        valid=finite[np.floor(query).astype(int)]&finite[np.ceil(query).astype(int)]
        # Internal placeholders only; every query touching a missing source
        # pose is invalidated, never raycast or supplied to training.
        rotations[~finite]=np.eye(3);positions[~finite]=0
        mesh_path=root/'Object_chairs/Object_mesh'/f'{name}.obj'
        r,p=interpolate_transforms(np.arange(hi-lo),rotations,positions,query)
        objects.append((name,mesh_path,r,p,'object_mat + variant mesh',valid,np.flatnonzero(~finite).tolist()))
        replaced.add(name.split('(')[0])
    for name,track in tracks.items():
        if name in replaced:
            continue
        n=len(track['location'])
        if n!=row['num_frames']:
            raise ValueError(f'{name}: track/source length mismatch')
        rotations=Rotation.from_euler('xyz',track['rotation']).as_matrix()
        r,p=interpolate_transforms(np.arange(n),rotations,track['location'],query)
        objects.append((name,root/'Object_all/Object_mesh'/f'{name}.obj',r,p,'raw Euler xyz radians',np.ones(len(query),bool),[]))
    return objects


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--training-only',action='store_true',help='Save only training inputs; omit duplicate display meshes/depth images')
    p.add_argument('--sequence',default='2023-02-13@18-03-43')
    p.add_argument('--source-start',type=int,required=True,
                   help='Explicit source 30 Hz frame; seated-chair example starts at 816')
    p.add_argument('--frames',type=int,default=128)
    p.add_argument('--output',type=Path,default=HERE/'data/scene_visibility_v2_oct05/trumans')
    args=p.parse_args();torch.set_num_threads(4);root=PROJECTS/'TRUMANS'
    row=next(r for r in map(json.loads,(root/'processed/scene_expert_v1/clips.jsonl').read_text().splitlines()) if r['clip_name']==args.sequence)
    joints_all=np.load(root/'human_joints.npy',mmap_mode='r')
    raw_joints=np.asarray(joints_all[row['global_start']:row['global_end_exclusive'],:22])
    start=args.source_start
    if start < 0: raise ValueError('source-start must be nonnegative')
    if not row['object_tracks']['path']:
        raise FileNotFoundError(f'{args.sequence}: missing object tracks; cannot reconstruct a moving chair')
    query=start+np.arange(args.frames)*1.5
    if query[-1]>row['num_frames']-1: raise ValueError('Clip extends beyond source')
    with (root/row['smplx_global']).open('rb') as f: params=pickle.load(f)
    model=smplx.create(str(PROJECTS/'SceneMI/body_models'),model_type='smplx',gender='male',
                       num_betas=20,use_pca=False,batch_size=args.frames)
    values={}
    for name,dims in [('global_orient',3),('body_pose',63),('left_hand_pose',45),('right_hand_pose',45),('jaw_pose',3)]:
        raw=np.asarray(params.get(name,np.zeros((row['num_frames'],dims))),np.float32).reshape(-1,dims//3,3)
        values[name]=torch.from_numpy(slerp_axis_angles(raw,query).reshape(args.frames,dims))
    values['transl']=torch.from_numpy(linear_resample(np.asarray(params['transl']),query).astype(np.float32))
    shape=np.asarray(params.get('betas',np.zeros(20)),np.float32).reshape(-1)
    if not len(shape): shape=np.zeros(20,np.float32)
    if len(shape)!=20: raise ValueError('Unexpected betas shape')
    values['betas']=torch.from_numpy(np.tile(shape,(args.frames,1)))
    expr=np.asarray(params.get('expression',np.zeros((row['num_frames'],10))),np.float32)
    values['expression']=torch.from_numpy(linear_resample(expr,query).astype(np.float32))
    with torch.no_grad(): out=model(**values,return_verts=True)
    vertices=out.vertices.numpy();joints=out.joints[:,:22].numpy();faces=model.faces
    self_faces,head_info=wearer_faces(faces,model.lbs_weights.numpy())
    poses=np.concatenate((values['global_orient'].numpy()[:,None],values['body_pose'].numpy().reshape(-1,21,3)),axis=1)
    local=Rotation.from_rotvec(poses.reshape(-1,3)).as_matrix().reshape(-1,22,3,3)
    head=local[:,0].copy()
    for k in HEAD_CHAIN[1:]: head=head@local[:,k]
    eyes=vertices[:,[9929,9448]].mean(1)
    offset=np.median(np.einsum('tji,tj->ti',head,eyes-joints[:,15])[:20],axis=0)
    cam,rotation=virtual_head_camera(joints[:,15],head,offset_head=offset)
    outdir=args.output/args.sequence;outdir.mkdir(parents=True,exist_ok=True)
    scene=trimesh.load(root/row['scene']['mesh'],force='mesh',process=False)
    scene_points,_=trimesh.sample.sample_surface(scene,65536,seed=2026)
    static_scene=static_ray_scene(scene.vertices,scene.faces)
    objects=[];rigid_objects=[];object_meta=[]
    official_bad=np.load(root/'bad_frames.npy');global_query=row['global_start']+query
    bad_body=np.isin(np.floor(global_query).astype(np.int64),official_bad)|np.isin(np.ceil(global_query).astype(np.int64),official_bad)
    observation_valid=~bad_body
    for i,(name,path,r,t,source,valid,invalid_source) in enumerate(load_objects(root,row,query)):
        observation_valid &= valid
        mesh=trimesh.load(path,force='mesh',process=False)
        rigid_objects.append((static_ray_scene(mesh.vertices,mesh.faces),r,t,i+1))
        verts=None if args.training_only else (np.asarray(mesh.vertices)[None]@r.transpose(0,2,1)+t[:,None]).astype(np.float32)
        if not args.training_only:
            np.save(outdir/f'object_{i}_vertices.npy',verts)
            np.save(outdir/f'object_{i}_faces.npy',mesh.faces)
            np.save(outdir/f'object_{i}_rotation.npy',r);np.save(outdir/f'object_{i}_position.npy',t)
        if not args.training_only:objects.append((verts,mesh.faces,i+1))
        object_meta.append(dict(name=name,owner=i+1,source=source,mesh=str(path),
                                displacement_m=float(np.linalg.norm(t[valid][-1]-t[valid][0])) if valid.any() else 0.,invalid_source_frames=invalid_source,invalid_observation_frames=np.flatnonzero(~valid).tolist()))
    clouds=[];cloud_owners=[];depths=[];owners=[];counts=[]
    for frame in range(args.frames):
        if not observation_valid[frame]:
            clouds.append(np.zeros((512,3),np.float32));cloud_owners.append(np.full(512,-1,np.int32));depths.append(np.full((96,128),np.inf,np.float32));owners.append(np.full((96,128),-1,np.int32));counts.append({'invalid_source_pose':1,'official_bad_body_pose':int(bad_body[frame])});continue
        meshes=[(vertices[frame],self_faces,100)]
        # Object geometry is raycast at the actual frame transform using cached local meshes.
        rendered=render_scene(meshes,cam[frame],rotation[frame],static_scene=static_scene,rigid_objects=[(sc,r[frame],t[frame],owner) for sc,r,t,owner in rigid_objects])
        points,ids=sample_scene_hits(rendered)
        clouds.append(points);cloud_owners.append(ids);depths.append(rendered['depth']);owners.append(rendered['owner'])
        counts.append({str(i):int((rendered['owner']==i).sum()) for i in [0,100]+list(range(1,len(rigid_objects)+1))})
        if frame % 100 == 0:print(f'visibility {frame}/{args.frames}',flush=True)
    arrays=dict(observation_valid=observation_valid,body_vertices_scenemi_yup=vertices,body_faces=faces,joints_scenemi_yup=joints,
                camera_position_scenemi_yup=cam,camera_rotation_scenemi_yup=rotation,
                head_rotation_scenemi_yup=head,face_local_rotation_scenemi_yup=np.eye(3),
                eye_position_scenemi_yup=eyes,nose_position_scenemi_yup=vertices[:,9120],
                scene_points_scenemi_yup=scene_points,visible_frame_points_scenemi_yup=np.asarray(clouds),
                visible_frame_mask=np.asarray(cloud_owners)>=0,visible_frame_owner=np.asarray(cloud_owners),
                depth=np.asarray(depths),owner=np.asarray(owners),source_frame_ids=query,
                self_occlusion_faces=self_faces)
    clouds=np.asarray(clouds);ids=np.asarray(cloud_owners)
    # No dynamic surfaces enter the historical static map.
    arrays['visible_static_points_scenemi_yup']=clouds[ids==0]
    if args.training_only:
        keep={'observation_valid','visible_frame_points_scenemi_yup','visible_frame_owner','camera_position_scenemi_yup','camera_rotation_scenemi_yup','source_frame_ids','visible_static_points_scenemi_yup'}
        arrays={key:val for key,val in arrays.items() if key in keep}
    for key,val in arrays.items():np.save(outdir/f'{key}.npy',val)
    meta=dict(dataset='trumans',sequence_id=args.sequence,output_fps=20,frames=args.frames,
              source_start_30fps=start,source_end_30fps=float(query[-1]),objects=object_meta,
              camera_protocol='TRUMANS head joint + fixed eye offset + 2 cm forward clearance',
              occlusion_version=2,head_exclusion=head_info,invalid_official_body_observation_frames=np.flatnonzero(bad_body).tolist(),invalid_observation_count=int((~observation_valid).sum()),invalid_observation_policy='drop entire observation; exclude overlapping target windows; no imputed object pose enters geometry',
              scene_protocol='exact mesh nearest hit; static history separate from per-frame dynamic surfaces',
              body_joints_vs_archive_max_m=float(np.linalg.norm(joints-linear_resample(raw_joints,query),axis=-1).max()),
              dynamic_occluder_subjects=['selected'],caveats=['Wearer head skin only excluded; other tracked bodies unavailable for this TRUMANS export'])
    (outdir/'metadata.json').write_text(json.dumps(meta,indent=2))
    (outdir/'occlusion_audit.json').write_text(json.dumps(dict(frames=counts),indent=2))
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':main()
