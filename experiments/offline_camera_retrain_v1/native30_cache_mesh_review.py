"""Read-only native GT meshes for cache visualization, never model inputs."""
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh

from experiments.offline_camera_retrain_v1.native30_export import inventory,raw_scene,rows
from experiments.offline_camera_retrain_v1.export_native20_training_scenes import forward_chunk
from experiments.offline_camera_retrain_v1.data import SOURCE,PROJECTS


def reference_geometry(meta,start,frames=192):
    row=next(r for r in inventory(meta['split']) if r['sequence_id']==meta['sequence_id'])
    q=np.arange(row['raw_first'],row['raw_last_exclusive'])
    for attempt in range(3):
        try:
            people,transform,_,_,_,_,_,_,_,_,_=raw_scene(row,q,hashlib.sha256());break
        except ValueError as exc:
            if 'invalid \\x escape' not in str(exc) or attempt==2:raise
    bodies=[]
    for model,params in people:
        vertices,joints=forward_chunk(model,params,start,start+frames,transform)
        bodies.append(dict(vertices=vertices,faces=np.asarray(model.faces,np.int32),joints=joints))
    expected=np.load(Path(meta['path'])/'motion_world.npy',mmap_mode='r')[start:start+frames,135:].reshape(frames,22,3)*2
    discrepancy=float(np.abs(bodies[0]['joints'][:,:22]-expected).max())
    if discrepancy>1e-4:raise AssertionError(f'Native mesh/cache joint coordinates differ: {discrepancy} m')
    objects=[]
    if row['dataset']=='trumans':
        from experiments.offline_camera_retrain_v1.prepare_trumans_dynamic_v2 import load_objects
        root=PROJECTS/'TRUMANS';clip=next(r for r in rows(root/'processed/scene_expert_v1/clips.jsonl') if r['clip_name']==row['sequence_id'])
        scan=root/clip['scene']['mesh']
        for name,path,r,p,_,valid,_ in load_objects(root,clip,q[start:start+frames]):
            mesh=trimesh.load(path,force='mesh',process=False)
            objects.append(dict(name=name,vertices=np.asarray(mesh.vertices,np.float32),
                faces=np.asarray(mesh.faces,np.int32),rotations=r,positions=p,valid=valid))
    elif row['dataset']=='egobody':
        sources={r['sequence_id']:r for r in rows(SOURCE/'sequences.jsonl')}
        scan=Path(sources[f"{row['recording']}/{row['group']}"]['scene_mesh'])
    else:
        scene=row['scene'];variant='chair' if 'chair' in row['sequence_id'] else 'yoga'
        scan=PROJECTS/'RICH/extracted/scan_calibration'/scene/(f'scan_{variant}_scene_camcoord.ply' if scene=='LectureHall' else 'scan_camcoord.ply')
    mesh=trimesh.load(scan,force='mesh',process=False);vertices=np.asarray(mesh.vertices,np.float32)
    if row['dataset']!='egobody':
        scale,r,p=transform;vertices=(scale*vertices@r.T+p).astype(np.float32)
    # Scene crop is strictly display-only. The model continues to consume the
    # separately exported observed point cloud, without any source scan mesh.
    lower=expected.min((0,1))-[1.5,.3,1.5];upper=expected.max((0,1))+[1.5,.5,1.5]
    source_faces=np.asarray(mesh.faces,np.int32);selected=[]
    for first in range(0,len(source_faces),100000):
        faces=source_faces[first:first+100000];triangles=vertices[faces]
        selected.append(faces[((triangles.max(1)>=lower)&(triangles.min(1)<=upper)).all(1)])
    faces=np.concatenate(selected)
    ids,inverse=np.unique(faces,return_inverse=True)
    result=dict(bodies=bodies,objects=objects,scene_vertices=vertices[ids],
        scene_faces=inverse.reshape(-1,3).astype(np.int32),maximum_mesh_joint_error_m=discrepancy,
        source_mesh=str(scan),reference_only=True)
    del people,mesh;gc.collect()
    return result
