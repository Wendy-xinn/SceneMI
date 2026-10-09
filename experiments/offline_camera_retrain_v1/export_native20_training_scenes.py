"""Direct raw30 -> native20 meshes, camera and exact first-hit observations.
No 5Hz arrays are read. Full fits include hands/face for occlusion only.
"""
import argparse,csv,hashlib,json,pickle,sys,time
from pathlib import Path
import numpy as np
import torch,trimesh
from scipy.spatial.transform import Rotation,Slerp
from experiments.offline_camera_retrain_v1.data import HERE,PROJECTS,SOURCE,PARENTS
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from experiments.offline_camera_retrain_v1.export_pose_sequences import slerp_rotvec,linear_sample
from experiments.offline_camera_retrain_v1.scene_visibility_v2 import render_scene,sample_scene_hits,static_ray_scene,wearer_faces
from experiments.offline_camera_retrain_v1.prepare_rich_pseudo_ego import RICH_GENDER,RICH_TO_SCENEMI,_read_transform,_load_smplx_model
sys.path.insert(0,str(PROJECTS/'diffusion-motion-inbetweening'))
from sample.virtual_head_camera import virtual_head_camera

PROTOCOL='native20-scene-v4-raw30-full-body-first-hit'
FILES=('source_frame_ids.npy','visible_frame_points_scenemi_yup.npy','visible_frame_owner.npy','camera_position_scenemi_yup.npy','camera_rotation_scenemi_yup.npy')

def read_fits(paths,digest):
    fits=[]
    for p in paths:
        raw=p.read_bytes();digest.update(str(p).encode());digest.update(raw);fits.append(pickle.loads(raw,encoding='latin1'))
    return fits

def resample_fits(fits,q,kind):
    keys=['global_orient','body_pose','transl','betas']
    if kind=='smplx':keys+=['left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression']
    values={}
    for key in keys:
        a=np.array([np.asarray(f[key]).reshape(-1) for f in fits],np.float32)
        if not np.isfinite(a).all():raise ValueError(f'Nonfinite raw {key}')
        if key in ['global_orient','body_pose','jaw_pose','leye_pose','reye_pose']:
            values[key]=slerp_rotvec(a.reshape(len(a),-1,3),q).reshape(len(q),-1)
        else:values[key]=linear_sample(a,q)
    return values

def forward_chunk(model,params,lo,hi,transform):
    values={k:torch.from_numpy(v[lo:hi].copy()) for k,v in params.items()}
    with torch.inference_mode():out=model(**values,return_verts=True)
    scale,r,t=transform
    return ((out.vertices.numpy()*scale)@r.T+t).astype(np.float32),((out.joints[:,:22].numpy()*scale)@r.T+t).astype(np.float32)

def setup_rich(split,name,digest):
    original='train' if split=='train' else 'val';root=PROJECTS/'RICH/extracted';native=HERE/f'data/scene_visibility_v2_oct05/rich_{original}_smpl_native20_faceout_oct07'/name
    meta=json.loads((native/'metadata.json').read_text());q=np.load(native/'source_frame_ids.npy').astype(float)
    source=root/f'{original}_body'/name;dirs=sorted((p for p in source.iterdir() if p.is_dir()),key=lambda p:int(p.name));ids=np.array([int(p.name) for p in dirs]);assert np.all(np.diff(ids)==1)
    subjects=sorted(p.stem for p in dirs[0].glob('*.pkl'));wearer=str(meta['subject_id']);subjects.remove(wearer);subjects.insert(0,wearer)
    people=[]
    for subject in subjects:
        fits=read_fits([d/f'{subject}.pkl' for d in dirs],digest);model=_load_smplx_model(RICH_GENDER[int(subject)]);params=resample_fits(fits,q-ids[0],'smplx')
        if not np.allclose(params['betas'],params['betas'][:1],atol=1e-4):raise ValueError('RICH shape changes')
        people.append((model,params))
    scene=meta['scene'];variant='chair' if 'chair' in name else 'yoga';scan=root/'scan_calibration'/scene/(f'scan_{variant}_scene_camcoord.ply' if scene=='LectureHall' else 'scan_camcoord.ply');tf=root/'multicam2world'/(f'LectureHall_{variant}_multicam2world.json' if scene=='LectureHall' else f'{scene}_multicam2world.json')
    scale,rot,t=_read_transform(tf);transform=(scale,RICH_TO_SCENEMI@rot.T,t@RICH_TO_SCENEMI.T)
    mesh=trimesh.load(scan,force='mesh',process=False);sv=mesh.vertices*scale@transform[1].T+transform[2]
    digest.update(scan.read_bytes());digest.update(tf.read_bytes())
    model,params=people[0];_,j=forward_chunk(model,params,0,min(20,len(q)),transform)
    v,_=forward_chunk(model,params,0,min(20,len(q)),transform);pose=np.concatenate((params['global_orient'][:,None],params['body_pose'].reshape(-1,21,3)),1)
    lr=Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix().reshape(-1,22,3,3);global_r=[lr[:,0]]
    for joint in range(1,22):global_r.append(global_r[PARENTS[joint]]@lr[:,joint])
    head=transform[1][None]@global_r[15];eyes=v[:,[9929,9448]].mean(1);offset=np.median(np.einsum('tji,tj->ti',head[:len(j)],eyes-j[:,15]),axis=0)
    return q,people,transform,static_ray_scene(sv,mesh.faces),None,head,offset,dict(dataset='rich',subject_ids=subjects,scene_mesh=str(scan),world_transform=str(tf),camera_protocol='raw30 poses -> 20Hz head; fixed first20 native20 eye offset; identity face mount; 2cm clearance',hfov=66.56,vfov=40.49)

def raw_pv_camera(root,record,scene,q,digest):
    import ast
    paths=list((root/'egocentric_color'/record).glob('202*/*_pv.txt'))
    if len(paths)!=1:raise ValueError('Expected one PV stream')
    raw=paths[0].read_bytes();digest.update(raw);lines=raw.decode().splitlines();cx,cy,w,h=ast.literal_eval(lines[0]);timestamps={}
    for line in lines[1:]:
        f=line.split(',');timestamps[f[0]]=(float(f[1]),float(f[2]),np.array(f[3:19],float).reshape(4,4))
    frames={}
    for image in (paths[0].parent/'PV').glob('*_frame_*.jpg'):
        timestamp,frame=image.stem.rsplit('_frame_',1)
        if timestamp in timestamps:frames[int(frame)]=timestamps[timestamp]
    ids=np.array(sorted(frames));ids=ids[(ids>=q[0]-10)&(ids<=q[-1]+10)]
    if len(ids)<2 or q[0]<ids[0] or q[-1]>ids[-1] or np.max(np.diff(ids))>6:raise ValueError('PV query uncovered or gap>0.2s; no extrapolation')
    cal=root/'calibrations'/record/'cal_trans';sp=cal/'kinect12_to_world'/f'{scene}.json';hp=cal/'holo_to_kinect12.json';sm=np.array(json.loads(sp.read_text())['trans']);hm=np.array(json.loads(hp.read_text())['trans']);digest.update(hp.read_bytes())
    poses=np.stack([sm@hm@frames[int(i)][2] for i in ids]);rot=Slerp(ids,Rotation.from_matrix(poses[:,:3,:3]))(q).as_matrix()@np.diag([-1.,1.,-1.]);pos=np.stack([np.interp(q,ids,poses[:,j,3]) for j in range(3)],1)
    fx=np.interp(q,ids,[frames[int(i)][0] for i in ids]);fy=np.interp(q,ids,[frames[int(i)][1] for i in ids]);intr=np.stack((fx*128/w,fy*96/h,np.full(len(q),(w-cx)*128/w),np.full(len(q),cy*96/h)),1)
    return pos.astype(np.float32),rot.astype(np.float32),intr.astype(np.float32)

def setup_egobody(split,name,digest):
    rows=list(map(json.loads,(SOURCE/f'egobody_sequences/{split}.jsonl').read_text().splitlines()));row=next(r for r in rows if r['sequence_id']==name);record=row['recording'];root=PROJECTS/'egobody'
    sources={r['sequence_id']:r for r in map(json.loads,(SOURCE/'sequences.jsonl').read_text().splitlines())};info={r['recording_name']:r for r in csv.DictReader((root/'data_info_release.csv').open())}[record]
    q=row['source_first_frame']+np.arange(row['frames_20fps'])*1.5;first=int(np.floor(q[0]));last=int(np.ceil(q[-1]));roles=[row['role'],'interactee' if row['role']=='camera_wearer' else 'camera_wearer'];people=[]
    pose_meta={r['sequence_id']:r for r in map(json.loads,(HERE/f'data/poses/{split}.jsonl').read_text().splitlines())}
    for role in roles:
        fit_dir=Path(sources[f'{record}/{role}']['body_fits']);idx=int(fit_dir.name.rsplit('_',1)[1]);gender=info[f'body_idx_{idx}'].split()[-1].lower();model=load_model('smpl',gender)
        fits=read_fits([fit_dir/f'results/frame_{f:05d}/000.pkl' for f in range(first,last+1)],digest);params=resample_fits(fits,q-first,'smpl')
        # Same fixed native shape as training for wearer; fixed median for partner.
        beta=np.array(pose_meta[name]['betas'],np.float32) if role==row['role'] else np.median(params['betas'],axis=0)
        params['betas']=np.broadcast_to(beta,params['betas'].shape).copy();people.append((model,params))
    tf=root/'calibrations'/record/'cal_trans/kinect12_to_world'/f"{row['scene_id']}.json";m=np.array(json.loads(tf.read_text())['trans']);assert np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-3)
    scan=Path(sources[f'{record}/{row["role"]}']['scene_mesh']);mesh=trimesh.load(scan,force='mesh',process=False);digest.update(scan.read_bytes());digest.update(tf.read_bytes());folder=Path(row['folder'])
    cam,rot,intr=raw_pv_camera(root,record,row['scene_id'],q,digest);head=None;offset=None;camera=(cam,rot)
    if row['role']=='interactee':
        model,params=people[0];pose=np.concatenate((params['global_orient'][:,None],params['body_pose'].reshape(-1,23,3)),1)
        lr=Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix().reshape(-1,24,3,3);glob=[lr[:,0]]
        for joint in range(1,24):glob.append(glob[int(model.parents[joint])]@lr[:,joint])
        head=m[:3,:3][None]@glob[15];v,j=forward_chunk(model,params,0,min(20,len(q)),(1.,m[:3,:3],m[:3,3]))
        from smplx.vertex_ids import vertex_ids
        landmarks=vertex_ids['smplh'];eyes=v[:,[landmarks['leye'],landmarks['reye']]].mean(1);offset=np.median(np.einsum('tji,tj->ti',head[:len(j)],eyes-j[:,15]),axis=0);camera=None
        intr=np.broadcast_to(np.median(intr[:20],axis=0),intr.shape).copy()
    return q,people,(1.,m[:3,:3],m[:3,3]),static_ray_scene(mesh.vertices,mesh.faces),camera,head,offset,dict(dataset='egobody',subject_ids=roles,recording=record,scene_mesh=str(scan),world_transform=str(tf),camera_protocol=('raw PV extrinsics/timestamps -> exact20' if camera is not None else 'raw30 native head -> exact20; fixed eye offset; 2cm clearance'),hfov=None,vfov=None,_intrinsics=intr,projection_intrinsics='same-recording official PV; wearer per-query, interactee fixed first20 median')


def export(dataset,split,name,output):
    torch.set_num_threads(1);start=time.monotonic();folder=output/split/name;folder.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256()
    setup=setup_rich if dataset=='rich' else setup_egobody
    q,people,transform,static,camera,head,offset,meta=setup(split,name,digest)
    n=len(q);intrinsics=meta.pop('_intrinsics',None)
    if intrinsics is None:
        intrinsics=np.broadcast_to([64/np.tan(np.deg2rad(meta['hfov']/2)),48/np.tan(np.deg2rad(meta['vfov']/2)),64,48],(n,4)).copy().astype(np.float32)
        meta['projection_intrinsics']='fixed virtual 66.56x40.49; exact pinhole center128x96'
    cloud=np.lib.format.open_memmap(folder/FILES[1],mode='w+',dtype='float32',shape=(n,512,3));owner=np.lib.format.open_memmap(folder/FILES[2],mode='w+',dtype='int32',shape=(n,512));cams=[];rots=[];counts=[];selected={};fullhead=[]
    self_faces,head_info=wearer_faces(people[0][0].faces,people[0][0].lbs_weights.numpy(),model_type='smplx' if dataset=='rich' else 'smpl')
    for lo in range(0,n,64):
        hi=min(lo+64,n);bodies=[forward_chunk(model,params,lo,hi,transform) for model,params in people]
        if camera is None:cam,rot=virtual_head_camera(bodies[0][1][:,15],head[lo:hi],offset_head=offset)
        else:cam,rot=camera[0][lo:hi],camera[1][lo:hi]
        cams.append(cam);rots.append(rot)
        for k in range(hi-lo):
            meshes=[(bodies[0][0][k],self_faces,100)]+[(bodies[a][0][k],people[a][0].faces,100+a) for a in range(1,len(people))]
            rendered=render_scene(meshes,cam[k],rot[k],static_scene=static,intrinsics=intrinsics[lo+k])
            points,ids=sample_scene_hits(rendered);cloud[lo+k]=points;owner[lo+k]=ids;counts.append({str(a):int((rendered['owner']==a).sum()) for a in [-1,0]+list(range(100,100+len(people)))})
            if lo+k in [0,n//2,n-1]:selected[f'depth_{lo+k}']=rendered['depth'];selected[f'owner_{lo+k}']=rendered['owner']
        print(f'{name}: {hi}/{n}',flush=True)
    np.save(folder/'camera_intrinsics_128x96.npy',intrinsics)
    cloud.flush();owner.flush();np.save(folder/FILES[0],q);np.save(folder/FILES[3],np.concatenate(cams));np.save(folder/FILES[4],np.concatenate(rots));np.savez_compressed(folder/'selected_raycast_audit.npz',**selected)
    meta.update(sequence_id=name,split=split,output_fps=20,source_fps=30,frames=n,occlusion_version=2,protocol=PROTOCOL,source_stride_for_scene=1,pointcloud_interpolation=False,body_parameter_resampling='Direct consecutive raw30 to query20; rotation SLERP, translation linear; RICH hand PCA coefficients linear',head_exclusion=head_info,input_sha256=digest.hexdigest(),elapsed_s=time.monotonic()-start)
    (folder/'occlusion_audit.json').write_text(json.dumps(dict(frames=counts)));meta['output_sha256']={f:hashlib.sha256((folder/f).read_bytes()).hexdigest() for f in FILES+('camera_intrinsics_128x96.npy',)};temp=folder/'metadata.tmp';temp.write_text(json.dumps(meta,indent=2));temp.replace(folder/'metadata.json');return meta

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['rich','egobody'],required=True);p.add_argument('--split',choices=['train','validation'],required=True);p.add_argument('--sequence',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();export(a.dataset,a.split,a.sequence,a.output)
if __name__=='__main__':main()
