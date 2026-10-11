"""Native SMPL VideoMimic import gate. Never interpret G1 DOFs as human pose.

The group name 'our_pred_humans_smplx_params' is misleading in the released
code: its visualizer instantiates SMPL and consumes 23 body rotations. This
adapter deliberately validates actual topology and source frame metadata.
"""
import argparse, json, re
from pathlib import Path
import h5py, numpy as np, torch
from scipy.spatial.transform import Rotation, Slerp
from smplx.lbs import blend_shapes, vertices2joints
from smplx.vertex_ids import vertex_ids
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from experiments.offline_camera_retrain_v1.native_body_data import native_fk

Z_TO_Y=np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32)

def time_queries(frame_ids,source_fps,target_fps=20):
    ids=np.asarray(frame_ids,dtype=float)
    if source_fps<=0 or len(ids)<2 or not np.isfinite(ids).all() or np.any(np.diff(ids)<=0):raise ValueError('Need strictly increasing original frame IDs and confirmed original RGB FPS')
    source_hz=source_fps/np.median(np.diff(ids))
    if source_hz<target_fps-1e-5:raise ValueError('Human reconstruction already below20Hz; reconstruct original frames at>=20Hz. Do not upsample low-rate reconstructions.')
    if np.max(np.diff(ids))/source_fps>.1:raise ValueError('Human tracking gap>0.1s; split recording, do not interpolate across it')
    duration=(ids[-1]-ids[0])/source_fps;q=ids[0]+np.arange(int(np.floor(duration*target_fps+1e-8))+1)*source_fps/target_fps
    return q,dict(source_fps=source_fps,native_fit_hz=float(source_hz),output_fps=target_fps,output_frames=len(q),duration_seconds=duration,direct_native_conversion=True)

def validate_rotations(a,name):
    a=np.asarray(a,dtype=np.float32)
    if a.shape[-2:]!=(3,3) or not np.isfinite(a).all():raise ValueError(f'{name}: require finite native SO(3) matrices')
    if np.max(np.abs(np.swapaxes(a,-1,-2)@a-np.eye(3)))>1e-3 or np.max(np.abs(np.linalg.det(a)-1))>1e-3:raise ValueError(f'{name}: invalid rotations/reflected coordinate frame')
    return a

def import_smpl(path,person_id,*,source_fps,gender,source_up_axis):
    with h5py.File(path) as f:
        group=f'our_pred_humans_smplx_params/{person_id}'
        if group not in f:raise ValueError('Native human parameters absent. retarget_poses_g1.h5 is robot-only and cannot continue SMPL55k.')
        h=f[group];body=validate_rotations(h['body_pose'][:],'body_pose');root=validate_rotations(h['global_orient'][:],'global_orient').reshape(len(body),1,3,3);pelvis=np.asarray(h['root_transl'][:],np.float32).reshape(-1,3);beta=np.asarray(h['betas'][:],np.float32).reshape(-1,10)
        if body.shape[1:]!=(23,3,3):raise ValueError('This gate requires native SMPL23 body rotations. Supply explicit SMPL-X adapter for21; do not copy model parameters.')
        if len(pelvis)!=len(body) or len(root)!=len(body) or not np.isfinite(pelvis).all() or not np.isfinite(beta).all():raise ValueError('Native human shape/time mismatch')
        if len(beta) not in (1,len(body)) or not np.allclose(beta,beta[:1],atol=1e-4):raise ValueError('Changing body shape: stabilize reconstruction before training')
        names=f[f'person_frame_info_list/{person_id}'][:].astype(str)
        if len(names)!=len(body):raise ValueError('Human source frame metadata mismatch')
        ids=np.array([int(re.search(r'(\d+)(?:\.[^.]+)?$',Path(n).name).group(1)) for n in names],float)
    q,audit=time_queries(ids,source_fps)
    local=np.concatenate((root,body),1);resampled=np.stack([Slerp(ids,Rotation.from_matrix(local[:,j]))(q).as_matrix() for j in range(24)],1).astype(np.float32)
    rp=np.stack([np.interp(q,ids,pelvis[:,j]) for j in range(3)],1).astype(np.float32);world=Z_TO_Y if source_up_axis=='z' else np.eye(3,dtype=np.float32)
    model=load_model('smpl',gender).cpu();betas=torch.tensor(beta[:1])
    with torch.no_grad():rest=vertices2joints(model.J_regressor,model.v_template[None]+blend_shapes(betas,model.shapedirs[:,:,:10]))[0,:22].numpy()
    rotations=resampled[:,:22].copy();rotations[:,0]=world[None]@rotations[:,0];rp=rp@world.T;pose=Rotation.from_matrix(rotations.reshape(-1,3,3)).as_rotvec().reshape(-1,22,3).astype(np.float32)
    translation=rp-rest[0];joints,global_r=native_fk(pose,translation,rest)
    # Cross-check native FK against actual full SMPL forward, not another
    # copied skeleton. Output root_transl in VideoMimic is absolute pelvis.
    v=[];gt=[]
    for start in range(0,len(q),32):
        rr=torch.from_numpy(resampled[start:start+32]);n=len(rr)
        with torch.no_grad():out=model(global_orient=rr[:,:1],body_pose=rr[:,1:],betas=betas.expand(n,-1),pose2rot=False,return_verts=True)
        zero_root=out.joints[:,0:1].numpy();v.append((out.vertices.numpy()-zero_root)@world.T+rp[start:start+n,None]);gt.append((out.joints[:,:22].numpy()-zero_root)@world.T+rp[start:start+n,None])
    vertices=np.concatenate(v).astype(np.float32);reference=np.concatenate(gt);error=float(np.linalg.norm(reference-joints,axis=-1).max())
    if error>1e-4:raise ValueError(f'Native FK differs from full SMPL by{error}m')
    eye_ids=[vertex_ids['smplh']['leye'],vertex_ids['smplh']['reye']];eyes=vertices[:,eye_ids].mean(1);first=min(20,len(q));offset=np.median(np.einsum('tji,tj->ti',global_r[:first,15],eyes[:first]-joints[:first,15]),axis=0);offset[2]+=.02
    camera=joints[:,15]+np.einsum('tij,j->ti',global_r[:,15],offset)
    audit.update(model='smpl',gender=gender,native_body_type='original human',source=str(path),person_id=str(person_id),source_up_axis=source_up_axis,world_to_scenemi=world.tolist(),native_FK_max_error_m=error,camera='native head15 rigid first20-eye offset +2cm clearance, identity face mount',camera_mount_m=offset.tolist(),training_ready=False,pending='render matching gravity-aligned scene and same-frame native body occlusion; recording-level split; causal memory; terrain support QC')
    arrays=dict(source_frame_ids=q,pose_axis_angle_world=pose,translation_world=translation,rest_joints=rest,joints_world=joints,camera_position_scenemi_yup=camera,camera_rotation_scenemi_yup=global_r[:,15],betas=beta[0],world_to_scenemi=world)
    return arrays,audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('human_h5',type=Path);p.add_argument('--person-id',required=True);p.add_argument('--source-fps',required=True,type=float);p.add_argument('--gender',required=True,choices=['male','female','neutral']);p.add_argument('--source-up-axis',required=True,choices=['y','z']);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4);a,audit=import_smpl(args.human_h5,args.person_id,source_fps=args.source_fps,gender=args.gender,source_up_axis=args.source_up_axis);args.output.mkdir(parents=True,exist_ok=True)
    for name,array in a.items():np.save(args.output/(name+'.npy'),array)
    (args.output/'human_import_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
