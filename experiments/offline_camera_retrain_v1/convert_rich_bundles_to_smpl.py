"""Convert prepared RICH SMPL-X bundles to the canonical SMPL proxy."""
import argparse, json, pickle, shutil
from pathlib import Path
import numpy as np
import torch
import inspect
for name, value in dict(bool=np.bool_, int=int, float=float, complex=complex,
                        object=object, unicode=str, str=str).items():
    if name not in np.__dict__: setattr(np, name, value)
import smplx
from scipy.spatial.transform import Rotation
from prepare_rich_pseudo_ego import RICH_GENDER, RICH_TO_SCENEMI, _read_transform

HERE = Path(__file__).resolve().parent
RICH = Path('/home/wenxin/projects/RICH')
SMPL_ROOT = Path('/home/wenxin/projects/ProtoMotions/data/smpl')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--src-root',type=Path,required=True)
    ap.add_argument('--dst-root',type=Path,required=True); args=ap.parse_args()
    args.dst_root.mkdir(parents=True,exist_ok=True)
    for src in sorted(p for p in args.src_root.iterdir() if p.is_dir()):
        meta_path=src/'metadata.json'
        if not meta_path.is_file(): continue
        meta=json.loads(meta_path.read_text()); seq=meta['sequence_id']; sid=str(meta['subject_id'])
        dst=args.dst_root/seq
        if (dst/'metadata.json').is_file(): print('SKIP',seq,flush=True); continue
        shutil.copytree(src,dst)
        split='train' if 'rich_train' in str(args.src_root) else 'val'
        scene=meta['scene']; variant='chair' if 'chair' in seq else 'yoga'
        tname=f'LectureHall_{variant}_multicam2world.json' if scene=='LectureHall' else f'{scene}_multicam2world.json'
        scale,rot,trans=_read_transform(RICH/'extracted/multicam2world'/tname)
        c2s=RICH_TO_SCENEMI @ rot.T
        gender=RICH_GENDER[int(sid)]
        model=smplx.SMPL(str(SMPL_ROOT/f'SMPL_{gender.upper()}.pkl'),num_betas=10,batch_size=1)
        src_frames=np.load(src/'source_frame_ids.npy')
        pkl_paths=[]
        body_root=RICH/'extracted'/f'{split}_body'/seq
        for frame in src_frames:
            matches=list(body_root.glob(f'{int(frame):05d}/{sid}.pkl')) or list(body_root.glob(f'{int(frame)}/{sid}.pkl'))
            if not matches: raise FileNotFoundError(f'{seq} frame {frame} subject {sid}')
            pkl_paths.append(matches[0])
        poses=[]; betas=[]; transl=[]
        for p in pkl_paths:
            fit=pickle.load(p.open('rb'),encoding='latin1')
            poses.append(np.concatenate((np.asarray(fit['global_orient']).reshape(1,3),np.asarray(fit['body_pose']).reshape(21,3)),0))
            betas.append(np.asarray(fit['betas']).reshape(-1)[:10]); transl.append(np.asarray(fit['transl']).reshape(3))
        poses=np.asarray(poses,np.float32); betas=np.asarray(betas,np.float32); transl=np.asarray(transl,np.float32)
        # SMPL has two terminal hand joints not present in SMPL-X body_pose.
        body_pose=np.pad(poses[:,1:],((0,0),(0,2),(0,0)))
        with torch.inference_mode():
            out=model(global_orient=torch.from_numpy(poses[:,0]),body_pose=torch.from_numpy(body_pose.reshape(len(poses),-1)),betas=torch.from_numpy(betas),transl=torch.from_numpy(transl),return_verts=True)
        v=out.vertices.numpy(); j=out.joints[:,:22].numpy()
        v=scale*np.einsum('tvi,ij->tvj',v,rot)+trans
        j=scale*np.einsum('tvi,ij->tvj',j,rot)+trans
        v=np.einsum('ij,tvj->tvi',RICH_TO_SCENEMI,v); j=np.einsum('ij,tvj->tvi',RICH_TO_SCENEMI,j)
        oldj=np.load(src/'joints_scenemi_yup.npy',mmap_mode='r')
        root_delta=np.asarray(oldj[:,0]-j[:,0])
        v+=root_delta[:,None]; j+=root_delta[:,None]
        with torch.inference_mode():
            shape=torch.from_numpy(betas[:1]); shaped=model.v_template[None]+smplx.lbs.blend_shapes(shape,model.shapedirs[:,:,:10]); rest=smplx.lbs.vertices2joints(model.J_regressor,shaped)[0,:22].numpy()
        rest=(scale*(rest@rot)+trans) @ RICH_TO_SCENEMI.T; rest+=root_delta[0]
        # Training FK expects a root-relative canonical rest template; the
        # transformed RICH rest joints are in absolute scene coordinates.
        rest = rest - rest[0]
        local=Rotation.from_rotvec(poses.reshape(-1,3)).as_matrix().reshape(len(poses),22,3,3)
        local_scene=np.einsum('ik,tjkl,lm->tjim',c2s,local,c2s.T)
        np.save(dst/'pose_axis_angle_scenemi_yup.npy',Rotation.from_matrix(local_scene.reshape(-1,3,3)).as_rotvec().reshape(len(poses),22,3).astype(np.float32))
        np.save(dst/'body_vertices_scenemi_yup.npy',v.astype(np.float32)); np.save(dst/'joints_scenemi_yup.npy',j.astype(np.float32)); np.save(dst/'rest_joints_scenemi_yup.npy',rest.astype(np.float32)); np.save(dst/'body_faces.npy',np.asarray(model.faces,np.int32)); np.save(dst/'self_occlusion_faces.npy',np.asarray(model.faces,np.int32))
        oldtrans=np.load(src/'translation_scenemi_yup.npy')
        np.save(dst/'translation_scenemi_yup.npy',(j[:,0]-rest[0]).astype(np.float32))
        meta['body_model']='SMPL canonical proxy'; meta['smpl_conversion']='SMPL-X pose mapped to SMPL; root trajectory preserved'; meta['smpl_conversion_note']='SMPL-X scene occlusion archive retained; training body/FK arrays are canonical SMPL'
        (dst/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        print('OK',seq,len(v),flush=True)

if __name__=='__main__': main()
