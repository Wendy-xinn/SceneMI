"""Six held-out native-mesh comparisons with the original world placement."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from experiments.offline_camera_retrain_v1.native30_memory import ObservedStaticMemory
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,rotation_from_6d
from experiments.offline_camera_retrain_v1.data import HERE


def render(source,output,regime='clean'):
    saved=np.load(source,allow_pickle=False);identities=json.loads(str(saved['identities_json']))
    step=int(saved['step']);chosen=[]
    for mode in ('ego','exo'):
        for group in ('camera_wearer','trumans','rich'):
            chosen.append(next(i for i,m in enumerate(identities) if m['mode']==mode and m['group']==group))
    entries=[json.loads(line) for line in (HERE/'data/native30_official_memory_v3/manifest.jsonl').read_text().splitlines()]
    arrays={}
    def load(folder,key):
        if (folder,key) not in arrays:arrays[(folder,key)]=np.load(Path(folder)/(key+'.npy'),mmap_mode='r')
        return arrays[(folder,key)]
    memory=ObservedStaticMemory(load,entries);models={};audit=[];frame=96
    fig=plt.figure(figsize=(18,12))
    for panel,i in enumerate(chosen):
        identity=identities[i];info=identity['native_body'];kind,gender=info['model'],info['gender']
        if (kind,gender) not in models:models[(kind,gender)]=load_model(kind,gender).cpu()
        model=models[(kind,gender)];scale=info['scale']
        pose=np.stack([saved[f'{i}_truth'][frame],saved[f'{i}_{regime}_reference_motion'][frame],saved[f'{i}_{regime}_prediction'][frame]])
        motion=torch.from_numpy(pose.astype(np.float32))[None]
        local=rotation_from_6d(motion[...,3:135].reshape(1,3,22,6))[0].numpy()
        angles=Rotation.from_matrix(local.reshape(-1,3,3)).as_rotvec().reshape(3,22,3)
        body=angles[:,1:].reshape(3,63)
        if kind=='smpl':body=np.concatenate((body,np.zeros((3,6))),-1)
        extra={} if kind=='smpl' else dict(left_hand_pose=torch.zeros(3,45),right_hand_pose=torch.zeros(3,45),
            jaw_pose=torch.zeros(3,3),leye_pose=torch.zeros(3,3),reye_pose=torch.zeros(3,3),expression=torch.zeros(3,model.num_expression_coeffs))
        with torch.no_grad():
            native=model(betas=torch.tensor([info['betas']]*3,dtype=torch.float32),global_orient=torch.tensor(angles[:,0],dtype=torch.float32),
                body_pose=torch.tensor(body,dtype=torch.float32),transl=torch.tensor(pose[:,:3]*2/scale,dtype=torch.float32),return_verts=True,**extra)
        vertices=native.vertices.numpy()*scale;joints=native.joints[:,:22].numpy()*scale
        rest=torch.from_numpy(saved[f'{i}_rest'].astype(np.float32))[None]
        error=float(np.max(np.abs(joints-forward_kinematics(motion,rest)[0].numpy())))
        if error>1e-4:raise AssertionError(f'Mesh/FK disagreement: {error}')
        ax=fig.add_subplot(2,3,panel+1,projection='3d')
        static,_=memory.window(identity,identity['mode'],192)
        if len(static):
            static=static[np.linspace(0,len(static)-1,min(2500,len(static)),dtype=int)]
            ax.scatter(static[:,0],static[:,2],static[:,1],s=.4,c='gray',alpha=.18)
        for v,color,alpha in zip(vertices,('royalblue','gray','darkmagenta'),(.38,.18,.55)):
            ax.add_collection3d(Poly3DCollection(v[np.asarray(model.faces)][:,:,[0,2,1]],facecolors=color,edgecolors='none',alpha=alpha))
        available=saved[f'{i}_{regime}_observation_meta'][frame,:,0]>0
        positions=saved[f'{i}_{regime}_observations'][frame,:,:3]*2
        ax.scatter(*positions[available][:,[0,2,1]].T,c='darkorange',s=18,depthshade=False)
        error_mm=float(np.linalg.norm(joints[2]-joints[0],axis=-1).mean()*1000)
        prior_mm=float(np.linalg.norm(joints[1]-joints[0],axis=-1).mean()*1000)
        ax.set_title(f"{identity['group']} / {identity['mode']} ({regime})\nframe error: prior {prior_mm:.1f} -> diffusion {error_mm:.1f} mm")
        centre=joints[0,0];radius=max(1.2,float(np.linalg.norm(joints-joints[0,0],axis=-1).max()))
        ax.set_xlim(centre[0]-radius,centre[0]+radius);ax.set_ylim(centre[2]-radius,centre[2]+radius)
        ax.set_zlim(centre[1]-radius,centre[1]+radius);ax.set_box_aspect((1,1,1));ax.view_init(elev=18,azim=-65)
        ax.set_xlabel('world X (m)');ax.set_ylabel('world Z (m)');ax.set_zlabel('world Y (m)')
        audit.append(dict(case=i,group=identity['group'],mode=identity['mode'],sequence_id=identity['sequence_id'],
            source_frame=identity['source_start_30fps']+frame,mesh_fk_error_m=error,frame_mpjpe_mm=error_mm,reference_mpjpe_mm=prior_mm))
        arrays.clear()
    fig.suptitle(f'Official validation / step {step} / frame {frame} / DDIM20 / no GT alignment')
    fig.legend(handles=[Patch(color='royalblue',label='GT'),Patch(color='gray',label='Input reference'),
                        Patch(color='darkmagenta',label='Scene-conditioned diffusion'),Patch(color='darkorange',label='Trusted joints')],
               loc='lower center',ncol=4)
    fig.tight_layout(rect=(0,.04,1,.96));output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/f'VALIDATION_MESH_{regime.upper()}_STEP{step}.png';fig.savefig(path,dpi=110);plt.close(fig);saved.close()
    (path.with_suffix('.json')).write_text(json.dumps(dict(image=str(path),split='validation',frame=frame,cases=audit,
        hands_face_neutral=True,GT_alignment=False),indent=2)+'\n')
    return str(path)
