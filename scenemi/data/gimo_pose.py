"""Decode GIMO's original VPoser v1.0 latent and verify exact 22-joint GT.

Only the body pose is certified by the released OBJ: the saved lhand/rhand
parameters do not reproduce its fingers, so original meshes remain the skin
reference and decoded meshes never replace them for visibility or playback.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import yaml

from scenemi.data.gimo import ROOT, atomic_json, digest, load_settings, transform_points


def decode(latent, weights):
    """The released vposer_smpl.py decoder in evaluation mode, no dropout."""
    value=latent
    for key in ('fc1','fc2'):
        value=F.leaky_relu(F.linear(value,weights[f'bodyprior_dec_{key}.weight'],weights[f'bodyprior_dec_{key}.bias']),negative_slope=.2)
    value=F.linear(value,weights['bodyprior_dec_out.weight'],weights['bodyprior_dec_out.bias']).reshape(-1,3,2)
    b1=F.normalize(value[:,:,0],dim=1)
    b2=F.normalize(value[:,:,1]-(b1*value[:,:,1]).sum(1,keepdim=True)*b1,dim=1)
    return torch.stack((b1,b2,torch.cross(b1,b2,dim=1)),dim=-1).reshape(-1,21,3,3)


class Exporter:
    def __init__(self,path,device):
        from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
        torch.set_num_threads(1)
        if device=='cuda':
            from scenemi.gpu_policy import configure_cuda
            configure_cuda()
        self.device=torch.device(device)
        self.path=Path(path)
        self.weights_path=self.path/'snapshots/TR00_E096.pt'
        self.weights_sha256=digest(self.weights_path)
        self.weights={k:v.to(self.device) for k,v in torch.load(self.weights_path,map_location='cpu',weights_only=True).items()}
        self.model=load_model('smplx','neutral').eval().to(self.device)
        self.parents=self.model.parents[:22].cpu().numpy()
        dominant=self.model.lbs_weights.argmax(1).cpu().numpy()
        self.certified_vertices=np.flatnonzero(dominant<20)
        self.implementation=digest(__file__)
        self.cpu_cuda_decode_max_abs=None
        if device=='cuda':
            latent=torch.linspace(-1,1,128).reshape(4,32)
            with torch.inference_mode():
                cpu=decode(latent,{k:v.cpu() for k,v in self.weights.items()}).numpy()
                gpu=decode(latent.to(self.device),self.weights).cpu().numpy()
            self.cpu_cuda_decode_max_abs=float(np.max(np.abs(cpu-gpu)))
            if self.cpu_cuda_decode_max_abs>1e-4:raise ValueError('CPU/CUDA decoder mismatch')

    def export(self,row):
        folder=Path(row['folder']);record=folder/'exact_pose.json'
        if record.exists():
            saved=json.loads(record.read_text())
            if saved['vposer_sha256']!=self.weights_sha256 or saved['implementation_sha256']!=self.implementation:
                raise ValueError('Pose decoder changed after export')
            return saved
        with np.load(folder/'source_parameters.npz',allow_pickle=False) as data:
            params={k:np.asarray(data[k]) for k in data.files}
        valid=np.load(folder/'observation_valid.npy').copy()
        transform=np.load(folder/'body_to_world.npy').astype(np.float64)
        truth=np.load(folder/'vertices_world.npy',mmap_mode='r')
        n=row['frames'];pose=np.zeros((n,22,3),np.float32);joints=np.zeros((n,22,3),np.float32)
        translation=np.zeros((n,3),np.float32);head=np.broadcast_to(np.eye(4,dtype=np.float32),(n,4,4)).copy()
        tracker_r=np.zeros((n,3,3,3),np.float32);rest=np.zeros((n,22,3),np.float32)
        mesh_error=np.zeros(n,np.float32);finger_error=np.zeros(n,np.float32)
        from smplx.lbs import blend_shapes,vertices2joints
        for lo in range(0,n,32):
            hi=min(n,lo+32);ix=np.arange(lo,hi);good=valid[ix]
            if not good.any():continue
            ids=ix[good]
            values={k:torch.as_tensor(v[ids],device=self.device) for k,v in params.items()}
            with torch.inference_mode():
                rotations=decode(values['latent'],self.weights).cpu().numpy()
                aa=Rotation.from_matrix(rotations.reshape(-1,3,3)).as_rotvec().reshape(-1,63)
                forward=self.model(global_orient=values['orient'],transl=values['trans'],betas=values['beta'],
                    body_pose=torch.as_tensor(aa,dtype=torch.float32,device=self.device),left_hand_pose=values['lhand'],
                    right_hand_pose=values['rhand'],jaw_pose=values['jaw_pose'],expression=values['expression'],
                    leye_pose=torch.zeros_like(values['orient']),reye_pose=torch.zeros_like(values['orient']))
                v=transform_points(forward.vertices.cpu().numpy(),transform)
                j=transform_points(forward.joints[:,:22].cpu().numpy(),transform)
                native_rest=vertices2joints(self.model.J_regressor,self.model.v_template[None]+
                    blend_shapes(values['beta'],self.model.shapedirs[:,:,:10]))[:,:22].cpu().numpy()
            mesh_error[ids]=np.linalg.norm(v[:,self.certified_vertices]-truth[ids][:,self.certified_vertices],axis=-1).max(axis=1)
            finger_error[ids]=np.linalg.norm(v-np.asarray(truth[ids]),axis=-1).max(axis=1)
            valid[ids]&=mesh_error[ids]<.0005
            body=np.zeros((len(ids),22,3),np.float32);body[:,1:]=aa.reshape(-1,21,3)
            root=transform[:3,:3]@Rotation.from_rotvec(params['orient'][ids]).as_matrix()
            body[:,0]=Rotation.from_matrix(root).as_rotvec()
            local=Rotation.from_rotvec(body.reshape(-1,3)).as_matrix().reshape(-1,22,3,3)
            global_r=local.copy()
            for joint in range(1,22):global_r[:,joint]=global_r[:,self.parents[joint]]@local[:,joint]
            pose[ids]=body;joints[ids]=j;rest[ids]=native_rest
            # SMPL-X rotates about its shape-dependent rest pelvis.
            translation[ids]=transform_points(params['trans'][ids]+native_rest[:,0],transform)-native_rest[:,0]
            head[ids,:3,:3]=global_r[:,15];head[ids,:3,3]=j[:,15]
            tracker_r[ids]=global_r[:,[15,20,21]]
        arrays=dict(pose_axis_angle_world=pose,translation_world=translation,joints_world=joints,
            rest_joints=rest,head_joint_world=head,synthetic_gt_tracker_rotations_world=tracker_r,
            synthetic_gt_tracker_positions_world=joints[:,[15,20,21]],pose_valid=valid,
            decoded_body_mesh_max_error_m=mesh_error,decoded_full_mesh_max_error_m=finger_error)
        for name,value in arrays.items():np.save(folder/(name+'.npy'),value)
        tested=np.load(folder/'observation_valid.npy')
        saved=dict(id=row['id'],frames=n,valid_frames=int(valid.sum()),
            matched_body_vertices=len(self.certified_vertices),
            body_mesh_error_max_m=float(mesh_error[tested].max()),
            full_mesh_error_max_m=float(finger_error[tested].max()),
            excluded_pose_frames=np.flatnonzero(tested&~valid).tolist(),
            vposer=str(self.path),vposer_sha256=self.weights_sha256,implementation_sha256=self.implementation,
            source_export_sha256=digest(folder/'metadata.json'),
            pose_convention='22 joints, root world rotation + 21 local axis-angle rotations; SMPL-X neutral, original shape',
            translation_convention='world SMPL-X transl with rest-pelvis rotation correction',
            hands='Fingers in original mesh differ from saved hand parameters. Original mesh retained; exact body/FK certified separately.',
            tracker_provenance='synthetic from GT FK; not the measured head/hand sensor and not camera-only observations',
            output_sha256={name+'.npy':digest(folder/(name+'.npy')) for name in arrays})
        atomic_json(record,saved)
        return saved


def run(config,device,watch=False):
    _,_,output=load_settings(config)
    paths=yaml.safe_load((ROOT/'configs/paths.local.yaml').read_text())
    exporter=Exporter(paths['gimo_vposer'],device)
    records={}
    while True:
        metas=sorted((output/'segments').glob('*/metadata.json'))
        for file in metas:
            row=json.loads(file.read_text())
            if row['id'] in records:continue
            saved=exporter.export(row);records[row['id']]=saved
            print(json.dumps(dict(poses_done=len(records),id=row['id'],body_max_m=saved['body_mesh_error_max_m'],invalid=saved['excluded_pose_frames'])),flush=True)
        summary_path=output/'summary.json'
        summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if not watch or summary.get('complete'):break
        time.sleep(5)
    result=dict(complete=bool(summary.get('complete')),segments=len(records),frames=sum(r['frames'] for r in records.values()),
        valid_frames=sum(r['valid_frames'] for r in records.values()),body_mesh_max_error_m=max(r['body_mesh_error_max_m'] for r in records.values()),
        full_mesh_max_error_m=max(r['full_mesh_error_max_m'] for r in records.values()),
        cpu_cuda_decode_max_abs=exporter.cpu_cuda_decode_max_abs,vposer_sha256=exporter.weights_sha256,
        records=list(records.values()))
    atomic_json(output/'pose_summary.json',result)
    if (output/'windows.jsonl').exists():
        windows=[json.loads(line) for line in (output/'windows.jsonl').read_text().splitlines()]
        good=[]
        for window in windows:
            mask=np.load(Path(window['folder'])/'pose_valid.npy',mmap_mode='r')
            if mask[window['start']:window['start']+128].all():good.append(window)
        (output/'pose_ready_windows.jsonl').write_text(''.join(json.dumps(w)+'\n' for w in good))
        result['pose_ready_windows']=len(good);result['pose_rejected_windows']=len(windows)-len(good)
        atomic_json(output/'pose_summary.json',result)
        summary['exact_pose_FK_export_ready']=result['complete'] and not result['pose_rejected_windows']
        summary['caveat']='Exact VPoser body pose and 22-joint FK added and mesh checked. Original finger skin differs from saved hand parameters; use original skin for visibility and review.'
        atomic_json(output/'summary.json',summary)
    return {k:v for k,v in result.items() if k!='records'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/gimo128.yaml')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args(argv)
    print(json.dumps(run(args.config,args.device,args.watch),indent=2),flush=True)


if __name__=='__main__':main()
