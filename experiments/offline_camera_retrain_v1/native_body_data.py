"""Native SMPL/SMPL-X coexistence without cross-model parameter copying.

Shared 22-body-joint motion. Root alone changes under world rotations; local
body rotations and native rest joints retain their model coordinate convention.
Archived source joints remain an audit reference, not conflicting supervision.
"""
import json,pickle,sys,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from smplx.lbs import blend_shapes,vertices2joints
from experiments.offline_camera_retrain_v1.data import OfflineSceneMIData,HERE,PARENTS,rotation6d
from experiments.offline_camera_retrain_v1.export_pose_sequences import slerp_rotvec,linear_sample
from experiments.offline_camera_retrain_v1.training_contract_v2 import covered_frames
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from prepare_rich_pseudo_ego import RICH_TO_SCENEMI,_read_transform
from experiments.offline_sequence_v1.data_loader import anchor_rotation


def native_fk(pose,translation,rest):
    local=Rotation.from_rotvec(np.asarray(pose).reshape(-1,3)).as_matrix().reshape(-1,22,3,3)
    rotations=[local[:,0]];positions=[translation+rest[0]]
    for j in range(1,22):
        parent=PARENTS[j];rotations.append(rotations[parent]@local[:,j]);positions.append(positions[parent]+np.einsum('tij,j->ti',rotations[parent],rest[j]-rest[parent]))
    return np.stack(positions,1).astype(np.float32),np.stack(rotations,1).astype(np.float32)


class NativeBodyData(OfflineSceneMIData):
    def __init__(self,split,**kwargs):
        if split=='test':raise ValueError('native_v1 mixed test is undefined: RICH has official train/val only; do not relabel val as test')
        self.temporal_scene_manifest=kwargs.pop('temporal_scene_manifest',None)
        self.temporal_scenes={}
        self.temporal_predecessors={}
        self.rich_causal_scene=kwargs.pop('rich_causal_scene',False)
        self.contact_root=kwargs.pop('contact_root',None)
        if self.contact_root is not None:self.contact_root=Path(self.contact_root)
        self.contact_cache={}
        kwargs.pop('skeleton_profile',None);kwargs['rich_source']='native20_faceout_oct07'
        kwargs.setdefault('trumans_window_protocol','temporal_valid_v1' if kwargs.get('trumans_scene_manifest') and self.temporal_scene_manifest else 'legacy_stable_v1')
        super().__init__(split,skeleton_profile='trumans_male_v2',**kwargs)
        self.native_models={};self.native_cache={};self.native_meta={}
        if self.temporal_scene_manifest is not None:
            manifest=Path(self.temporal_scene_manifest).resolve()
            for row in map(json.loads,manifest.read_text().splitlines()):
                if row['split']!=split:continue
                name=row['sequence_id'];folder=Path(row['path'])
                if name in self.temporal_scenes:raise ValueError('Duplicate temporal scene sequence')
                if not folder.is_absolute():folder=manifest.parent/folder
                meta=json.loads((folder/'metadata.json').read_text())
                if meta.get('split')!=split or meta.get('sequence_id')!=name or meta.get('dataset')!=row.get('dataset'):raise ValueError('Temporal manifest scope mismatch')
                self.temporal_scenes[name]=folder
            required={name for name,m in self.prepared.items() if m['dataset']=='egobody'}|{m['sequence_id'] for m,v in self.rich_members}
            missing=required-self.temporal_scenes.keys()
            if missing:raise ValueError(f'Exact20 native scenes missing {len(missing)} sequences; no 5Hz/union fallback')
            for name in required:
                folder=self.temporal_scenes[name];meta=json.loads((folder/'metadata.json').read_text());ids=np.load(folder/'source_frame_ids.npy')
                if name in self.prepared:
                    m=self.prepared[name];expected=m['source_first_frame']+np.arange(m['frames_20fps'])*1.5
                else:expected=np.load(self.rich_root/name/'source_frame_ids.npy')
                if meta.get('memory_protocol')!='causal20-first-world-voxel25mm-current-dynamic-v1':raise ValueError(f'{name}: causal recording features incomplete')
                if meta.get('protocol')!='native20-scene-v4-raw30-full-body-first-hit' or meta.get('memory_protocol')!='causal20-first-world-voxel25mm-current-dynamic-v1' or not meta.get('projection_intrinsics') or not (folder/'camera_intrinsics_128x96.npy').is_file() or len(ids)!=len(expected) or not np.allclose(ids,expected,atol=1e-6):raise ValueError(f'{name}: stale, partial or non20 scene bundle')

            # Continue static memory across valid contiguous segments of the
            # same EgoBody recording AND observer role, scoped to this split.
            groups={}
            for name,folder in self.temporal_scenes.items():
                meta=json.loads((folder/'metadata.json').read_text())
                if meta['dataset']!='egobody':continue
                key=(meta['recording'],meta['subject_ids'][0]);ids=np.load(folder/'source_frame_ids.npy')
                groups.setdefault(key,[]).append((name,folder,float(ids[0])))
            for entries in groups.values():
                for name,folder,start in entries:self.temporal_predecessors[name]=[other for other_name,other,first in entries if other_name!=name and first<start]
    def model(self,kind,gender):
        key=(kind,gender)
        if key not in self.native_models:self.native_models[key]=load_model(*key)
        return self.native_models[key]
    def _native(self,name,group):
        if name in self.native_cache:return self.native_cache[name]
        if group=='rich':
            meta=json.loads((self.rich_root/name/'metadata.json').read_text());subject=meta['subject_id'];gender=meta['gender'];kind='smplx'
            source=HERE.parents[2]/'RICH/extracted'/('train_body' if self.split=='train' else 'val_body')/name
            dirs=sorted((p for p in source.iterdir() if p.is_dir()),key=lambda p:int(p.name));ids=np.array([int(p.name) for p in dirs],float)
            if not np.all(np.diff(ids)==1):raise ValueError('Nonconsecutive native RICH fits')
            fits=[]
            for folder in dirs:
                with (folder/f'{subject}.pkl').open('rb') as stream:
                    fits.append(pickle.load(stream,encoding='latin1'))
            beta=np.asarray(fits[0]['betas'],np.float32).reshape(-1)[:10]
            if any(not np.allclose(np.asarray(f['betas']).reshape(-1)[:10],beta,atol=1e-4) for f in fits):raise ValueError('RICH shape changes')
            pose=np.array([np.concatenate((f['global_orient'].reshape(1,3),f['body_pose'].reshape(21,3))) for f in fits],np.float32)
            query=np.load(self.rich_root/name/'source_frame_ids.npy').astype(float)
            pose=slerp_rotvec(pose,query-ids[0]);raw_t=linear_sample(np.array([f['transl'].reshape(3) for f in fits]),query-ids[0])
            scene=meta['scene'];variant='chair' if 'chair' in name else 'yoga';tf=f'LectureHall_{variant}_multicam2world.json' if scene=='LectureHall' else f'{scene}_multicam2world.json'
            scale,world_rotation,world_offset=_read_transform(HERE.parents[2]/'RICH/extracted/multicam2world'/tf)
            transform=RICH_TO_SCENEMI@world_rotation.T
            pose[:,0]=Rotation.from_matrix(transform[None]@Rotation.from_rotvec(pose[:,0]).as_matrix()).as_rotvec()
            model=self.model(kind,gender)
            with torch.no_grad():rest0=vertices2joints(model.J_regressor,model.v_template[None]+blend_shapes(torch.tensor(beta[None]),model.shapedirs[:,:,:10]))[0,:22].numpy()
            rest=(rest0*scale).astype(np.float32)
            root=(scale*((raw_t+rest0[0])@world_rotation)+world_offset)@RICH_TO_SCENEMI.T
            translation=root-rest[0]
        else:
            meta=self.pose_meta[name];kind='smplx' if group=='trumans' else 'smpl';gender='male' if group=='trumans' else meta['gender'];scale=1.
            pose,translation,_=self._pose(name);pose=np.asarray(pose).copy();translation=np.asarray(translation).copy();beta=np.asarray(meta['betas'],np.float32)
            model=self.model(kind,gender)
            with torch.no_grad():rest=vertices2joints(model.J_regressor,model.v_template[None]+blend_shapes(torch.tensor(beta[None]),model.shapedirs[:,:,:10]))[0,:22].numpy()
            if group=='trumans':
                # Released joints agree with male native template. Explicitly
                # retain measured pelvis; source translation bias is audited separately.
                root=np.load(Path(meta['prepared_folder'])/'joints_world.npy',mmap_mode='r')[:,0]
                translation=root-rest[0]
        joints,rotations=native_fk(pose,translation,rest)
        self.native_meta[name]=dict(model=kind,gender=gender,betas=beta.tolist(),scale=float(scale),root_translation_protocol='native rest-root compensation; TRUMANS measured pelvis',missing_hand_face_supervision=True)
        value=(pose.astype(np.float32),translation.astype(np.float32),rest.astype(np.float32),joints,rotations)
        self.native_cache[name]=value
        return value
    def sample(self,length,group,**kwargs):
        rich_scene_override=kwargs.pop('scene_bundle',None) if group=='rich' else None
        sample,identity=super().sample(length,group,**kwargs);name=identity['sequence_id'];pose,trans,rest,joints,rotations=self._native(name,group)
        if group=='rich':
            folder=self.rich_root/name;ids=np.load(folder/'source_frame_ids.npy');start=int(np.searchsorted(ids,identity['source_start_30fps']))
            camera=np.load(folder/'camera_position_scenemi_yup.npy')[start:start+length];camera_r=np.load(folder/'camera_rotation_scenemi_yup.npy')[start:start+length]
        else:
            meta=self.prepared[name];start=int(round((identity['source_start_30fps']-meta.get('source_first_frame',0))/1.5));_,camera_all,r_all,_=self.base._load(meta)
            camera=camera_all[start:start+length];camera_r=r_all[start:start+length]
            if identity.get('scene_bundle'):
                folder=Path(identity['scene_bundle']);ids=np.load(folder/'source_frame_ids.npy');ix=np.searchsorted(ids,identity['source_start_30fps']+np.arange(length)*1.5)
                camera=np.load(folder/'camera_position_scenemi_yup.npy')[ix];camera_r=np.load(folder/'camera_rotation_scenemi_yup.npy')[ix]
        scene_folder=self.temporal_scenes.get(name,rich_scene_override)
        if scene_folder is not None:
            from experiments.offline_camera_retrain_v1.causal_scene import temporal_scene_inputs
            queries=ids[start:start+length] if group=='rich' else identity['source_start_30fps']+np.arange(length)*1.5
            inputs,camera,camera_r,scene_identity=temporal_scene_inputs(scene_folder,queries,history_bundles=self.temporal_predecessors.get(name))
            sample.update(inputs);identity.update(scene_identity)
            a=anchor_rotation(camera_r);r=np.einsum('ij,tjk->tik',a,camera_r)
            sample['camera']=np.concatenate(((camera-camera[0])@a.T/2,r[:,:,0],r[:,:,1]),axis=1).astype(np.float32)
        elif group=='rich' and self.rich_causal_scene:
            from experiments.offline_camera_retrain_v1.causal_scene import static_scene_inputs
            meta=json.loads((folder/'metadata.json').read_text());observed=Path(meta['scene_bundle'])
            occ,bps,valid,n=static_scene_inputs(np.load(observed/'source_frame_ids.npy'),np.load(observed/'visible_frame_points_scenemi_yup.npy',mmap_mode='r'),np.load(observed/'visible_frame_mask.npy',mmap_mode='r'),ids[start:start+length],camera,camera_r)
            sample.update(occupancy=occ,bps=bps,bps_valid=valid)
            identity.update(scene_protocol='causal static 5Hz observations; recording prefix; start-fixed occupancy',scene_history_points=n,scene_observation_fps=5)
        if start+length>len(pose):raise ValueError('Native pose coverage mismatch')
        anchor=anchor_rotation(camera_r);origin=camera[0];local=Rotation.from_rotvec(pose[start:start+length].reshape(-1,3)).as_matrix().reshape(length,22,3,3)
        local[:,0]=anchor[None]@local[:,0]
        native_joints=(joints[start:start+length]-origin)@anchor.T
        transl=((trans[start:start+length]+rest[0]-origin)@anchor.T-rest[0])/2
        pose6=(rotation6d(local)-np.array([1,0,0,0,1,0])).reshape(length,132)
        archive_joints=sample['joints']*2
        sample['motion']=np.concatenate((transl,pose6,native_joints.reshape(length,66)/2),axis=-1).astype(np.float32)
        sample['joints']=(native_joints/2).astype(np.float32);sample['rest']=rest.copy()
        global_r=np.einsum('ij,tqjk->tqik',anchor,rotations[start:start+length])
        sample['trajectory']=np.concatenate((native_joints/2,rotation6d(global_r)),axis=-1).astype(np.float32);sample['trajectory'][:,15]=sample['camera']
        info=self.native_meta[name];sample['body_type']=np.eye(2,dtype=np.float32)[int(info['model']=='smplx')]
        sample['body_scale']=np.array([info['scale']],np.float32);sample['body_betas']=np.asarray(info['betas'],np.float32)
        if self.contact_root is not None:
            sample['contact_target']=np.zeros((length,22),np.float32);sample['contact_valid']=np.zeros((length,22),bool)
            if group=='rich':
                if name not in self.contact_cache:
                    from experiments.offline_camera_retrain_v1.contact_supervision import vertex_regions,aggregate_contacts
                    contact_path=self.contact_root/('train' if self.split=='train' else 'val')/name/'contacts.npz'
                    with np.load(contact_path) as labels:
                        if not np.allclose(labels['source_frame_ids'],np.load(self.rich_root/name/'source_frame_ids.npy')):raise ValueError('Contact timestamps mismatch')
                        self.contact_cache[name]=aggregate_contacts(labels['smplx_packed'],labels['valid'][:,1],vertex_regions(self.model(info['model'],info['gender'])),10475)
                target,valid=self.contact_cache[name];sample['contact_target']=target[start:start+length].copy();sample['contact_valid']=valid[start:start+length].copy()
        identity=dict(identity,native_body=info,archived_joint_difference_mean_m=float(np.linalg.norm(native_joints-archive_joints,axis=-1).mean()))
        return sample,identity
