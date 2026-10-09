"""Recover native-only runtime indices from retained poses, raw pelvis, and exact20 scenes.

Does not recreate or claim byte identity of the deleted historical camera archive.
Legacy/test consumers must not use the resulting archive. READY is issued only by
an independent semantic/provenance audit, never by this recovery writer.
"""
import json
import re
import os
from pathlib import Path
import numpy as np
import torch
from smplx.lbs import blend_shapes, vertices2joints
from experiments.offline_camera_retrain_v1.data import HERE, SOURCE, PROJECTS, PARENTS
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from experiments.offline_camera_retrain_v1.native_body_data import native_fk
from sample.render_trumans_gt_smpl import linear_resample


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(row)+'\n' for row in rows))


def main():
    torch.set_num_threads(2)
    source = {(row['dataset'], row['sequence_id']): row for row in read(SOURCE/'sequences.jsonl')}
    bundles = {}
    for file in [HERE/'data/native20_temporal_scenes_v4/manifest.jsonl', HERE/'data/trumans_temporal_training_v3/manifest.jsonl']:
        for row in read(file):
            if row.get('status') != 'excluded': bundles[(row['split'], row['sequence_id'])] = Path(row['path'])
    human = np.load(PROJECTS/'TRUMANS/human_joints.npy', mmap_mode='r')
    models = {}
    report = {'protocol': 'native-source-recovery-v1', 'scope': 'native_v1 exact20 train/validation only; no legacy/test recovery', 'splits': {}}
    for split in ['train', 'validation', 'test']:
        output = {'trumans': [], 'egobody': []}
        for pose_meta in read(HERE/f'data/poses/{split}.jsonl'):
            name=pose_meta['sequence_id'];dataset=pose_meta['dataset'];frames=pose_meta['frames_20fps']
            if dataset=='trumans':
                original=source[(dataset,name)];first=0
                meta=dict(dataset=dataset,sequence_id=name,split=split,scene_id=original['scene_id'],scene_family=original['scene_family'],
                          frames_20fps=frames,fps=20,camera_source=original['camera_source'],object_tracks=original['object_tracks'],
                          dynamic_objects_present=original['dynamic_objects_present'])
            else:
                match=re.fullmatch(r'(recording_.+)_(camera_wearer|interactee)_(\d+)-(\d+)',name)
                if not match:raise ValueError(name)
                recording,role,begin,end=match.groups();first=int(begin);original=source[(dataset,f'{recording}/{role}')]
                meta=dict(dataset=dataset,sequence_id=name,split=split,recording=recording,role=role,scene_id=original['scene_id'],
                          scene_family=original['scene_family'],source_first_frame=first,source_last_frame_exclusive=int(end),
                          frames_20fps=frames,fps=20,camera_source='real_PV_pose_synthetic_scene_depth' if role=='camera_wearer' else 'synthetic_head_xright_yup_zforward',
                          object_tracks=None,dynamic_objects_present=False)
            folder=Path(pose_meta['prepared_folder']);folder.mkdir(parents=True,exist_ok=True)
            meta.update(folder=str(folder),reconstruction_protocol=report['protocol'],
                        observation_protocol='recovered exact20 first-hit projected rays; native causal inputs required; historical archive is unavailable')
            (folder/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');output[dataset].append(meta)
            if split=='test':continue  # metadata inventory only, never fake test cameras.
            if dataset=='trumans':
                raw=human[original['source_global_start']:original['source_global_end_exclusive'],:22]
                query=np.arange(0,original['source_frames']-1,1.5,dtype=np.float64)
                joints=linear_resample(raw,query).astype(np.float32)
                assert len(joints)==frames
                np.save(folder/'joints_world.npy',joints)
            scene=bundles.get((split,name))
            if scene is None:continue  # excluded/unusable recordings retain only real pelvis provenance.
            ids=np.load(scene/'source_frame_ids.npy');expected=first+np.arange(frames)*1.5
            if not np.array_equal(ids,expected):raise ValueError(f'{name}: exact20 clock mismatch')
            if dataset=='egobody':
                gender=pose_meta['gender'];key=('smpl',gender)
                if key not in models:models[key]=load_model(*key)
                model=models[key];beta=np.array(pose_meta['betas'],np.float32)[None]
                with torch.no_grad():rest=vertices2joints(model.J_regressor,model.v_template[None]+blend_shapes(torch.tensor(beta),model.shapedirs[:,:,:10]))[0,:22].numpy()
                pose=np.load(Path(pose_meta['folder'])/'pose_axis_angle_world.npy');trans=np.load(Path(pose_meta['folder'])/'translation_world.npy')
                joints,_=native_fk(pose,trans,rest);np.save(folder/'joints_world.npy',joints)
            for source_name,dest_name in [('camera_position_scenemi_yup.npy','camera_position_world.npy'),('camera_rotation_scenemi_yup.npy','camera_rotation_world.npy')]:
                target=folder/dest_name
                if target.exists() or target.is_symlink():target.unlink()
                target.symlink_to((scene/source_name).resolve())
            visible=HERE/f'data/visible_{dataset}_5hz/{split}/{name}/visible_static_points_world.npy'
            if dataset=='egobody':visible=HERE/f'data/visible_egobody_5hz/{split}/{name}/visible_static_points_world.npy'
            if not visible.is_file():raise FileNotFoundError(visible)
            target=folder/'visible_static_points_world.npy'
            if target.exists() or target.is_symlink():target.unlink()
            target.symlink_to(visible.resolve())
        for dataset,rows in output.items():write(SOURCE/f'{dataset}_sequences/{split}.jsonl',rows)
        report['splits'][split]={dataset:len(rows) for dataset,rows in output.items()}
        print(split,report['splits'][split],flush=True)
    (SOURCE/'RECOVERY.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Runtime archive reconstructed; READY not issued. Independent audit still required.',flush=True)

if __name__=='__main__':main()
