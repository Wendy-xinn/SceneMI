"""Fixed prior failure cases; save thin native motions, not duplicate meshes."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch,trimesh
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.body_history_condition import HistorySceneMI,attach_executed_history
from experiments.offline_camera_retrain_v1.executed_prefix_sampling import sample_with_executed_prefix
from experiments.offline_camera_retrain_v1.state_relative_spline_refinement import refine_from_state
from experiments.offline_camera_retrain_v1.bounded_head_observation import prepare_bounded_observation
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.prepare_trumans_dynamic_v2 import load_objects
from experiments.offline_camera_retrain_v1.evaluate_body_history import measure,crop
from experiments.offline_sequence_v1.data_loader import anchor_rotation
HERE=Path(__file__).parent;RUN=HERE/'runs/state_relative_spline_oct10';OUT=RUN/'demo'


@torch.inference_mode()
def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True);selected=json.loads((RUN/'frozen_selection.json').read_text())
    assert hashlib.sha256((HERE/'state_relative_spline_refinement.py').read_bytes()).hexdigest()==selected['source_sha256']
    cp=torch.load(HERE/'runs/body_history_replan_oct10/delta_history/last.pt',map_location='cpu',weights_only=False);c=cp['config']
    model=HistorySceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True,conditioning_trial='delta_history').cuda().eval();model.load_state_dict(cp['model']);del cp
    d=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'])
    rows=[json.loads(s) for s in (HERE/'runs/native_dynamic_scene20_contact_55k_oct07/full_validation/standard_rows.jsonl').read_text().splitlines()]
    projects=HERE.resolve().parents[2];clips={x['clip_name']:x for x in map(json.loads,(projects/'TRUMANS/processed/scene_expert_v1/clips.jsonl').read_text().splitlines())};cases=[]
    for index in [850,906]:
        row=rows[index];w=row['window'];sample,identity=d.sample(128,w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);identity_difference={k:{'historical':row['identity'].get(k),'current':identity.get(k)} for k in identity.keys()|row['identity'].keys() if identity.get(k)!=row['identity'].get(k)}
        assert set(identity_difference)<= {'dynamic_object_points'},identity_difference
        batch={k:v.cuda() for k,v in collate([sample]).items()};history=batch['motion'][:,:16];observed=attach_executed_history(prepare_bounded_observation(batch,'joint'),history);mask=fixed_control_mask(1,128,'head','cuda');seed=2026101031+index
        original=sample_with_executed_prefix(model,observed,history,steps=20,seed=seed,control_mask=mask)
        refined,audit=refine_from_state(original,history,observed,pose_frames=32,iterations=60,relative_root=True)
        folder=Path(identity['scene_bundle']);q=identity['source_start_30fps']+np.arange(128)*1.5;ix=np.searchsorted(np.load(folder/'source_frame_ids.npy'),q)
        cam=np.load(folder/'camera_position_scenemi_yup.npy')[ix];rot=np.load(folder/'camera_rotation_scenemi_yup.npy')[ix]
        arrays=dict(gt_motion=sample['motion'],original_motion=original[0].cpu().numpy(),refined_motion=refined[0].cpu().numpy(),camera=cam,rotation=rot,anchor_rotation=anchor_rotation(rot),source_frames=q,
                    visible_points=np.load(folder/'visible_frame_points_scenemi_yup.npy')[ix],visible_owners=np.load(folder/'visible_frame_owner.npy')[ix])
        memory=Path(identity['memory_bundle']);points=np.load(memory/'static_points.npy');times=np.load(memory/'first_observed_source_frames.npy');keep=np.flatnonzero(times<=q[-1]);keep=keep[::max(1,int(np.ceil(len(keep)/60000)))];arrays['static_points']=points[keep];arrays['static_times']=times[keep]
        objects=[]
        if w['group']=='trumans':
            for i,(name,path,r,t,source,valid,bad) in enumerate(load_objects(projects/'TRUMANS',clips[identity['sequence_id']],q)):
                assert valid.all();mesh=trimesh.load(path,force='mesh',process=False);arrays[f'obj_{i}_v']=np.asarray(mesh.vertices,np.float32);arrays[f'obj_{i}_f']=mesh.faces;arrays[f'obj_{i}_r']=r;arrays[f'obj_{i}_t']=t;objects.append(name)
        np.savez_compressed(OUT/f'{index}.npz',**arrays)
        values={label:{h:measure(motion[:,start:end],crop(batch,start,end),turn_threshold=15 if h=='short32' else 30) for h,start,end in [('short32',16,48),('future112',16,None)]} for label,motion in [('original',original),('refined',refined)]}
        cases.append(dict(index=index,identity=identity,group=w['group'],seed=seed,objects=objects,metrics=values,refinement_audit=audit,legacy_identity_difference=identity_difference,actual_dynamic_observation_points=int(((arrays['visible_owners']>0)&(arrays['visible_owners']<100)).sum())));print('demo prepared',index,flush=True)
    (OUT/'manifest.json').write_text(json.dumps(dict(cases=cases,selection='fixed original failure indices850/906, no new-score screening; illustrative, not benchmark',body_history='first16 native GT as simulator-state proxy, exact prefix for both variants',future_head='known anatomical joint trajectory, soft refinement',dynamic_objects='actual transform of corresponding frame, never a history trail',weights='same body_history/delta_history500 model for both variants',new_checkpoint=0),indent=2))

if __name__=='__main__':main()
