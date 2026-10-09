"""Full-manifest sample/FK/mesh and recording-history checks across all lengths."""
import json,time
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE,collate
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.causal_scene import temporal_scene_inputs

def main():
 torch.set_num_threads(2);start=time.monotonic();cases=[];scopes=[]
 for split in ['train','validation']:
  d=NativeBodyData(split,seed=2026,contact_root=HERE/'data/rich_contact_native20_v1',temporal_scene_manifest=HERE/'data/native20_temporal_scenes_v4/manifest.jsonl',trumans_scene_manifest=HERE/'data/trumans_temporal_training_v3/manifest.jsonl')
  for group in ['trumans','camera_wearer','interactee','rich']:
   for length in [64,128,192]:
    s,i=d.sample(length,group);b=collate([s]);fk=forward_kinematics(b['motion'],b['rest'])[0].numpy();assert np.linalg.norm(fk-s['joints']*2,axis=-1).max()<1e-4
    info=i['native_body'];frames=[0,length//2,length-1];_,_,j=decode_native_mesh(s['motion'][frames],info,d.model(info['model'],info['gender']));error=float(np.linalg.norm(j-fk[frames],axis=-1).max());assert error<1e-4
    assert all(np.isfinite(v).all() for v in s.values())
    assert i.get('scene_protocol','').startswith(('exact20','v3:'))
    folder=Path(i['scene_bundle']);ids=np.load(folder/'source_frame_ids.npy');q=i['source_start_30fps']+np.arange(length)*1.5;ix=np.searchsorted(ids,q)
    cam=np.load(folder/'camera_position_scenemi_yup.npy')[ix];rot=np.load(folder/'camera_rotation_scenemi_yup.npy')[ix]
    from experiments.offline_sequence_v1.data_loader import anchor_rotation
    a=anchor_rotation(rot);r=np.einsum('ij,tjk->tik',a,rot);expected=np.concatenate(((cam-cam[0])@a.T/2,r[:,:,0],r[:,:,1]),axis=1);np.testing.assert_allclose(s['camera'],expected,atol=1e-6)
    cases.append(dict(split=split,group=group,length=length,identity=i,native_mesh_fk_max_m=error,contact_entries=int(s['contact_valid'].sum())))
  for name,prior in d.temporal_predecessors.items():
   current=json.loads((d.temporal_scenes[name]/'metadata.json').read_text())
   for folder in prior:
    m=json.loads((folder/'metadata.json').read_text());assert (m['recording'],m['subject_ids'][0],m['split'])==(current['recording'],current['subject_ids'][0],split)
  # Select a target segment with a genuine earlier segment, not synthetic fixture.
  eligible=[(m,v) for g in ['camera_wearer','interactee'] for m,v in d.base.groups[g] if v[64] and d.temporal_predecessors.get(m['sequence_id'])]
  assert eligible
  m,v=eligible[0];name=m['sequence_id'];folder=d.temporal_scenes[name];ids=np.load(folder/'source_frame_ids.npy');q=ids[:64]
  before,cam,r,i=temporal_scene_inputs(folder,q,history_bundles=d.temporal_predecessors[name])
  from experiments.offline_camera_retrain_v1.causal_scene import static_scene_inputs
  p=np.load(folder/'visible_frame_points_scenemi_yup.npy');o=np.load(folder/'visible_frame_owner.npy');reset_occ,*_=static_scene_inputs(ids,p,o==0,q,cam,r)
  assert i['scene_history_points']>0;assert np.abs(before['occupancy']-reset_occ).sum()>0
  scopes.append(dict(split=split,sequence=name,prior_segments=[str(p) for p in d.temporal_predecessors[name]],historical_static_points=i['scene_history_points'],extra_occupied_voxels=int((before['occupancy']!=reset_occ).sum()),same_recording_observer_split=True))
 out=dict(status='passed',cases=cases,recording_history=scopes,elapsed_s=time.monotonic()-start,full_manifests=True)
 (HERE/'runs/training_contract_audit_oct07/scene20_training_inputs_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(status='passed',sample_cases=len(cases),maximum_native_mesh_fk_error_m=max(c['native_mesh_fk_max_m'] for c in cases),history=scopes,elapsed_s=out['elapsed_s'])))
if __name__=='__main__':main()
