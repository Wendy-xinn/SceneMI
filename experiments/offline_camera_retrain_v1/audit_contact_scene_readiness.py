"""Data-wide gate for native contact + causal scene training; no launch side effect."""
import json
from pathlib import Path
import numpy as np
from experiments.offline_camera_retrain_v1.data import HERE,PROJECTS
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.contact_supervision import vertex_regions,aggregate_contacts

def main():
 out=HERE/'runs/training_contract_audit_oct07';out.mkdir(exist_ok=True)
 contacts={};rich_scene={};missing=[]
 for split in ('train','val'):
  data=NativeBodyData('train' if split=='train' else 'validation');pos=np.zeros(22,int);valid=np.zeros(22,int);count=0;observations=0;maximum_age=0.
  for p in sorted((HERE/f'data/rich_contact_native20_v1/{split}').glob('*/contacts.npz')):
   meta=json.loads((data.rich_root/p.parent.name/'metadata.json').read_text());model=data.model('smplx',meta['gender'])
   with np.load(p) as z:
    q=np.load(data.rich_root/p.parent.name/'source_frame_ids.npy');assert np.array_equal(q,z['source_frame_ids'])
    target,mask=aggregate_contacts(z['smplx_packed'],z['valid'][:,1],vertex_regions(model),10475);pos+=(target*mask).sum(0).astype(int);valid+=mask.sum(0);count+=len(q)
   folder=Path(meta['scene_bundle']);ids=np.load(folder/'source_frame_ids.npy');points=np.load(folder/'visible_frame_points_scenemi_yup.npy',mmap_mode='r');m=np.load(folder/'visible_frame_mask.npy',mmap_mode='r')
   assert len(ids)==len(points) and m.shape==points.shape[:2] and np.all(np.diff(ids)>0) and np.isfinite(points[m]).all()
   before=np.searchsorted(ids,q,side='right')-1;good=before>=0
   maximum_age=max(maximum_age,float(np.max((q[good]-ids[before[good]])/30)));observations+=len(ids)
  contacts[split]=dict(frames=count,positive_entries=pos.tolist(),valid_entries=valid.tolist(),positive_fraction=(pos/np.maximum(valid,1)).tolist())
  rich_scene[split]=dict(observations=observations,max_observation_age_s=maximum_age)
 # Audit chair source matrices before spending raycasting time.
 root=PROJECTS/'TRUMANS';rows=list(map(json.loads,(root/'processed/scene_expert_v1/clips.jsonl').read_text().splitlines()));flags=np.load(root/'object_flag.npy',mmap_mode='r');mats=np.load(root/'object_mat.npy',mmap_mode='r');names=np.load(root/'object_list.npy');issues=[]
 prepared=set()
 for split in ('train','validation'):
  d=NativeBodyData(split);prepared.update(d.prepared)
 for row in rows:
  if row['clip_name'] not in prepared:continue
  lo,hi=row['global_start'],row['global_end_exclusive'];f=flags[lo:hi]
  for idx in np.flatnonzero((f>=0).any(0)):
   slots=f[:,idx];exists=slots>=0;mat=np.asarray(mats[np.arange(lo,hi)[exists],slots[exists]]);r=mat[:,:3,:3];finite=np.isfinite(mat).all((1,2));bad=~finite
   rigid=np.zeros(len(mat),bool);rigid[finite]=np.max(np.abs(r[finite].transpose(0,2,1)@r[finite]-np.eye(3)),axis=(1,2))<=1e-4
   improper=np.zeros(len(mat),bool);improper[finite]=np.abs(np.linalg.det(r[finite])-1)>1e-4
   if bad.any() or (~rigid).any() or improper.any() or (~exists).any():issues.append(dict(sequence=row['clip_name'],object=str(names[idx]),nan_frames=np.flatnonzero(exists)[bad].tolist(),nonrigid_frames=np.flatnonzero(exists)[finite&~rigid].tolist(),improper_frames=np.flatnonzero(exists)[improper].tolist(),absent_frames=np.flatnonzero(~exists).tolist()))
 status_path=HERE/'data/trumans_temporal_training_v3/status.json';status=json.loads(status_path.read_text())
 blockers=['EgoBody only has offline full-sequence scene unions; timestamped occlusion-aware observations not exported','Full TRUMANS temporal export is incomplete','Invalid TRUMANS chair matrices/presence require explicit handling']
 report=dict(ready_for_long_training=False,contact=contacts,rich_causal_observations=rich_scene,trumans_export=status,trumans_source_issues=issues,blockers=blockers,limitations=['Contact is 22 body-region classification, not vertex correspondence or collision enforcement','Partial head-conditioned observation poses use GT fits for synthetic depth; this is an offline benchmark','Static prefix memory assumes scan surfaces remain static; movable untracked scan objects may leave stale surfaces','No ray-carved known-free space in current occupancy; zero means unknown','No recurrent motion state across target windows; cumulative geometric scene memory is not learned long-term motion memory'])
 (out/'contact_scene_readiness.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(ready=False,trumans_completed=status['completed'],trumans_planned=status['planned'],chair_source_issues=len(issues),rich_scene=rich_scene,contact_positive_fraction={k:v['positive_fraction'] for k,v in contacts.items()})))
if __name__=='__main__':main()
