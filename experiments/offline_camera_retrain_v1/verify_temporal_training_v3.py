"""Full-recording integration check: cross-window memory, no future, dynamic motion."""
import json,tempfile
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import OfflineSceneMIData,collate
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.training_contract_v2 import duration_sampling_plan,covered_frames
HERE=Path(__file__).parent;folder=HERE/'data/trumans_temporal_training_v3/validation/2023-02-18@22-17-40'
torch.set_num_threads(4)
data=OfflineSceneMIData('validation',skeleton_profile='trumans_male_v2',rich_source='native20_faceout_oct07')
members=[(m,v) for m,v in data.base.groups['trumans'] if v[128]]
si=next(i for i,(m,v) in enumerate(members) if m['sequence_id']==folder.name)
wi=next(i for i,(_,s) in enumerate(members[si][1][128]) if s==300)
sample,identity=data.sample(128,'trumans',sequence_index=si,start_index=wi,scene_bundle=folder)
points=np.load(folder/'visible_frame_points_scenemi_yup.npy');owners=np.load(folder/'visible_frame_owner.npy');ids=np.load(folder/'source_frame_ids.npy');ix=int(np.searchsorted(ids,300))
assert ix>0 and (owners[:ix]==0).any()
assert all(np.isfinite(v).all() for v in sample.values())
# Contract must refuse a partial manifest as a full training input.
try:
 with tempfile.TemporaryDirectory(prefix='scenemi_partial_manifest_') as tmp:
  manifest=Path(tmp)/'manifest.jsonl';manifest.write_text(json.dumps(dict(split='validation',sequence_id=folder.name,path=str(folder.resolve())))+'\n')
  OfflineSceneMIData('validation',skeleton_profile='trumans_male_v2',rich_source='native20_faceout_oct07',trumans_scene_manifest=manifest)
 raise AssertionError('Partial manifest accepted')
except ValueError as e:assert 'missing' in str(e)
# Changing future static/dynamic surfaces cannot change an already computed BPS.
from experiments.offline_camera_retrain_v1.scene_visibility_v2 import temporal_bps
static=[p[o==0] for p,o in zip(points[ix:ix+128],owners[ix:ix+128])];dynamic=[p[(o>0)&(o<100)] for p,o in zip(points[ix:ix+128],owners[ix:ix+128])]
cam=np.load(folder/'camera_position_scenemi_yup.npy')[ix:ix+128];rot=np.load(folder/'camera_rotation_scenemi_yup.npy')[ix:ix+128]
from experiments.offline_camera_retrain_v1.data import ANCHORS
history=points[:ix][owners[:ix]==0]
a=temporal_bps(static,dynamic,cam,rot,ANCHORS,initial_static=history)
changed_static=static[:64]+[p+100 for p in static[64:]];changed_dynamic=dynamic[:64]+[p+100 for p in dynamic[64:]]
b=temporal_bps(changed_static,changed_dynamic,cam,rot,ANCHORS,initial_static=history)
assert np.array_equal(a[:64],b[:64])
# Real forward/backward, CPU only. No optimizer or checkpoint changes.
batch=collate([sample]);torch.manual_seed(2026);model=OfflineSceneMI();prediction=model(torch.randn_like(batch['motion']),torch.tensor([500]),batch);loss=(prediction-batch['motion']).square().mean();loss.backward()
assert torch.isfinite(prediction).all() and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
train=OfflineSceneMIData('train',skeleton_profile='trumans_male_v2',rich_source='native20_faceout_oct07',window_sampling='duration')
plan=duration_sampling_plan(train,.5)
for group in ['trumans','camera_wearer','interactee','rich']:
 for length in [64,128,192]:
  ex,ident=train.sample(length,group);assert ex['motion'].shape==(length,201)
assert covered_frames([0,1,2],64)==66
report=dict(status='passed',identity=identity,recording_frames=len(ids),history_static_observations=int((owners[:ix]==0).sum()),future_invariance=True,partial_manifest_rejected=True,finite_forward_backward=True,duration_sampling_plan=plan)
(HERE/'runs/training_contract_audit_oct07/temporal_v3_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
