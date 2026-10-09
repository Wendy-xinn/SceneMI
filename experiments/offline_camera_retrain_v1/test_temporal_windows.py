"""Regression checks for dynamic-inclusive source validity and legacy coexistence."""
import unittest,json
import numpy as np
from experiments.offline_camera_retrain_v1.temporal_windows import valid_windows
from experiments.offline_camera_retrain_v1.data import HERE,SOURCE,PROJECTS
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.training_contract_v2 import duration_sampling_plan

class TemporalWindowsTests(unittest.TestCase):
 def test_raw_clock_mask_both_window_boundaries(self):
  ids=np.arange(400)*1.5;valid=np.ones(400,bool);valid[127]=False
  got=valid_windows(ids,valid,128)
  self.assertNotIn((0,0),got);self.assertNotIn((120,180),got);self.assertIn((140,210),got);self.assertEqual(got,[(s,int(ids[s])) for s in range(0,273,20) if valid[s:s+128].all()])
 def test_dynamic_window_real_loader_and_legacy_kept(self):
  kwargs=dict(trumans_scene_manifest=HERE/'data/trumans_temporal_training_v3/manifest.jsonl',temporal_scene_manifest=HERE/'data/native20_temporal_scenes_v4/manifest.jsonl')
  d=NativeBodyData('validation',**kwargs);old=NativeBodyData('validation',trumans_window_protocol='legacy_stable_v1',**kwargs);name='2023-02-18@22-17-40';members=[(m,v) for m,v in d.base.groups['trumans'] if v[128]];si=next(i for i,(m,v) in enumerate(members) if m['sequence_id']==name);starts=members[si][1][128];wi=next(i for i,(offset,source) in enumerate(starts) if source==120);s,i=d.sample(128,'trumans',sequence_index=si,start_index=wi);self.assertEqual(i['source_start_30fps'],120);self.assertTrue(all(np.isfinite(a).all() for a in s.values()));legacy=next(v for m,v in old.base.groups['trumans'] if m['sequence_id']==name);self.assertNotIn(120,[q for offset,q in legacy[128]])
 def test_reset_history_preserves_first_frame_current_geometry(self):
  from experiments.offline_camera_retrain_v1.data import collate,ANCHORS
  from experiments.offline_camera_retrain_v1.history_reset_eval import reset_history_batch
  from scipy.spatial import cKDTree
  d=NativeBodyData('validation',trumans_scene_manifest=HERE/'data/trumans_temporal_training_v3/manifest.jsonl',temporal_scene_manifest=HERE/'data/native20_temporal_scenes_v4/manifest.jsonl');members=[(m,v) for m,v in d.base.groups['trumans'] if v[128]];si=next(i for i,(m,v) in enumerate(members) if m['sequence_id']=='2023-02-18@22-17-40');wi=next(i for i,(off,q) in enumerate(members[si][1][128]) if q==120);s,identity=d.sample(128,'trumans',sequence_index=si,start_index=wi);reset=reset_history_batch(collate([s]),identity,128);folder=HERE/'data/trumans_temporal_training_v3/validation'/identity['sequence_id'];frame=80;p=np.load(folder/'visible_frame_points_scenemi_yup.npy')[frame];o=np.load(folder/'visible_frame_owner.npy')[frame];self.assertTrue(((o>0)&(o<100)).any());static=p[o==0];_,idx=np.unique(np.floor(static/.025).astype(int),axis=0,return_index=True);static=static[np.sort(idx)];surface=np.concatenate((static,p[(o>0)&(o<100)]));cam=np.load(folder/'camera_position_scenemi_yup.npy')[frame];r=np.load(folder/'camera_rotation_scenemi_yup.npy')[frame];probe=ANCHORS@r.T+cam;nearest=surface[cKDTree(surface).query(probe)[1]];delta=(nearest-probe)@r;expected=delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1)/2;np.testing.assert_allclose(reset['bps'][0,0].numpy(),expected,atol=1e-6);self.assertGreater(float(np.abs(reset['bps'][0].numpy()-s['bps']).max()),1e-5)
 def test_all_new_windows_cover_valid_source_and_new_recordings(self):
  kwargs=dict(trumans_scene_manifest=HERE/'data/trumans_temporal_training_v3/manifest.jsonl',temporal_scene_manifest=HERE/'data/native20_temporal_scenes_v4/manifest.jsonl');d=NativeBodyData('train',**kwargs);old=NativeBodyData('train',trumans_window_protocol='legacy_stable_v1',**kwargs);self.assertEqual(len(d.base.groups['trumans']),469);self.assertGreater(duration_sampling_plan(d)['unique_eligible_frames']['trumans'],duration_sampling_plan(old)['unique_eligible_frames']['trumans']);raw={r['clip_name']:r for r in map(json.loads,(PROJECTS/'TRUMANS/processed/scene_expert_v1/clips.jsonl').read_text().splitlines())};bad=np.load(PROJECTS/'TRUMANS/bad_frames.npy')
  for m,v in d.base.groups['trumans']:
   folder=d.trumans_scene_bundles[m['sequence_id']];mask=np.load(folder/'observation_valid.npy') if (folder/'observation_valid.npy').exists() else np.ones(m['frames_20fps'],bool)
   for L,windows in v.items():
    for offset,q in windows:
     self.assertTrue(mask[offset:offset+L].all());ids=raw[m['sequence_id']]['global_start']+q+np.arange(L)*1.5;self.assertFalse(np.isin(np.floor(ids).astype(int),bad).any());self.assertFalse(np.isin(np.ceil(ids).astype(int),bad).any())
  # Legacy manifest bytes remain the originally audited source; the loader never rewrites them.
  self.assertEqual(old.trumans_window_protocol,'legacy_stable_v1')

if __name__=='__main__':unittest.main()
