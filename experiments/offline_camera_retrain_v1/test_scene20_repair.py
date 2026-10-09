"""Independent ray geometry, causal input and raw-missing-pose regression checks."""
import hashlib,json,pickle,tempfile,unittest
from pathlib import Path
import numpy as np
import torch,trimesh
from scipy.spatial import cKDTree
from experiments.offline_camera_retrain_v1.data import HERE,PROJECTS,ANCHORS,collate
from experiments.offline_camera_retrain_v1.causal_scene import temporal_scene_inputs
from experiments.offline_camera_retrain_v1.scene_visibility_v2 import render_scene,ray_grid,wearer_faces,static_ray_scene
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.prepare_rich_pseudo_ego import RICH_GENDER,_read_transform,RICH_TO_SCENEMI,_load_smplx_model,_smplx_forward
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI,ddim_sample
ROOT=HERE/'data/native20_temporal_scenes_v4'
RICH=ROOT/'validation/LectureHall_003_wipingchairs1'
EGO=ROOT/'validation/recording_20210907_S02_S01_01_camera_wearer_1551-2811'
TRU=HERE/'data/trumans_temporal_training_v3/train/2023-01-16@23-06-03'

class Scene20RepairTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2)
 def test_rich_raw_released_ply_firsthit_reference(self):
  m=json.loads((RICH/'metadata.json').read_text());q=np.load(RICH/'source_frame_ids.npy');intr=np.load(RICH/'camera_intrinsics_128x96.npy');cam=np.load(RICH/'camera_position_scenemi_yup.npy');rot=np.load(RICH/'camera_rotation_scenemi_yup.npy')
  scale,r,t=_read_transform(Path(m['world_transform']));tf=RICH_TO_SCENEMI@r.T;offset=t@RICH_TO_SCENEMI.T;scan=trimesh.load(m['scene_mesh'],force='mesh',process=False);meshes=[(scan.vertices*scale@tf.T+offset,scan.faces,0)]
  for a,s in enumerate(m['subject_ids']):
   folder=PROJECTS/f'RICH/extracted/val_body/{RICH.name}/{int(q[0]):05d}';ply=trimesh.load(folder/f'{s}.ply',force='mesh',process=False)
   model=_load_smplx_model(RICH_GENDER[int(s)]);faces,_=wearer_faces(ply.faces,model.lbs_weights.numpy()) if a==0 else (ply.faces,{})
   meshes.append((ply.vertices*scale@tf.T+offset,faces,100+a))
  reference=render_scene(meshes,cam[0],rot[0],intrinsics=intr[0]);z=np.load(RICH/'selected_raycast_audit.npz');expected=z['depth_0'];both=np.isfinite(reference['depth'])&np.isfinite(expected)
  self.assertGreater(both.sum(),0);self.assertLess(np.quantile(np.abs(reference['depth'][both]-expected[both]),.999),.001)
  self.assertLess(np.mean(reference['owner']!=z['owner_0']),.001)
 def test_current_bps_matches_scalar_world_oracle_and_no_future(self):
  for folder in [RICH,EGO]:
   with self.subTest(folder=folder.name):
    ids=np.load(folder/'source_frame_ids.npy');q=ids[20:84];a,cam,r,identity=temporal_scene_inputs(folder,q);p=np.load(folder/'visible_frame_points_scenemi_yup.npy');o=np.load(folder/'visible_frame_owner.npy')
    for frame in [0,11,63]:
     metadata=json.loads((folder/'metadata.json').read_text());memory=json.loads((Path(metadata['memory_bundle'])/'metadata.json').read_text());sources=[folder.parent/name for name in memory['source_sequences']];sources.sort(key=lambda f:np.load(f/'source_frame_ids.npy')[0]);chunks=[];timestamps=[]
     for source in sources:
      sid=np.load(source/'source_frame_ids.npy');sp=np.load(source/'visible_frame_points_scenemi_yup.npy');so=np.load(source/'visible_frame_owner.npy');available=sid<=q[frame];mask=so[available]==0;chunks.append(sp[available][mask]);timestamps.append(np.broadcast_to(sid[available,None],mask.shape)[mask])
     known=np.concatenate(chunks);time=np.concatenate(timestamps);known=known[np.argsort(time,kind='stable')];_,unique=np.unique(np.floor(known/.025).astype(np.int32),axis=0,return_index=True);known=known[np.sort(unique)]
     probe=ANCHORS@r[frame].T+cam[frame];dist,ix=cKDTree(known).query(probe);delta=(known[ix]-probe)@r[frame];oracle=delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1)/2
     self.assertLess(np.linalg.norm(a['bps'][frame]-oracle,axis=1).max(),1e-5)
    with tempfile.TemporaryDirectory(prefix='scenemi_scene20_verify_') as temp:
     tmp=Path(temp)
     for filename in ['source_frame_ids.npy','camera_position_scenemi_yup.npy','camera_rotation_scenemi_yup.npy']:np.save(tmp/filename,np.load(folder/filename))
     np.save(tmp/'visible_frame_points_scenemi_yup.npy',p);np.save(tmp/'visible_frame_owner.npy',o);metadata=json.loads((folder/'metadata.json').read_text());metadata.pop('memory_protocol',None);metadata.pop('memory_bundle',None);(tmp/'metadata.json').write_text(json.dumps(metadata))
     prior=[source for source in sources if source!=folder and np.load(source/'source_frame_ids.npy')[0]<ids[0]];baseline,*_=temporal_scene_inputs(tmp,q,history_bundles=prior)
     changed=p.copy();changed[ids>q[15]]+=100;np.save(tmp/'visible_frame_points_scenemi_yup.npy',changed);b,*_=temporal_scene_inputs(tmp,q,history_bundles=prior)
     np.testing.assert_array_equal(baseline['occupancy'],b['occupancy']);np.testing.assert_array_equal(baseline['bps'][:16],b['bps'][:16]);np.testing.assert_array_equal(baseline['bps_valid'][:16],b['bps_valid'][:16])
 def test_nan_raw_neighbors_are_masked_without_filling(self):
  root=PROJECTS/'TRUMANS';row=next(r for r in map(json.loads,(root/'processed/scene_expert_v1/clips.jsonl').read_text().splitlines()) if r['clip_name']==TRU.name);flags=np.load(root/'object_flag.npy',mmap_mode='r');mat=np.load(root/'object_mat.npy',mmap_mode='r');lo,hi=row['global_start'],row['global_end_exclusive'];valid=np.ones(hi-lo,bool)
  for i in np.flatnonzero((flags[lo:hi]>=0).any(0)):
   slots=flags[lo:hi,i];valid &= (slots>=0)&np.isfinite(mat[np.arange(lo,hi),np.maximum(slots,0)]).all((1,2))
  q=np.load(TRU/'source_frame_ids.npy');expected=valid[np.floor(q).astype(int)]&valid[np.ceil(q).astype(int)];actual=np.load(TRU/'observation_valid.npy');np.testing.assert_array_equal(actual,expected);self.assertTrue((~actual).any())
  p=np.load(TRU/'visible_frame_points_scenemi_yup.npy');o=np.load(TRU/'visible_frame_owner.npy');self.assertTrue((o[~actual]==-1).all());self.assertTrue((p[~actual]==0).all())
  bad=int(np.flatnonzero(~actual)[0]);
  with self.assertRaisesRegex(ValueError,'invalid observation'):temporal_scene_inputs(TRU,q[bad:bad+64])
 def test_overlapping_windows_have_identical_absolute_frame_features(self):
  for folder in [RICH,EGO,TRU]:
   ids=np.load(folder/'source_frame_ids.npy');a,*_=temporal_scene_inputs(folder,ids[:64]);b,*_=temporal_scene_inputs(folder,ids[20:84]);np.testing.assert_array_equal(a['bps'][20:],b['bps'][:44]);np.testing.assert_array_equal(a['bps_valid'][20:],b['bps_valid'][:44])
 def test_old_5hz_union_cannot_change_exact20_inputs(self):
  from unittest.mock import patch
  data=NativeBodyData('validation');members=[(m,v) for m,v in data.rich_members if v[64]];si=next(i for i,(m,v) in enumerate(members) if m['sequence_id']==RICH.name)
  a,_=data.sample(64,'rich',sequence_index=si,start_index=20,scene_bundle=RICH);original=np.load
  def changed(path,*args,**kwargs):
   if str(path).endswith('visible_static_points_scenemi_yup.npy'):return np.ones((64,3),np.float32)*123
   return original(path,*args,**kwargs)
  with patch('numpy.load',side_effect=changed):b,_=data.sample(64,'rich',sequence_index=si,start_index=20,scene_bundle=RICH)
  for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
 def test_exclusion_must_match_actual_missing_source(self):
  with tempfile.TemporaryDirectory(prefix='scenemi_exclusion_verify_') as tmp:
   p=Path(tmp)/'manifest.jsonl';p.write_text(json.dumps(dict(split='validation',sequence_id='2023-02-18@22-17-40',status='excluded',reason='missing_source_object_tracks'))+'\n')
   with self.assertRaisesRegex(ValueError,'Cannot exclude'):NativeBodyData('validation',trumans_scene_manifest=p)
 def test_native_training_reads_exact20_scene_and_fk(self):
  data=NativeBodyData('validation',contact_root=HERE/'data/rich_contact_native20_v1');members=[(m,v) for m,v in data.rich_members if v[64]];si=next(i for i,(m,v) in enumerate(members) if m['sequence_id']==RICH.name)
  a,i=data.sample(64,'rich',sequence_index=si,start_index=20,scene_bundle=RICH);self.assertEqual(i['scene_observation_fps'],20);self.assertEqual(i['scene_bundle'],str(RICH));self.assertGreater(i['scene_history_points'],0)
  members=[(m,v) for m,v in data.base.groups['camera_wearer'] if v[64]];si=next(i for i,(m,v) in enumerate(members) if m['sequence_id']==EGO.name);b,j=data.sample(64,'camera_wearer',sequence_index=si,start_index=0,scene_bundle=EGO)
  batch=collate([a,b]);model=OfflineSceneMI(64,body_conditioning=True,contact_prediction=True);pred=model(torch.randn_like(batch['motion']),torch.tensor([50,50]),batch)
  from experiments.offline_camera_retrain_v1.contact_supervision import contact_loss
  loss=(pred-batch['motion']).square().mean()+contact_loss(model.contact_logits(pred,batch),batch);loss.backward();self.assertTrue(torch.isfinite(loss));model.eval();gen=ddim_sample(model,batch,64,steps=2);self.assertTrue(torch.isfinite(gen).all())
  from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
  np.testing.assert_allclose(forward_kinematics(batch['motion'],batch['rest']).detach(),batch['joints']*2,atol=1e-5)
if __name__=='__main__':unittest.main()
