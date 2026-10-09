"""Full coverage/hash/time/mask/projection gate; emits explicit failure list.
No status file or export process count is treated as evidence of correctness.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from experiments.offline_camera_retrain_v1.data import HERE,SOURCE,LENGTHS
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.training_contract_v2 import duration_sampling_plan


def main():
 p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');a=p.parse_args();root=HERE/'data/native20_temporal_scenes_v4';tru=HERE/'data/trumans_temporal_training_v3';out=HERE/'runs/training_contract_audit_oct07/scene20_full_verification.json';problems=[];summary={};expected=[];quarantines=json.loads((tru/'exclusions.json').read_text()) if (tru/'exclusions.json').exists() else [];excluded={(r['split'],r['sequence_id']) for r in quarantines};start=time.monotonic()
 for split in ('train','validation'):
  rs='train' if split=='train' else 'val';expected.extend(('rich',split,m.parent.name,np.load(m.parent/'source_frame_ids.npy')) for m in sorted((HERE/f'data/scene_visibility_v2_oct05/rich_{rs}_smpl_native20_faceout_oct07').glob('*/metadata.json')))
  for dataset in ['egobody','trumans']:
   catalog=NativeBodyData(split)
   rows=(list(map(json.loads,(SOURCE/f'trumans_sequences/{split}.jsonl').read_text().splitlines())) if dataset=='trumans' else list(map(json.loads,(SOURCE/f'{dataset}_sequences/{split}.jsonl').read_text().splitlines())))
   for m in rows:
    if dataset=='trumans' and (split,m['sequence_id']) in excluded:continue
    if dataset=='trumans':
     # Only recordings with target windows are required by strict manifest.
     d=None
     if m['frames_20fps']<64:continue
    q=m.get('source_first_frame',0)+np.arange(m['frames_20fps'])*1.5;expected.append((dataset,split,m['sequence_id'],q))
 for dataset,split,name,q in expected:
  folder=(tru if dataset=='trumans' else root)/split/name;mp=folder/'metadata.json'
  if not mp.exists():problems.append(dict(dataset=dataset,split=split,sequence=name,error='missing completed metadata'));continue
  try:
   meta=json.loads(mp.read_text());ids=np.load(folder/'source_frame_ids.npy');points=np.load(folder/'visible_frame_points_scenemi_yup.npy',mmap_mode='r');owners=np.load(folder/'visible_frame_owner.npy',mmap_mode='r');cam=np.load(folder/'camera_position_scenemi_yup.npy');rot=np.load(folder/'camera_rotation_scenemi_yup.npy')
   assert meta.get('split',split)==split and meta['output_fps']==20 and meta['occlusion_version']==2 and meta['sequence_id']==name
   assert np.array_equal(ids,q) and (len(ids)<2 or np.allclose(np.diff(ids),1.5,atol=1e-6))
   assert points.shape==(len(q),512,3) and owners.shape==(len(q),512) and cam.shape==(len(q),3) and rot.shape==(len(q),3,3)
   assert np.isfinite(points).all() and np.isfinite(cam).all() and np.isfinite(rot).all()
   assert np.allclose(rot.transpose(0,2,1)@rot,np.eye(3),atol=1e-4) and np.allclose(np.linalg.det(rot),1,atol=1e-4)
   assert meta.get('memory_protocol')=='causal20-first-world-voxel25mm-current-dynamic-v1'
   memory=Path(meta['memory_bundle']);mpoints=np.load(memory/'static_points.npy',mmap_mode='r');mtimes=np.load(memory/'first_observed_source_frames.npy');assert mpoints.shape==(len(mtimes),3) and np.isfinite(mpoints).all() and np.isfinite(mtimes).all() and (np.diff(mtimes)>=0).all()
   bps=np.load(folder/'causal_bps_20.npy',mmap_mode='r');bv=np.load(folder/'causal_bps_valid_20.npy',mmap_mode='r');assert bps.shape==(len(q),67,3) and bv.shape==(len(q),67) and bv.dtype==bool and np.isfinite(bps).all() and (np.linalg.norm(bps,axis=2)<=.500001).all()
   input_hashes=meta.get('input_sha256',{}) if isinstance(meta.get('input_sha256'),dict) else {}
   for f,sha in {**input_hashes,**meta.get('output_sha256',{})}.items():assert hashlib.sha256((folder/f).read_bytes()).hexdigest()==sha,f'{f} hash mismatch'
   mm=json.loads((memory/'metadata.json').read_text());assert mm['protocol']==meta['memory_protocol']
   for f,sha in mm['output_sha256'].items():assert hashlib.sha256((memory/f).read_bytes()).hexdigest()==sha,f'memory {f} hash mismatch'
   assert np.isin(owners,np.arange(-1,100)).all() # human first hits are never scene points
   known=owners>=0;delta=points-cam[:,None];local=np.einsum('tpi,tij->tpj',delta,rot)
   if dataset!='trumans':
    assert meta['protocol']=='native20-scene-v4-raw30-full-body-first-hit' and not meta['pointcloud_interpolation']
    intr=np.load(folder/'camera_intrinsics_128x96.npy');assert intr.shape==(len(q),4) and np.isfinite(intr).all() and (intr[:,:2]>0).all()
    projected=np.stack((local[:,:,0]/np.maximum(local[:,:,2],1e-10)*intr[:,None,0]+intr[:,None,2],-local[:,:,1]/np.maximum(local[:,:,2],1e-10)*intr[:,None,1]+intr[:,None,3]),2)
    assert ((projected[known]>=-1e-3)&(projected[known]<[128.001,96.001])).all()
    assert np.allclose(projected[known]%1,.5,atol=.003) # actual ray pixel centers, not arbitrary scan samples
    for f,sha in meta['output_sha256'].items():assert hashlib.sha256((folder/f).read_bytes()).hexdigest()==sha,f'{f} hash mismatch'
   assert (local[:,:,2][known]>=.05-1e-4).all() and (local[:,:,2][known]<=4+1e-4).all()
   if (folder/'observation_valid.npy').exists():
    valid=np.load(folder/'observation_valid.npy');assert valid.shape==(len(q),) and valid.dtype==bool;assert (owners[~valid]==-1).all() and (points[~valid]==0).all();assert int((~valid).sum())==meta.get('invalid_observation_count',0)
   bucket=summary.setdefault(dataset+'/'+split,dict(sequences=0,frames=0,observed_points=0,invalid_observations=0));bucket['sequences']+=1;bucket['frames']+=len(q);bucket['observed_points']+=int(known.sum());bucket['invalid_observations']+=meta.get('invalid_observation_count',0)
  except Exception as e:problems.append(dict(dataset=dataset,split=split,sequence=name,error=repr(e)))
 report=dict(excluded_sequences=quarantines,status='failed' if problems else 'passed',expected_sequences=len(expected),checked_sequences=sum(v['sequences'] for v in summary.values()),summary=summary,problems=problems,elapsed_s=time.monotonic()-start,rate=20,pointcloud_interpolation=False)
 if not problems:
  counts={}
  for split in ('train','validation'):
   d=NativeBodyData(split,trumans_scene_manifest=tru/'manifest.jsonl',temporal_scene_manifest=root/'manifest.jsonl',contact_root=HERE/'data/rich_contact_native20_v1',window_sampling='duration');counts[split]=d.summary()
   for m,v in d.base.groups['trumans']:
    folder=d.trumans_scene_bundles[m['sequence_id']]
    if (folder/'observation_valid.npy').exists():
     mask=np.load(folder/'observation_valid.npy')
     assert all(mask[s:s+L].all() for L in LENGTHS for s,source in v[L])
   if split=='train':report['duration_sampling_plan']=duration_sampling_plan(d,.5)
  report['training_window_summary']=counts;report['invalid_target_windows_excluded']=True
 out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['problems','training_window_summary']}));print('problems',len(problems))
 if a.require_complete and problems:raise SystemExit(1)
if __name__=='__main__':main()
