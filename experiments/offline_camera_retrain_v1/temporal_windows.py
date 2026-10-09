"""Versioned dynamic-inclusive target windows; legacy audit manifests stay immutable."""
import json
from pathlib import Path
import numpy as np
PROTOCOL='temporal_valid_v1'
LEGACY='legacy_stable_v1'

def valid_windows(ids,frame_valid,length,stride=20):
 ids=np.asarray(ids);frame_valid=np.asarray(frame_valid,bool)
 if ids.ndim!=1 or frame_valid.shape!=ids.shape or not np.allclose(np.diff(ids),1.5):raise ValueError('Invalid native20 clock/mask')
 bad=np.r_[0,np.cumsum(~frame_valid)];starts=np.arange(0,len(ids)-length+1,stride,dtype=int)
 starts=starts[(bad[starts+length]-bad[starts])==0]
 return [(int(s),int(ids[s]) if float(ids[s]).is_integer() else float(ids[s])) for s in starts]

def rebuild_trumans(split,bundles,excluded,source,here,projects,lengths):
 prepared=[json.loads(l) for l in (source/'trumans_sequences'/f'{split}.jsonl').read_text().splitlines()];audit={(r['dataset'],r['sequence_id']):r for r in map(json.loads,(source/'intervals.jsonl').read_text().splitlines())};poses={r['sequence_id']:r for r in map(json.loads,(here/'data/poses'/f'{split}.jsonl').read_text().splitlines())};raw={r['clip_name']:r for r in map(json.loads,(projects/'TRUMANS/processed/scene_expert_v1/clips.jsonl').read_text().splitlines())};bad_global=np.load(projects/'TRUMANS/bad_frames.npy');groups=[];records=[]
 for m in prepared:
  name=m['sequence_id']
  if name in excluded:continue
  original=audit[('trumans',name)];r=raw[name]
  if m['dataset']!='trumans' or m['split']!=split or original['split']!=split or original['scene_family']!=m['scene_family'] or original['camera_source']!=m['camera_source']:raise ValueError('Raw/prepared scope mismatch')
  if not r['object_tracks']['path']:raise ValueError(f'{name}: missing raw object tracks')
  if name not in bundles:raise ValueError(f'{name}: dynamic-inclusive scene coverage missing')
  folder=bundles[name];meta=json.loads((folder/'metadata.json').read_text());ids=np.load(folder/'source_frame_ids.npy');n=m['frames_20fps'];expected=m.get('source_first_frame',0)+np.arange(n)*1.5
  if meta.get('split')!=split or meta['sequence_id']!=name or meta['output_fps']!=20 or not np.array_equal(ids,expected) or ids[-1]>r['num_frames']-1:raise ValueError('Incomplete source/scene clock')
  valid=np.load(folder/'observation_valid.npy') if (folder/'observation_valid.npy').exists() else np.ones(n,bool)
  if valid.shape!=(n,) or valid.dtype!=bool:raise ValueError('Invalid object observation mask')
  pose=Path(poses[name]['folder']);p=np.load(pose/'pose_axis_angle_world.npy',mmap_mode='r');t=np.load(pose/'translation_world.npy',mmap_mode='r');j=np.load(Path(m['folder'])/'joints_world.npy',mmap_mode='r');cam=np.load(folder/'camera_position_scenemi_yup.npy',mmap_mode='r');rot=np.load(folder/'camera_rotation_scenemi_yup.npy',mmap_mode='r')
  for a in [p,t,j,cam,rot]:
   if len(a)!=n:raise ValueError('Target fit coverage mismatch')
   valid=valid & np.isfinite(a.reshape(n,-1)).all(1)
  # Released bad_frames are global raw30 IDs. Fractional targets require both source neighbors.
  raw_ids=r['global_start']+ids;raw_bad=np.isin(np.floor(raw_ids).astype(np.int64),bad_global)|np.isin(np.ceil(raw_ids).astype(np.int64),bad_global);valid &= ~raw_bad
  windows={L:valid_windows(ids,valid,L) for L in lengths}
  record={'sequence_id':name,'split':split,'protocol':PROTOCOL,'frames':n,'invalid_target_frames':int((~valid).sum()),'raw_bad_query_frames':int(raw_bad.sum()),'windows':{str(L):v for L,v in windows.items()},'legacy_starts':{str(L):len(original['valid_starts_30fps'][str(L)]) for L in lengths}}
  records.append(record)
  if any(windows.values()):groups.append((m,windows))
 return groups,records
