"""Independent lexicographic first-observation oracle plus direct nearest surfaces.
Uses raw per-frame observations, not window sampling or the feature builder.
"""
import hashlib,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.spatial import cKDTree
from experiments.offline_camera_retrain_v1.data import HERE,ANCHORS

def check(job):
 memory,folders=job;memory=Path(memory);raw=[];times=[]
 folders=sorted(map(Path,folders),key=lambda f:np.load(f/'source_frame_ids.npy')[0])
 for folder in folders:
  p=np.load(folder/'visible_frame_points_scenemi_yup.npy',mmap_mode='r');o=np.load(folder/'visible_frame_owner.npy',mmap_mode='r');q=np.load(folder/'source_frame_ids.npy');mask=o==0;raw.append(p[mask]);times.append(np.broadcast_to(q[:,None],o.shape)[mask])
 points=np.concatenate(raw);ts=np.concatenate(times);cells=np.floor(points/.025).astype(np.int32)
 order=np.lexsort((np.arange(len(ts)),ts,cells[:,2],cells[:,1],cells[:,0]));sorted_cells=cells[order];first=np.r_[True,(sorted_cells[1:]!=sorted_cells[:-1]).any(1)] if len(order) else np.zeros(0,bool);expected=order[first]
 stored=np.load(memory/'static_points.npy');st=np.load(memory/'first_observed_source_frames.npy');sc=np.floor(stored/.025).astype(np.int32);so=np.lexsort((sc[:,2],sc[:,1],sc[:,0]))
 assert np.array_equal(points[expected],stored[so]),'First observed world point mismatch';assert np.array_equal(ts[expected],st[so]),'Earliest observation timestamp mismatch'
 maximum=0.;queries=0
 for folder in folders:
  p=np.load(folder/'visible_frame_points_scenemi_yup.npy',mmap_mode='r');o=np.load(folder/'visible_frame_owner.npy',mmap_mode='r');q=np.load(folder/'source_frame_ids.npy');cam=np.load(folder/'camera_position_scenemi_yup.npy');r=np.load(folder/'camera_rotation_scenemi_yup.npy');bps=np.load(folder/'causal_bps_20.npy');bv=np.load(folder/'causal_bps_valid_20.npy')
  for frame in sorted(set([0,len(q)//2,len(q)-1])):
   end=np.searchsorted(st,q[frame],side='right');known=stored[:end];dyn=p[frame][(o[frame]>0)&(o[frame]<100)];surfaces=np.concatenate((known,dyn));oracle=np.zeros((67,3),np.float32)
   if len(surfaces):
    probes=ANCHORS@r[frame].T+cam[frame];idx=cKDTree(surfaces).query(probes)[1];delta=(surfaces[idx]-probes)@r[frame];oracle=delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1)/2
   error=float(np.linalg.norm(oracle-bps[frame],axis=1).max());maximum=max(maximum,error);assert error<1e-5,f'BPS oracle mismatch: {folder.name} {frame}: {error}';assert bv[frame].all()==bool(len(surfaces));queries+=1
 return dict(memory=str(memory),sequences=len(folders),queries=queries,max_bps_error=maximum,static_points=len(stored))

def main():
 groups={};start=time.monotonic()
 for manifest in [HERE/'data/native20_temporal_scenes_v4/manifest.jsonl',HERE/'data/trumans_temporal_training_v3/manifest.jsonl']:
  for row in map(json.loads,manifest.read_text().splitlines()):
   if row.get('status')=='excluded':continue
   folder=Path(row['path']);m=json.loads((folder/'metadata.json').read_text());groups.setdefault(m['memory_bundle'],[]).append(str(folder))
 results=[]
 with ProcessPoolExecutor(max_workers=6) as pool:
  for f in as_completed([pool.submit(check,j) for j in groups.items()]):
   results.append(f.result())
   if len(results)%50==0:print('independent groups',len(results),len(groups),flush=True)
 out=dict(status='passed',groups=len(results),sequences=sum(r['sequences'] for r in results),reference_query_frames=sum(r['queries'] for r in results),maximum_bps_error=max(r['max_bps_error'] for r in results),elapsed_s=time.monotonic()-start,checks='Original observation lexsort oracle: first world-voxel point/time; available static prefix + current observed dynamic nearest-surface recomputation',details=results)
 (HERE/'runs/training_contract_audit_oct07/scene20_causal_features_independent.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='details'}))
if __name__=='__main__':main()
