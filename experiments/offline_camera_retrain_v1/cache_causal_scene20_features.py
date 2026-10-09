"""One recording/observer memory, first surface per world25mm voxel.
Per-frame BPS is independent of target-window boundaries; dynamic stays current.
"""
import argparse,hashlib,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.spatial import cKDTree
from experiments.offline_camera_retrain_v1.data import HERE,ANCHORS
FILES=('source_frame_ids.npy','visible_frame_points_scenemi_yup.npy','visible_frame_owner.npy','camera_position_scenemi_yup.npy','camera_rotation_scenemi_yup.npy')
PROTOCOL='causal20-first-world-voxel25mm-current-dynamic-v1'

def cache_group(job):
 key,rows,memory_dir=job;memory_dir=Path(memory_dir);memory_dir.mkdir(parents=True,exist_ok=True);observed=[];times=[];source_hash=hashlib.sha256()
 rows=sorted(rows,key=lambda r:float(np.load(Path(r['path'])/'source_frame_ids.npy')[0]))
 for row in rows:
  folder=Path(row['path']);p=np.load(folder/FILES[1],mmap_mode='r');o=np.load(folder/FILES[2],mmap_mode='r');ids=np.load(folder/FILES[0]);known=o==0;observed.append(np.asarray(p[known]));times.append(np.broadcast_to(ids[:,None],o.shape)[known])
  for f in FILES:source_hash.update(hashlib.sha256((folder/f).read_bytes()).digest())
 points=np.concatenate(observed);ts=np.concatenate(times)
 if len(ts)>1 and (np.diff(ts)<0).any():order=np.argsort(ts,kind='stable');points=points[order];ts=ts[order]
 vox=np.ascontiguousarray(np.floor(points/.025).astype(np.int32));_,ix=np.unique(vox.view(np.dtype((np.void,12))).reshape(-1),return_index=True);ix.sort();points=points[ix];ts=ts[ix]
 np.save(memory_dir/'static_points.npy',points);np.save(memory_dir/'first_observed_source_frames.npy',ts)
 meta=dict(protocol=PROTOCOL,key=key,voxel_size_m=.025,static_points=len(points),source_bundle_sha256=source_hash.hexdigest(),source_sequences=[r['sequence_id'] for r in rows]);meta['output_sha256']={f:hashlib.sha256((memory_dir/f).read_bytes()).hexdigest() for f in ['static_points.npy','first_observed_source_frames.npy']};(memory_dir/'metadata.json').write_text(json.dumps(meta,indent=2))
 tree=None;end=-1;features=0
 for row in rows:
  folder=Path(row['path']);ids=np.load(folder/FILES[0]);p=np.load(folder/FILES[1],mmap_mode='r');o=np.load(folder/FILES[2],mmap_mode='r');camera=np.load(folder/FILES[3]);rot=np.load(folder/FILES[4]);bps=np.zeros((len(ids),67,3),np.float32);valid=np.zeros((len(ids),67),bool)
  for frame,q in enumerate(ids):
   new_end=int(np.searchsorted(ts,q,side='right'))
   if new_end!=end:tree=cKDTree(points[:new_end]) if new_end else None;end=new_end
   probe=ANCHORS@rot[frame].T+camera[frame];nearest=np.zeros_like(probe);best=np.full(67,np.inf)
   if tree is not None:dist,idx=tree.query(probe);nearest=points[idx].copy();best=dist
   dyn=p[frame][(o[frame]>0)&(o[frame]<100)]
   if len(dyn):
    dist,idx=cKDTree(dyn).query(probe);choose=dist<best;nearest[choose]=dyn[idx[choose]];best[choose]=dist[choose]
   good=np.isfinite(best);delta=(nearest-probe)@rot[frame];delta/=np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1);bps[frame,good]=delta[good]/2;valid[frame]=good
  np.save(folder/'causal_bps_20.npy',bps);np.save(folder/'causal_bps_valid_20.npy',valid);m=json.loads((folder/'metadata.json').read_text());m.update(memory_bundle=str(memory_dir.resolve()),memory_protocol=PROTOCOL);m.setdefault('output_sha256',{}).update({f:hashlib.sha256((folder/f).read_bytes()).hexdigest() for f in ['causal_bps_20.npy','causal_bps_valid_20.npy']});tmp=folder/'memory_metadata.tmp';tmp.write_text(json.dumps(m,indent=2));tmp.replace(folder/'metadata.json');features+=len(ids)
 return dict(key=key,sequences=len(rows),frames=features,static_points=len(points))

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--resume',action='store_true');a=p.parse_args();groups={};start=time.monotonic();root=HERE/'data/causal_scene20_memory_v1';root.mkdir(parents=True,exist_ok=True)
 for manifest in [HERE/'data/native20_temporal_scenes_v4/manifest.jsonl',HERE/'data/trumans_temporal_training_v3/manifest.jsonl']:
  for row in map(json.loads,manifest.read_text().splitlines()):
   if row.get('status')=='excluded':continue
   m=json.loads((Path(row['path'])/'metadata.json').read_text());key=f"{row['split']}/{m['dataset']}/{m['recording']}_{m['subject_ids'][0]}" if m['dataset']=='egobody' else f"{row['split']}/{m['dataset']}/{row['sequence_id']}"
   groups.setdefault(key,[]).append(row)
 jobs=[];done=[]
 for k,rows in groups.items():
  memory=root/k;cached=a.resume and (memory/'metadata.json').exists()
  if cached:
   m=json.loads((memory/'metadata.json').read_text());digest=hashlib.sha256()
   for row in sorted(rows,key=lambda r:float(np.load(Path(r['path'])/'source_frame_ids.npy')[0])):
    folder=Path(row['path']);meta=json.loads((folder/'metadata.json').read_text())
    for f in FILES:digest.update(hashlib.sha256((folder/f).read_bytes()).digest())
    cached &= meta.get('memory_protocol')==PROTOCOL and all((folder/f).exists() for f in ['causal_bps_20.npy','causal_bps_valid_20.npy'])
   cached &= m.get('protocol')==PROTOCOL and m.get('source_bundle_sha256')==digest.hexdigest()
  if cached:done.append(dict(key=k,sequences=len(rows),frames=sum(r['frames'] for r in rows),static_points=m['static_points']))
  else:jobs.append((k,rows,str(memory)))
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for f in as_completed([pool.submit(cache_group,j) for j in jobs]):
   done.append(f.result());info=dict(status='running',completed=len(done),planned=len(groups),frames=sum(r['frames'] for r in done),elapsed_s=time.monotonic()-start);(root/'status.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
 info['status']='completed';info['groups']=done;(root/'status.json').write_text(json.dumps(info,indent=2))
if __name__=='__main__':main()
