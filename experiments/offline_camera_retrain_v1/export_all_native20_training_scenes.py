"""Resume complete direct20 RICH/EgoBody exports with a strict manifest."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from experiments.offline_camera_retrain_v1.data import HERE,SOURCE
from experiments.offline_camera_retrain_v1.export_native20_training_scenes import FILES,PROTOCOL

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'data/native20_temporal_scenes_v4');p.add_argument('--workers',type=int,default=3);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);jobs=[];rows={};failed=[];start=time.monotonic()
 for split in ('train','validation'):
  rsplit='train' if split=='train' else 'val'
  jobs.extend(('rich',split,p.parent.name) for p in sorted((HERE/f'data/scene_visibility_v2_oct05/rich_{rsplit}_smpl_native20_faceout_oct07').glob('*/metadata.json')))
  jobs.extend(('egobody',split,r['sequence_id']) for r in map(json.loads,(SOURCE/f'egobody_sequences/{split}.jsonl').read_text().splitlines()))
 def export(job):
  dataset,split,name=job;folder=a.output/split/name;path=folder/'metadata.json';meta=json.loads(path.read_text()) if path.exists() else {}
  if meta.get('protocol')!=PROTOCOL or not meta.get('projection_intrinsics') or not (folder/'camera_intrinsics_128x96.npy').is_file() or not all((folder/f).is_file() for f in FILES):
   folder.mkdir(parents=True,exist_ok=True)
   with (folder/'export.log').open('w') as log:subprocess.run([sys.executable,str(HERE/'export_native20_training_scenes.py'),'--dataset',dataset,'--split',split,'--sequence',name,'--output',str(a.output)],check=True,stdout=log,stderr=subprocess.STDOUT)
   meta=json.loads(path.read_text())
  return (split,name),dict(dataset=dataset,split=split,sequence_id=name,path=str(folder.resolve()),frames=meta['frames'],protocol=PROTOCOL)
 def save(status='running'):
  info=dict(status=status,completed=len(rows),planned=len(jobs),failed=len(failed),failures=failed,elapsed_s=time.monotonic()-start,workers=a.workers);(a.output/'status.json').write_text(json.dumps(info,indent=2));tmp=a.output/'manifest.tmp';tmp.write_text(''.join(json.dumps(r)+'\n' for k,r in sorted(rows.items())));tmp.replace(a.output/'manifest.jsonl');print(json.dumps({k:v for k,v in info.items() if k!='failures'}),flush=True)
 save()
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(export,j):j for j in jobs}
  for f in as_completed(futures):
   try:k,row=f.result();rows[k]=row
   except Exception as e:failed.append(dict(job=futures[f],error=str(e)))
   save()
 save('failed' if failed else 'completed')
 if failed:raise RuntimeError('Incomplete scene export; refuse training')
if __name__=='__main__':main()
