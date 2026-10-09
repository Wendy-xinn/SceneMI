"""Export complete prepared recordings and an atomic, strict training manifest.

Limited/failed manifests are diagnostic only: train rejects missing coverage.
Existing completed exports resume. Workers use cached rigid-mesh raycasting.
"""
import argparse,json,subprocess,sys,hashlib,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from experiments.offline_camera_retrain_v1.data import SOURCE
HERE=Path(__file__).parent.resolve()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'data/trumans_temporal_training_v3');p.add_argument('--splits',nargs='+',default=['train','validation']);p.add_argument('--limit',type=int,default=0);p.add_argument('--sequence');p.add_argument('--workers',type=int,default=1);a=p.parse_args()
 if a.workers<1:raise ValueError('Positive workers required')
 a.output.mkdir(parents=True,exist_ok=True);manifest=a.output/'manifest.jsonl';existing={};jobs=[]
 if manifest.exists():existing={(r['split'],r['sequence_id']):r for r in map(json.loads,manifest.read_text().splitlines())}
 files=['source_frame_ids.npy','visible_frame_points_scenemi_yup.npy','visible_frame_owner.npy','camera_position_scenemi_yup.npy','camera_rotation_scenemi_yup.npy']
 for split in a.splits:
  members=[(json.loads(line),None) for line in (SOURCE/'trumans_sequences'/f'{split}.jsonl').read_text().splitlines()]
  if a.sequence:members=[(m,v) for m,v in members if m['sequence_id']==a.sequence]
  if not members:raise ValueError('No matching sequences')
  if a.limit:members=members[:a.limit]
  jobs.extend((split,m) for m,v in members)
 exclusions=json.loads((a.output/'exclusions.json').read_text()) if (a.output/'exclusions.json').exists() else []
 for row in exclusions:existing[(row['split'],row['sequence_id'])]=row
 excluded={(r['split'],r['sequence_id']) for r in exclusions};jobs=[j for j in jobs if (j[0],j[1]['sequence_id']) not in excluded]
 start=time.monotonic();failures=[];done=0
 def export(job):
  split,m=job;folder=a.output/split/m['sequence_id'];key=(split,m['sequence_id'])
  if key not in existing or not all((folder/f).exists() for f in files):
   folder.mkdir(parents=True,exist_ok=True)
   with (folder/'export.log').open('w') as log:
    subprocess.run([sys.executable,str(HERE/'prepare_trumans_dynamic_v2.py'),'--training-only','--sequence',m['sequence_id'],'--source-start',str(m.get('source_first_frame',0)),'--frames',str(m['frames_20fps']),'--output',str(a.output/split)],stdout=log,stderr=subprocess.STDOUT,check=True)
  meta=json.loads((folder/'metadata.json').read_text());meta['training_export_protocol']='v3 complete recording; native 20 Hz dynamic objects and first-hit visibility; static recording memory'
  meta['split']=split;meta['input_sha256']={f:hashlib.sha256((folder/f).read_bytes()).hexdigest() for f in files+(['observation_valid.npy'] if (folder/'observation_valid.npy').exists() else [])}
  (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
  return key,dict(split=split,sequence_id=m['sequence_id'],path=str(folder.resolve()),frames=m['frames_20fps'])
 def status():
  info=dict(status='running',completed=done,planned=len(jobs),excluded=len(exclusions),failed=len(failures),elapsed_s=time.monotonic()-start,workers=a.workers,failures=failures)
  (a.output/'status.json').write_text(json.dumps(info,indent=2));return info
 status()
 with ThreadPoolExecutor(max_workers=a.workers) as executor:
  futures={executor.submit(export,j):j for j in jobs}
  for future in as_completed(futures):
   split,m=futures[future]
   try:
    key,row=future.result();existing[key]=row;done+=1
    temp=manifest.with_suffix('.tmp');temp.write_text(''.join(json.dumps(r)+'\n' for r in existing.values()));temp.replace(manifest)
   except Exception as e:failures.append(dict(split=split,sequence_id=m['sequence_id'],error=str(e),log=str(a.output/split/m['sequence_id']/'export.log')))
   print(json.dumps(status()),flush=True)
 info=status();info['status']='failed' if failures else 'completed';(a.output/'status.json').write_text(json.dumps(info,indent=2))
 if failures:raise RuntimeError(f'{len(failures)} scene exports failed; training must reject incomplete manifest')
if __name__=='__main__':main()
