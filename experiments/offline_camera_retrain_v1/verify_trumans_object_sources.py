"""Audit full TRUMANS object coverage, raw clocks and documented Euler convention."""
import json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
base=Path(__file__).resolve().parent;root=base.parents[2]/'TRUMANS';flags=np.load(root/'object_flag.npy',mmap_mode='r');names=np.load(root/'object_list.npy');fids=np.load(root/'frame_id.npy',mmap_mode='r');rows={r['clip_name']:r for r in map(json.loads,(root/'processed/scene_expert_v1/clips.jsonl').read_text().splitlines())};report={'sequences':0,'raw_nonchair_objects':0,'problems':[],'max_analytic_rotation_error':0.,'official_source':'https://github.com/jnnan/trumans_utils#trumans-dataset','note':'Raw Object_all Euler xyz is documented separately from Object_chairs flag/matrix/variant assets. Chair representations are not interchangeable; chair reconstruction always uses the latter.'};t=time.monotonic()
for mp in sorted((base/'data/trumans_temporal_training_v3').glob('*/*/metadata.json')):
 meta=json.loads(mp.read_text());row=rows[meta['sequence_id']];lo,hi=row['global_start'],row['global_end_exclusive'];tracks=np.load(root/row['object_tracks']['path'],allow_pickle=True).item();report['sequences']+=1
 try:
  assert np.array_equal(fids[lo:hi],np.arange(row['num_frames'])),'raw time indices mismatch'
  chairs={str(names[i]) for i in np.flatnonzero((flags[lo:hi]>=0).any(0))};replaced={n.split('(')[0] for n in chairs};expected=chairs| (set(tracks)-replaced);assert expected=={o['name'] for o in meta['objects']},'object coverage mismatch'
  for obj in meta['objects']:
   n=obj['name'];assert Path(obj['mesh']).exists()
   if n in chairs:assert obj['source']=='object_mat + variant mesh';continue
   assert obj['source']=='raw Euler xyz radians';track=tracks[n];e=np.asarray(track['rotation']);p=np.asarray(track['location']);assert e.shape==p.shape==(row['num_frames'],3);assert np.isfinite(e).all() and np.isfinite(p).all();report['raw_nonchair_objects']+=1
   for j in (0,len(e)//2,len(e)-1):
    x,y,z=e[j];cx,sx=np.cos(x),np.sin(x);cy,sy=np.cos(y),np.sin(y);cz,sz=np.cos(z),np.sin(z);rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]);ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]);rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]);err=np.max(np.abs(rz@ry@rx-Rotation.from_euler('xyz',e[j]).as_matrix()));report['max_analytic_rotation_error']=max(report['max_analytic_rotation_error'],float(err));assert err<1e-12
 except Exception as ex:report['problems'].append({'sequence':row['clip_name'],'error':str(ex)})
report['elapsed_s']=time.monotonic()-t;report['status']='passed' if report['sequences']==sum(r.get('status')!='excluded' for r in map(json.loads,(base/'data/trumans_temporal_training_v3/manifest.jsonl').read_text().splitlines())) and not report['problems'] else 'failed';out=base/'runs/training_contract_audit_oct07/trumans_object_sources_verification.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report));raise SystemExit(report['status']!='passed')
