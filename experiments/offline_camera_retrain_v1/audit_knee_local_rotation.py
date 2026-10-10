"""Native-template X-axis swing/twist diagnostic, not a hard joint limit."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
OUT=Path(__file__).parent/'runs/soft_coupled_contact_oct10'

def measures(motion):
 x=motion[:,3:135].reshape(-1,22,6)[:,(4,5)]+np.array([1,0,0,0,1,0])
 a=x[...,:3];b=x[...,3:];n1=np.linalg.norm(a,axis=-1);a=a/np.maximum(n1[...,None],1e-8)
 b=b-(a*b).sum(-1,keepdims=True)*a;n2=np.linalg.norm(b,axis=-1);b=b/np.maximum(n2[...,None],1e-8)
 r=np.stack((a,b,np.cross(a,b)),axis=-1);q=Rotation.from_matrix(r.reshape(-1,3,3)).as_quat().reshape(*x.shape[:2],4)
 twist=(2*np.arctan2(q[...,0],q[...,3])+np.pi)%(2*np.pi)-np.pi
 swing=2*np.arccos(np.clip(np.sqrt(q[...,0]**2+q[...,3]**2),0,1))
 return np.rad2deg(twist).ravel(),np.rad2deg(swing).ravel(),((n1<.1)|(n2<.1)).ravel()

def main():
 data=defaultdict(list);sequence=defaultdict(lambda:defaultdict(list))
 rows=[json.loads(x) for x in (OUT/'rows.jsonl').read_text().splitlines()]
 for row in rows:
  if row['condition']!='clean' or row['length']!=128:continue
  with np.load(OUT/row['motion_file']) as f:
   for label,key in (('GT','truth_motion'),('baseline','baseline'),('coupled','coupled'),('coupled_contact','coupled_contact')):
    twist,swing,bad=measures(f[key]);data[label].append((twist,swing,bad));sequence[label][row['identity']['sequence_id']].append((twist,swing))
 result={}
 for label,parts in data.items():
  twist=np.concatenate([p[0] for p in parts]);swing=np.concatenate([p[1] for p in parts]);bad=np.concatenate([p[2] for p in parts])
  result[label]={'knee_frames':len(twist),'twist_p05_p50_p95_deg':np.quantile(twist,[.05,.5,.95]).tolist(),'swing_p50_p95_deg':np.quantile(swing,[.5,.95]).tolist(),'twist_below_minus10_fraction':float((twist< -10).mean()),'swing_above30_fraction':float((swing>30).mean()),'raw_rotation_norm_below_point1_fraction':float(bad.mean())}
 report={'scope':'clean128, both native knee joints4/5, two seeds; X-axis is a template proxy, not an anatomical hard limit; compare against GT before choosing a regularizer','results':result}
 (OUT/'knee_rotation_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
