"""Check degeneracy of raw residual root rotation columns before normalization."""
import json
from pathlib import Path
import numpy as np
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D

def main():
 root=Path(__file__).parent/'runs/bounded_head_adaptation_oct10'
 rows=[json.loads(line) for line in (root/'rows.jsonl').read_text().splitlines()]
 results={}
 for label in ('joint_all','joint_staged','rotation_only','mild_only'):
  first=[];second=[]
  for row in rows:
   if row['condition']!='clean' or row['length']!=128:continue
   with np.load(root/row['motion_file']) as saved:x=saved[label][:,3:9]+np.asarray(IDENTITY_6D)
   a=x[:,:3];b=x[:,3:];na=np.linalg.norm(a,axis=-1);unit=a/np.maximum(na[:,None],1e-8)
   orth=b-np.sum(unit*b,axis=-1,keepdims=True)*unit
   first.extend(na);second.extend(np.linalg.norm(orth,axis=-1))
  first=np.asarray(first);second=np.asarray(second)
  results[label]={'frames':len(first),'first_norm_p01':float(np.quantile(first,.01)),'orthogonal_second_norm_p01':float(np.quantile(second,.01)),'either_norm_below_point1_fraction':float(np.mean((first<.1)|(second<.1)))}
 out={'scope':'clean128 development predictions, root only; does not prove all joints or all inputs stable','results':results}
 (root/'root_6d_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
