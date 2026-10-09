"""Audit known camera mounts and neck compensation on all fixed cached v2 motions."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.head_constraint import project_head_orientation
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,rotation_from_6d,IDENTITY_6D
BASE=Path(__file__).parent/'runs';FIXED=BASE/'turn_generalization_oct09';OUT=BASE/'turn_balance_oct09'
def angles(r):return torch.rad2deg(torch.acos(((r.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)))
def main():
 torch.set_num_threads(2);c=json.loads((BASE/'turn_repair_oct08/orientation_v2/config.json').read_text());d=NativeBodyData('validation',seed=20261009,skeleton_profile=c['skeleton_profile'],rich_source=c['rich_source'],trumans_scene_manifest=c['trumans_scene_manifest'],trumans_window_protocol=c['trumans_window_protocol'],contact_root=c['rich_contact_root'],temporal_scene_manifest=c['temporal_scene_manifest']);windows=json.loads((FIXED/'manifest.json').read_text())['selected'];rows=[]
 for index,w in enumerate(windows):
  sample,identity=d.sample(128,w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);b=collate([sample]);cam=rotation_from_6d(b['camera'][...,3:]-b['camera'].new_tensor(IDENTITY_6D));truth=global_rotations(b['motion'])[:,:,15];residual=angles(truth.transpose(-1,-2)@cam);enabled=w['group']!='camera_wearer';valid=torch.full((1,128),enabled,dtype=torch.bool)
  for replicate in range(2):
   with np.load(FIXED/f'motions/{index:04d}_{replicate}.npz') as z:m=torch.tensor(z['orientation_v2'][None])
   p=project_head_orientation(m,b['camera'],valid);before=forward_kinematics(m,b['rest']);after=forward_kinematics(p,b['rest']);delta=float((before-after).abs().max());assert delta==0
   rawlocal=rotation_from_6d(m[...,93:99]);newlocal=rotation_from_6d(p[...,93:99]);correction=angles(rawlocal.transpose(-1,-2)@newlocal)
   rows.append({'index':index,'replicate':replicate,'group':w['group'],'sequence_id':identity['sequence_id'],'enabled':enabled,'camera_gt_head_identity_mount_error_mean_deg':float(residual.mean()),'camera_gt_head_identity_mount_error_max_deg':float(residual.max()),'gt_head_to_neck_local_p95_deg':float(torch.quantile(angles(rotation_from_6d(b['motion'][...,93:99])),.95)),'head_to_neck_local_p95_before_deg':float(torch.quantile(angles(rawlocal),.95)),'head_to_neck_local_p95_after_deg':float(torch.quantile(angles(newlocal),.95)),'projection_rotation_change_mean_deg':float(correction.mean()),'all_joint_fk_max_change_m':delta})
  if index%24==0:print('audited',index,flush=True)
 summary={}
 for group in sorted({r['group'] for r in rows}):
  rs=[r for r in rows if r['group']==group];keys=[k for k,v in rs[0].items() if isinstance(v,float)]
  summary[group]={k:float(np.mean([r[k] for r in rs])) for k in keys};summary[group]['max_camera_gt_head_identity_mount_error_deg']=max(r['camera_gt_head_identity_mount_error_max_deg'] for r in rs)
 (OUT/'head_constraint_audit.json').write_text(json.dumps({'scope':'144 windows, two cached v2 seeds; evaluation-only GT calibration residual, never passed to projection','summary':summary,'rows':rows},indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
