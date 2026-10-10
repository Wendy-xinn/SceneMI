import json
from pathlib import Path
import torch,numpy as np
from experiments.offline_camera_retrain_v1.turn_foot_coordination import coordinate_turn
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
H=Path('experiments/offline_camera_retrain_v1');O=H/'runs/turn_foot_coordination_oct10'
torch.set_num_threads(4)
c=torch.load(H/'runs/native_dynamic_scene20_contact_55k_oct07/last.pt',map_location='cpu',weights_only=False)['config']
d=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'])
rows=json.loads((O/'rows.json').read_text());tests=[]
for r in rows[:2]:
 w=r['window'];s,_=d.sample(128,w['group'],sequence_index=w['sequence_index'],start_index=w['start_index'])
 with np.load(H/f"runs/state_relative_spline_oct10/demo/{r['index']}.npz") as a:m=torch.from_numpy(a['official55k_motion'])
 with np.load(O/f"{r['index']}.npz") as a:saved=torch.from_numpy(a['camera_soft'])
 s['rest']=torch.as_tensor(s['rest']);s['camera']=torch.as_tensor(s['camera'])
 original_j=forward_kinematics(m[None],s['rest'][None])[0];original_r=global_rotations(m[None])[0]
 identity,_=coordinate_turn(m,s['camera'],s['rest'],mode='identity');assert torch.max(abs(global_rotations(identity[None])[0]-original_r))<2e-5
 out,audit=coordinate_turn(m,s['camera'],s['rest'],mode='camera_soft');assert torch.max(abs(out-saved))<2e-5
 j=forward_kinematics(out[None],s['rest'][None])[0];rot=global_rotations(out[None])[0]
 foot=float((j[:,[7,8,10,11]]-original_j[:,[7,8,10,11]]).norm(dim=-1).max());head=float((rot[:,15]-original_r[:,15]).abs().max())
 assert foot<2e-5 and head<2e-5 and torch.max(abs(rot[0]-original_r[0]))<2e-5 and torch.isfinite(out).all()
 assert audit['generated_head_position_max_change_cm']<=2.0001
 tests.append({'index':r['index'],'identity_roundtrip_pass':True,'saved_reproduction_within_2e_minus5':True,'own_initial_pose_preserved':True,'world_foot_error_m':foot,'head_global_rotation_matrix_max_error':head,'head_displacement_budget_pass':True})
(O/'tests.json').write_text(json.dumps(tests,indent=2));print(json.dumps(tests,indent=2))
