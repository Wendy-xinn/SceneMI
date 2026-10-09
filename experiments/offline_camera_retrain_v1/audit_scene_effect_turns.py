import json,pathlib,numpy as np,torch
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d,PARENTS
p=pathlib.Path('/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native_dynamic_scene20_contact_55k_oct07');c=json.loads((p/'config.json').read_text());torch.set_num_threads(4)
d=NativeBodyData('validation',seed=777,skeleton_profile=c['skeleton_profile'],rich_source=c['rich_source'],trumans_scene_manifest=c['trumans_scene_manifest'],trumans_window_protocol=c['trumans_window_protocol'],contact_root=c['rich_contact_root'],temporal_scene_manifest=c['temporal_scene_manifest'])
manifest=json.loads((p/'scene_effect_demo/manifest.json').read_text());results=[]
def rotations(m):
 local=rotation_from_6d(torch.tensor(m[:,3:135].reshape(-1,22,6))).numpy();g=[local[:,0]]
 for j in range(1,22):g.append(g[PARENTS[j]]@local[:,j])
 return local,np.stack(g,1)
def angle(r,s):return np.rad2deg(Rotation.from_matrix((r.transpose(0,2,1)@s).copy()).magnitude())
def turn(g,j):
 v=g[:,j,:,2];yaw=np.unwrap(np.arctan2(v[:,0],v[:,2]));return float(np.rad2deg(yaw[-1]-yaw[0])),yaw
for case in manifest['cases']:
 row=case['row'];w=row['window'];s,identity=d.sample(w['length'],w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);original=np.load(p/'full_validation'/row['motion_file']);tl,tg=rotations(s['motion']);r={'index':case['index'],'gt_root_turn_deg':turn(tg,0)[0],'gt_head_turn_deg':turn(tg,15)[0],'variants':{}}
 # Encoder -> decoder roundtrip against native FK global orientations.
 pose,trans,rest,joints,global_r=d._native(identity['sequence_id'],w['group']);folder=pathlib.Path(identity['scene_bundle']);ids=np.load(folder/'source_frame_ids.npy');q=identity['source_start_30fps']+np.arange(w['length'])*1.5;ix=np.searchsorted(ids,q)
 from experiments.offline_sequence_v1.data_loader import anchor_rotation
 a=anchor_rotation(np.load(folder/'camera_rotation_scenemi_yup.npy')[ix]);start=int(round((identity['source_start_30fps']-d.prepared[identity['sequence_id']].get('source_first_frame',0))/1.5))
 expected=np.einsum('ij,tqjk->tqik',a,global_r[start:start+w['length']]);r['gt_rotation_roundtrip_max_abs']=float(np.max(np.abs(expected-tg)));r['anchor_determinant']=float(np.linalg.det(a));assert r['gt_rotation_roundtrip_max_abs']<1e-5 and r['anchor_determinant']>.999
 for k in ['scene_on','scene_off']:
  l,g=rotations(original[k+'_motion']);headerr=angle(tg[:,15],g[:,15]);rooterr=angle(tg[:,0],g[:,0]);jump=angle(l[:-1,1],l[1:,1]);hjump=angle(l[:-1,2],l[1:,2]);rr=np.asarray(original[k+'_motion'][:,3:135]).reshape(-1,22,6)+np.array([1,0,0,0,1,0]);v=rr[:,:,:3];z=rr[:,:,3:];v=v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12);second=z-(z*v).sum(-1,keepdims=True)*v
  r['variants'][k]={'root_turn_deg':turn(g,0)[0],'head_turn_deg':turn(g,15)[0],'head_rotation_error_mean_deg':float(headerr.mean()),'head_rotation_error_max_deg':float(headerr.max()),'pelvis_rotation_error_mean_deg':float(rooterr.mean()),'hip_rotation_step_max_deg':float(max(jump.max(),hjump.max())),'sixd_orthogonal_second_min_norm':float(np.linalg.norm(second,axis=-1).min()),'feet_support_height_proxy_m':row[k]['support_floating_m'],'head_position_error_cm':row[k]['head_cm']}
 results.append(r);print(json.dumps(r),flush=True)
(p/'scene_effect_demo/turn_audit.json').write_text(json.dumps({'cases':results,'yaw_definition':'Unwrapped atan2 projected global rotation +Z; endpoint turn descriptive only, not full trajectory direction proof','head_condition':'Camera pose, soft input; not hard head joint rotation constraint'},indent=2))
