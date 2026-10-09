"""Export paired short-training native meshes against unchanged GT/context."""
import json
from pathlib import Path
import numpy as np,torch
from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
from experiments.offline_camera_retrain_v1.data import anchor_rotation
BASE=Path(__file__).parent/'runs';OLD=BASE/'native_dynamic_scene20_contact_55k_oct07/scene_effect_demo';TRIAL=BASE/'turn_repair_oct08';OUT=TRIAL/'mesh_comparison'
def main():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);original=json.loads((OLD/'manifest.json').read_text());new={x['index']:x for x in json.loads((TRIAL/'orientation_v2/turn_evaluation/metrics.json').read_text())['cases']};control={x['index']:x for x in json.loads((TRIAL/'control/turn_evaluation/metrics.json').read_text())['cases']};cases=[];models={}
 for c in original['cases']:
  index=c['index'];g=dict(np.load(OLD/f'{index}.npz'));body=c['row']['identity']['native_body'];a=anchor_rotation(g['rotation']);modelkey=(body['model'],body['gender']);row=dict(c['row']);maxerr=0.
  for label,trial in [('on','orientation_v2'),('off','control')]:
   with np.load(TRIAL/trial/'turn_evaluation'/f'{index}.npz') as saved:
    motion=saved['scene_on_motion'];reference=saved['scene_on'];truth=saved['truth']
   if modelkey not in models:
    from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
    models[modelkey]=load_model(*modelkey)
   vertices,faces,joints=decode_native_mesh(motion,body,models[modelkey]);maxerr=max(maxerr,float(np.linalg.norm(joints-reference,axis=-1).max()));g[label+'_vertices']=(vertices@a+g['camera'][0]).astype(np.float32);g[label+'_joints']=(joints@a+g['camera'][0]).astype(np.float32)
   from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
   rotations=global_rotations(torch.tensor(motion[None])).numpy()[0];g[label+'_head_forward']=(rotations[:,15,:,2]@a).astype(np.float32)
   assert np.max(np.abs(truth@a+g['camera'][0]-g['gt_joints']))<1e-4
  assert maxerr<1e-4;row['scene_on']=new[index]['variants']['scene_on'];row['scene_off']=control[index]['variants']['scene_on'];on=g['on_joints'];off=g['off_joints'];row['scene_effect_cm']=float(np.linalg.norm(on-off,axis=-1).mean()*100);row['root_relative_scene_effect_cm']=float(np.linalg.norm((on-on[:,:1])-(off-off[:,:1]),axis=-1).mean()*100);row['scene_on']['gt_root_path_length_m']=c['row']['scene_on']['gt_root_path_length_m'];g['old55k_vertices']=np.load(OLD/f'{index}.npz')['on_vertices'];np.savez_compressed(OUT/f'{index}.npz',**g);cases.append({**c,'row':row,'mesh_fk_max_error_m':maxerr});print('exported',index,maxerr,flush=True)
 (OUT/'manifest.json').write_text(json.dumps({'comparison':'orientation_repair','cases':cases,'green':'orientation_v2 1k, scene on','red':'coordination_v1 control 1k, scene on','blue':'unchanged native GT'},indent=2))
if __name__=='__main__':main()
