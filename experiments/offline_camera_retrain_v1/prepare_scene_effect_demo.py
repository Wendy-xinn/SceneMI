"""Selected paired held-out generations; selection is illustrative, not benchmark."""
import json
from pathlib import Path
import numpy as np
import torch,trimesh
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
from experiments.offline_camera_retrain_v1.prepare_trumans_dynamic_v2 import load_objects
from experiments.offline_sequence_v1.data_loader import anchor_rotation
ROOT=Path('/home/wenxin/projects/SceneMI');RUN=ROOT/'experiments/offline_camera_retrain_v1/runs/native_dynamic_scene20_contact_55k_oct07';OUT=RUN/'scene_effect_demo'
def main():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);config=json.loads((RUN/'config.json').read_text());d=NativeBodyData('validation',seed=777,skeleton_profile=config['skeleton_profile'],rich_source=config['rich_source'],trumans_scene_manifest=config['trumans_scene_manifest'],trumans_window_protocol=config['trumans_window_protocol'],contact_root=config['rich_contact_root'],temporal_scene_manifest=config['temporal_scene_manifest'])
 rows=[json.loads(s) for s in (RUN/'full_validation/standard_rows.jsonl').read_text().splitlines()];clips={x['clip_name']:x for x in map(json.loads,(ROOT.parent/'TRUMANS/processed/scene_expert_v1/clips.jsonl').read_text().splitlines())};cases=[]
 for index in [850,906,607,3487]:
  row=rows[index];w=row['window'];ident=row['identity'];sample,actual=d.sample(w['length'],w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);assert actual==ident
  folder=Path(ident['scene_bundle']);q=ident['source_start_30fps']+np.arange(w['length'])*1.5;ix=np.searchsorted(np.load(folder/'source_frame_ids.npy'),q);cam=np.load(folder/'camera_position_scenemi_yup.npy')[ix];rot=np.load(folder/'camera_rotation_scenemi_yup.npy')[ix];a=anchor_rotation(rot)
  saved=np.load(RUN/'full_validation'/row['motion_file']);arrays={'camera':cam,'rotation':rot,'visible_points':np.load(folder/'visible_frame_points_scenemi_yup.npy')[ix],'visible_owners':np.load(folder/'visible_frame_owner.npy')[ix]};err=0.
  for k,motion in [('gt',sample['motion']),('on',saved['scene_on_motion']),('off',saved['scene_off_motion'])]:
   v,f,j=decode_native_mesh(motion,ident['native_body'],d.model(ident['native_body']['model'],ident['native_body']['gender']));expected=saved[{'gt':'truth','on':'scene_on','off':'scene_off'}[k]];err=max(err,float(np.linalg.norm(j-expected,axis=-1).max()));arrays[k+'_vertices']=(v@a+cam[0]).astype(np.float32);arrays[k+'_joints']=(j@a+cam[0]).astype(np.float32);arrays['faces']=f
  assert err<1e-4
  memory=Path(ident['memory_bundle']);mp=np.load(memory/'static_points.npy',mmap_mode='r');mt=np.load(memory/'first_observed_source_frames.npy',mmap_mode='r');prefix=mp[mt<=q[-1]];arrays['context']=prefix[::max(1,int(np.ceil(len(prefix)/90000)))];objects=[]
  if w['group']=='trumans':
   clip=clips[ident['sequence_id']]
   for i,(name,path,r,t,source,valid,bad) in enumerate(load_objects(ROOT.parent/'TRUMANS',clip,q)):
    assert valid.all();mesh=trimesh.load(path,force='mesh',process=False);arrays[f'obj_{i}_v']=np.asarray(mesh.vertices,np.float32);arrays[f'obj_{i}_f']=mesh.faces;arrays[f'obj_{i}_r']=r;arrays[f'obj_{i}_t']=t;objects.append({'name':name,'max_displacement_m':float(np.linalg.norm(t-t[:1],axis=-1).max())})
  np.savez_compressed(OUT/f'{index}.npz',**arrays);cases.append({'index':index,'row':row,'objects':objects,'mesh_fk_max_error_m':err});print('exported',index,'mesh error',err,flush=True)
 (OUT/'manifest.json').write_text(json.dumps({'cases':cases,'selection':'Chosen for low scene-on world error, moving GT and large paired benefit; selected examples, not unbiased estimates','context':'Static observed prefix up to window end displayed as reference; not all available at frame0; current points separately shown'},indent=2))
if __name__=='__main__':main()
