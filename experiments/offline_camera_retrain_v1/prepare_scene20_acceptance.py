"""Export actual native_v1 loader samples and matching world-space inspection geometry."""
import json,pickle,time
from pathlib import Path
import numpy as np
import torch,trimesh,smplx
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.data import HERE,PROJECTS,ANCHORS
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
from experiments.offline_camera_retrain_v1.export_native20_training_scenes import setup_rich,setup_egobody,forward_chunk
from experiments.offline_camera_retrain_v1.prepare_trumans_dynamic_v2 import load_objects
from experiments.offline_camera_retrain_v1.scene_visibility_v2 import static_ray_scene,render_scene,wearer_faces
from experiments.offline_camera_retrain_v1.export_pose_sequences import slerp_rotvec,linear_sample
from experiments.offline_sequence_v1.data_loader import anchor_rotation
import hashlib
OUT=HERE/'runs/scene20_visual_acceptance_oct07'

def choose(d,group,name_hint=None,diagnostic=False):
 members=[(m,v) for m,v in (d.rich_members if group=='rich' else d.base.groups[group]) if v[128]]
 candidates=[(i,m,v) for i,(m,v) in enumerate(members) if name_hint is None or name_hint in m['sequence_id']]
 if not candidates:raise ValueError((group,name_hint))
 si,m,v=candidates[0];name=m['sequence_id'];folder=d.temporal_scenes[name] if group!='trumans' else HERE/'data/trumans_temporal_training_v3'/d.split/name
 ids=np.load(folder/'source_frame_ids.npy');cam=np.load(folder/'camera_position_scenemi_yup.npy');owners=np.load(folder/'visible_frame_owner.npy',mmap_mode='r')
 # Choose a legal audited window with visible objects / movement, with existing history.
 starts=np.array(v[128]);original_starts=list(v[128])
 if diagnostic:
  offsets=np.arange(0,len(ids)-127,20);starts=np.stack((offsets,ids[offsets]),1);v[128]=[(int(i),int(q)) for i,q in starts]
 q=starts
 # RICH starts index native20 frames; other groups store raw30 absolute source IDs.
 ix=starts.astype(int) if group=='rich' else np.searchsorted(ids,starts[:,1])
 dyn=((owners>0)&(owners<100)).sum(1);sums=np.r_[0,np.cumsum(dyn)]
 scores=np.linalg.norm(cam[ix+127]-cam[ix],axis=-1)+(sums[ix+128]-sums[ix])/128/100
 scores+=np.minimum(ix,100)/1000
 if group=='trumans':
  row=next(r for r in map(json.loads,(PROJECTS/'TRUMANS/processed/scene_expert_v1/clips.jsonl').read_text().splitlines()) if r['clip_name']==name)
  activity=np.zeros(len(ids))
  for owner,(n,path,r,p,source,valid,bad) in enumerate(load_objects(PROJECTS/'TRUMANS',row,ids),1):
   velocity=np.r_[0,np.linalg.norm(np.diff(p,axis=0),axis=-1)]
   angular=np.r_[0,(Rotation.from_matrix(r[:-1]).inv()*Rotation.from_matrix(r[1:])).magnitude()]*.3
   activity+=(velocity+angular)*np.minimum((owners==owner).sum(1)/10,1)
  integral=np.r_[0,np.cumsum(activity)];scores+=15*(integral[ix+128]-integral[ix])
 if group=='interactee':
  matching=np.flatnonzero(starts[:,1]==2061)
  if len(matching):scores[:]=-1;scores[matching[0]]=1
 k=int(np.argmax(scores))
 try:sample,identity=d.sample(128,group,sequence_index=si,start_index=k)
 finally:
  if diagnostic:v[128]=original_starts
 if diagnostic:assert identity['source_start_30fps'] not in [q for _,q in original_starts]
 identity['visualization_scope']='source_dynamic_diagnostic_excluded_by_old_stability_filter' if diagnostic else 'actual_eligible_training_window'
 assert identity['sequence_id']==name
 return sample,identity

def export_case(key,d,group,hint,diagnostic=False):
 start=time.monotonic();sample,identity=choose(d,group,hint,diagnostic=diagnostic);folder=Path(identity['scene_bundle']);meta=json.loads((folder/'metadata.json').read_text());ids=np.load(folder/'source_frame_ids.npy');q=identity['source_start_30fps']+np.arange(128)*1.5;ix=np.searchsorted(ids,q);assert np.array_equal(ids[ix],q)
 cam=np.load(folder/'camera_position_scenemi_yup.npy')[ix];rot=np.load(folder/'camera_rotation_scenemi_yup.npy')[ix];a=anchor_rotation(rot)
 v,f,j=decode_native_mesh(sample['motion'],identity['native_body'],d.model(identity['native_body']['model'],identity['native_body']['gender']));v=v@a+cam[0];j=j@a+cam[0];expected=sample['joints']*2@a+cam[0];err=float(np.linalg.norm(j-expected,axis=-1).max());assert err<1e-4
 target=OUT/key;target.mkdir(parents=True,exist_ok=True);np.savez_compressed(target/'training_input.npz',**sample)
 arrays={'training_vertices':v.astype(np.float32),'training_faces':f,'joints':j.astype(np.float32),'source_frame_ids':q,'camera':cam,'rotation':rot,'visible_points':np.load(folder/'visible_frame_points_scenemi_yup.npy')[ix],'visible_owners':np.load(folder/'visible_frame_owner.npy')[ix]}
 objects=[]
 if group=='trumans':
  row=next(r for r in map(json.loads,(PROJECTS/'TRUMANS/processed/scene_expert_v1/clips.jsonl').read_text().splitlines()) if r['clip_name']==identity['sequence_id']);root=PROJECTS/'TRUMANS'
  with (root/row['smplx_global']).open('rb') as file:params=pickle.load(file)
  model=smplx.create(str(PROJECTS/'SceneMI/body_models'),model_type='smplx',gender='male',num_betas=20,use_pca=False,batch_size=128)
  values={}
  for n,dims in [('global_orient',3),('body_pose',63),('left_hand_pose',45),('right_hand_pose',45),('jaw_pose',3)]:values[n]=torch.from_numpy(slerp_rotvec(np.asarray(params.get(n,np.zeros((row['num_frames'],dims))),np.float32).reshape(-1,dims//3,3),q).reshape(128,dims))
  values['transl']=torch.from_numpy(linear_sample(np.asarray(params['transl']),q).astype(np.float32));beta=np.asarray(params.get('betas',np.zeros(20)),np.float32).reshape(-1);beta=beta if len(beta) else np.zeros(20,np.float32);values['betas']=torch.from_numpy(np.tile(beta,(128,1)));values['expression']=torch.from_numpy(linear_sample(np.asarray(params.get('expression',np.zeros((row['num_frames'],10))),np.float32),q).astype(np.float32))
  with torch.inference_mode():body=model(**values,return_verts=True).vertices.numpy()
  people=[(body,model.faces,model)];scene=trimesh.load(root/row['scene']['mesh'],force='mesh',process=False)
  static=static_ray_scene(scene.vertices,scene.faces)
  for n,path,r,p,source,valid,bad in load_objects(root,row,q):
   assert valid.all();mesh=trimesh.load(path,force='mesh',process=False);objects.append((n,mesh,r,p,static_ray_scene(mesh.vertices,mesh.faces)))
  intr=np.tile([64/np.tan(np.deg2rad(66.56/2)),48/np.tan(np.deg2rad(40.49/2)),64,48],(128,1))
 else:
  setup=setup_rich if group=='rich' else setup_egobody
  allq,rawpeople,transform,static,_,_,_,rawmeta=setup(d.split,identity['sequence_id'],hashlib.sha256());assert np.array_equal(allq[ix],q);people=[]
  for model,params in rawpeople:
   vv=np.concatenate([forward_chunk(model,params,int(ix[k]),int(ix[min(k+63,127)])+1,transform)[0] for k in range(0,128,64)]);people.append((vv,model.faces,model))
  scene=trimesh.load(rawmeta['scene_mesh'],force='mesh',process=False)
  if group=='rich':scale,r,t=transform;scene.vertices=scene.vertices*scale@r.T+t
  intr=np.load(folder/'camera_intrinsics_128x96.npy')[ix]
 arrays['intrinsics']=intr
 arrays['scene_reference']=trimesh.sample.sample_surface(scene,65000,seed=2026)[0].astype(np.float32)
 for i,(vv,ff,model) in enumerate(people):arrays[f'body_{i}_vertices']=vv;arrays[f'body_{i}_faces']=ff
 self_faces,_=wearer_faces(people[0][1],people[0][2].lbs_weights.numpy(),model_type=identity['native_body']['model']);arrays['self_faces']=self_faces
 depths=[];owners=[];max_depth_error=0.;cached_point_owner_disagreements=0
 for t in range(128):
  meshes=[(vv[t],self_faces if i==0 else ff,100+i) for i,(vv,ff,_) in enumerate(people)]
  render=render_scene(meshes,cam[t],rot[t],static_scene=static,rigid_objects=[(cached,r[t],p[t],i+1) for i,(_,mesh,r,p,cached) in enumerate(objects)],intrinsics=intr[t]);depths.append(render['depth']);owners.append(render['owner'])
  good=arrays['visible_owners'][t]>=0;local=(arrays['visible_points'][t][good]-cam[t])@rot[t];fx,fy,cx,cy=intr[t];u=np.rint(fx*local[:,0]/local[:,2]+cx-.5).astype(int);w=np.rint(-fy*local[:,1]/local[:,2]+cy-.5).astype(int);assert ((u>=0)&(u<128)&(w>=0)&(w<96)).all();pred=render['depth'][w,u];assert np.isfinite(pred).all();max_depth_error=max(max_depth_error,float(np.max(np.abs(pred-local[:,2]),initial=0)));cached_point_owner_disagreements+=int((render['owner'][w,u]!=arrays['visible_owners'][t][good]).sum())
 assert max_depth_error<.003,(key,max_depth_error);assert cached_point_owner_disagreements==0,(key,cached_point_owner_disagreements)
 arrays['depth']=np.array(depths);arrays['ray_owner']=np.array(owners)
 for i,(n,mesh,r,p,_) in enumerate(objects):arrays[f'object_{i}_vertices']=np.asarray(mesh.vertices,np.float32);arrays[f'object_{i}_faces']=mesh.faces;arrays[f'object_{i}_rotation']=r;arrays[f'object_{i}_position']=p
 memory=Path(meta['memory_bundle']);arrays['memory_points']=np.load(memory/'static_points.npy');arrays['memory_times']=np.load(memory/'first_observed_source_frames.npy')
 occ=np.argwhere(sample['occupancy']>0);local=np.stack((occ[:,1],occ[:,0],occ[:,2]),1)*[.26666667,.2,.26666667]+np.array([.133333335,.1,.133333335])+[0,-.7,0]-[6.4,2.4,6.4];arrays['occupancy_centers']=(local@a+cam[0]).astype(np.float32)
 anchors=ANCHORS[None]@rot.transpose(0,2,1)+cam[:,None];arrays['bps_anchors']=anchors.astype(np.float32);arrays['bps_endpoints']=(anchors+(sample['bps']*2)@rot.transpose(0,2,1)).astype(np.float32)
 np.savez_compressed(target/'geometry.npz',**arrays)
 report={'key':key,'identity':identity,'frame_count':128,'fps':20,'duration_s':6.4,'source_bundle':str(folder),'native_mesh_fk_max_error_m':err,'cached_visible_point_recast_max_depth_error_m':max_depth_error,'cached_visible_point_recast_owner_disagreements':cached_point_owner_disagreements,'objects':[{'name':n,'source':meta['objects'][i]['source'],'displacement_m':float(np.linalg.norm(p[-1]-p[0])),'rotation_change_degrees':float(np.rad2deg((Rotation.from_matrix(r[0]).inv()*Rotation.from_matrix(r[-1])).magnitude()))} for i,(n,mesh,r,p,_) in enumerate(objects)],'body_count':len(people),'camera_protocol':meta['camera_protocol'],'memory_protocol':meta['memory_protocol'],'history_points_at_start':int((arrays['memory_times']<q[0]).sum()),'elapsed_s':time.monotonic()-start}
 (target/'metadata.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);return report

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);d=NativeBodyData('validation',seed=2026,contact_root=HERE/'data/rich_contact_native20_v1',temporal_scene_manifest=HERE/'data/native20_temporal_scenes_v4/manifest.jsonl',trumans_scene_manifest=HERE/'data/trumans_temporal_training_v3/manifest.jsonl')
 cases=[('rich_contact','rich','LectureHall_020_wipingchairs1'),('rich_motion','rich','Pavallion_018_yoga1'),('trumans_dynamic','trumans','2023-02-18@22-17-40'),('egobody_wearer','camera_wearer','recording_20210907_S02_S01_01'),('egobody_interactee','interactee','recording_20210907_S02_S01_01')]
 cases.append(('trumans_source_dynamic','trumans','2023-02-18@22-17-40'))
 reports=[json.loads((OUT/k/'metadata.json').read_text()) if (OUT/k/'metadata.json').exists() else export_case(k,d,g,h) for k,g,h in cases];(OUT/'manifest.json').write_text(json.dumps({'status':'passed','cases':reports,'note':'Actual training input verification, not generation performance. Viewer visual acceptance remains for user review.'},indent=2))
if __name__=='__main__':main()
