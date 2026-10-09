"""Failure-case paired audit, reproducing original full-validation batch noise."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI,ddim_sample
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.standard_motion_metrics import standard_motion_metrics
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.compare_feet import metrics
RUN=Path(__file__).parent/'runs/native_dynamic_scene20_contact_55k_oct07'
def angle(a,b):return np.rad2deg(Rotation.from_matrix((a.transpose(0,2,1)@b).copy()).magnitude())
def diagnostics(m,truth,batch):
 p=global_rotations(m).cpu().numpy()[0];t=global_rotations(truth).cpu().numpy()[0];result={}
 for j,name in [(0,'pelvis'),(15,'head')]:
  e=angle(t[:,j],p[:,j]);result[name+'_orientation_mean_deg']=float(e.mean());result[name+'_orientation_over90_fraction']=float((e>90).mean())
  def yaw(r):return np.unwrap(np.arctan2(r[:,j,0,2],r[:,j,2,2]))
  py,ty=yaw(p),yaw(t);result[name+'_turn_deg']=float(np.rad2deg(py[-1]-py[0]));result[name+'_gt_turn_deg']=float(np.rad2deg(ty[-1]-ty[0]));dp=np.diff(py,n=1);dt=np.diff(ty,n=1);valid=np.abs(dt)>np.deg2rad(.5);result[name+'_turn_sign_agreement']=float((np.sign(dp[valid])==np.sign(dt[valid])).mean()) if valid.any() else None
 result['leg_orientation_mean_deg']=float(np.mean([angle(t[:,j],p[:,j]).mean() for j in (1,2,4,5,7,8)]))
 result['leg_global_rotation_step_max_deg']=float(max(angle(p[:-1,j],p[1:,j]).max() for j in (1,2,4,5,7,8)))
 result['camera_mount_variation_max_deg']=float(angle((t[:,15].transpose(0,2,1)@camera_matrix(batch))[0:1].repeat(len(t),0),t[:,15].transpose(0,2,1)@camera_matrix(batch)).max())
 values,joints=metrics(m,batch);values.update(standard_motion_metrics(joints[0].cpu().numpy(),batch['joints'][0].cpu().numpy()*2));return {**values,**result}
def camera_matrix(batch):
 c=batch['camera'][0,:,3:].cpu().numpy();return np.stack((c[:,:3],c[:,3:],np.cross(c[:,:3],c[:,3:])),axis=-1)
@torch.inference_mode()
def main():
 a=argparse.ArgumentParser();a.add_argument('--checkpoint',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--saved-original',action='store_true');args=a.parse_args();torch.set_num_threads(4)
 cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False);c=cp['config'];model=None
 if not args.saved_original:
  model=OfflineSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=c.get('body_conditioning',False),contact_prediction=c.get('contact_prediction',False)).cuda().eval();model.load_state_dict(cp['model'])
 d=NativeBodyData('validation',seed=777,skeleton_profile=c['skeleton_profile'],rich_source=c['rich_source'],trumans_scene_manifest=c['trumans_scene_manifest'],trumans_window_protocol=c['trumans_window_protocol'],contact_root=c['rich_contact_root'],temporal_scene_manifest=c['temporal_scene_manifest']);rows=[json.loads(s) for s in (RUN/'full_validation/standard_rows.jsonl').read_text().splitlines()];cases=[];args.output.mkdir(parents=True,exist_ok=True)
 for index in [850,906,607,3487]:
  row=rows[index];batch_rows=[r for r in rows if r['batch_seed']==row['batch_seed']];samples=[]
  for r in batch_rows:
   w=r['window'];s,identity=d.sample(w['length'],w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);assert identity==r['identity'];samples.append(s)
  batch={k:v.cuda() for k,v in collate(samples).items()};pos=row['batch_position'];one={k:v[pos:pos+1] for k,v in batch.items()};L=row['window']['length'];result={'index':index,'identity':row['identity'],'variants':{}};saved={}
  for variant,on in [('scene_on',True),('scene_off',False)]:
   if args.saved_original:
    with np.load(RUN/'full_validation'/row['motion_file']) as f:generated=torch.as_tensor(f[variant+'_motion'][None],device='cuda')
   else:
    full=ddim_sample(model,batch,L,steps=20,seed=row['batch_seed'],control_mask=fixed_control_mask(len(samples),L,'head','cuda'),use_scene=on);generated=full[pos:pos+1]
   result['variants'][variant]=diagnostics(generated,one['motion'],one);saved[variant+'_motion']=generated[0].cpu().numpy();saved[variant]=forward_kinematics(generated,one['rest'])[0].cpu().numpy()
  saved['truth']=one['joints'][0].cpu().numpy()*2;saved['truth_motion']=one['motion'][0].cpu().numpy();np.savez_compressed(args.output/f'{index}.npz',**saved);cases.append(result);print(json.dumps({'index':index,'head_error':result['variants']['scene_on']['head_orientation_mean_deg']}),flush=True)
 # Separate fixed, unbiased small validation set, all groups, same seed for all checkpoints.
 from experiments.offline_camera_retrain_v1.data import GROUPS
 if model is not None:
  fixed=[];d.base.rng=np.random.default_rng(919)
  for gi,g in enumerate(GROUPS):
   for si in range(4):
    sample,identity=d.sample(128,g);b={k:v.cuda() for k,v in collate([sample]).items()};motion=ddim_sample(model,b,128,steps=20,seed=919+gi*100+si,control_mask=fixed_control_mask(1,128,'head','cuda'));fixed.append({'group':g,'identity':identity,'metrics':diagnostics(motion,b['motion'],b)})
 else:fixed=[]
 (args.output/'metrics.json').write_text(json.dumps({'checkpoint':str(args.checkpoint),'step':cp['step'],'original_noise_batch_reproduced':True,'cases':cases,'fixed_validation':fixed},indent=2))
if __name__=='__main__':main()
