"""Native body decode gates and a small shared-model overfit trial, not a 55k."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate,HERE
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,rotation_from_6d,supervised_losses
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI,cosine_alphas,ddim_sample
from experiments.offline_camera_retrain_v1.control import fixed_control_mask

def main():
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=300);a=p.parse_args();torch.set_num_threads(4)
 out=HERE/'runs/native_body_coexistence_trial_oct07';out.mkdir(exist_ok=True)
 data=NativeBodyData('validation',seed=777);samples=[];identities=[];gates=[]
 for group in ['trumans','camera_wearer','interactee','rich']:
  sample,identity=data.sample(64,group);info=identity['native_body'];model=data.model(info['model'],info['gender']);m=torch.from_numpy(sample['motion']);local=rotation_from_6d(m[:,3:135].reshape(-1,22,6));scale=info['scale']
  aa=torch.from_numpy(Rotation.from_matrix(local.numpy().reshape(-1,3,3)).as_rotvec().astype(np.float32).reshape(64,22,3))
  body=aa[:,1:]
  extras={}
  if info['model']=='smpl':body=torch.cat((body,torch.zeros(64,2,3)),1)
  else:extras=dict(left_hand_pose=torch.zeros(64,45),right_hand_pose=torch.zeros(64,45),jaw_pose=torch.zeros(64,3),leye_pose=torch.zeros(64,3),reye_pose=torch.zeros(64,3),expression=torch.zeros(64,10))
  with torch.no_grad():
   decoded=model(global_orient=aa[:,0],body_pose=body.reshape(64,-1),betas=torch.from_numpy(sample['body_betas'][None]).expand(64,-1),**extras)
  joints=decoded.joints[:,:22].numpy()*scale+sample['motion'][:,:3,None].transpose(0,2,1)*2
  fk=forward_kinematics(collate([sample])['motion'],collate([sample])['rest'])[0].numpy();error=float(np.linalg.norm(joints-fk,axis=-1).max());target_error=float(np.linalg.norm(fk-sample['joints']*2,axis=-1).max())
  assert error<1e-4 and target_error<1e-4,(group,error,target_error)
  # Save actual native meshes and predicted pose source for visual review.
  vertices=decoded.vertices.numpy()*scale+sample['motion'][:,:3,None].transpose(0,2,1)*2
  np.savez_compressed(out/f'{group}_native_gt.npz',vertices=vertices,faces=model.faces,joints=fk,motion=sample['motion'])
  gates.append(dict(group=group,model=info['model'],max_mesh_fk_joint_error_m=error,max_fk_target_error_m=target_error,archived_joint_difference_mean_m=identity['archived_joint_difference_mean_m']))
  samples.append(sample);identities.append(identity);print('native decode gate',gates[-1],flush=True)
 # Mixed SMPL and SMPL-X in ONE batch, one optimizer and one shared U-Net.
 device='cuda' if torch.cuda.is_available() else 'cpu';torch.manual_seed(2026)
 model=OfflineSceneMI(latent_dim=128,body_conditioning=True).to(device)
 batch={k:v.to(device) for k,v in collate(samples).items()};clean=batch['motion'];mask=fixed_control_mask(4,64,'head',device);alpha=cosine_alphas().to(device);t=torch.full((4,),50,device=device,dtype=torch.long);torch.manual_seed(777);noise=torch.randn_like(clean);noisy=alpha[t,None,None].sqrt()*clean+(1-alpha[t,None,None]).sqrt()*noise
 opt=torch.optim.AdamW(model.parameters(),lr=2e-4)
 def assess():
  model.eval()
  with torch.no_grad():pred=model(noisy,t,batch,control_mask=mask);fk=forward_kinematics(pred,batch['rest']);e=(fk-batch['joints']*2).norm(dim=-1).mean((1,2))*100
  return e.cpu().tolist()
 before=assess();start=time.monotonic();log=[]
 model.train()
 for step in range(a.steps):
  opt.zero_grad();pred=model(noisy,t,batch,control_mask=mask);losses=supervised_losses(pred,clean,batch);losses['total'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
  if step%50==0:print('trial step',step,'loss',float(losses['total']),flush=True)
 after=assess();assert np.mean(after)<np.mean(before),(before,after)
 from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
 with torch.no_grad():final_prediction=model(noisy,t,batch,control_mask=mask).cpu().numpy()
 for i,group in enumerate(['trumans','camera_wearer','interactee','rich']):
  v,f,j=decode_native_mesh(final_prediction[i],identities[i]['native_body'],data.model(identities[i]['native_body']['model'],identities[i]['native_body']['gender']))
  np.savez_compressed(out/f'{group}_low_noise_fit.npz',vertices=v,faces=f,joints=j,motion=final_prediction[i])
 config=dict(latent_dim=128,dim_mults=[1,2,4],body_conditioning=True,body_protocol='native SMPL/SMPL-X public22',steps=a.steps,split='validation',trial='fixed low-noise four-example overfit, not held-out generalization',shape_condition='known subject shape/rest for this diagnostic; deployment needs estimated/calibrated body shape')
 torch.save(dict(model=model.state_dict(),config=config),out/'trial.pt')
 report=dict(status='passed',native_decode_gates=gates,identities=identities,shared_batch_body_types=batch['body_type'].cpu().tolist(),before_low_noise_fk_mpjpe_cm=before,after_low_noise_fk_mpjpe_cm=after,steps=a.steps,elapsed_s=time.monotonic()-start,config=config)
 (out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
