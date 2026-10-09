"""Output-gradient diagnostics on training data; no optimization or new checkpoint."""
import json
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import GROUPS,collate
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI,cosine_alphas
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.supervision import supervised_losses
from experiments.offline_camera_retrain_v1.orientation_supervision import orientation_losses
BASE=Path(__file__).parent/'runs';OUT=BASE/'turn_generalization_oct09'
def main():
 torch.set_num_threads(4);cp=torch.load(BASE/'native_dynamic_scene20_contact_55k_oct07/last.pt',map_location='cpu',weights_only=False);c=cp['config'];model=OfflineSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda().eval();model.load_state_dict(cp['model'])
 d=NativeBodyData('train',seed=1009,skeleton_profile=c['skeleton_profile'],rich_source=c['rich_source'],trumans_scene_manifest=c['trumans_scene_manifest'],trumans_window_protocol=c['trumans_window_protocol'],contact_root=c['rich_contact_root'],temporal_scene_manifest=c['temporal_scene_manifest']);rows=[];alphas=cosine_alphas().cuda();samples=[];identities=[]
 for g in GROUPS:
  for _ in range(2):s,i=d.sample(128,g);samples.append(s);identities.append(i)
 batch={k:v.cuda() for k,v in collate(samples).items()}
 for time in [100,500,900]:
  alpha=alphas[time];generator=torch.Generator(device='cuda').manual_seed(1009+time);noisy=alpha.sqrt()*batch['motion']+(1-alpha).sqrt()*torch.randn(batch['motion'].shape,device='cuda',generator=generator)
  with torch.no_grad():pred=model(noisy,torch.full((8,),time,device='cuda'),batch,control_mask=fixed_control_mask(8,128,'head','cuda'))
  for index,identity in enumerate(identities):
   one={k:v[index:index+1] for k,v in batch.items()};p=pred[index:index+1].detach().requires_grad_();base=supervised_losses(p,one['motion'],one,profile='coordination_v1',signal_weight=alpha[None]);ori=orientation_losses(p,one['motion'],alpha[None],accumulated=True);physical=base['gait_total']+base['phase_free_support_total'];leg=.1*ori['global_leg_orientation_loss'];grad={k:torch.autograd.grad(value,p,retain_graph=True)[0] for k,value in [('orientation',ori['orientation_total']),('support',physical),('global_legs',leg)]}
   row={'timestep':time,'alpha':float(alpha),'identity':identity,'diagnostics':{}}
   for region,indices in [('all_pose',list(range(135))),('root',list(range(9))),('legs',[3+j*6+i for j in (1,2,4,5,7,8) for i in range(6)])]:
    a=grad['orientation'][...,indices].flatten();b=grad['support'][...,indices].flatten();l=grad['global_legs'][...,indices].flatten();row['diagnostics'][region]={'orientation_norm':float(a.norm()),'support_norm':float(b.norm()),'norm_ratio':float(a.norm()/b.norm().clamp_min(1e-12)),'orientation_support_cosine':float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-12)),'global_legs_support_cosine':float(torch.dot(l,b)/(l.norm()*b.norm()).clamp_min(1e-12))}
   rows.append(row)
 # Same 10 deg/s bias over different duration demonstrates loss scaling.
 length_scaling=[]
 for length in [64,128,192]:
  p=torch.zeros(1,length,201);target=p.clone();r=Rotation.from_euler('y',np.arange(length)/20*np.deg2rad(10)).as_matrix();p[0,:,3:9]=torch.tensor(np.concatenate((r[:,:,0],r[:,:,1]),-1)-np.array([1,0,0,0,1,0]),dtype=torch.float32);v=orientation_losses(p,target,accumulated=True);length_scaling.append({'frames':length,'duration_s':length/20,'accumulated_heading_loss':float(v['accumulated_heading_loss'])})
 (OUT/'gradient_balance.json').write_text(json.dumps({'scope':'24 examples: 8 fixed training windows x 3 diffusion timesteps; output-space gradients, not shared-parameter gradients or causal proof','rows':rows,'fixed_rate_bias_length_scaling':length_scaling},indent=2));print('completed output gradient audit',len(rows))
if __name__=='__main__':main()
