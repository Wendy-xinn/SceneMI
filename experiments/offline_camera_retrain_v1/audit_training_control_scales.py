"""TRAIN-only unit, output-gradient and shared condition-encoder gradient audit."""
import json,math
import numpy as np,torch
from experiments.offline_camera_retrain_v1.data import HERE,GROUPS,collate
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI,cosine_alphas
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.known_input_generation import camera_inputs_only,generate_without_body_initialization
from experiments.offline_camera_retrain_v1.supervision import supervised_losses
from experiments.offline_camera_retrain_v1.winding_guard_objective import winding_guard_losses
from experiments.offline_camera_retrain_v1.calibrated_control_objective import calibrated_control_losses
from experiments.offline_camera_retrain_v1.support_coverage_objective import FullFootCache,coverage_physics_losses
from experiments.offline_camera_retrain_v1.scene_floor_physics import FloorCache
O=HERE/'runs/control_scale_body_scene_oct11'
def main():
 torch.set_num_threads(4);O.mkdir(exist_ok=True);cp=torch.load(HERE/'runs/native_dynamic_scene20_contact_55k_oct07/last.pt',map_location='cpu',weights_only=False);c=cp['config'];model=OfflineSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda().eval();model.load_state_dict(cp['model']);model.requires_grad_(False)
 shared=list(model.core.sparse_control_process.parameters())+list(model.core.cond_process.parameters())
 for p in shared:p.requires_grad_(True)
 data=NativeBodyData('train',seed=2026101133,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'],window_sampling='duration');floors=FloorCache();feet=FullFootCache();alphas=cosine_alphas().cuda();rows=[]
 for group in GROUPS:
  for index in range(2):
   sample,identity=data.sample(128,group);batch={k:v.cuda() for k,v in collate([sample]).items()};known=camera_inputs_only(batch)
   replay=generate_without_body_initialization(model,known,seed=2026101133+index,steps=20)
   for t,basis_name in [(50,'GT_qnoise'),(500,'GT_qnoise'),(950,'GT_qnoise'),(50,'frozen55k_replay')]:
    a=alphas[t:t+1];generator=torch.Generator(device='cuda').manual_seed(2026101133+t+index);basis=replay if basis_name=='frozen55k_replay' else batch['motion'];noisy=a[:,None,None].sqrt()*basis+(1-a[:,None,None]).sqrt()*torch.randn(basis.shape,device='cuda',generator=generator)
    pred=model(noisy,torch.tensor([t],device='cuda'),known);base=supervised_losses(pred,batch['motion'],batch,profile='coordination_v1',signal_weight=a);cal=calibrated_control_losses(pred,batch['motion'],batch['rest'],a);wind=winding_guard_losses(pred,batch['motion'],a);physics=coverage_physics_losses(pred,batch['motion'],batch['rest'],[identity],a,floors,feet)
    terms=dict(base=base['total'],head_position_existing=base['head_position_mse'],winding=wind['winding_guard_total'],support=physics['scene_physics_total'],calibrated=cal['calibrated_control_total']);grads={};values={}
    for name,value in terms.items():
     parts=torch.autograd.grad(value,[pred,*shared],retain_graph=True,allow_unused=True);out=parts[0];encoder=torch.cat([g.flatten() if g is not None else torch.zeros_like(p).flatten() for g,p in zip(parts[1:],shared)])
     grads[name]=(out,encoder);values[name]=dict(weighted_loss=float(value.detach()),output_translation_norm=float(out[...,:3].norm()),output_rotation_norm=float(out[...,3:135].norm()),condition_encoder_norm=float(encoder.norm()))
    cos={}
    for l,r in [('calibrated','support'),('calibrated','winding'),('head_position_existing','support')]:
     x,y=grads[l][1],grads[r][1];cos[l+'_'+r]=float(torch.dot(x,y)/(x.norm()*y.norm()).clamp_min(1e-12))
    rows.append(dict(identity=identity,timestep=t,basis=basis_name,alpha_bar=float(a[0]),terms=values,condition_encoder_cosine=cos,valid_floor=bool(physics['scene_floor_valid_examples'])));print(group,index,t,basis_name,flush=True)
 result=dict(scope='8 TRAIN windows x4 noise/replay settings=32; shared sparse-control and BPS encoder gradients only, not full-backbone gradients; no optimizer or validation-based coefficient selection',rows=rows,units=dict(root_motion='half meters, FK restores x2',head_position_existing='m squared',rotation='dimensionless chordal',physical_height='3cm normalization',calibrated='5cm/5deg and15deg Huber; fixed .20/.02/.10',winding='cumulative radians squared, scales with duration'))
 (O/'training_gradient_audit.json').write_text(json.dumps(result,indent=2));print('TRAIN_GRADIENT_AUDIT_COMPLETED',flush=True)
if __name__=='__main__':main()
