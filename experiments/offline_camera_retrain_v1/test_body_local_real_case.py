"""Real 850 zero-adapter/native-coordinate smoke without saving weights."""
import json,torch,numpy as np
from experiments.offline_camera_retrain_v1.data import HERE,collate
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.body_local_scene_model import BodyLocalSceneMI
from experiments.offline_camera_retrain_v1.body_local_scene import BodySceneCache
from experiments.offline_camera_retrain_v1.known_input_generation import generate_without_body_initialization
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

def main():
 torch.set_num_threads(2);cp=torch.load(HERE/'runs/native_dynamic_scene20_contact_55k_oct07/last.pt',map_location='cpu',weights_only=False);c=cp['config'];data=NativeBodyData('validation',**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root']);row=json.loads((HERE/'runs/fresh55k_support_coverage_oct11/rows.json').read_text())[0];w=row['window'];s,identity=data.sample(128,w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);batch={k:v.cuda() for k,v in collate([s]).items()};scene=BodySceneCache().get(identity);batch['body_scene_query']=[scene]
 models=[]
 for cls in (OfflineSceneMI,BodyLocalSceneMI):
  model=cls(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda().eval();model.load_state_dict(cp['model'],strict=cls is OfflineSceneMI);models.append(model)
 a=generate_without_body_initialization(models[0],batch,seed=777,steps=2);b=generate_without_body_initialization(models[1],batch,seed=777,steps=2);assert torch.equal(a,b)
 j=forward_kinematics(a,batch['rest'])[0].cpu().numpy();r=global_rotations(a)[0].cpu().numpy();f=scene.features(j,r);assert np.isfinite(f).all()
 poisoned=dict(batch,motion=torch.randn_like(batch['motion'])*100,joints=torch.randn_like(batch['joints'])*100,trajectory=torch.randn_like(batch['trajectory'])*100);p=generate_without_body_initialization(models[1],poisoned,seed=777,steps=2);assert torch.equal(b,p)
 record=dict(index=850,zero_adapter_generation_max_diff=0,GT_poison_output_max_diff=0,body_query_available_fraction=float(f[...,4].mean()),moving_surface_fraction=float(f[...,5].mean()),static_tree_prefixes=len(scene.trees),queries_follow_generated_FK=True,GT_body_query=False)
 (HERE/'runs/control_scale_body_scene_oct11/body_query_preflight.json').write_text(json.dumps(record,indent=2));print('PASS REAL BODY QUERY',record,flush=True)
if __name__=='__main__':main()
