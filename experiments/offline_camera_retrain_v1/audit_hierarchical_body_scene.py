"""Real original55k preflight of the untrained scene-attention architecture."""
import json,time
from pathlib import Path
import torch
from experiments.offline_camera_retrain_v1.hierarchical_body_scene import HierarchicalBodySceneMI
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.body_local_scene import BodySceneCache
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.known_input_generation import camera_inputs_only
H=Path(__file__).parent;OUT=H/'runs/hierarchical_constraint_oct11'
def main():
    torch.set_num_threads(4)
    cp=torch.load(H/'runs/native_dynamic_scene20_contact_55k_oct07/last.pt',map_location='cpu',weights_only=False);c=cp['config']
    args=(c['latent_dim'],tuple(c['dim_mults']));kw=dict(body_conditioning=True,contact_prediction=True)
    base=OfflineSceneMI(*args,**kw).cuda().eval();base.load_state_dict(cp['model'])
    model=HierarchicalBodySceneMI(*args,**kw).cuda().eval();missing,extra=model.load_state_dict(cp['model'],strict=False)
    assert not extra and all(k.startswith('body_surface_attention.') for k in missing);del cp
    data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'])
    members=json.loads((OUT/'protocol.json').read_text())['members'][:2];scenes=BodySceneCache();rows=[]
    for member in members:
        w=member['window'];sample,identity=data.sample(128,w['group'],sequence_index=w['sequence_index'],start_index=w['start_index'])
        batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in collate([sample]).items()};batch['body_scene_query']=[scenes.get(identity)];batch=camera_inputs_only(batch)
        noise=torch.randn(1,128,201,device='cuda',generator=torch.Generator(device='cuda').manual_seed(2026101155));t=torch.tensor([50],device='cuda')
        started=time.monotonic()
        with torch.no_grad():
            a=base(noise,t,batch);b=model(noise,t,batch);assert torch.equal(a,b)
            contacts_a=base.contact_logits(a,batch);contacts_b=model.contact_logits(a,batch);assert torch.equal(contacts_a,contacts_b)
        rows.append(dict(index=member['index'],original55k_zero_adapter_max_difference=0.,contact_zero_adapter_max_difference=0.,available_by_joint=model.last_surface_available.float().mean((0,1)).tolist(),available_by_joint_and_radius=model.last_surface_available_by_scale.float().mean((0,1)).tolist(),comparison_and_contact_queries_seconds=time.monotonic()-started))
        print('PASS_REAL',rows[-1],flush=True)
    (OUT/'attention_preflight.json').write_text(json.dumps(dict(status='prototype_verified_not_trained',rows=rows,new_parameters=sum(p.numel() for p in model.body_surface_attention.parameters()),radii_m=[.25,.75,2.],surfaces_per_radius=8,query_source='generated preliminary x0 FK, never validation body',scene_token_policy='causal static prefix plus current-time dynamics; no signed SDF; CPU discrete selection',new_checkpoint_files=0),indent=2))
if __name__=='__main__':main()
