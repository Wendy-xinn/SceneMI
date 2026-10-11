"""Inference-only adapter mute/GT-poison checks, never checkpoint selection."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.body_local_scene_model import BodyLocalSceneMI
from experiments.offline_camera_retrain_v1.body_local_scene import BodySceneCache
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.known_input_generation import generate_without_body_initialization
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.evaluate_body_history import measure

H=Path(__file__).parent;O=H/'runs/control_scale_body_scene_oct11'
@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    cp=torch.load(O/'body_local/last.pt',map_location='cpu',weights_only=False);c=cp['config']
    model=BodyLocalSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda().eval();model.load_state_dict(cp['model'])
    data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root']);cache=BodySceneCache();rows=[]
    for row in json.loads((O/'rows.json').read_text()):
        if row['scope']!='demo':continue
        w=row['window'];sample,identity=data.sample(128,w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);batch={k:v.cuda() for k,v in collate([sample]).items()};batch['body_scene_query']=[cache.get(identity)]
        normal=generate_without_body_initialization(model,batch,seed=row['seed']);saved=np.load(O/f"{row['index']}.npz")['body_local'];difference=float(np.max(abs(saved-normal[0].cpu().numpy())));assert difference<1e-5
        features=torch.as_tensor(model.last_body_query_features,device='cuda');embedding=model.body_local_encoder(features.flatten(2))*features[...,4].any(-1)[...,None]
        stats=dict(model.last_body_query_stats);stats['embedding_l2_mean']=float(embedding.norm(dim=-1).mean());stats['embedding_l2_max']=float(embedding.norm(dim=-1).max())
        handle=model.body_local_encoder.register_forward_hook(lambda module,args,out:torch.zeros_like(out))
        try:muted=generate_without_body_initialization(model,batch,seed=row['seed'])
        finally:handle.remove()
        poisoned=dict(batch)
        for key in ('motion','joints','trajectory','contact_target','contact_valid'):
            if key in poisoned:poisoned[key]=torch.rand_like(poisoned[key].float())*100
        corrupt=generate_without_body_initialization(model,poisoned,seed=row['seed']);assert torch.equal(corrupt,normal)
        nj=forward_kinematics(normal,batch['rest']);mj=forward_kinematics(muted,batch['rest'])
        record=dict(index=row['index'],seed=row['seed'],saved_generation_max_diff=difference,GT_poison_max_diff=float((corrupt-normal).abs().max()),normal_metrics=measure(normal,batch,turn_threshold=30),muted_metrics=measure(muted,batch,turn_threshold=30),adapter_mute_FK_difference_mean_cm=float((nj-mj).norm(dim=-1).mean()*100),adapter_mute_FK_difference_max_cm=float((nj-mj).norm(dim=-1).max()*100),query=stats)
        rows.append(record);print(json.dumps(record),flush=True)
    last=model.body_local_encoder[-1]
    result=dict(scope='850/906 post-training mechanism diagnostic only; not an adoption gate or independent test',training_changed=False,checkpoint_saved=False,adapter_last_weight_norm=float(last.weight.norm()),adapter_last_bias_norm=float(last.bias.norm()),rows=rows)
    (O/'body_local_intervention_audit.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
