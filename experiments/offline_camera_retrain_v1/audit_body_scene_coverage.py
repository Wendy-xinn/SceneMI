"""TRAIN-only diagnostic separates absent geometry from drifting x0 queries.

GT FK queries are diagnostic measurements ONLY, never generator inputs.
"""
import json
from pathlib import Path
import torch
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI,cosine_alphas
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate,GROUPS
from experiments.offline_camera_retrain_v1.body_local_scene import BodySceneCache
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.known_input_generation import camera_inputs_only,generate_without_body_initialization
H=Path(__file__).parent;O=H/'runs/control_scale_body_scene_oct11'
@torch.inference_mode()
def main():
    torch.set_num_threads(2);cp=torch.load(H/'runs/native_dynamic_scene20_contact_55k_oct07/last.pt',map_location='cpu',weights_only=False);c=cp['config'];model=OfflineSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda().eval();model.load_state_dict(cp['model']);del cp
    data=NativeBodyData('train',seed=2026101133,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'],window_sampling='duration');cache=BodySceneCache(limit=1);rows=[];alphas=cosine_alphas().cuda()
    for group in GROUPS:
        for index in range(2):
            sample,identity=data.sample(128,group);batch={k:v.cuda() for k,v in collate([sample]).items()};known=camera_inputs_only(batch);scene=cache.get(identity)
            motions={'GT_diagnostic_only':batch['motion'],'original55k_full_noise':generate_without_body_initialization(model,known,seed=2026101133+index,steps=20)}
            for t in (50,500,950):
                alpha=alphas[t];generator=torch.Generator(device='cuda').manual_seed(2026101133+t+index);basis=batch['motion'];noisy=alpha.sqrt()*basis+(1-alpha).sqrt()*torch.randn(basis.shape,device='cuda',generator=generator)
                motions[f'TRAIN_qnoise_x0_t{t}']=model(noisy,torch.tensor([t],device='cuda'),known)
            values={}
            for name,motion in motions.items():
                joints=forward_kinematics(motion,batch['rest'])[0].cpu().numpy();rotations=global_rotations(motion)[0].cpu().numpy();features=scene.features(joints,rotations)
                values[name]=dict(available_fraction=float(features[...,4].mean()),available_by_joint=features[...,4].mean(0).tolist(),dynamic_fraction=float(features[...,5].mean()))
            rows.append(dict(group=group,index=index,identity=identity,coverage=values));print(group,index,{k:round(v['available_fraction'],3) for k,v in values.items()},flush=True)
    (O/'TRAIN_body_scene_coverage_audit.json').write_text(json.dumps(dict(scope='same8 TRAIN windows as gradient audit; original55k diagnostic only; no optimization or validation selection',GT_query_inference=False,weights_saved=False,radius_m=.75,rows=rows),indent=2))
if __name__=='__main__':main()
