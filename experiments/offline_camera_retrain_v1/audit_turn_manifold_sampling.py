"""Small pre-registered frozen-model probe; retain metrics, no checkpoints."""
import json,time
from pathlib import Path
import torch
from experiments.offline_camera_retrain_v1.audit_hard_head_coordination import angle
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.known_input_generation import camera_inputs_only,generate_without_body_initialization
from experiments.offline_camera_retrain_v1.typed_head_condition import camera_to_head
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d,IDENTITY_6D
from experiments.offline_camera_retrain_v1.turn_manifold_sampling import sample_turn_manifold
from experiments.offline_camera_retrain_v1.hard_head_coordination import coordinate_hard_head
from experiments.offline_camera_retrain_v1.scene_floor_physics import FloorCache
from experiments.offline_camera_retrain_v1.support_coverage_objective import FullFootCache
from experiments.offline_camera_retrain_v1.evaluate_body_history import measure
from experiments.offline_camera_retrain_v1.evaluate_turn_path_trial import extra_metrics
from experiments.offline_camera_retrain_v1.audit_native_support_scene import stats
H=Path(__file__).parent;SOURCE=H/'runs/hierarchical_constraint_oct11';OUT=H/'runs/turn_manifold_sampling_oct11'

def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True);start=time.monotonic()
    members=json.loads((SOURCE/'protocol.json').read_text())['members'];reference=json.loads((SOURCE/'rows.json').read_text())
    protocol=dict(status='registered_before_scoring',members=members,source='original55k frozen',frames=128,seed_matched=True,steps=20,projection_start=500,polish_iterations=160,GT_initialization_or_root_track=False,scope='850/906 two seeds feasibility only; no new training and no held-out claim')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    cp=torch.load(H/'runs/native_dynamic_scene20_contact_55k_oct07/last.pt',map_location='cpu',weights_only=False);c=cp['config'];model=OfflineSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda().eval();model.load_state_dict(cp['model']);del cp
    data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'])
    mount_t=torch.tensor(json.loads((SOURCE/'TRAIN_mount_calibration.json').read_text())['translation_m'],device='cuda');mount_r=torch.eye(3,device='cuda');floors=FloorCache();feet=FullFootCache();rows=[]
    for member,ref in zip(members,reference):
        w=member['window'];sample,identity=data.sample(128,w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in collate([sample]).items()};inputs=camera_inputs_only(batch)
        with torch.no_grad():
            original=generate_without_body_initialization(model,inputs,seed=member['seed']);head=camera_to_head(inputs['camera'],mount_r,mount_t);hp=head[...,:3]*2;hr=rotation_from_6d(head[...,3:]-head.new_tensor(IDENTITY_6D))
            generated,audit=sample_turn_manifold(model,inputs,hp,hr,original,seed=member['seed'])
        floor=floors.get(identity);skin=feet.get(identity['native_body'])
        polished,polish_audit=coordinate_hard_head(generated,inputs['rest'],hp,hr,floor=floor,skin=skin,iterations=160)
        values={k:ref['metrics'][k] for k in ('original55k','hard_head_turn_scaffold')}
        with torch.no_grad():
            for label,motion in (('during_sampling',generated),('during_sampling_support',polished)):
                m=measure(motion,batch,turn_threshold=30);m.update(extra_metrics(motion,inputs['rest']));m['observed_floor']=stats(skin(motion[0]).cpu().numpy(),skin.labels.cpu().numpy(),floor[0],floor[1].data);values[label]=m
        rows.append(dict(**member,identity=identity,metrics=values,sampling=audit,support_polish=polish_audit));(OUT/'rows.json').write_text(json.dumps(rows,indent=2))
        print('DONE',member['index'],member['scope'],{k:(round(v['pelvis_orientation_mean_deg'],2),round(v['gt_stance_slide_cm_frame'],3),round(v['ankle_cross_fraction'],3)) for k,v in values.items()},flush=True)
    poisoned=dict(batch,motion=torch.rand_like(batch['motion'])*100,trajectory=torch.rand_like(batch['trajectory'])*100,contact_target=torch.ones_like(batch['contact_target']))
    repeated,_=sample_turn_manifold(model,poisoned,hp,hr,original,seed=members[-1]['seed']);assert torch.equal(repeated,generated)
    protocol.update(status='complete',elapsed_seconds=time.monotonic()-start,GT_poison_sampling_max_diff=0.,new_checkpoints=0,saved_motion_arrays=0)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
