"""TRAIN-only mount fit, frozen original55k controls, known-input repair audit."""
import json,time
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.known_input_generation import generate_without_body_initialization,camera_inputs_only
from experiments.offline_camera_retrain_v1.typed_head_condition import camera_to_head
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,rotation_from_6d,IDENTITY_6D
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.hard_head_coordination import project_head_pose,coordinate_hard_head
from experiments.offline_camera_retrain_v1.scene_floor_physics import FloorCache
from experiments.offline_camera_retrain_v1.support_coverage_objective import FullFootCache
from experiments.offline_camera_retrain_v1.evaluate_body_history import measure
from experiments.offline_camera_retrain_v1.evaluate_turn_path_trial import extra_metrics
from experiments.offline_camera_retrain_v1.audit_native_support_scene import stats as native_scene_stats

H=Path(__file__).parent;OUT=H/'runs/hierarchical_constraint_oct11'

def angle(r):return torch.rad2deg(torch.acos(((r.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)))

def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True);started=time.monotonic()
    cp=torch.load(H/'runs/native_dynamic_scene20_contact_55k_oct07/last.pt',map_location='cpu',weights_only=False);c=cp['config']
    kwargs={k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')};kwargs['contact_root']=c['rich_contact_root']
    train=NativeBodyData('train',seed=2026101133,**kwargs);mounts=[];train_rows=[]
    for _ in range(4):
        sample,identity=train.sample(64,'trumans');batch=collate([sample]);hp=forward_kinematics(batch['motion'],batch['rest'])[:,:,15];hr=global_rotations(batch['motion'])[:,:,15]
        cr=rotation_from_6d(batch['camera'][...,3:]-batch['camera'].new_tensor(IDENTITY_6D))
        translation=(hr.transpose(-1,-2)@(batch['camera'][...,:3]*2-hp)[...,None]).squeeze(-1)
        mounts.append(translation.flatten(0,1));train_rows.append(dict(sequence=identity['sequence_id'],source_start=identity['source_start_30fps'],rotation_identity_error_max_deg=float(angle(hr.transpose(-1,-2)@cr).max()),mount_translation_m=translation.mean((0,1)).tolist()))
    mount_t=torch.cat(mounts).median(0).values.cuda();mount_r=torch.eye(3,device='cuda')
    (OUT/'TRAIN_mount_calibration.json').write_text(json.dumps(dict(scope='Four TRAIN windows only; shared TRUMANS male zero-shape source. Approximate population calibration, not per-validation GT mount',rows=train_rows,translation_m=mount_t.tolist(),rotation='identity, fixed by synthetic source protocol',validation_GT_used_for_fit=False),indent=2))
    del train
    model=OfflineSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda().eval();model.load_state_dict(cp['model']);del cp
    data=NativeBodyData('validation',seed=777,**kwargs);floor_cache=FloorCache();feet=FullFootCache()
    demos=json.loads((H/'runs/state_relative_spline_oct10/demo/manifest.json').read_text())['cases']
    historical=[json.loads(s) for s in (H/'runs/native_dynamic_scene20_contact_55k_oct07/full_validation/standard_rows.jsonl').read_text().splitlines()]
    members=[dict(index=d['index'],seed=d['seed'],window=historical[d['index']]['window'],scope='demo') for d in demos]
    # One additional seed on each fixed case; no re-selection by GT scores.
    members += [dict(index=d['index'],seed=2026101144,window=historical[d['index']]['window'],scope='second_seed') for d in demos]
    protocol=dict(status='registered_before_scoring',members=members,frames=128,iterations=160,model='original55k frozen; no candidate checkpoint warm-start',constraints='calibrated known head pose only; support inferred once from generated native feet and observed static plane',turn_scaffold='separate candidate from original55k: gravity-twist continuous head path, initial root-head offset inferred from original motion clipped to 60deg, relative turn-path drift soft bound45deg, intrinsic gait velocity; not GT pelvis or a universally valid head==body assumption',GT='TRAIN calibration and final scoring only; no validation body prefix/initialization/root/contacts/mount fit',limitations='two exposed development windows/two seeds, no statistical generalization claim; unknown/collision outside support patch not solved; head-track calibration approximation reported separately')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
    for member in members:
        w=member['window'];sample,identity=data.sample(128,w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in collate([sample]).items()}
        assert identity['sequence_id'] not in {r['sequence'] for r in train_rows}
        inputs=camera_inputs_only(batch)
        with torch.no_grad():
            generated=generate_without_body_initialization(model,inputs,seed=member['seed'])
            head=camera_to_head(inputs['camera'],mount_r,mount_t);hp=head[...,:3]*2;hr=rotation_from_6d(head[...,3:]-head.new_tensor(IDENTITY_6D));valid=torch.ones(hp.shape[:2],dtype=torch.bool,device='cuda')
            hard=project_head_pose(generated,inputs['rest'],hp,hr,valid,valid)
        floor=floor_cache.get(identity);skin=feet.get(identity['native_body'])
        repaired,audit=coordinate_hard_head(generated,inputs['rest'],hp,hr,floor=floor,skin=skin,iterations=160)
        scaffolded,scaffold_audit=coordinate_hard_head(generated,inputs['rest'],hp,hr,floor=floor,skin=skin,iterations=160,turn_scaffold=True)
        values={}
        with torch.no_grad():
            for label,motion in (('original55k',generated),('hard_head_only',hard),('hard_head_coordinated',repaired),('hard_head_turn_scaffold',scaffolded)):
                m=measure(motion,batch,turn_threshold=30);m.update(extra_metrics(motion,inputs['rest']))
                joints=forward_kinematics(motion,inputs['rest']);g=global_rotations(motion)
                m['known_calibrated_head_position_max_cm']=float((joints[:,:,15]-hp).norm(dim=-1).max()*100)
                m['known_head_orientation_mean_deg']=float(angle(hr.transpose(-1,-2)@g[:,:,15]).mean())
                m['head_local_neck_p95_deg']=float(torch.quantile(angle(rotation_from_6d(motion[...,93:99])),.95))
                m['root_absolute_final_rotation_error_deg']=float(angle(global_rotations(batch['motion'])[:,-1,0].transpose(-1,-2)@g[:,-1,0]).mean())
                m['observed_floor']=native_scene_stats(skin(motion[0]).cpu().numpy(),skin.labels.cpu().numpy(),floor[0],floor[1].data)
                values[label]=m
            GT_head=forward_kinematics(batch['motion'],inputs['rest'])[:,:,15]
            calibration_residual=dict(native_GT_head_vs_TRAIN_calibrated_track_mean_cm=float((GT_head-hp).norm(dim=-1).mean()*100),native_GT_head_vs_TRAIN_calibrated_track_max_cm=float((GT_head-hp).norm(dim=-1).max()*100))
        row=dict(**member,identity=identity,calibration_residual_scoring_only=calibration_residual,metrics=values,solver=audit,scaffold_solver=scaffold_audit)
        rows.append(row);(OUT/'rows.json').write_text(json.dumps(rows,indent=2))
        if member['scope']=='demo':np.savez_compressed(OUT/f"{member['index']}.npz",hard_head_only=hard[0].detach().cpu().numpy(),hard_head_coordinated=repaired[0].cpu().numpy(),hard_head_turn_scaffold=scaffolded[0].cpu().numpy())
        print('DONE',member['index'],member['scope'],{k:(round(v['pelvis_orientation_mean_deg'],1),round(v['gt_stance_slide_cm_frame'],3),round(v['head_local_neck_p95_deg'],1)) for k,v in values.items()},flush=True)
    # Validation-body poison audit: sanitizer eliminates all body/label fields;
    # same seed generation and deterministic repair therefore get same inputs.
    poisoned=dict(batch,motion=torch.rand_like(batch['motion'])*100,trajectory=torch.rand_like(batch['trajectory'])*100,contact_target=torch.ones_like(batch['contact_target']))
    clean=camera_inputs_only(batch);dirty=camera_inputs_only(poisoned)
    assert clean.keys()==dirty.keys() and all(torch.equal(clean[k],dirty[k]) for k in clean if isinstance(clean[k],torch.Tensor))
    with torch.no_grad():regenerated=generate_without_body_initialization(model,dirty,seed=members[-1]['seed'])
    assert torch.equal(regenerated,generated)
    again,_=coordinate_hard_head(regenerated,dirty['rest'],hp,hr,floor=floor,skin=skin,iterations=160)
    assert torch.equal(again,repaired)
    again,_=coordinate_hard_head(regenerated,dirty['rest'],hp,hr,floor=floor,skin=skin,iterations=160,turn_scaffold=True)
    assert torch.equal(again,scaffolded)
    protocol.update(status='complete',elapsed_seconds=time.monotonic()-started,GT_poison_generation_max_diff=0.,GT_poison_repair_max_diff=0.,new_checkpoints=0)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));print('COMPLETE',protocol['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
