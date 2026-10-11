"""Fixed all-noise final-trial comparison; GT is scoring-only."""
import json, time
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.train_projected_replay_final import H,OUT,BASE,observed_head
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.known_input_generation import camera_inputs_only,generate_without_body_initialization
from experiments.offline_camera_retrain_v1.turn_manifold_sampling import sample_turn_manifold
from experiments.offline_camera_retrain_v1.native_surface_points import NativeSurfacePoints
from experiments.offline_camera_retrain_v1.evaluate_body_history import measure
from experiments.offline_camera_retrain_v1.evaluate_turn_path_trial import extra_metrics
from experiments.offline_camera_retrain_v1.evaluate_state_spline_confirmation import leg_motion

def register():
    members=json.loads((H/'runs/hierarchical_constraint_oct11/protocol.json').read_text())['members']
    selected=json.loads((H/'runs/state_relative_spline_oct10/confirmation/protocol.json').read_text())['selected']
    members += [dict(scope='panel',index=i,seed=seed,window=w) for i,w in enumerate(selected[::3]) for seed in (2026101201,2026101202)]
    protocol=dict(status='registered_before_scoring',members=members,frames=128,ddim_steps=20,projection_start=500,
        protocol='original55k and trained raw all-noise sampling. Primary candidate is constrained sampler for synthetic TRUMANS; other domains use trained raw sampling because no valid physical PV/head mount supplied.',
        metrics='Full128 output; native SMPL/SMPL-X sole GT stance, joint GT stance, actual inferred foot contact where observed planar support available; turning and crossing proxy. GT only for evaluation, never initializer.',
        scope='fixed development windows and two seeds; not an independent test set',
        no_GT_body_input=True,acceptance='all4 demos improve mean pelvis error>=20%, stance metrics<=original1.02 and crossing<=original+.02. Fixed panel aggregate same nonregression and heading<=original. Failed final trial ends turning tuning.')
    OUT.mkdir(exist_ok=True);(OUT/'evaluation_protocol.json').write_text(json.dumps(protocol,indent=2));return protocol

@torch.no_grad()
def main():
    torch.set_num_threads(4);protocol=json.loads((OUT/'evaluation_protocol.json').read_text()) if (OUT/'evaluation_protocol.json').exists() else register();start=time.monotonic();models={}
    for label,path in [('original55k',BASE),('trained_raw',OUT/'last.pt')]:
        cp=torch.load(path,map_location='cpu',weights_only=False);c=cp['config'];m=OfflineSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda().eval();m.load_state_dict(cp['model']);models[label]=m;del cp
    data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'])
    from experiments.offline_camera_retrain_v1.scene_floor_physics import FloorCache
    from experiments.offline_camera_retrain_v1.support_coverage_objective import FullFootCache
    from experiments.offline_camera_retrain_v1.audit_native_support_scene import stats
    floors=FloorCache();feet=FullFootCache();rows=[]
    for member in protocol['members']:
        w=member['window'];sample,identity=data.sample(128,w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);batch={k:v.cuda() for k,v in collate([sample]).items()};known=camera_inputs_only(batch)
        motions={label:generate_without_body_initialization(model,known,seed=member['seed']) for label,model in models.items()}
        constrained=w['group']=='trumans';sampling={}
        if constrained:
            hp,hr=observed_head(known)
            try:motions['primary'],sampling=sample_turn_manifold(models['trained_raw'],known,hp,hr,motions['original55k'],seed=member['seed'])
            except ValueError:
                constrained=False;motions['primary']=motions['trained_raw'];sampling={'projection_skipped':'unobservable gravity heading'}
        else:motions['primary']=motions['trained_raw']
        skin=NativeSurfacePoints(identity['native_body']);gt_soles=skin.soles(skin(batch['motion'][0]));stance=torch.diff(gt_soles,dim=0).norm(dim=-1)<.01
        floor=floors.get(identity);fullskin=feet.get(identity['native_body']) if floor else None;values={}
        for label,motion in motions.items():
            result=measure(motion,batch,turn_threshold=30);result.update(extra_metrics(motion,known['rest']));result.update(leg_motion(motion));soles=skin.soles(skin(motion[0]));slide=torch.diff(soles,dim=0)[...,[0,2]].norm(dim=-1)
            result['native_sole_stance_horizontal_cm_frame']=float((slide*stance).sum()/stance.sum().clamp_min(1)*100)
            result['observed_floor']=stats(fullskin(motion[0]).cpu().numpy(),fullskin.labels.cpu().numpy(),floor[0],floor[1].data) if floor else None
            values[label]=result
        if member['scope']=='demo':
            poisoned=dict(batch)
            for key in ('motion','joints','trajectory','contact_target','contact_valid'):
                if key in poisoned:poisoned[key]=torch.rand_like(poisoned[key].float())*100
            if constrained:repeated,_=sample_turn_manifold(models['trained_raw'],poisoned,hp,hr,motions['original55k'],seed=member['seed'])
            else:repeated=generate_without_body_initialization(models['trained_raw'],poisoned,seed=member['seed'])
            assert torch.equal(repeated,motions['primary']);protocol.setdefault('GT_poison_max_diff',{})[str(member['index'])]=0.
            np.savez_compressed(OUT/f"{member['index']}.npz",trained_raw=motions['trained_raw'][0].cpu().numpy(),primary=motions['primary'][0].cpu().numpy())
        rows.append(dict(**member,identity=identity,constrained=constrained,sampling=sampling,metrics=values));(OUT/'rows.json').write_text(json.dumps(rows,indent=2))
        print(member['scope'],member['index'],{k:(round(v['pelvis_orientation_mean_deg'],2),round(v['gt_stance_slide_cm_frame'],3),round(v['ankle_cross_fraction'],3)) for k,v in values.items()},flush=True)
    keys=('pelvis_orientation_mean_deg','gt_stance_slide_cm_frame','native_sole_stance_horizontal_cm_frame','ankle_cross_fraction','mpjpe_cm','head_orientation_mean_deg','root_step_max_deg','leg_angular_accel_p95_deg_frame2')
    means={scope:{label:{key:float(np.mean([r['metrics'][label][key] for r in rows if (r['scope']=='panel')==(scope=='panel')])) for key in keys} for label in ('original55k','trained_raw','primary')} for scope in ('demos','panel')}
    gates=[]
    for r in rows:
        if r['scope']=='panel':continue
        a=r['metrics']['primary'];b=r['metrics']['original55k'];g=dict(index=r['index'],seed=r['seed'],heading=a['pelvis_orientation_mean_deg']<=.8*b['pelvis_orientation_mean_deg'],joint_stance=a['gt_stance_slide_cm_frame']<=1.02*b['gt_stance_slide_cm_frame'],native_stance=a['native_sole_stance_horizontal_cm_frame']<=1.02*b['native_sole_stance_horizontal_cm_frame'],crossing=a['ankle_cross_fraction']<=b['ankle_cross_fraction']+.02);g['pass']=all(g[k] for k in ('heading','joint_stance','native_stance','crossing'));gates.append(g)
    a=means['panel']['primary'];b=means['panel']['original55k'];panel=dict(heading=a['pelvis_orientation_mean_deg']<=b['pelvis_orientation_mean_deg'],joint_stance=a['gt_stance_slide_cm_frame']<=1.02*b['gt_stance_slide_cm_frame'],native_stance=a['native_sole_stance_horizontal_cm_frame']<=1.02*b['native_sole_stance_horizontal_cm_frame'],crossing=a['ankle_cross_fraction']<=b['ankle_cross_fraction']+.02)
    adopt=all(g['pass'] for g in gates) and all(panel.values());acceptance=dict(adopt=adopt,demo_gates=gates,panel_gates=panel,stop_turn_tuning_if_failed=not adopt)
    (OUT/'summary.json').write_text(json.dumps(means,indent=2));(OUT/'acceptance.json').write_text(json.dumps(acceptance,indent=2));protocol.update(status='complete',elapsed_s=time.monotonic()-start);(OUT/'evaluation_protocol.json').write_text(json.dumps(protocol,indent=2));print('COMPLETE',json.dumps(acceptance),flush=True)

if __name__=='__main__':main()
