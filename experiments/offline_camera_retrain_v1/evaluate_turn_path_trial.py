"""Matched all-noise 128-frame generation: original55k vs two short-train arms."""
import argparse,json,time,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.known_input_generation import generate_without_body_initialization,camera_inputs_only
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.evaluate_body_history import measure
from experiments.offline_camera_retrain_v1.evaluate_state_spline_confirmation import leg_motion
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.native_surface_points import NativeSurfacePoints
H=Path(__file__).parent;O=H/'runs/turn_path_training_oct10'
METRICS=('mpjpe_cm','pa_mpjpe_mm','pelvis_orientation_mean_deg','head_orientation_mean_deg','gt_stance_slide_cm_frame','meaningful_opposite_turn','under_turn','leg_angular_step_p95_deg','leg_angular_accel_p95_deg_frame2','root_step_max_deg','root_step_p95_deg','leg_global_rotation_step_max_deg','ankle_cross_fraction','native_sole_stance_horizontal_cm_frame','contact_coverage','support_floating_m','support_penetration_m','native_sole_envelope_abs_cm','native_sole_positive_envelope_cm','post80_stance_slide_cm_frame','post80_floating_proxy_cm')
def extra_metrics(m,rest):
    g=global_rotations(m)[0].cpu().numpy();delta=g[1:,0]@g[:-1,0].transpose(0,2,1)
    angles=np.rad2deg(Rotation.from_matrix(delta.copy()).magnitude())
    j=forward_kinematics(m,rest)[0];hip_axis=j[:,1]-j[:,2];ankle_axis=j[:,7]-j[:,8];cross=float(((hip_axis*ankle_axis).sum(-1)<0).float().mean())
    return dict(root_step_max_deg=float(angles.max()),root_step_p95_deg=float(np.quantile(angles,.95)),ankle_cross_fraction=cross)
@torch.inference_mode()
def main():
    global O
    parser=argparse.ArgumentParser();parser.add_argument('--support-refinement',action='store_true');parser.add_argument('--fresh55k-replay',action='store_true');parser.add_argument('--fresh55k-winding',action='store_true');parser.add_argument('--fresh55k-physics',action='store_true');parser.add_argument('--fresh55k-coverage',action='store_true');parser.add_argument('--scale-body-scene',action='store_true');parser.add_argument('--root-facing',action='store_true');parser.add_argument('--angular-balance',action='store_true');args=parser.parse_args()
    if args.root_facing:args.angular_balance=True
    if args.angular_balance:args.scale_body_scene=True
    if args.scale_body_scene:args.fresh55k_coverage=True
    if args.fresh55k_coverage:
        if args.fresh55k_physics or args.fresh55k_winding or args.fresh55k_replay or args.support_refinement:parser.error('Choose coverage alone')
        args.fresh55k_physics=True
    if args.fresh55k_physics:args.fresh55k_replay=True
    if args.fresh55k_winding:args.fresh55k_replay=True
    if args.support_refinement and args.fresh55k_replay:parser.error('Choose one trial')
    if args.support_refinement:O=H/'runs/sole_support_oct11'
    if args.fresh55k_replay:O=H/'runs/fresh55k_replay_oct11'
    if args.fresh55k_winding:O=H/'runs/fresh55k_winding_oct11'
    if args.fresh55k_physics:O=H/'runs/fresh55k_scene_physics_oct11'
    if args.fresh55k_coverage:O=H/'runs/fresh55k_support_coverage_oct11'
    labels=['original55k','prior_turn600','turn_control','sole_support'] if args.support_refinement else ['original55k','continued_control','turn_path']
    if args.fresh55k_replay:labels=['original55k','denoise_control','rollout_replay']
    if args.fresh55k_winding:labels=['original55k','replay_reference','winding_guard']
    if args.fresh55k_physics:labels=['original55k','guard_reference','winding_scene']
    if args.fresh55k_coverage:labels=['original55k','scene_reference','coverage_support']
    if args.scale_body_scene:O=H/'runs/control_scale_body_scene_oct11';labels=['original55k','support_reference','scale_calibrated','body_local']
    training_root=O
    if args.angular_balance:
        labels=['original55k','support_reference','angular_balanced','root_facing'] if args.root_facing else ['original55k','support_reference','scale_calibrated','angular_balanced'];O=O/('root_facing_eval' if args.root_facing else 'angular_balance_eval');O.mkdir(exist_ok=True)
    torch.set_num_threads(4);start=time.monotonic();models={};digests={}
    for label in labels:
        path=H/'runs/native_dynamic_scene20_contact_55k_oct07/last.pt' if label=='original55k' else (training_root if args.angular_balance else O)/label/'last.pt'
        if label=='prior_turn600':path=H/'runs/turn_path_training_oct10/turn_path/last.pt'
        if label=='replay_reference':path=H/'runs/fresh55k_replay_oct11/rollout_replay/last.pt'
        if label=='support_reference':path=H/'runs/fresh55k_support_coverage_oct11/coverage_support/last.pt'
        if label=='scene_reference':path=H/'runs/fresh55k_scene_physics_oct11/winding_scene/last.pt'
        if label=='guard_reference':path=H/'runs/fresh55k_winding_oct11/winding_guard/last.pt'
        cp=torch.load(path,map_location='cpu',weights_only=False);c=cp['config'];cls=OfflineSceneMI
        if c.get('body_local_queries'):
            from experiments.offline_camera_retrain_v1.body_local_scene_model import BodyLocalSceneMI
            cls=BodyLocalSceneMI
        model=cls(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda().eval();model.load_state_dict(cp['model']);models[label]=model
        digest=hashlib.sha256()
        for name,t in sorted(cp['model'].items()):digest.update(name.encode());digest.update(t.numpy().tobytes())
        digests[label]=digest.hexdigest();del cp
    if args.fresh55k_replay and not args.fresh55k_physics:
        aliases=[('replay_terminal','replay_reference'),('guard_terminal','winding_guard')] if args.fresh55k_winding else [('denoise_terminal','denoise_control'),('replay_terminal','rollout_replay')]
        for label,source in aliases:models[label]=models[source];digests[label]=digests[source]
    if args.fresh55k_physics:
        from experiments.offline_camera_retrain_v1.scene_floor_physics import FloorCache
        from experiments.offline_camera_retrain_v1.audit_native_support_scene import stats as native_scene_stats
        floor_cache=FloorCache()
        if args.fresh55k_coverage:
            from experiments.offline_camera_retrain_v1.support_coverage_objective import FullFootCache
            full_feet=FullFootCache()
        if args.scale_body_scene:
            from experiments.offline_camera_retrain_v1.body_local_scene import BodySceneCache
            body_scene_cache=BodySceneCache()
    data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'])
    selected=json.loads((H/'runs/state_relative_spline_oct10/confirmation/protocol.json').read_text())['selected'];panel=[selected[i] for i in range(0,48,3)]
    demos=json.loads((H/'runs/state_relative_spline_oct10/demo/manifest.json').read_text())['cases'];historic=[json.loads(s) for s in (H/'runs/native_dynamic_scene20_contact_55k_oct07/full_validation/standard_rows.jsonl').read_text().splitlines()]
    protocol=dict(status='registered_before_scoring',panel=panel,seeds=[2026101201,2026101202],source_model_sha256=digests,scope='16 exposed validation body sequences, one window each, two seeds; 850/906 development; not pristine holdout',frames='full128=6.4s; all noise initialized; no GT body prefix',metrics='GT stance scoring only; ankle_cross heuristic can include legitimate crossed steps; no physical certificate',batch_size=1)
    (O/'evaluation_protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
    members=[dict(scope='demo',index=d['index'],seed=d['seed'],window=historic[d['index']]['window']) for d in demos]+[dict(scope='panel',index=i,seed=seed,window=w) for seed in protocol['seeds'] for i,w in enumerate(panel)]
    if args.fresh55k_replay:
        confirmation=[w for i,w in enumerate(selected) if i%3!=0]
        members.extend(dict(scope='confirmation',index=i,seed=seed,window=w) for seed in protocol['seeds'] for i,w in enumerate(confirmation))
        protocol.update(confirmation=confirmation,scope='16 exposed primary +32 other fixed development windows, two seeds; same48 body sequences, not independent holdout')
        (O/'evaluation_protocol.json').write_text(json.dumps(protocol,indent=2))
    for member in members:
        w=member['window'];sample,identity=data.sample(128,w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);batch={k:v.cuda() for k,v in collate([sample]).items()};values={};saved={}
        skin=NativeSurfacePoints(identity['native_body']);gt_soles=skin.soles(skin(batch['motion'][0]));sole_stance=torch.diff(gt_soles,dim=0).norm(dim=-1)<.01
        base_motion=None
        for label,model in models.items():
            if label.endswith('_terminal'):
                motion=model(base_motion,torch.zeros(len(base_motion),device='cuda',dtype=torch.long),camera_inputs_only(batch),control_mask=fixed_control_mask(len(base_motion),128,'head','cuda'),use_scene=True)
            else:
                if label=='body_local':batch['body_scene_query']=[body_scene_cache.get(identity)]
                else:batch.pop('body_scene_query',None)
                motion=generate_without_body_initialization(model,batch,seed=member['seed'])
            if label=='original55k':base_motion=motion
            result=measure(motion,batch,turn_threshold=30);
            if label=='body_local':result['body_local_query']=model.last_body_query_stats
            result.update(leg_motion(motion));result.update(extra_metrics(motion,batch['rest']));soles=skin.soles(skin(motion[0]));slide=torch.diff(soles,dim=0)[...,[0,2]].norm(dim=-1);result['native_sole_stance_horizontal_cm_frame']=float((slide*sole_stance).sum()/sole_stance.sum().clamp_min(1)*100)
            if args.fresh55k_replay:
                feet=forward_kinematics(motion,batch['rest'])[:,:,(10,11)];support=torch.quantile((batch['joints'][:,:,(10,11),1]*2).flatten(1),.05,dim=1)
                low=(feet[:,1:,:,1]-support[:,None,None]).abs()<.05;slow=torch.diff(feet,dim=1).norm(dim=-1)<.01
                result['joint_low_and_slow_fraction']=float((low&slow).float().mean())
                gt_head=global_rotations(batch['motion'])[0,:,15,:,2];head_valid=float((gt_head[:,[0,2]].norm(dim=-1)>.3).float().mean())>=.95
                result['head_opposite_turn']=float(result['head_gt_turn_deg']*result['head_turn_deg']<0) if head_valid and abs(result['head_gt_turn_deg'])>=30 else None
            envelope=soles[:,:,1].min(-1).values-gt_soles[:,:,1].min(-1).values;result['native_sole_envelope_abs_cm']=float(envelope.abs().mean()*100);result['native_sole_positive_envelope_cm']=float(envelope.clamp_min(0).mean()*100)
            if args.fresh55k_coverage:
                pr=global_rotations(motion)[0,:,0,:,2].cpu().numpy();tr=global_rotations(batch['motion'])[0,:,0,:,2].cpu().numpy()
                py=np.unwrap(np.arctan2(pr[:,0],pr[:,2]));ty=np.unwrap(np.arctan2(tr[:,0],tr[:,2]));err=(np.rad2deg(py-ty)+180)%360-180
                result['horizontal_root_yaw_abs_mean_deg']=float(np.mean(abs(err)));result['horizontal_root_yaw_final_abs_deg']=float(abs(err[-1]));result['root_turn_progress_error_deg']=float(np.rad2deg((py[-1]-py[0])-(ty[-1]-ty[0])))
            if args.fresh55k_physics:
                floor=floor_cache.get(identity,128)
                support_skin=full_feet.get(identity['native_body']) if args.fresh55k_coverage else skin
                result['observed_floor_support']=native_scene_stats(support_skin(motion[0]).cpu().numpy(),support_skin.labels.cpu().numpy(),floor[0],floor[1].data) if floor else None
                if args.fresh55k_coverage:result['floor_support_topology']='ALL native dominant ankle/foot vertices; identical rescoring of all variants'
                result['floor_audit']=floor[2] if floor else None
            from experiments.offline_camera_retrain_v1.evaluate_body_history import crop
            late=measure(motion[:,80:],crop(batch,80),turn_threshold=30);result['post80_stance_slide_cm_frame']=late['gt_stance_slide_cm_frame'];result['post80_floating_proxy_cm']=late['support_floating_m']*100;values[label]=result
            if member['scope']=='demo':
                if label=='original55k':
                    with np.load(H/f"runs/state_relative_spline_oct10/demo/{member['index']}.npz") as a:assert np.max(abs(motion[0].cpu().numpy()-a['official55k_motion']))<1e-5
                elif label!='prior_turn600':saved[label]=motion[0].cpu().numpy()
                if label=='body_local':
                    qj=model.last_body_query_joints[0];features=model.last_body_query_features[0];r=model.last_body_query_rotations[0]
                    saved['body_query_joints']=qj;saved['body_query_surface_points']=qj+np.einsum('tjik,tjk->tji',r,features[...,:3]*.75);saved['body_query_valid']=features[...,4]>0;saved['body_query_dynamic']=features[...,5]>0
            if member['scope']=='demo' and (member['index']==850 or args.angular_balance) and (label in ((['angular_balanced','root_facing'] if args.root_facing else ['scale_calibrated','angular_balanced'] if args.angular_balance else ['scale_calibrated','body_local']) if args.scale_body_scene else (['coverage_support'] if args.fresh55k_coverage else (['winding_scene'] if args.fresh55k_physics else (['winding_guard','guard_terminal'] if args.fresh55k_winding else ['rollout_replay','replay_terminal'])))) if args.fresh55k_replay else label==('sole_support' if args.support_refinement else 'turn_path')):
                poisoned=dict(batch)
                for key in ['motion','joints','trajectory','contact_target','contact_valid']:
                    if key in poisoned:poisoned[key]=torch.rand_like(poisoned[key].float())*100
                if label.endswith('_terminal'):
                    regenerated_base=generate_without_body_initialization(models['original55k'],poisoned,seed=member['seed']);regenerated=model(regenerated_base,torch.zeros(len(regenerated_base),device='cuda',dtype=torch.long),camera_inputs_only(poisoned),control_mask=fixed_control_mask(len(regenerated_base),128,'head','cuda'),use_scene=True)
                else:regenerated=generate_without_body_initialization(model,poisoned,seed=member['seed'])
                assert torch.equal(regenerated,motion);protocol['body_GT_poison_output_max_diff']=float((regenerated-motion).abs().max())
                protocol.setdefault('GT_poison_by_case',{}).setdefault(str(member['index']),{})[label]=float((regenerated-motion).abs().max())
                protocol.setdefault('GT_poison_by_variant',{})[label]=float((regenerated-motion).abs().max())
        if saved:np.savez_compressed(O/f"{member['index']}.npz",**saved)
        rows.append(dict(**member,identity=identity,metrics=values,gt_geometry=extra_metrics(batch['motion'],batch['rest'])));(O/'rows.json').write_text(json.dumps(rows,indent=2))
        print(member['scope'],member['index'],{k:(round(v['pelvis_orientation_mean_deg'],2),round(v['gt_stance_slide_cm_frame'],3)) for k,v in values.items()},flush=True)
    summary={};rng=np.random.default_rng(1031)
    for label in models:
        summary[label]={m:float(np.mean([r['metrics'][label][m] for r in rows if r['scope']=='panel' and r['metrics'][label][m] is not None])) for m in METRICS if any(r['metrics'][label][m] is not None for r in rows if r['scope']=='panel')}
    paired={}
    pairs=[('sole_support','original55k'),('sole_support','prior_turn600'),('sole_support','turn_control'),('turn_control','prior_turn600')] if args.support_refinement else [('turn_path','original55k'),('turn_path','continued_control'),('continued_control','original55k')]
    if args.fresh55k_replay:pairs=[('rollout_replay','original55k'),('rollout_replay','denoise_control'),('denoise_control','original55k')]
    if args.fresh55k_replay:pairs.extend([('replay_terminal','original55k'),('replay_terminal','denoise_terminal'),('denoise_terminal','original55k')])
    if args.fresh55k_winding:pairs=[('winding_guard','original55k'),('winding_guard','replay_reference'),('guard_terminal','original55k'),('guard_terminal','replay_terminal')]
    if args.fresh55k_physics:pairs=[('winding_scene','original55k'),('winding_scene','guard_reference')]
    if args.fresh55k_coverage:pairs=[('coverage_support','original55k'),('coverage_support','scene_reference')]
    if args.scale_body_scene:pairs=[('scale_calibrated','support_reference'),('body_local','scale_calibrated'),('body_local','original55k')]
    if args.angular_balance:pairs=[('angular_balanced','original55k'),('angular_balanced','scale_calibrated'),('angular_balanced','support_reference')]
    if args.root_facing:pairs=[('root_facing','original55k'),('root_facing','angular_balanced'),('root_facing','support_reference')]
    for label,base in pairs:
        paired[label+'-'+base]={}
        for m in METRICS:
            groups=defaultdict(list)
            for r in rows:
                a,b=r['metrics'][label][m],r['metrics'][base][m]
                if r['scope']=='panel' and a is not None and b is not None:groups[r['identity']['group']+'/'+r['identity']['sequence_id']].append(a-b)
            ds=np.array([np.mean(v) for v in groups.values()])
            if len(ds):paired[label+'-'+base][m]=dict(mean_difference=float(ds.mean()),exploratory_ci95=np.quantile(rng.choice(ds,(2000,len(ds))).mean(1),[.025,.975]).tolist(),sequences=len(ds))
    result=dict(means=summary,paired=paired)
    if args.fresh55k_replay:
        for label in models:
            summary[label]['joint_low_and_slow_fraction']=float(np.mean([r['metrics'][label]['joint_low_and_slow_fraction'] for r in rows if r['scope']=='panel']))
            eligible=[r['metrics'][label]['head_opposite_turn'] for r in rows if r['scope']=='panel' and r['metrics'][label]['head_opposite_turn'] is not None]
            summary[label]['head_opposite_turn']=float(np.mean(eligible)) if eligible else None
            summary[label]['head_turn_eligible_seed_cases']=len(eligible)
    if args.fresh55k_replay:result['confirmation_means']={label:{m:float(np.mean([r['metrics'][label][m] for r in rows if r['scope']=='confirmation' and r['metrics'][label][m] is not None])) for m in METRICS if any(r['metrics'][label][m] is not None for r in rows if r['scope']=='confirmation')} for label in models}
    if args.fresh55k_coverage:
        for label in models:
            for key in ('horizontal_root_yaw_abs_mean_deg','horizontal_root_yaw_final_abs_deg','root_turn_progress_error_deg'):
                summary[label][key]=float(np.mean([r['metrics'][label][key] for r in rows if r['scope']=='panel']))
    (O/'summary.json').write_text(json.dumps(result,indent=2));protocol.update(status='completed',elapsed_s=time.monotonic()-start);(O/'evaluation_protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
