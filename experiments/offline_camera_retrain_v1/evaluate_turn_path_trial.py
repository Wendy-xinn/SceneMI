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
from experiments.offline_camera_retrain_v1.known_input_generation import generate_without_body_initialization
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
    parser=argparse.ArgumentParser();parser.add_argument('--support-refinement',action='store_true');args=parser.parse_args()
    if args.support_refinement:O=H/'runs/sole_support_oct11'
    labels=['original55k','prior_turn600','turn_control','sole_support'] if args.support_refinement else ['original55k','continued_control','turn_path']
    torch.set_num_threads(4);start=time.monotonic();models={};digests={}
    for label in labels:
        path=H/'runs/native_dynamic_scene20_contact_55k_oct07/last.pt' if label=='original55k' else O/label/'last.pt'
        if label=='prior_turn600':path=H/'runs/turn_path_training_oct10/turn_path/last.pt'
        cp=torch.load(path,map_location='cpu',weights_only=False);c=cp['config'];model=OfflineSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda().eval();model.load_state_dict(cp['model']);models[label]=model
        digest=hashlib.sha256()
        for name,t in sorted(cp['model'].items()):digest.update(name.encode());digest.update(t.numpy().tobytes())
        digests[label]=digest.hexdigest();del cp
    data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'])
    selected=json.loads((H/'runs/state_relative_spline_oct10/confirmation/protocol.json').read_text())['selected'];panel=[selected[i] for i in range(0,48,3)]
    demos=json.loads((H/'runs/state_relative_spline_oct10/demo/manifest.json').read_text())['cases'];historic=[json.loads(s) for s in (H/'runs/native_dynamic_scene20_contact_55k_oct07/full_validation/standard_rows.jsonl').read_text().splitlines()]
    protocol=dict(status='registered_before_scoring',panel=panel,seeds=[2026101201,2026101202],source_model_sha256=digests,scope='16 exposed validation body sequences, one window each, two seeds; 850/906 development; not pristine holdout',frames='full128=6.4s; all noise initialized; no GT body prefix',metrics='GT stance scoring only; ankle_cross heuristic can include legitimate crossed steps; no physical certificate',batch_size=1)
    (O/'evaluation_protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
    members=[dict(scope='demo',index=d['index'],seed=d['seed'],window=historic[d['index']]['window']) for d in demos]+[dict(scope='panel',index=i,seed=seed,window=w) for seed in protocol['seeds'] for i,w in enumerate(panel)]
    for member in members:
        w=member['window'];sample,identity=data.sample(128,w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);batch={k:v.cuda() for k,v in collate([sample]).items()};values={};saved={}
        skin=NativeSurfacePoints(identity['native_body']);gt_soles=skin.soles(skin(batch['motion'][0]));sole_stance=torch.diff(gt_soles,dim=0).norm(dim=-1)<.01
        for label,model in models.items():
            motion=generate_without_body_initialization(model,batch,seed=member['seed']);result=measure(motion,batch,turn_threshold=30);result.update(leg_motion(motion));result.update(extra_metrics(motion,batch['rest']));soles=skin.soles(skin(motion[0]));slide=torch.diff(soles,dim=0)[...,[0,2]].norm(dim=-1);result['native_sole_stance_horizontal_cm_frame']=float((slide*sole_stance).sum()/sole_stance.sum().clamp_min(1)*100)
            envelope=soles[:,:,1].min(-1).values-gt_soles[:,:,1].min(-1).values;result['native_sole_envelope_abs_cm']=float(envelope.abs().mean()*100);result['native_sole_positive_envelope_cm']=float(envelope.clamp_min(0).mean()*100)
            from experiments.offline_camera_retrain_v1.evaluate_body_history import crop
            late=measure(motion[:,80:],crop(batch,80),turn_threshold=30);result['post80_stance_slide_cm_frame']=late['gt_stance_slide_cm_frame'];result['post80_floating_proxy_cm']=late['support_floating_m']*100;values[label]=result
            if member['scope']=='demo':
                if label=='original55k':
                    with np.load(H/f"runs/state_relative_spline_oct10/demo/{member['index']}.npz") as a:assert np.max(abs(motion[0].cpu().numpy()-a['official55k_motion']))<1e-5
                elif label!='prior_turn600':saved[label]=motion[0].cpu().numpy()
            if member['scope']=='demo' and member['index']==850 and label==('sole_support' if args.support_refinement else 'turn_path'):
                poisoned=dict(batch)
                for key in ['motion','joints','trajectory','contact_target','contact_valid']:
                    if key in poisoned:poisoned[key]=torch.rand_like(poisoned[key].float())*100
                regenerated=generate_without_body_initialization(model,poisoned,seed=member['seed']);assert torch.equal(regenerated,motion);protocol['body_GT_poison_output_max_diff']=float((regenerated-motion).abs().max())
        if saved:np.savez_compressed(O/f"{member['index']}.npz",**saved)
        rows.append(dict(**member,identity=identity,metrics=values,gt_geometry=extra_metrics(batch['motion'],batch['rest'])));(O/'rows.json').write_text(json.dumps(rows,indent=2))
        print(member['scope'],member['index'],{k:(round(v['pelvis_orientation_mean_deg'],2),round(v['gt_stance_slide_cm_frame'],3)) for k,v in values.items()},flush=True)
    summary={};rng=np.random.default_rng(1031)
    for label in models:
        summary[label]={m:float(np.mean([r['metrics'][label][m] for r in rows if r['scope']=='panel' and r['metrics'][label][m] is not None])) for m in METRICS if any(r['metrics'][label][m] is not None for r in rows if r['scope']=='panel')}
    paired={}
    pairs=[('sole_support','original55k'),('sole_support','prior_turn600'),('sole_support','turn_control'),('turn_control','prior_turn600')] if args.support_refinement else [('turn_path','original55k'),('turn_path','continued_control'),('continued_control','original55k')]
    for label,base in pairs:
        paired[label+'-'+base]={}
        for m in METRICS:
            groups=defaultdict(list)
            for r in rows:
                a,b=r['metrics'][label][m],r['metrics'][base][m]
                if r['scope']=='panel' and a is not None and b is not None:groups[r['identity']['group']+'/'+r['identity']['sequence_id']].append(a-b)
            ds=np.array([np.mean(v) for v in groups.values()])
            if len(ds):paired[label+'-'+base][m]=dict(mean_difference=float(ds.mean()),exploratory_ci95=np.quantile(rng.choice(ds,(2000,len(ds))).mean(1),[.025,.975]).tolist(),sequences=len(ds))
    (O/'summary.json').write_text(json.dumps(dict(means=summary,paired=paired),indent=2));protocol.update(status='completed',elapsed_s=time.monotonic()-start);(O/'evaluation_protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
