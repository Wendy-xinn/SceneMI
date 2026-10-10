"""Fixed-foot ablations: exposed two demos and eight sequences/two seeds."""
import json,time,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.known_input_generation import generate_without_body_initialization
from experiments.offline_camera_retrain_v1.turn_foot_coordination import coordinate_turn
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.evaluate_body_history import measure
from experiments.offline_camera_retrain_v1.evaluate_state_spline_confirmation import leg_motion
from experiments.offline_camera_retrain_v1.native_surface_points import NativeSurfacePoints
HERE=Path(__file__).parent;OUT=HERE/'runs/turn_foot_coordination_oct10'
@torch.inference_mode()
def main():
    torch.set_num_threads(4);began=time.monotonic();OUT.mkdir(exist_ok=True);free=shutil.disk_usage(HERE).free
    cp=torch.load(HERE/'runs/native_dynamic_scene20_contact_55k_oct07/last.pt',map_location='cpu',weights_only=False);c=cp['config'];model=OfflineSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda().eval();model.load_state_dict(cp['model']);del cp
    data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'])
    demos=json.loads((HERE/'runs/state_relative_spline_oct10/demo/manifest.json').read_text())['cases'];historic=[json.loads(s) for s in (HERE/'runs/native_dynamic_scene20_contact_55k_oct07/full_validation/standard_rows.jsonl').read_text().splitlines()]
    selected=json.loads((HERE/'runs/state_relative_spline_oct10/confirmation/protocol.json').read_text())['selected'];panel=[selected[i] for i in [0,3,12,15,24,27,36,39]]
    protocol={'status':'registered_before_scoring','hypothesis':'small root-yaw path correction while exactly retaining own world ankle/toe trajectories can reduce turning spikes without sliding','source_sha256':hashlib.sha256((HERE/'turn_foot_coordination.py').read_bytes()).hexdigest(),'variants':['original55k','smooth','camera_soft'],'weights':'untouched original55k; no body GT prefix; all128 noise initialization','panel':panel,'panel_seeds':[2026101201,2026101202],'limits':{'root_yaw_deg':15,'generated_head_displacement_m':.02,'camera_relative_weight':.25,'sigma_frames':3,'DP_velocity_weight':10},'scope':'two development demos, eight exposed sequence windows, two seeds; not holdout','acceptance':'foot world preservation20micrometers; slide numerical tolerance1e-5cm/frame, no world/PA or local-joint dynamics regression for adoption; physical collision pending','GT':'scoring only; camera/scene/native template allowed inputs','new_checkpoints':0}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
    members=[{'scope':'demo','index':d['index'],'seed':d['seed'],'window':historic[d['index']]['window']} for d in demos]+[{'scope':'panel','index':i,'seed':seed,'window':m} for seed in protocol['panel_seeds'] for i,m in enumerate(panel)]
    for member in members:
        w=member['window'];sample,identity=data.sample(128,w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);batch={k:v.cuda() for k,v in collate([sample]).items()};original=generate_without_body_initialization(model,batch,seed=member['seed']);skin=NativeSurfacePoints(identity['native_body']);base_soles=skin.soles(skin(original[0])).detach();variants={'original55k':original[0]};audits={}
        for label in ['smooth','camera_soft']:
            value,audit=coordinate_turn(original[0],batch['camera'][0],batch['rest'][0],mode=label);variants[label]=value;audits[label]=audit
        metrics={}
        for label,value in variants.items():
            metrics[label]=measure(value[None],batch,turn_threshold=30);metrics[label].update(leg_motion(value[None]));soles=skin.soles(skin(value));metrics[label]['native_sole_change_max_cm']=float((soles-base_soles).norm(dim=-1).max()*100);metrics[label]['native_sole_velocity_change_max_cm_frame']=float((torch.diff(soles,dim=0)-torch.diff(base_soles,dim=0)).norm(dim=-1).max()*100)
        if member['scope']=='demo':
            # Regenerate from whitelisted input; cached arrays are reference only.
            with np.load(HERE/f"runs/state_relative_spline_oct10/demo/{member['index']}.npz") as a:assert np.max(abs(original[0].cpu().numpy()-a['official55k_motion']))<1e-5
            np.savez_compressed(OUT/f"{member['index']}.npz",**{k:v.cpu().numpy() for k,v in variants.items()})
        rows.append({**member,'identity':identity,'metrics':metrics,'audits':audits});(OUT/'rows.json').write_text(json.dumps(rows,indent=2));print(member['scope'],member['index'],member['seed'],{k:round(v['pelvis_orientation_mean_deg'],2) for k,v in metrics.items()},flush=True)
    result={}
    for label in protocol['variants']:
        rr=[r for r in rows if r['scope']=='panel'];result[label]={k:float(np.mean([r['metrics'][label][k] for r in rr])) for k in ['mpjpe_cm','pa_mpjpe_mm','pelvis_orientation_mean_deg','gt_stance_slide_cm_frame','leg_angular_step_p95_deg','leg_angular_accel_p95_deg_frame2','native_sole_change_max_cm','native_sole_velocity_change_max_cm_frame']}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2));protocol.update(status='completed',elapsed_s=time.monotonic()-began,free_disk_before=free,free_disk_after=shutil.disk_usage(HERE).free);(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
