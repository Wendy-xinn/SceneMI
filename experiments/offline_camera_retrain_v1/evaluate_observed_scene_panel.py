"""Eight identity-selected, already exposed sequences; preliminary robustness only."""
import json,time
from pathlib import Path
import torch
from experiments.offline_camera_retrain_v1.audit_state_plan_comparison import load_models,inputs_only
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.bounded_head_observation import prepare_bounded_observation
from experiments.offline_camera_retrain_v1.body_history_condition import attach_executed_history
from experiments.offline_camera_retrain_v1.executed_prefix_sampling import sample_with_executed_prefix
from experiments.offline_camera_retrain_v1.state_relative_spline_refinement import refine_from_state
from experiments.offline_camera_retrain_v1.observed_body_surfaces import from_identity
from experiments.offline_camera_retrain_v1.native_surface_points import NativeSurfacePoints
from experiments.offline_camera_retrain_v1.observed_scene_refinement import refine_observed_scene,geometry_metrics
from experiments.offline_camera_retrain_v1.evaluate_body_history import measure,crop
HERE=Path(__file__).parent;OUT=HERE/'runs/observed_surface_refine_oct10'
def summarize(rows, **metadata):
    keys=['mpjpe_cm','pa_mpjpe_mm','pelvis_orientation_mean_deg','head_cm','head_orientation_mean_deg','gt_stance_slide_cm_frame','observed_halfspace_p95_depth_cm','observed_halfspace_query_coverage']
    labels=['history_raw','history_spline','scene_contact']
    def average(label, key):
        values=[r['metrics'][label][key] for r in rows if r['metrics'][label][key] is not None]
        return sum(values)/len(values) if values else None
    comparisons={k:[r for r in rows if all(r['metrics'][label][k] is not None for label in ('history_spline','scene_contact'))] for k in ['mpjpe_cm','gt_stance_slide_cm_frame','observed_halfspace_p95_depth_cm']}
    result={'scope':'8 exposed sequences, one seed, preliminary; no holdout/generalization claim','identity_selection':[r['selected_index'] for r in rows],'summary':{label:{k:average(label,k) for k in keys} for label in labels},'wins':{k:sum(r['metrics']['scene_contact'][k]<r['metrics']['history_spline'][k] for r in eligible) for k,eligible in comparisons.items()},'eligible_comparisons':{k:len(eligible) for k,eligible in comparisons.items()},'new_checkpoints':0,**metadata}
    (OUT/'panel_summary.json').write_text(json.dumps(result,indent=2));return result
def main():
    import sys
    if '--summarize-only' in sys.argv:
        print(json.dumps(summarize(json.loads((OUT/'panel_rows.json').read_text()),summary_recovered_from_completed_rows=True),indent=2));return
    torch.set_num_threads(4);started=time.monotonic();models,configs,hashes=load_models();del models['official'];c=configs['history'];members=json.loads((HERE/'runs/state_relative_spline_oct10/confirmation/protocol.json').read_text())['selected'];indices=[0,3,12,15,24,27,36,39];members=[members[i] for i in indices];data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root']);samples=[];identities=[]
    for member in members:
        sample,identity=data.sample(128,member['group'],sequence_index=member['sequence_index'],start_index=member['start_index']);assert identity['sequence_id']==member['sequence_id'];samples.append(sample);identities.append(identity)
    batch={k:v.cuda() for k,v in collate(samples).items()};past=batch['motion'][:,:16].clone();obs=attach_executed_history(inputs_only(prepare_bounded_observation(batch,'joint')),past)
    with torch.inference_mode():
        original=sample_with_executed_prefix(models['history'],obs,past,seed=2026101099,steps=20);spline,_=refine_from_state(original,past,obs,pose_frames=32,iterations=60,relative_root=True)
    del models;rows=[]
    for i,(member,identity) in enumerate(zip(members,identities)):
        one={k:v[i:i+1].detach().clone() for k,v in batch.items()};allowed={k:v[i:i+1].detach().clone() if torch.is_tensor(v) and len(v)==8 else v for k,v in obs.items()};scene,audit=from_identity(identity,frames=48);skin=NativeSurfacePoints(identity['native_body']);old=spline[i:i+1,:48].detach().clone();refined,solver=refine_observed_scene(old,past[i:i+1],allowed,scene,skin)
        variants={'history_raw':original[i:i+1,:48],'history_spline':old,'scene_contact':refined};values={}
        for label,value in variants.items():
            metrics=measure(value[:,16:48],crop(one,16,48),turn_threshold=15);metrics.update(geometry_metrics(value[0,16:48],skin,scene));values[label]=metrics
        reference=geometry_metrics(one['motion'][0,16:48],skin,scene);row={'selected_index':indices[i],'group':member['group'],'identity':identity,'metrics':values,'solver':solver,'scene':audit,'GT_surface_reference':reference};rows.append(row);(OUT/'panel_rows.json').write_text(json.dumps(rows,indent=2));print('panel',i+1,member['group'],{k:round(v['gt_stance_slide_cm_frame'],3) for k,v in values.items()},flush=True)
    print(summarize(rows,weights=hashes['history'],elapsed_s=time.monotonic()-started),flush=True)
if __name__=='__main__':main()
