"""Kinematic feedback proxy: body GT ONLY initializes history, never refreshes it.

Known head/scene remain oracle diagnostics. No physics, contact tracking, or
new scene-cache reconstruction. Shared original anchor/initial static map.
"""
import json
from pathlib import Path
import torch
from experiments.offline_camera_retrain_v1.audit_state_plan_comparison import load_models,inputs_only
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.bounded_head_observation import prepare_bounded_observation
from experiments.offline_camera_retrain_v1.body_history_condition import attach_executed_history
from experiments.offline_camera_retrain_v1.executed_prefix_sampling import sample_with_executed_prefix
from experiments.offline_camera_retrain_v1.state_relative_planner import generate_state_relative_plan
from experiments.offline_camera_retrain_v1.evaluate_body_history import crop,measure
HERE=Path(__file__).parent;OUT=HERE/'runs/state_relative_spline_oct10/paired_55k_audit'
@torch.inference_mode()
def main():
    torch.set_num_threads(4);models,configs,hashes=load_models();del models['official'];c=configs['history'];selected=json.loads((OUT/'protocol.json').read_text())['selected'];indices=[0,3,12,15,24,27,36,39];members=[selected[i] for i in indices]
    data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root']);samples=[];ids=[]
    for m in members:
        sample,identity=data.sample(128,m['group'],sequence_index=m['sequence_index'],start_index=m['start_index']);assert identity['sequence_id']==m['sequence_id'];samples.append(sample);ids.append(identity)
    batch={k:v.cuda() for k,v in collate(samples).items()};allowed=inputs_only(prepare_bounded_observation(batch,'joint'));rows=[]
    for label in ['history_prefix','history_refined']:
        past=batch['motion'][:,:16].clone();executed=[]
        for step in range(4):
            start=step*8;end=128 if step==0 else start+64
            obs={k:(v[:,start:end] if k in ['trajectory','observation_meta','bps','bps_valid','camera'] else v) for k,v in allowed.items()};seed=2026101088+step
            if label=='history_refined':future=generate_state_relative_plan(models['history'],obs,past,seed=seed,execute_frames=8)['execute_motion']
            else:future=sample_with_executed_prefix(models['history'],attach_executed_history(obs,past),past,seed=seed,steps=20)[:,16:24]
            executed.append(future);past=torch.cat((past[:,8:],future),dim=1)
            # No target/body/contact truth is referenced during feedback update.
        combined=torch.cat(executed,dim=1)
        for i,identity in enumerate(ids):
            one={k:v[i:i+1] for k,v in batch.items()};rows.append({'label':label,'identity':identity,'group':members[i]['group'],'metrics':measure(combined[i:i+1],crop(one,16,48),turn_threshold=15)})
    result={'rows':rows,'selected_indices':indices,'four_replans_execute8_frames_each':True,'total_executed_s':1.6,'initial_history_s':.8,'body_GT_refresh_after_initialization':False,'feedback':'previous generated executed chunk; exact kinematic execution proxy, NOT tracked simulator state','known_head_scene':'GT simulated anatomical head and known-time BPS; original anchored frame and initial static occupancy retained','no_physics_or_collision_verified':True,'new_checkpoints':0,'models':{}}
    for label in ['history_prefix','history_refined']:
        r=[x for x in rows if x['label']==label];result['models'][label]={k:sum(x['metrics'][k] for x in r)/len(r) for k in ['mpjpe_cm','pa_mpjpe_mm','pelvis_orientation_mean_deg','head_cm','gt_stance_slide_cm_frame']}
    (OUT/'kinematic_feedback_proxy.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['models']))
if __name__=='__main__':main()
