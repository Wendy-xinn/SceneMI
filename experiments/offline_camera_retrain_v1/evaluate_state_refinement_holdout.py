"""Frozen selected method, reserved sequence panel and synthetic noisy head."""
import hashlib,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.evaluate_body_history import crop,measure,METRICS as BASIC
from experiments.offline_camera_retrain_v1.body_history_condition import HistorySceneMI,attach_executed_history
from experiments.offline_camera_retrain_v1.executed_prefix_sampling import sample_with_executed_prefix
from experiments.offline_camera_retrain_v1.state_relative_refinement import refine_from_state
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.bounded_head_observation import prepare_bounded_observation
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
HERE=Path(__file__).parent;OUT=HERE/'runs/state_relative_refinement_oct10/holdout'
METRICS=BASIC+('head_cm','leg_orientation_mean_deg','leg_global_rotation_step_max_deg','leg_angular_step_p95_deg','leg_angular_accel_p95_deg_frame2')


def leg_motion(motion):
    rot=global_rotations(motion)[:,:,(1,2,4,5,7,8)]
    relative=rot[:,1:]@rot[:,:-1].transpose(-1,-2)
    # Principal SO(3) log in a fixed anchored world frame; no Euler unwrap.
    from scipy.spatial.transform import Rotation
    rv=Rotation.from_matrix(relative.detach().cpu().numpy().reshape(-1,3,3).copy()).as_rotvec().reshape(len(motion),motion.shape[1]-1,6,3)
    return dict(leg_angular_step_p95_deg=float(np.rad2deg(np.quantile(np.linalg.norm(rv,axis=-1),.95))),
                leg_angular_accel_p95_deg_frame2=float(np.rad2deg(np.quantile(np.linalg.norm(np.diff(rv,axis=1),axis=-1),.95))))


def noisy_head(observed,condition):
    out=dict(observed);out['trajectory']=observed['trajectory'].clone();out['observation_meta']=observed['observation_meta'].clone()
    if condition=='clean':return out
    b,t=out['trajectory'].shape[:2];phase=torch.linspace(0,1,t,device='cuda')
    amplitude=.03 if condition=='head_mild' else .08
    yaw=10. if condition=='head_mild' else 30.
    confidence=.5 if condition=='head_mild' else .2
    out['trajectory'][:,:,15,0]+=amplitude*phase/2
    angle=torch.deg2rad(yaw*phase);r=torch.zeros(t,3,3,device='cuda');r[:,0,0]=r[:,2,2]=angle.cos();r[:,0,2]=angle.sin();r[:,2,0]=-angle.sin();r[:,1,1]=1
    columns=out['trajectory'][:,:,15,3:].reshape(b,t,2,3).transpose(-1,-2)
    out['trajectory'][:,:,15,3:]=(r[None]@columns).transpose(-1,-2).reshape(b,t,6)
    out['observation_meta'][:,:,15,2:4]=confidence
    return out


@torch.inference_mode()
def main():
    torch.set_num_threads(4);started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    chosen=json.loads((OUT.parent/'frozen_selection.json').read_text())
    assert hashlib.sha256((HERE/'state_relative_refinement.py').read_bytes()).hexdigest()==chosen['source_sha256']
    holdout=json.loads((HERE/'runs/bounded_head_adaptation_oct10/holdout_manifest.json').read_text());selected=holdout['selected']
    dev=json.loads((HERE/'runs/body_history_replan_oct10/screen_protocol.json').read_text())['selected']
    assert not {r['sequence_id'] for r in selected}&{r['sequence_id'] for r in dev}
    (OUT/'protocol.json').write_text(json.dumps(dict(selected=selected,source_manifest_sha256=hashlib.sha256((HERE/'runs/bounded_head_adaptation_oct10/holdout_manifest.json').read_bytes()).hexdigest(),frozen_method=chosen,conditions=['clean','head_mild','head_large'],noisy_head='synthetic future drift 3cm/10deg confidence.5;8cm/30deg confidence.2; not calibrated estimator',future_scene='known-time recorded inputs retained',seeds_per_window=2,new_training_steps=0,scope='reserved official-validation sequences, not a pristine external test; paired same seed'),indent=2))
    cp=torch.load(HERE/'runs/body_history_replan_oct10/delta_history/last.pt',map_location='cpu',weights_only=False);c=cp['config']
    model=HistorySceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True,conditioning_trial='delta_history').cuda().eval();model.load_state_dict(cp['model']);del cp
    data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'])
    rows=[];native=[];latencies=[]
    with (OUT/'rows.jsonl').open('w') as stream:
        for start in range(0,len(selected),8):
            members=selected[start:start+8];samples=[];identities=[]
            for m in members:
                sample,identity=data.sample(128,m['group'],sequence_index=m['sequence_index'],start_index=m['start_index']);assert identity['sequence_id']==m['sequence_id'] and identity['source_start_30fps']==m['expected_source_start_30fps'];samples.append(sample);identities.append(identity)
            batch={k:v.cuda() for k,v in collate(samples).items()};observed=prepare_bounded_observation(batch,'joint');history=batch['motion'][:,:16]
            for i,identity in enumerate(identities):
                one={k:v[i:i+1] for k,v in batch.items()}
                native.append(dict(index=start+i,identity=identity,short32=leg_motion(one['motion'][:,16:48]),future112=leg_motion(one['motion'][:,16:])))
            for rep in range(2):
                seed=2026101017+start+rep*100000
                for condition in ['clean','head_mild','head_large']:
                    altered=attach_executed_history(noisy_head(observed,condition),history);mask=fixed_control_mask(len(members),128,'head','cuda')
                    torch.cuda.synchronize();begin=time.monotonic()
                    original=sample_with_executed_prefix(model,altered,history,steps=20,seed=seed,control_mask=mask)
                    torch.cuda.synchronize();generated=time.monotonic()
                    refined,audit=refine_from_state(original,history,altered,pose_frames=chosen['pose_frames'],iterations=chosen['iterations'])
                    torch.cuda.synchronize();end=time.monotonic();latencies.append(dict(batch_size=len(members),generation_s=generated-begin,refinement_s=end-generated))
                    for i,identity in enumerate(identities):
                        one={k:v[i:i+1] for k,v in batch.items()};variants={}
                        for label,motion in [('previous',original),('refine32',refined)]:
                            full=measure(motion[i:i+1,16:],crop(one,16),turn_threshold=30);short=measure(motion[i:i+1,16:48],crop(one,16,48),turn_threshold=15)
                            full.update(leg_motion(motion[i:i+1,16:]));short.update(leg_motion(motion[i:i+1,16:48]))
                            previous=altered['executed_history'][i,-1,:,:3]*2;joints=forward_kinematics(motion[i:i+1,16:17],one['rest'])[0,0]
                            for metrics in [short,full]:metrics.update(boundary_root_step_cm=float((joints[0]-previous[0]).norm()*100),boundary_joint_step_cm=float((joints-previous).norm(dim=-1).mean()*100))
                            variants[label]=dict(short32=short,future112=full)
                        row=dict(index=start+i,replicate=rep,seed=seed,condition=condition,group=members[i]['group'],identity=identity,variants=variants);rows.append(row);stream.write(json.dumps(row)+'\n')
            stream.flush();print('holdout evaluated',start+len(members),flush=True)
    summary={};rng=np.random.default_rng(1017)
    for condition in ['clean','head_mild','head_large']:
        summary[condition]={}
        for horizon in ['short32','future112']:
            result={'models':{},'paired':{}};per_label={}
            for label in ['previous','refine32']:
                per_label[label]={};result['models'][label]={}
                for metric in METRICS:
                    groups=defaultdict(list)
                    for row in rows:
                        if row['condition']!=condition:continue
                        value=row['variants'][label][horizon].get(metric)
                        if value is not None:groups[row['group']+'/'+row['identity']['sequence_id']].append(value)
                    per_label[label][metric]={k:float(np.mean(v)) for k,v in groups.items()}
                    result['models'][label][metric]=float(np.mean(list(per_label[label][metric].values()))) if groups else None
            for metric in METRICS:
                a=per_label['refine32'][metric];b=per_label['previous'][metric];keys=sorted(a.keys()&b.keys())
                if not keys:continue
                delta=np.array([a[k]-b[k] for k in keys]);bootstrap=rng.choice(delta,(2000,len(delta))).mean(1)
                result['paired'][metric]=dict(difference=float(delta.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),sequences=len(keys))
            summary[condition][horizon]=result
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'native_leg_motion.json').write_text(json.dumps(native,indent=2));(OUT/'latencies.json').write_text(json.dumps(latencies,indent=2))
    (OUT/'verification.json').write_text(json.dumps(dict(status='completed',windows=len(selected),sequence_roles=len({r['group']+'/'+r['sequence_id'] for r in selected}),new_generation_predictions=len(rows),derived_predictions=len(rows),no_development_sequence_overlap=True,parameters_unchanged_after_selection=True,elapsed_s=time.monotonic()-started,not_simulator_tracking=True),indent=2))

if __name__=='__main__':main()
