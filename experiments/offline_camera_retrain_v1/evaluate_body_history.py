"""Future-only, matched-seed history/feedback probes; no motion-cache copies."""
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.body_history_condition import (
    HistorySceneMI, attach_executed_history, perturb_history_motion, HISTORY_FRAMES)
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.bounded_head_observation import prepare_bounded_observation
from experiments.offline_camera_retrain_v1.scene_model import ddim_sample
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.evaluate_turn_repair import diagnostics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.train_body_history import TIME_KEYS

HERE=Path(__file__).parent
OUT=HERE/'runs/body_history_replan_oct10'
LABELS=('baseline','delta','delta_history')
METRICS=('mpjpe_cm','pa_mpjpe_mm','pelvis_orientation_mean_deg','head_orientation_mean_deg',
         'gt_stance_slide_cm_frame','support_floating_m','support_penetration_m','opposite_turn',
         'under_turn','meaningful_opposite_turn','boundary_root_step_cm','boundary_joint_step_cm')


def crop(batch, start, end=None):
    return {k:(v[:,start:end] if k in TIME_KEYS else v) for k,v in batch.items()}


def measure(motion, batch, *, turn_threshold):
    result=diagnostics(motion,batch['motion'],batch)
    forward=global_rotations(batch['motion'])[0,:,0,:,2]
    valid=float((forward[:,[0,2]].norm(dim=-1)>.3).float().mean())>=.95
    gt=result['pelvis_gt_turn_deg']; turn=result['pelvis_turn_deg']
    eligible=valid and abs(gt)>=turn_threshold
    result['opposite_turn']=float(gt*turn<0) if eligible else None
    result['meaningful_opposite_turn']=float(gt*turn<0 and abs(turn)>=15) if eligible else None
    result['under_turn']=float(abs(turn)<.5*abs(gt)) if eligible else None
    return result


def seq_values(rows,label,metric,horizon):
    values=defaultdict(list)
    for row in rows:
        value=row['variants'][label][horizon].get(metric)
        if value is not None:values[row['group']+'/'+row['identity']['sequence_id']].append(value)
    return {k:float(np.mean(v)) for k,v in values.items()}


def summarize(rows):
    output={};rng=np.random.default_rng(20261010)
    for condition in sorted({r['condition'] for r in rows}):
        members=[r for r in rows if r['condition']==condition]
        output[condition]={}
        for horizon in ('future112','short32'):
            result={'windows':len({r['index'] for r in members}),'sequences':len({r['group']+'/'+r['identity']['sequence_id'] for r in members}),'models':{},'paired_differences':{}}
            for label in LABELS:
                result['models'][label]={m:float(np.mean(list(values.values()))) if values else None
                    for m in METRICS for values in [seq_values(members,label,m,horizon)]}
            for new,reference in [('delta','baseline'),('delta_history','delta')]:
                paired={}
                for metric in METRICS:
                    a=seq_values(members,new,metric,horizon);b=seq_values(members,reference,metric,horizon)
                    keys=sorted(a.keys()&b.keys())
                    if not keys:continue
                    differences=np.array([a[k]-b[k] for k in keys])
                    bootstrap=np.mean(rng.choice(differences,(2000,len(differences))),axis=1)
                    paired[metric]={'difference':float(differences.mean()),'ci95':np.quantile(bootstrap,[.025,.975]).tolist(),'sequences':len(keys)}
                result['paired_differences'][new+'-'+reference]=paired
            output[condition][horizon]=result
    (OUT/'summary.json').write_text(json.dumps(output,indent=2))
    return output


@torch.inference_mode()
def main():
    torch.set_num_threads(4);started=time.monotonic()
    protocol=json.loads((OUT/'screen_protocol.json').read_text()); selected=protocol['selected']
    conditions=['head_only','oracle_history','execution_error','large_execution_error','prior_generated_history','history_only_forecast']
    models={};configs={};fingerprints={}
    for label in LABELS:
        cp=torch.load(OUT/label/'last.pt',map_location='cpu',weights_only=False)
        c=cp['config'];assert cp['step']==protocol['steps'] and c['conditioning_trial']==label
        model=HistorySceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True,conditioning_trial=label).cuda().eval()
        model.load_state_dict(cp['model']);models[label]=model;configs[label]=c
        digest=hashlib.sha256()
        for name,t in sorted(cp['model'].items()):digest.update(name.encode());digest.update(t.numpy().tobytes())
        fingerprints[label]=digest.hexdigest();del cp
    assert len({c['source_hash'] for c in configs.values()})==1
    c=configs['baseline']
    data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'])
    rows=[];prediction_count=0
    with (OUT/'rows.jsonl').open('w') as stream:
        for start in range(0,len(selected),8):
            members=selected[start:start+8];samples=[];identities=[]
            for member in members:
                sample,identity=data.sample(128,member['group'],sequence_index=member['sequence_index'],start_index=member['start_index'])
                assert identity['sequence_id']==member['sequence_id']
                samples.append(sample);identities.append(identity)
            batch={k:v.cuda() for k,v in collate(samples).items()}
            observed=prepare_bounded_observation(batch,'joint')
            mask=fixed_control_mask(len(members),128,'head','cuda')
            for rep in range(2):
                seed=20261010+start+rep*100000
                base={label:ddim_sample(model,observed,128,steps=20,seed=seed,control_mask=mask) for label,model in models.items()}
                prediction_count+=len(members)*3
                for condition in conditions:
                    altered=observed
                    if condition!='head_only':
                        history=batch['motion'][:,:HISTORY_FRAMES]
                        if condition=='execution_error':history=perturb_history_motion(history,seed=seed+77)
                        if condition=='large_execution_error':history=perturb_history_motion(history,seed=seed+77,translation_m=.10,yaw_deg=20.,joint_deg=6.)
                        if condition=='prior_generated_history':history=base['baseline'][:,:HISTORY_FRAMES]
                        altered=attach_executed_history(observed,history)
                    motions=dict(base)
                    if condition=='history_only_forecast':
                        # Window-wise time-causal BPS still contains later camera
                        # queries/observations. Replanning may use only the past.
                        altered=dict(altered)
                        for key in ('bps','bps_valid'):
                            if key in altered:
                                snapshot=altered[key].clone()
                                snapshot[:,HISTORY_FRAMES:]=snapshot[:,HISTORY_FRAMES-1:HISTORY_FRAMES]
                                altered[key]=snapshot
                        future_unknown=mask.clone();future_unknown[:,HISTORY_FRAMES:]=False
                        motions={label:ddim_sample(model,altered,128,steps=20,seed=seed,control_mask=future_unknown) for label,model in models.items()}
                        prediction_count+=len(members)*3
                    elif condition!='head_only':
                        motions['delta_history']=ddim_sample(models['delta_history'],altered,128,steps=20,seed=seed,control_mask=mask)
                        prediction_count+=len(members)
                    for i,identity in enumerate(identities):
                        one={k:(v[i:i+1] if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
                        future=crop(one,HISTORY_FRAMES);short=crop(one,HISTORY_FRAMES,HISTORY_FRAMES+32)
                        variants={}
                        for label,motion in motions.items():
                            value=motion[i:i+1]; full=measure(value[:,HISTORY_FRAMES:],future,turn_threshold=30)
                            small=measure(value[:,HISTORY_FRAMES:HISTORY_FRAMES+32],short,turn_threshold=15)
                            if condition!='head_only':
                                previous=altered['executed_history'][i,-1,:,:3]*2
                                next_joints=forward_kinematics(value[:,HISTORY_FRAMES:HISTORY_FRAMES+1],one['rest'])[0,0]
                                rootstep=float((next_joints[0]-previous[0]).norm()*100)
                                jointstep=float((next_joints-previous).norm(dim=-1).mean()*100)
                                for metrics in (full,small):metrics.update(boundary_root_step_cm=rootstep,boundary_joint_step_cm=jointstep)
                            variants[label]={'future112':full,'short32':small}
                        row={'index':start+i,'replicate':rep,'seed':seed,'identity':identity,'group':members[i]['group'],'condition':condition,'variants':variants}
                        stream.write(json.dumps(row)+'\n');rows.append(row)
            stream.flush()
            print('evaluated',start+len(members),flush=True)
            (OUT/'evaluation_status.json').write_text(json.dumps({'status':'running','windows':start+len(members),'prediction_count':prediction_count}))
    summarize(rows)
    traces={label:[json.loads(x) for x in (OUT/label/'sample_trace.jsonl').read_text().splitlines()] for label in LABELS}
    assert all(len(t)==500 for t in traces.values())
    for label in LABELS[1:]:
        assert all(a['pairing']==b['pairing'] and a['samples']==b['samples'] for a,b in zip(traces['baseline'],traces[label]))
    verification={'sample_noise_timestep_head_mask_scene_exact_pairing':True,'model_sha256':fingerprints,
                  'windows':len(selected),'replicates':2,'conditions':conditions,'unique_predictions':prediction_count,
                  'paired_rows':len(rows),'metric_scope':'exclude16 history; future112 and first32 forecast frames',
                  'future_causal_probe':'history_only_forecast masks future head and repeats only t15 BPS/valid; initial occupancy is already past-only; no future camera queries or observed surfaces',
                  'limitations':'synthetic execution errors and generated-prefix feedback; no simulator rollout or action-tracking success measured; oracle past history is diagnostic only'}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2))
    (OUT/'evaluation_status.json').write_text(json.dumps({'status':'completed','elapsed_s':time.monotonic()-started,**verification},indent=2))


if __name__=='__main__':main()
