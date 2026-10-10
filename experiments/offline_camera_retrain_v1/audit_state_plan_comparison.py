"""Paired 55k/history/refinement controls and real-input GT mutation audit.

Frozen method; previously exposed confirmation panel, not a new holdout.
No weights or per-window motion archives are created.
"""
import hashlib,json,time,shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI,ddim_sample
from experiments.offline_camera_retrain_v1.body_history_condition import HistorySceneMI,attach_executed_history,perturb_history_motion
from experiments.offline_camera_retrain_v1.executed_prefix_sampling import sample_with_executed_prefix
from experiments.offline_camera_retrain_v1.state_relative_spline_refinement import refine_from_state
from experiments.offline_camera_retrain_v1.state_relative_planner import generate_state_relative_plan,INPUT_KEYS
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.bounded_head_observation import prepare_bounded_observation
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.evaluate_body_history import crop,measure
from experiments.offline_camera_retrain_v1.evaluate_state_spline_confirmation import noisy_head,leg_motion
HERE=Path(__file__).parent;ROOT=HERE/'runs/state_relative_spline_oct10';OUT=ROOT/'paired_55k_audit'
LABELS=('official55k','official55k_prefix','history_prefix','official55k_refined','history_refined')
HORIZONS={'short32':(16,48),'after_initial_0p4s':(24,48),'future112':(16,128)}
METRICS=('mpjpe_cm','pa_mpjpe_mm','pelvis_orientation_mean_deg','head_orientation_mean_deg','head_cm','gt_stance_slide_cm_frame','support_floating_m','support_penetration_m','meaningful_opposite_turn','under_turn','leg_angular_step_p95_deg','leg_angular_accel_p95_deg_frame2')
CONDITIONS=('clean','head_large','biased_history')

def load_models():
    result={};configs={};hashes={}
    for label,path,cls,extra in [('official',HERE/'runs/native_dynamic_scene20_contact_55k_oct07/last.pt',OfflineSceneMI,{}),('history',HERE/'runs/body_history_replan_oct10/delta_history/last.pt',HistorySceneMI,{'conditioning_trial':'delta_history'})]:
        cp=torch.load(path,map_location='cpu',weights_only=False);c=cp['config'];m=cls(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True,**extra).cuda().eval();m.load_state_dict(cp['model']);result[label]=m;configs[label]=c
        digest=hashlib.sha256()
        for k,v in sorted(cp['model'].items()):digest.update(k.encode());digest.update(v.numpy().tobytes())
        hashes[label]={'path':str(path),'model_sha256':digest.hexdigest(),'step':cp['step']};del cp
    return result,configs,hashes

def inputs_only(observed):
    x={k:observed[k] for k in INPUT_KEYS if k in observed};x['trajectory']=x['trajectory'].clone();x['observation_meta']=x['observation_meta'].clone();other=[j for j in range(22) if j!=15];x['trajectory'][:,:,other]=0;x['observation_meta'][:,:,other]=0
    return x

def summarize(rows,selected):
    result={};rng=np.random.default_rng(20261011)
    for condition in CONDITIONS:
        result[condition]={}
        for horizon in HORIZONS:
            x={'models':{},'paired':{},'per_group':{}}
            for label in LABELS:
                x['models'][label]={}
                for metric in METRICS:
                    grouped=defaultdict(list)
                    for row in rows:
                        if row['condition']!=condition:continue
                        value=row['variants'][label][horizon].get(metric)
                        if value is not None:grouped[row['group']+'/'+row['identity']['sequence_id']].append(value)
                    x['models'][label][metric]=float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None
                for group in sorted({r['group'] for r in rows}):
                    x['per_group'].setdefault(group,{})[label]={}
                    for metric in METRICS:
                        grouped=defaultdict(list)
                        for row in rows:
                            if row['condition']==condition and row['group']==group:
                                v=row['variants'][label][horizon].get(metric)
                                if v is not None:grouped[row['identity']['sequence_id']].append(v)
                        x['per_group'][group][label][metric]=float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None
            for a,b in [('official55k_prefix','official55k'),('history_prefix','official55k_prefix'),('official55k_refined','official55k_prefix'),('history_refined','history_prefix'),('history_refined','official55k')]:
                pair={}
                for metric in METRICS:
                    sequence=defaultdict(list);records={}
                    for row in rows:
                        if row['condition']!=condition:continue
                        av=row['variants'][a][horizon].get(metric);bv=row['variants'][b][horizon].get(metric)
                        if av is None or bv is None:continue
                        key=row['group']+'/'+row['identity']['sequence_id'];sequence[key].append(av-bv);records[key]=selected[row['index']]['holdout_recording']
                    keys=sorted(sequence)
                    if not keys:continue
                    values=np.array([np.mean(sequence[k]) for k in keys]);clusters=sorted(set(records.values()));boot=[]
                    for draw in rng.integers(0,len(clusters),size=(2000,len(clusters))):
                        picked=[v for j in draw for k,v in zip(keys,values) if records[k]==clusters[j]];boot.append(np.mean(picked))
                    pair[metric]={'difference':float(values.mean()),'recording_cluster_ci95':np.quantile(boot,[.025,.975]).tolist(),'sequence_roles':len(keys),'recordings':len(clusters),'improved_sequence_roles':int((values<0).sum()),'sequence_differences':dict(zip(keys,values.tolist()))}
                x['paired'][a+'-'+b]=pair
            result[condition][horizon]=x
    return result

@torch.inference_mode()
def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True);started=time.monotonic();disk_before=shutil.disk_usage(HERE).free
    chosen=json.loads((ROOT/'frozen_selection.json').read_text());assert hashlib.sha256((HERE/'state_relative_spline_refinement.py').read_bytes()).hexdigest()==chosen['source_sha256']
    selected=json.loads((ROOT/'confirmation/protocol.json').read_text())['selected'];models,configs,hashes=load_models();c=configs['history']
    data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'])
    protocol={'selected':selected,'labels':LABELS,'conditions':CONDITIONS,'weights':hashes,'frozen_solver':chosen,'scope':'previously exposed 48-window confirmation panel; matched input/seed comparison, no new holdout claim','head':'anatomical GT-derived ideal-input diagnostic; noisy synthetic drift for head_large; not real-video estimator','history':'GT first16 simulator-state proxy; biased_history applies coherent3cm/8deg/3deg bias, not physics','allowed_known_future':'head position/orientation and scene arrays per original recorded time; future body/contact excluded','horizons':HORIZONS,'seeds':'2026101031+batch_start+replicate*100000; B8; two replicates','data_source_hash':c['source_hash'],'no_training_or_checkpoints':True};(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];audit_rows=[]
    with (OUT/'rows.jsonl').open('w') as stream:
        for start in range(0,len(selected),8):
            members=selected[start:start+8];samples=[];identities=[]
            for member in members:
                sample,identity=data.sample(128,member['group'],sequence_index=member['sequence_index'],start_index=member['start_index']);assert identity['sequence_id']==member['sequence_id'] and identity['source_start_30fps']==member['expected_source_start_30fps'];samples.append(sample);identities.append(identity)
            batch={k:v.cuda() for k,v in collate(samples).items()};observed=prepare_bounded_observation(batch,'joint');history=batch['motion'][:,:16].clone();mask=fixed_control_mask(len(members),128,'head','cuda')
            for rep in range(2):
                seed=2026101031+start+rep*100000
                for condition in CONDITIONS:
                    inputs=inputs_only(noisy_head(observed,'head_large' if condition=='head_large' else 'clean'))
                    past=perturb_history_motion(history,seed=seed+77) if condition=='biased_history' else history
                    altered=attach_executed_history(inputs,past)
                    variants={'official55k':ddim_sample(models['official'],inputs,128,steps=20,seed=seed,control_mask=mask)}
                    variants['official55k_prefix']=sample_with_executed_prefix(models['official'],altered,past,steps=20,seed=seed,control_mask=mask)
                    variants['history_prefix']=sample_with_executed_prefix(models['history'],altered,past,steps=20,seed=seed,control_mask=mask)
                    for base,label in [('official55k_prefix','official55k_refined'),('history_prefix','history_refined')]:
                        variants[label],audit=refine_from_state(variants[base],past,altered,pose_frames=32,iterations=60,relative_root=True)
                    for i,identity in enumerate(identities):
                        one={k:v[i:i+1] for k,v in batch.items()};metrics={}
                        for label,motion in variants.items():
                            metrics[label]={}
                            for horizon,(begin,end) in HORIZONS.items():
                                value=measure(motion[i:i+1,begin:end],crop(one,begin,end),turn_threshold=30 if horizon=='future112' else 15);value.update(leg_motion(motion[i:i+1,begin:end]));metrics[label][horizon]=value
                        row={'index':start+i,'replicate':rep,'seed':seed,'condition':condition,'group':members[i]['group'],'identity':identity,'variants':metrics};rows.append(row);stream.write(json.dumps(row)+'\n')
            # Real tensors, four-dataset role coverage. Poison future GT AFTER allowed inputs freeze.
            if start in (0,8,16,24,32,40):
                base_inputs=inputs_only(observed);reference=generate_state_relative_plan(models['history'],base_inputs,history,seed=987+start)
                poisoned={k:v.clone() for k,v in observed.items()}
                for key in ['motion','joints','contact_target','contact_valid','contact_labels']:
                    if key in poisoned:poisoned[key][:,16:]=1234
                other=[j for j in range(22) if j!=15];poisoned['trajectory'][:,16:,other]=1234;poisoned['observation_meta'][:,16:,other]=1
                changed=generate_state_relative_plan(models['history'],poisoned,history,seed=987+start)
                assert torch.equal(reference['full_motion'],changed['full_motion']),'GT mutation changed API result'
                assert torch.equal(reference['original_motion'],changed['original_motion']),'GT mutation changed sampler'
                audit_rows.append({'batch_start':start,'groups':sorted({m['group'] for m in members}),'windows':len(members),'future_GT_fields_mutated':[k for k in ['motion','joints','contact_target','contact_valid','contact_labels'] if k in poisoned],'nonhead_future_tracks_poisoned':True,'head_scene_body_parameters_and_actual_history_fixed':True,'sampler_max_difference':float((reference['original_motion']-changed['original_motion']).abs().max()),'final_max_difference':float((reference['full_motion']-changed['full_motion']).abs().max())})
            stream.flush();print('paired evaluated',start+len(members),flush=True)
    (OUT/'summary.json').write_text(json.dumps(summarize(rows,selected),indent=2));(OUT/'GT_mutation_audit.json').write_text(json.dumps({'status':'passed','rows':audit_rows,'claim':'No future body/contact/nonhead-track influence when allowed head/scene/rest/history remain fixed. Does not prove those allowed observations are available in deployment.'},indent=2))
    (OUT/'verification.json').write_text(json.dumps({'status':'completed','windows':len(selected),'paired_rows':len(rows),'prediction_variants':len(rows)*len(LABELS),'elapsed_s':time.monotonic()-started,'disk_free_before':disk_before,'disk_free_after':shutil.disk_usage(HERE).free,'new_checkpoints':0},indent=2));print('completed',flush=True)

if __name__=='__main__':main()
