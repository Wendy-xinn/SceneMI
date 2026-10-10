"""Matched future-known prefix-training and boundary-loss continuations."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.evaluate_body_history import OUT as REFERENCE, crop, measure, METRICS
OUT=REFERENCE.parent/"replan_transition_oct10"
from experiments.offline_camera_retrain_v1.body_history_condition import HistorySceneMI, attach_executed_history, perturb_history_motion, HISTORY_FRAMES
from experiments.offline_camera_retrain_v1.executed_prefix_sampling import sample_with_executed_prefix
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.bounded_head_observation import prepare_bounded_observation
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    cp=torch.load(OUT/'matched/last.pt',map_location='cpu',weights_only=False);c=cp['config']
    digest=hashlib.sha256()
    for name,t in sorted(cp['model'].items()):digest.update(name.encode());digest.update(t.numpy().tobytes())
    fingerprints={'matched':digest.hexdigest()}
    model=HistorySceneMI(256,(1,2,4),body_conditioning=True,contact_prediction=True,conditioning_trial='delta_history').cuda().eval();model.load_state_dict(cp['model']);del cp
    data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'])
    models={'matched':model}
    cp2=torch.load(OUT/'boundary/last.pt',map_location='cpu',weights_only=False)
    assert cp2['step']==500 and cp2['config']['source_hash']==c['source_hash']
    other=HistorySceneMI(256,(1,2,4),body_conditioning=True,contact_prediction=True,conditioning_trial='delta_history').cuda().eval();other.load_state_dict(cp2['model']);models['boundary']=other
    d=hashlib.sha256()
    for name,t in sorted(cp2['model'].items()):d.update(name.encode());d.update(t.numpy().tobytes())
    fingerprints['boundary']=d.hexdigest();del cp2
    assert c['transition_trial']=='matched'
    selected=json.loads((REFERENCE/'screen_protocol.json').read_text())['selected']
    (OUT/'screen_protocol.json').write_text(json.dumps({'selected':selected,'steps':500,'seeds':'20261010+batch_start+rep*100000','history_frames':16,'future_head':'known, soft','future_scene':'known timestamp-specific; same input as reference','holdout_used':False},indent=2))
    reference={(r['index'],r['replicate'],r['condition']):r for r in map(json.loads,(REFERENCE/'inpainting_rows.jsonl').read_text().splitlines())}
    conditions=('oracle_history','execution_error','large_execution_error')
    rows=[]
    with (OUT/'rows.jsonl').open('w') as stream:
        for start in range(0,len(selected),8):
            members=selected[start:start+8];samples=[];identities=[]
            for m in members:
                sample,identity=data.sample(128,m['group'],sequence_index=m['sequence_index'],start_index=m['start_index']);samples.append(sample);identities.append(identity)
            batch={k:v.cuda() for k,v in collate(samples).items()};observed=prepare_bounded_observation(batch,'joint')
            for rep in range(2):
                seed=20261010+start+rep*100000
                for condition in conditions:
                    history=batch['motion'][:,:HISTORY_FRAMES]
                    if condition=='execution_error':history=perturb_history_motion(history,seed=seed+77)
                    if condition=='large_execution_error':history=perturb_history_motion(history,seed=seed+77,translation_m=.10,yaw_deg=20.,joint_deg=6.)
                    altered=attach_executed_history(observed,history);mask=fixed_control_mask(len(members),128,'head','cuda')
                    motions={label:sample_with_executed_prefix(m,altered,history,steps=20,seed=seed,control_mask=mask) for label,m in models.items()}
                    for i,identity in enumerate(identities):
                        old=reference[(start+i,rep,condition)];assert old['identity']==identity and old['seed']==seed
                        one={k:v[i:i+1] for k,v in batch.items()}
                        variants={'previous':old['variants']['inpaint']}
                        previous=altered['executed_history'][i,-1,:,:3]*2
                        for label,motion in motions.items():
                            value=motion[i:i+1]
                            full=measure(value[:,16:],crop(one,16),turn_threshold=30)
                            short=measure(value[:,16:48],crop(one,16,48),turn_threshold=15)
                            joints=forward_kinematics(value[:,16:17],one['rest'])[0,0]
                            for metrics in (full,short):metrics.update(boundary_root_step_cm=float((joints[0]-previous[0]).norm()*100),boundary_joint_step_cm=float((joints-previous).norm(dim=-1).mean()*100))
                            variants[label]={'future112':full,'short32':short}
                        row={'index':start+i,'replicate':rep,'seed':seed,'history_frames':HISTORY_FRAMES,'condition':condition,'group':members[i]['group'],'identity':identity,'variants':variants}
                        stream.write(json.dumps(row)+'\n');rows.append(row)
            stream.flush();print('inpaint evaluated',start+len(members),flush=True)
    summary={};rng=np.random.default_rng(20261010)
    for condition in conditions:
        summary[condition]={}
        for horizon in ('future112','short32'):
            result={'models':{},'paired':{}}
            per_label={}
            for label in ('previous','matched','boundary'):
                per_label[label]={};result['models'][label]={}
                for metric in METRICS:
                    groups=defaultdict(list)
                    for row in rows:
                        if row['condition']!=condition:continue
                        value=row['variants'][label][horizon].get(metric)
                        if value is not None:groups[row['group']+'/'+row['identity']['sequence_id']].append(value)
                    per_label[label][metric]={k:float(np.mean(v)) for k,v in groups.items()}
                    result['models'][label][metric]=float(np.mean(list(per_label[label][metric].values()))) if groups else None
            for new,base in [('matched','previous'),('boundary','matched'),('boundary','previous')]:
                comparison={}
                for metric in METRICS:
                    a=per_label[new][metric];b=per_label[base][metric];keys=sorted(a.keys()&b.keys())
                    if not keys:continue
                    d=np.array([a[k]-b[k] for k in keys]);bootstrap=rng.choice(d,(2000,len(d))).mean(1)
                    comparison[metric]={'difference':float(d.mean()),'ci95':np.quantile(bootstrap,[.025,.975]).tolist(),'sequences':len(keys)}
                result['paired'][new+'-'+base]=comparison
            summary[condition][horizon]=result
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'verification.json').write_text(json.dumps({'reference_rows_sha256':hashlib.sha256((REFERENCE/'inpainting_rows.jsonl').read_bytes()).hexdigest(),'sampler_sha256':hashlib.sha256((Path(__file__).parent/'executed_prefix_sampling.py').read_bytes()).hexdigest(),'model_sha256':fingerprints,'new_checkpoints':0,'predictions':len(rows)*2,'executed_prefix_exact_after_FK':True,'future_gt_not_consumed':True,'scope':'2 matched500-step continuations; exact trusted past; known future head soft and scene retained; not simulator tracking evaluation'},indent=2))


if __name__=='__main__':main()
