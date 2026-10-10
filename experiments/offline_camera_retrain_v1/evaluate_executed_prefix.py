"""Compare trusted-history inpainting with the SAME completed soft-history model."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.evaluate_body_history import OUT, crop, measure, METRICS
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
    cp=torch.load(OUT/'delta_history/last.pt',map_location='cpu',weights_only=False);c=cp['config']
    digest=hashlib.sha256()
    for name,t in sorted(cp['model'].items()):digest.update(name.encode());digest.update(t.numpy().tobytes())
    assert digest.hexdigest()==json.loads((OUT/'verification.json').read_text())['model_sha256']['delta_history']
    model=HistorySceneMI(256,(1,2,4),body_conditioning=True,contact_prediction=True,conditioning_trial='delta_history').cuda().eval();model.load_state_dict(cp['model']);del cp
    data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'])
    selected=json.loads((OUT/'screen_protocol.json').read_text())['selected']
    reference={(r['index'],r['replicate'],r['condition']):r for r in map(json.loads,(OUT/'rows.jsonl').read_text().splitlines())}
    conditions=('oracle_history','execution_error','large_execution_error','history_only_forecast')
    rows=[]
    with (OUT/'inpainting_rows.jsonl').open('w') as stream:
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
                    if condition=='history_only_forecast':
                        mask[:,HISTORY_FRAMES:]=False
                        for key in ('bps','bps_valid'):
                            altered[key]=altered[key].clone();altered[key][:,HISTORY_FRAMES:]=altered[key][:,HISTORY_FRAMES-1:HISTORY_FRAMES]
                    motion=sample_with_executed_prefix(model,altered,history,steps=20,seed=seed,control_mask=mask)
                    for i,identity in enumerate(identities):
                        old=reference[(start+i,rep,condition)];assert old['identity']==identity and old['seed']==seed
                        one={k:v[i:i+1] for k,v in batch.items()};value=motion[i:i+1]
                        full=measure(value[:,16:],crop(one,16),turn_threshold=30)
                        short=measure(value[:,16:48],crop(one,16,48),turn_threshold=15)
                        previous=altered['executed_history'][i,-1,:,:3]*2
                        joints=forward_kinematics(value[:,16:17],one['rest'])[0,0]
                        for metrics in (full,short):metrics.update(boundary_root_step_cm=float((joints[0]-previous[0]).norm()*100),boundary_joint_step_cm=float((joints-previous).norm(dim=-1).mean()*100))
                        row={'index':start+i,'replicate':rep,'seed':seed,'history_frames':HISTORY_FRAMES,'condition':condition,'group':members[i]['group'],'identity':identity,
                             'variants':{'soft':old['variants']['delta_history'],'inpaint':{'future112':full,'short32':short}}}
                        stream.write(json.dumps(row)+'\n');rows.append(row)
            stream.flush();print('inpaint evaluated',start+len(members),flush=True)
    summary={};rng=np.random.default_rng(20261010)
    for condition in conditions:
        summary[condition]={}
        for horizon in ('future112','short32'):
            result={'models':{},'paired':{}}
            per_label={}
            for label in ('soft','inpaint'):
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
                a=per_label['inpaint'][metric];b=per_label['soft'][metric];keys=sorted(a.keys()&b.keys())
                if not keys:continue
                d=np.array([a[k]-b[k] for k in keys]);bootstrap=rng.choice(d,(2000,len(d))).mean(1)
                result['paired'][metric]={'difference':float(d.mean()),'ci95':np.quantile(bootstrap,[.025,.975]).tolist(),'sequences':len(keys)}
            summary[condition][horizon]=result
    (OUT/'inpainting_summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'inpainting_verification.json').write_text(json.dumps({'model_sha256':digest.hexdigest(),'new_checkpoints':0,'predictions':len(rows),'executed_prefix_exact_after_FK':True,'future_gt_not_consumed':True,'scope':'same500-step model; exact known past body only; future head stays soft or absent; not simulator tracking evaluation'},indent=2))


if __name__=='__main__':main()
