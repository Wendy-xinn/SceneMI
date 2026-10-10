"""Original 55k camera-conditioning reference, separated from matched head control."""
import json
from pathlib import Path
import torch
from experiments.offline_camera_retrain_v1.audit_state_plan_comparison import load_models,inputs_only,summarize,HORIZONS
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.typed_head_condition import prepare_observation
from experiments.offline_camera_retrain_v1.scene_model import ddim_sample
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.evaluate_body_history import crop,measure
from experiments.offline_camera_retrain_v1.evaluate_state_spline_confirmation import leg_motion
HERE=Path(__file__).parent;OUT=HERE/'runs/state_relative_spline_oct10/paired_55k_audit'
@torch.inference_mode()
def main():
    torch.set_num_threads(4);models,configs,_=load_models();del models['history'];c=configs['history'];selected=json.loads((OUT/'protocol.json').read_text())['selected'];data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root']);rows=[]
    for start in range(0,len(selected),8):
        members=selected[start:start+8];samples=[];identities=[]
        for m in members:
            sample,identity=data.sample(128,m['group'],sequence_index=m['sequence_index'],start_index=m['start_index']);assert identity['sequence_id']==m['sequence_id'];samples.append(sample);identities.append(identity)
        batch={k:v.cuda() for k,v in collate(samples).items()};obs=inputs_only(prepare_observation(batch,'camera'))
        for rep in range(2):
            seed=2026101031+start+rep*100000;motion=ddim_sample(models['official'],obs,128,steps=20,seed=seed,control_mask=fixed_control_mask(len(members),128,'head','cuda'))
            for i,identity in enumerate(identities):
                one={k:v[i:i+1] for k,v in batch.items()};metrics={}
                for h,(begin,end) in HORIZONS.items():
                    values=measure(motion[i:i+1,begin:end],crop(one,begin,end),turn_threshold=30 if h=='future112' else 15);values.update(leg_motion(motion[i:i+1,begin:end]));metrics[h]=values
                rows.append({'index':start+i,'replicate':rep,'seed':seed,'condition':'clean','group':members[i]['group'],'identity':identity,'variants':{'official55k_camera':metrics}})
    (OUT/'camera55k_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));groups={};summary={}
    for h in HORIZONS:
        summary[h]={}
        for metric in next(iter(rows))['variants']['official55k_camera'][h]:
            from collections import defaultdict
            seq=defaultdict(list)
            for r in rows:
                value=r['variants']['official55k_camera'][h][metric]
                if value is not None:seq[r['group']+'/'+r['identity']['sequence_id']].append(value)
            if seq:summary[h][metric]=sum(sum(v)/len(v) for v in seq.values())/len(seq)
    (OUT/'camera55k_summary.json').write_text(json.dumps({'models':summary,'conditioning':'original data trajectory15=camera; original55k weights/standard DDIM20; matched seed and scenes; camera/head semantics differ from five-way controlled table','windows':48,'draws':96},indent=2));print(json.dumps(summary['short32']))
if __name__=='__main__':main()
