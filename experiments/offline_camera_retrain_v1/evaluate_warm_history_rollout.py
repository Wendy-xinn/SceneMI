"""Follow-up to independent-noise rolling failures: reuse old prediction only."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.evaluate_long_history_rollout import HERE,OUT,TIME_KEYS,long_samples,evaluate
from experiments.offline_camera_retrain_v1.audit_state_plan_comparison import load_models,inputs_only
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.bounded_head_observation import prepare_bounded_observation
from experiments.offline_camera_retrain_v1.body_history_condition import attach_executed_history
from experiments.offline_camera_retrain_v1.executed_prefix_sampling import sample_with_executed_prefix
from experiments.offline_camera_retrain_v1.warm_history_sampling import shift_prior,sample_warm_prefix
from experiments.offline_camera_retrain_v1.evaluate_body_history import crop

@torch.inference_mode()
def main():
    torch.set_num_threads(4);began=time.monotonic();models,configs,hashes=load_models();del models['official'];model=models['history'];c=configs['history']
    data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'])
    samples,identities,_=long_samples(data);original=json.loads((OUT/'protocol.json').read_text())
    assert identities==original['identities']
    protocol={'status':'registered_before_generation','hypothesis':'Independent fresh DDIM plans disrupt body/foot phase; shifted own plan at lower noise may reduce boundary jumps',
        'not_isolated_noise_level_ablation':True,'label':'roll128_warm400','start_timestep':400,'steps':20,'identities':identities,'seeds':original['replicate_seeds'],'weights':hashes['history'],
        'GT':'Only first16 initializes body; past updates from own executed chunks, planned prior from own generation. Future head remains GT-derived diagnostic.',
        'unchanged':'128 window, execute8, head/scene/history, no IK/scene correction or contact target input',
        'scope':'follow-up on exposed development data; no holdout claim','new_checkpoints':0,'source_sha256':{f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['warm_history_sampling.py','evaluate_warm_history_rollout.py']}}
    (OUT/'warm_protocol.json').write_text(json.dumps(protocol,indent=2));batch={k:v.cuda() for k,v in collate(samples).items()};allowed=inputs_only(prepare_bounded_observation(batch,'joint'));rows=[]
    for rep,seed in enumerate(original['replicate_seeds']):
        initial=batch['motion'][:,:16].clone();past=initial.clone();previous=None;executed=[]
        for step in range(20):
            obs={k:v[:,step*8:step*8+128] if k in TIME_KEYS else v for k,v in allowed.items()};obs=attach_executed_history(obs,past)
            if previous is None:plan=sample_with_executed_prefix(model,obs,past,steps=20,seed=seed)
            else:plan=sample_warm_prefix(model,obs,past,shift_prior(previous,past,obs['rest']),seed=seed+step*1009)
            chunk=plan[:,16:24].clone();executed.append(chunk);past=torch.cat((past[:,8:],chunk),1);previous=plan
            if (step+1)%5==0:print('warm',rep,step+1,flush=True)
        combined=torch.cat(executed,1)
        for i,identity in enumerate(identities):
            one={k:v[i:i+1] for k,v in batch.items()};rows.append({'replicate':rep,'seed':seed,'label':'roll128_warm400','group':identity['group'],'identity':identity,'full':evaluate(combined[i:i+1],crop(one,16,176),initial[i:i+1],list(range(0,160,8)))})
        (OUT/'warm_rows.json').write_text(json.dumps(rows,indent=2));np.savez_compressed(OUT/f'warm_motions_rep{rep}.npz',roll128_warm400=torch.cat((initial,combined),1).cpu().numpy())
        print('warm slide',np.mean([r['full']['gt_stance_slide_cm_frame'] for r in rows if r['replicate']==rep]),flush=True)
    protocol.update(status='completed',elapsed_s=time.monotonic()-began);(OUT/'warm_protocol.json').write_text(json.dumps(protocol,indent=2))
if __name__=='__main__':main()
