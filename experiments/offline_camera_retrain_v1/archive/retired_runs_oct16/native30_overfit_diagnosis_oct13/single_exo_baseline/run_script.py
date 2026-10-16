"""Capacity diagnostic on fixed official TRAIN windows; never a benchmark.

Baseline single-exo control deliberately disables scenes to isolate the shared
denoiser and objective. The mixed experiment uses an exact seven/three cycle.
Only one atomic resumable last.pt is retained per independently named trial.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config, make_model, window_options, write_json, code_hashes
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder, inference_payload
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_losses import objective
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.native30_validation import raw_metrics
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--scene',action='store_true')
    p.add_argument('--mixed',action='store_true')
    p.add_argument('--groups',nargs='+',default=['camera_wearer'])
    args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists():raise ValueError('Use a new trial directory; do not overwrite a finished trial')
    torch.set_num_threads(4);torch.manual_seed(2026101371);np.random.seed(2026101371)
    c=read_config(HERE/'native30_stride64_config.json')
    data=Native30Data(HERE/c['train_manifest'],'train',c['seed'],frames=c['frames'],
        memory_protocol=c['scene']['memory_protocol'],quality_config=c['gt_reliability'],
        exo_max_empty_frames=c['exo_max_empty_frames'],**window_options(c))
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),json.loads((HERE/c['mount_defaults']).read_text()),c['observation']['input_noise'])
    bank=[];identities=[]
    for group in args.groups:
        for mode in ('ego','exo') if args.mixed else ('exo',):
            eligible=sorted([i for i,(m,_) in enumerate(data.members[group]) if len(data.starts_for(group,i,mode))],
                key=lambda i:data.members[group][i][0]['sequence_id'])
            i=eligible[0];starts=data.starts_for(group,i,mode);start=int(starts[len(starts)//2])
            sample,identity=data.sample(group,mode,member=i,start=start,yaw=0)
            observed,scene,asset=builder.build(data,sample,identity,mode,(15,) if mode=='ego' else None)
            bank.append((observed,scene,asset,mode));identities.append(identity)
            print('Prepared',group,mode,identity['sequence_id'],start,flush=True)
    model=make_model(c).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    alphas=cosine_alphas().cuda();rng=np.random.default_rng(2026101372)
    coefficients=json.loads((HERE/'runs/native30_short500_oct13/prepared/TRAIN_gradient_scales.json').read_text())['coefficients']
    setup=dict(config=c,code_sha256=code_hashes(),identities=identities,steps=args.steps,scene=args.scene,
        protocol='fixed TRAIN windows, clean inputs, yaw0, random diffusion noise, constant lr1e-4, no scene dropout',
        ego_exo_ratio='exact7:3 per10 steps' if args.mixed else 'exo-only capacity control',
        body_frontend_prior=False,initialization='from_scratch',loss='production native30 objective and recorded observation coefficients',
        optimizer_steps=0,benchmark=False,validation_data_used=False,test_data_used=False)
    write_json(out/'setup.json',setup)
    counts={'ego':0,'exo':0};started=time.monotonic()

    def evaluate(step):
        model.eval();rows=[];saved={};
        with torch.no_grad():
            for i,item in enumerate(bank):
                batch=collate_observations([item]);known=inference_payload(batch)
                prediction=sample_native30(model,known,steps=20,seed=2026101373+i,use_scene=args.scene)
                mask=batch['observation_meta'][...,0]>0
                metrics=raw_metrics(prediction,batch,mask)
                rows.append(dict(case=i,mode=identities[i]['mode'],group=identities[i]['group'],metrics=metrics))
                saved[f'{i}_truth']=batch['motion'][0].cpu().numpy()
                saved[f'{i}_clean_prediction']=prediction[0].cpu().numpy()
                saved[f'{i}_clean_observations']=batch['trajectory'][0].cpu().numpy()
                saved[f'{i}_clean_observation_meta']=batch['observation_meta'][0].cpu().numpy()
                saved[f'{i}_rest']=batch['rest'][0].cpu().numpy()
            saved['identities_json']=np.array(json.dumps(identities));saved['step']=np.array(step)
            np.savez_compressed(out/'validation_latest.npz',**saved)
        with (out/'evaluation.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,rows=rows))+'\n')
        print('EVAL',step,[(r['group'],r['mode'],round(r['metrics']['mpjpe_mm']['value'],2),round(r['metrics']['root_position_error_mm']['value'],2),round(r['metrics']['root_orientation_deg']['value'],2)) for r in rows],flush=True)
        model.train();return rows

    first=evaluate(0)
    with (out/'trace.jsonl').open('w') as trace:
        for step in range(1,args.steps+1):
            mode='ego' if args.mixed and (step-1)%10<7 else 'exo'
            candidates=[i for i,m in enumerate(identities) if m['mode']==mode]
            i=int(rng.choice(candidates));batch=collate_observations([bank[i]]);known=inference_payload(batch)
            t=torch.randint(1000,(1,),device='cuda')
            if rng.random()<.25:t=torch.randint(100,(1,),device='cuda')
            signal=alphas[t];noisy=signal[:,None,None].sqrt()*batch['motion']+(1-signal[:,None,None]).sqrt()*torch.randn_like(batch['motion'])
            opt.zero_grad(set_to_none=True)
            pred,details=model(noisy,t,known,use_scene=args.scene,return_details=True)
            loss,terms=objective(pred,batch['motion'],batch,signal,details,c,coefficients)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            counts[mode]+=1
            row=dict(step=step,case=i,mode=mode,t=int(t),loss=float(loss.detach()),gradient_norm=float(norm),
                     base={k:float(v.detach()) for k,v in terms['base'].items()})
            trace.write(json.dumps(row)+'\n');trace.flush()
            if step%25==0:print('STEP',step,row['loss'],counts,'seconds',round(time.monotonic()-started,1),flush=True)
            if step%100==0 or step==args.steps:
                latest=evaluate(step)
                save_checkpoint(dict(step=step,planned_steps=args.steps,model=model.state_dict(),optimizer=opt.state_dict(),
                    setup=setup,counts=counts,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                    condition_rng=rng.bit_generator.state,data_rng=data.sampler_state()),out/'last.pt')
                write_json(out/'result.json',dict(step=step,planned_steps=args.steps,status='complete' if step==args.steps else 'running',
                    counts=counts,elapsed_s=time.monotonic()-started,initial=first,latest=latest,
                    scope='TRAIN memorization diagnostic; not generalization/benchmark',scene=args.scene))


if __name__=='__main__':main()
