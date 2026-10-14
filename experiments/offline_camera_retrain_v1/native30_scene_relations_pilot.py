"""Matched pilot for support labels, metric relations and within-DDIM guidance.

Uses a registered TRAIN bank and untouched official VAL windows. No TEST or
automatic full-corpus training. New checkpoints each overwrite one last.pt.
"""
import argparse
import copy
import datetime as dt
import json
import math
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config,make_model,write_json,code_hashes,sha256
from experiments.offline_camera_retrain_v1.native30_overnight import scoped_data,build_bank,augment,evaluate,scene_gate
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1.unified_geometry import occupancy_from_points
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs,training_motion
from experiments.offline_camera_retrain_v1.native30_losses import objective
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint


def fit(model, config, bank, identities, builder, weights, out, warm_step, updates, *, scene):
    common=('reference_', 'observation_', 'camera_encoder.', 'body_encoder.', 'core.unet.final_conv.')
    adapters=('relation_', 'surface_', 'anchor_', 'region_condition.', 'contact_', 'correspondence_',
              'feedback_temporal.', 'root_feedback.', 'rotation_feedback.', 'patch_', 'dynamic_', 'owner_embedding.')
    slow=[];fast=[]
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(common+adapters))
        if p.requires_grad:(fast if name.startswith(adapters) else slow).append(p)
    opt=torch.optim.AdamW([dict(params=slow,lr=config['optimizer']['lr']),dict(params=fast,lr=.0001)],weight_decay=.01)
    rng=np.random.default_rng(config['seed']+871)
    by_mode={m:[i for i,v in enumerate(identities) if v['mode']==m] for m in ('ego','exo')}
    out.mkdir(parents=True,exist_ok=True);began=time.monotonic();counts={};trace=[]
    def checkpoint(step):
        save_checkpoint(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,config=config,
             warm_step=warm_step,updates=step-warm_step,condition_rng=rng.bit_generator.state,
             torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),counts=counts,
             meta=dict(train_plan=identities,trainable_common_prefixes=common,trainable_scene_prefixes=adapters,
                       scene=scene,full_corpus=False)),out/'last.pt')
        write_json(out/'progress.json',dict(step=step,updates=step-warm_step,elapsed_s=time.monotonic()-began,
             counts=counts,full_corpus=False))
    with (out/'trace.jsonl').open('w') as stream:
        for update in range(1,updates+1):
            step=warm_step+update;items=[];ids=[]
            for b in range(config['batch_size']):
                slot=(update-1)*config['batch_size']+b;mode='ego' if slot%10<7 else 'exo'
                i=int(rng.choice(by_mode[mode]));items.append(augment(bank[i],identities[i],builder,config,rng));ids.append(i)
            batch=collate_observations(items);known=model_inputs(config,batch);target=training_motion(config,batch,known)
            generator=torch.Generator(device='cuda').manual_seed(config['seed']+step*31)
            t=torch.randint(1000,(len(items),),device='cuda',generator=generator)
            low=torch.rand(len(items),device='cuda',generator=generator)<.25
            t=torch.where(low,torch.randint(100,t.shape,device='cuda',generator=generator),t)
            alpha=model.alphas[t];noise=torch.randn(target.shape,device='cuda',generator=generator)
            noisy=alpha[:,None,None].sqrt()*target+(1-alpha[:,None,None]).sqrt()*noise
            draw=rng.random();use_scene=scene and draw>=config['scene']['dropout_probability']
            model.train();model.core.scene_embedding.eval();opt.zero_grad(set_to_none=True)
            prediction,details=model(noisy,t,known,use_scene=use_scene,return_details=True)
            loss,terms=objective(prediction,batch['motion'],batch,alpha,details,config,weights)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite v4 objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            schedule=max(.2,.5*(1+math.cos(math.pi*update/max(updates,1))))
            opt.param_groups[0]['lr']=config['optimizer']['lr']*schedule;opt.param_groups[1]['lr']=.0001*schedule
            opt.step()
            for i in ids:
                k=identities[i]['group']+'/'+identities[i]['mode'];counts[k]=counts.get(k,0)+1
            row=dict(step=step,identities=ids,timesteps=t.tolist(),observation_checksum=float(batch['trajectory'].double().sum()),
                     diffusion_noise_checksum=float(noise.double().sum()),scene=use_scene,loss=float(loss.detach()),
                     gradient_norm=float(norm),elapsed_s=time.monotonic()-began,
                     geometry={k:float(v.detach()) for k,v in terms['geometry'].items()},
                     physical_accepted=float(details.get('physical_accepted_steps',alpha.new_zeros(1)).sum()))
            stream.write(json.dumps(row)+'\n');stream.flush();trace.append(row)
            if update==1 or update%25==0:print('STEP',out.name,update,round(row['loss'],5),'seconds',round(row['elapsed_s'],1),flush=True)
            if update==1 or update%100==0:checkpoint(step)
        checkpoint(warm_step+updates)
    return trace


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=HERE/'native30_scene_relations_config.json')
    parser.add_argument('--output',type=Path,default=HERE/'runs/native30_scene_relations_oct14/experiment')
    parser.add_argument('--warm',type=Path,default=HERE/'runs/native30_overnight_ik_oct14/experiment/candidate/last.pt')
    parser.add_argument('--steps',type=int,default=200);parser.add_argument('--ddim-steps',type=int,default=20)
    parser.add_argument('--seeds',type=int,nargs='+',default=[2026101401,2026101403])
    parser.add_argument('--diagnostic-ablations',action='store_true')
    args=parser.parse_args();out=args.output
    if (out/'protocol.json').exists():raise ValueError('Use a new output; preserve prior trials')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.30)
    config=read_config(args.config);prepared=HERE/config['prepared_output']
    checks=json.loads((prepared/'input_checks.json').read_text());effective=json.loads((prepared/'effective_config.json').read_text())
    if not checks.get('training_ready') or effective['config']!=config or effective['code_sha256']!=code_hashes():
        raise ValueError('Current complete zero-step model/data preparation must pass')
    warm=torch.load(args.warm,map_location='cpu',weights_only=False)
    if warm['config']['version']!='native30-official-overnight-observation-ik-v3':raise ValueError('Expected admitted v3 4000-step source')
    plan=json.loads((args.warm.parent.parent/'window_plan.json').read_text())
    if any(v['split']!='train' for v in plan['train']):raise ValueError('Warm source TRAIN contamination')
    protocol=dict(status='building_banks',started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
         config=config,code_sha256=code_hashes(),initialization=dict(path=str(args.warm),sha256=sha256(args.warm),step=warm['step']),
         new_optimizer=True,updates=args.steps,ego_exo='exact7:3 sample slots',full_training_started=False,test_used=False,
         train_windows=len(plan['train']),validation_windows=len(plan['validation']),free_disk_before=shutil.disk_usage(HERE).free)
    write_json(out/'protocol.json',protocol)
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),json.loads((HERE/config['mount_defaults']).read_text()),config['observation']['input_noise'])
    bank,ids=build_bank(scoped_data(config,'train'),builder,plan['train'],config)
    val,vids=build_bank(scoped_data(config,'validation'),builder,plan['validation'])
    write_json(out/'identities.json',dict(train=ids,validation=vids))
    labels=[dict(group=i['group'],mode=i['mode'],valid=int(item[0]['contact_valid'].sum()),
                 positive=int(((item[0]['contact_target']>.5)&item[0]['contact_valid']).sum())) for item,i in zip(bank,ids)]
    write_json(out/'support_labels.json',dict(rows=labels,scope='TRAIN only, no inference target keys'))
    weights=effective['observation_coefficients'];rows=[]
    def initialize():
        torch.manual_seed(config['seed']);model=make_model(config).cuda()
        missing,unexpected=model.load_state_dict(warm['model'],strict=False)
        if unexpected or any(not k.startswith('relation_') for k in missing):raise ValueError(f'Unexpected warm transfer: {missing}, {unexpected}')
        return model
    for name,scene in [('candidate',True),('control',False)]:
        protocol['status']=name+'_training';write_json(out/'protocol.json',protocol)
        model=initialize();tr=fit(model,config,bank,ids,builder,weights,out/name,warm['step'],args.steps,scene=scene);rows.append(tr)
        protocol['status']=name+'_validation';write_json(out/'protocol.json',protocol)
        report=evaluate(model,val,vids,builder,config,out/name,warm['step']+args.steps,steps=args.ddim_steps,
                        seeds=tuple(args.seeds),ablate=scene)
        if scene:
            candidate=report
            from experiments.offline_camera_retrain_v1.native30_overnight_mesh import render
            for regime in ('clean','noisy'):print('MESH',render(out/name/'validation_latest.npz',out/name,regime),flush=True)
            if args.diagnostic_ablations:
                model.guidance_enabled=False
                evaluate(model,val,vids,builder,config,out/'without_guidance',warm['step']+args.steps,
                         steps=args.ddim_steps,seeds=(args.seeds[0],),scene_modes=('full',))
                model.guidance_enabled=True
                wrong=[]
                for i,item in enumerate(val):
                    sample=dict(item[0]);scene_other=val[(i+4)%len(val)][1]
                    sample['occupancy']=occupancy_from_points(scene_other.points)
                    wrong.append((sample,scene_other,item[2],item[3]))
                evaluate(model,wrong,vids,builder,config,out/'mismatched_scene',warm['step']+args.steps,
                         steps=args.ddim_steps,seeds=(args.seeds[0],),scene_modes=('full',))
        else:control=report
        del model;torch.cuda.empty_cache()
    gate=scene_gate(candidate,control)
    keys=('step','identities','timesteps','observation_checksum','diffusion_noise_checksum')
    paired=len(rows[0])==len(rows[1]) and all(all(a[k]==b[k] for k in keys) for a,b in zip(*rows))
    gate['paired_training_inputs']=paired;gate['passed']=gate['passed'] and paired
    write_json(out/'scene_gate.json',gate)
    protocol.update(status='completed_gate_passed' if gate['passed'] else 'completed_gate_failed',
         last_gate=gate,candidate_step=warm['step']+args.steps,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
         free_disk_after=shutil.disk_usage(HERE).free)
    write_json(out/'protocol.json',protocol);print('FINISHED',protocol['status'],flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--output',type=Path,default=HERE/'runs/native30_scene_relations_oct14/experiment')
        args,_=parser.parse_known_args();args.output.mkdir(parents=True,exist_ok=True)
        write_json(args.output/'failure.json',dict(traceback=traceback.format_exc(),utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        raise
