"""Future explicit native30 training. Preparation never starts this entry point."""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import (
    CONFIG_PATH,read_config,make_model,code_hashes,sha256,write_json,sample_item,dataset_digest,window_options)
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder,inference_payload
from experiments.offline_camera_retrain_v1.native30_losses import objective
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.native30_validation import validate


def train(config_path,steps,output=None,resume=False):
    if steps<=0:raise ValueError('Positive explicit training budget required')
    config=read_config(config_path);prepared=HERE/config['prepared_output']
    checks=json.loads((prepared/'input_checks.json').read_text())
    effective=json.loads((prepared/'effective_config.json').read_text())
    if checks['status']!='passed' or effective['config']!=config or effective['data_mode']!='full' or not checks['training_ready']:
        raise ValueError('Full raw30 corpus and its zero-step preparation are required; review clips cannot train')
    hashes=code_hashes()
    if hashes!=effective['code_sha256'] or sha256(HERE/config['mount_defaults'])!=effective['mount_defaults_sha256']:
        raise ValueError('Code/calibration changed since zero-step preparation')
    folder=Path(output or prepared/'training')
    if folder.resolve()==prepared.resolve() or folder.resolve() in {
        (HERE/'runs/ego_exo_surface_continue_oct11').resolve(),(HERE/'runs/native_dynamic_scene20_contact_55k_oct07').resolve()}:
        raise ValueError('Retain prepared records and old resumable models in separate directories')
    if (folder/'last.pt').exists()!=bool(resume):
        raise ValueError('Existing checkpoints require explicit resume; missing checkpoint cannot resume')
    torch.set_num_threads(4);torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    data=Native30Data(HERE/config['train_manifest'],'train',config['seed'],frames=config['frames'],memory_protocol=config['scene']['memory_protocol'],quality_config=config['gt_reliability'],exo_max_empty_frames=config.get('exo_max_empty_frames'),**window_options(config))
    val=Native30Data(HERE/config['train_manifest'],'validation',config['seed']+1,frames=config['frames'],memory_protocol=config['scene']['memory_protocol'],quality_config=config['gt_reliability'],exo_max_empty_frames=config.get('exo_max_empty_frames'),**window_options(config))
    if dataset_digest(data)|dataset_digest(val)!=effective['data_sha256']:
        raise ValueError('Native30 data changed after preparation')
    plan=data.sampling_plan(config['ego_probability'])
    if plan!=effective['sampling_plan']:raise ValueError('Sampling duration/configuration changed')
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),json.loads((HERE/config['mount_defaults']).read_text()),config['observation']['input_noise'])
    model=make_model(config).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=config['optimizer']['lr'],weight_decay=config['optimizer']['weight_decay'])
    rng=np.random.default_rng(config['seed']+991);start=0;counts={}
    if resume:
        state=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if state['effective_config']!=effective or state['planned_steps']!=steps:
            raise ValueError('Resume requires identical configuration, dependencies, and schedule')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
        for entry in optimizer.state.values():
            for key,value in entry.items():
                if torch.is_tensor(value):entry[key]=value.cuda()
        data.load_sampler_state(state['data_rng']);rng.bit_generator.state=state['condition_rng']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);np.random.set_state(state['numpy_rng'])
        start,counts=state['step'],state['counts'];del state
    folder.mkdir(parents=True,exist_ok=True);started=time.monotonic();alphas=cosine_alphas().cuda()
    protocol=dict(status='training',planned_steps=steps,completed_steps=start,effective_config=effective,
        free_disk_before=shutil.disk_usage(HERE).free,checkpoint_policy='one atomic resumable last.pt')
    write_json(folder/'protocol.json',protocol)
    def checkpoint(step):
        save_checkpoint(dict(step=step,planned_steps=steps,model=model.state_dict(),optimizer=optimizer.state_dict(),
            effective_config=effective,counts=counts,data_rng=data.sampler_state(),condition_rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state()),folder/'last.pt')
    with (folder/'trace.jsonl').open('a' if resume else 'w') as trace:
        for step in range(start+1,steps+1):
            pairs=[sample_item(data,builder,config,plan,rng) for _ in range(config['batch_size'])]
            items,identities=zip(*pairs);batch=collate_observations(items);known=inference_payload(batch)
            t=torch.randint(1000,(len(items),),device='cuda')
            low=torch.rand(len(items),device='cuda')<config['noise']['extra_low_noise_probability']
            t=torch.where(low,torch.randint(config['noise']['low_noise_steps'],t.shape,device='cuda'),t)
            signal=alphas[t];noisy=signal[:,None,None].sqrt()*batch['motion']+(1-signal[:,None,None]).sqrt()*torch.randn_like(batch['motion'])
            scene_on=rng.random()>=config['scene']['dropout_probability']
            model.train();optimizer.zero_grad(set_to_none=True)
            prediction,details=model(noisy,t,known,use_scene=scene_on,return_details=True)
            loss,terms=objective(prediction,batch['motion'],batch,signal,details,config,effective['observation_coefficients'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native30 objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['optimizer']['clip_norm'],error_if_nonfinite=True)
            schedule=min(1.,step/config['optimizer']['warmup_steps'])*max(config['optimizer']['floor_fraction'],.5*(1+math.cos(math.pi*step/steps)))
            for group in optimizer.param_groups:group['lr']=config['optimizer']['lr']*schedule
            optimizer.step()
            for identity in identities:
                key=f"{identity['group']}/{identity['mode']}";counts[key]=counts.get(key,0)+1
            def numbers(values):return {k:float(v.detach()) for k,v in values.items()}
            row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),timesteps=t.tolist(),scene=bool(scene_on),
                     identities=identities,terms={k:numbers(v) for k,v in terms.items()},lr=optimizer.param_groups[0]['lr'])
            trace.write(json.dumps(row)+'\n');trace.flush()
            if step==1 or step%25==0:print(step,row['loss'],counts,flush=True)
            # Preserve the first real update before validation can fail or be
            # interrupted. Subsequent checkpoints overwrite this same last.pt.
            if step==1:
                checkpoint(step);protocol.update(completed_steps=step,counts=dict(counts),
                    elapsed_s=time.monotonic()-started,free_disk_after=shutil.disk_usage(HERE).free)
                write_json(folder/'protocol.json',protocol)
            if step==1 or step%config['validation']['interval_steps']==0 or step==steps:
                validate(model,val,builder,config,folder,step)
            if step%100==0 or step==steps:
                checkpoint(step);protocol.update(status='completed' if step==steps else 'training',completed_steps=step,
                    elapsed_s=time.monotonic()-started,counts=counts,free_disk_after=shutil.disk_usage(HERE).free)
                write_json(folder/'protocol.json',protocol)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG_PATH);parser.add_argument('--steps',type=int,required=True)
    parser.add_argument('--output',type=Path);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();train(args.config,args.steps,args.output,args.resume)
