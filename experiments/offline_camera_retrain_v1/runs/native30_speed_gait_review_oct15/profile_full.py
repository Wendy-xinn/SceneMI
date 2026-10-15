"""Measure full Native30 TRAIN forward/backward only; never update weights."""
import cProfile
import io
import json
from pathlib import Path
import pstats
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.train_native30 import (
    HERE, read_config, make_model, window_options, Native30Data, Native30SkinAnchorCache,
    Native30ObservationBuilder, collate_observations, model_inputs, training_motion,
    objective, write_json, sha256, code_hashes,
)
from experiments.offline_camera_retrain_v1.native30_support_cache import cached_support

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.28)
    config=read_config(HERE/'native30_fresh_ego_config.json')
    path=HERE/'runs/native30_fresh_ego_oct14/training/last.pt'
    parent=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    data=Native30Data(HERE/config['train_manifest'],'train',config['seed'],frames=192,
        memory_protocol=config['scene']['memory_protocol'],quality_config=config['gt_reliability'],
        exo_max_empty_frames=config['exo_max_empty_frames'],**window_options(config))
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),
        json.loads((HERE/config['mount_defaults']).read_text()),config['observation']['input_noise'])
    bank={};identities={};preparation={}
    for group in ('trumans','camera_wearer','rich'):
        began=time.monotonic()
        member=next(i for i in range(len(data.members[group])) if len(data.starts_for(group,i,'ego')))
        start=int(data.starts_for(group,member,'ego')[0])
        sample,identity=data.sample(group,'ego',member=member,start=start,yaw=.6)
        sample,scene,asset=builder.build(data,sample,identity,'ego',(15,))
        sample=cached_support(data,builder,sample,identity)
        bank[group]=(sample,scene,asset,'ego'); identities[group]=identity
        preparation[group]=time.monotonic()-began
        print('prepared',group,preparation[group],flush=True)
    report=dict(status='profiling',optimizer_steps=0,checkpoint_updated=False,
        model_source_step=parent['step'],source_sha256=sha256(path),identities=identities,
        preparation_seconds=preparation,cases=[],code_sha256=code_hashes(),
        scope='B2 FP32 30Hz/L192; same original full model and losses; F/B excludes Adam, preparation, saving, validation; no external training process')
    write_json(ROOT/'FULL_PROFILE.json',report)
    model=make_model(config).cuda(); model.load_state_dict(parent['model'],strict=True)
    coefficients=parent['effective_config']['observation_coefficients']
    del parent
    for pair in (('trumans','camera_wearer'),('trumans','rich'),('camera_wearer','rich')):
        batch=collate_observations([bank[k] for k in pair]);known=model_inputs(config,batch)
        t=torch.full((2,),300,device='cuda',dtype=torch.long);signal=model.alphas[t]
        target=training_motion(config,batch,known)
        generator=torch.Generator(device='cuda').manual_seed(6621)
        noisy=signal[:,None,None].sqrt()*target+(1-signal[:,None,None]).sqrt()*torch.randn(target.shape,device='cuda',generator=generator)
        def compute():
            model.train();model.zero_grad(set_to_none=True);torch.manual_seed(10047)
            torch.cuda.synchronize();began=time.monotonic()
            pred,details=model(noisy,t,known,use_scene=True,return_details=True)
            torch.cuda.synchronize();forward=time.monotonic()-began
            tick=time.monotonic()
            loss,terms=objective(pred,batch['motion'],batch,signal,details,config,coefficients)
            torch.cuda.synchronize();loss_time=time.monotonic()-tick;tick=time.monotonic()
            loss.backward();torch.cuda.synchronize();backward=time.monotonic()-tick
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite profiling loss')
            return dict(forward_seconds=forward,objective_seconds=loss_time,backward_seconds=backward,
                total_seconds=forward+loss_time+backward,loss=float(loss.detach()),
                accepted_physical_steps=details['physical_accepted_steps'].tolist(),
                peak_CUDA_GiB=torch.cuda.max_memory_allocated()/2**30)
        samples=[compute() for _ in range(3)]
        case=dict(pair=pair,warmup=samples[0],measured=samples[1:])
        report['cases'].append(case);write_json(ROOT/'FULL_PROFILE.json',report)
        print(json.dumps(case),flush=True)
        if pair==('trumans','rich'):
            profile=cProfile.Profile();profile.enable();result=compute();profile.disable()
            profile.dump_stats(str(ROOT/'full_backward.prof'))
            stream=io.StringIO();pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(80)
            (ROOT/'PROFILE_CUMULATIVE.txt').write_text(stream.getvalue())
            stream=io.StringIO();pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('tottime').print_stats(80)
            (ROOT/'PROFILE_SELF.txt').write_text(stream.getvalue())
            report['profiled_case']=result
    report.update(status='completed',support_cache_hits=data.support_cache.hits,
        support_cache_misses=data.support_cache.misses)
    write_json(ROOT/'FULL_PROFILE.json',report)


if __name__=='__main__':main()
