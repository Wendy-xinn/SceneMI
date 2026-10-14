"""Zero-update B2/B4 and CPU-thread comparison with the fresh complete model."""
import cProfile
import pstats
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config,make_model,window_options,write_json,training_plan
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_support_cache import cached_support
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs,training_motion
from experiments.offline_camera_retrain_v1.native30_losses import objective
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1.train_native30 import model_digest
from experiments.offline_camera_retrain_v1.native30_sampling import StratifiedRecordingSampler
from collections import Counter

ROOT=Path(__file__).resolve().parent

def main():
    config=read_config(HERE/'native30_fresh_ego_config.json')
    weights=json.loads((HERE/config['prepared_output']/'TRAIN_gradient_scales.json').read_text())['coefficients']
    torch.set_num_threads(4);torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    data=Native30Data(HERE/config['train_manifest'],'train',config['seed'],frames=config['frames'],memory_protocol=config['scene']['memory_protocol'],quality_config=config['gt_reliability'],exo_max_empty_frames=config['exo_max_empty_frames'],**window_options(config))
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),json.loads((HERE/config['mount_defaults']).read_text()),config['observation']['input_noise'])
    model=make_model(config).cuda()
    initial=model_digest(model)
    # Include both Adam moments in VRAM without constructing/updating an optimizer.
    moments=[torch.zeros_like(p) for p in model.parameters() for _ in range(2)]
    report=dict(status='running',optimizer_steps=0,initialization='from_scratch',initial_model_sha256=initial,
        config_sha256=hashlib.sha256((HERE/'native30_fresh_ego_config.json').read_bytes()).hexdigest(),
        model_parameters=sum(p.numel() for p in model.parameters()),all_parameters_trainable=all(p.requires_grad for p in model.parameters()),
        scope='One predefined TRAIN head-only B2 (TRUMANS + EgoBody); B4 duplicates this pair only to measure memory/throughput. Fixed t300; FP32, all scene/IK/physics settings unchanged. Timings exclude optimizer, checkpoint and validation; not a full-pool mean.',rows=[])
    sampler=StratifiedRecordingSampler(data,training_plan(data,config),config['sampler'],config['seed']+371)
    counts=Counter();windows=set();records=set()
    for _ in range(2*config['phase_plan']['planned_updates']):
        row=sampler.draw();meta,_=data.members[row['group']][row['member']]
        counts[row['group']+'/'+row['mode']]+=1;records.add((row['group'],row['recording_key']))
        windows.add((row['group'],meta['sequence_id'],row['mode'],row['start']))
    report['phase_sampling_audit']=dict(draws=sum(counts.values()),counts=dict(counts),unique_windows=len(windows),unique_group_recordings=len(records),eligible_ego_windows=sum(len(data.starts_for(g,i,'ego')) for g,members in data.members.items() for i in range(len(members))),training_pool='all eligible official TRAIN ego windows; recording cycles, not a complete-window epoch')
    if any(not k.endswith('/ego') for k in counts):raise AssertionError('Exo entered the ego phase')
    items=[];identities=[]
    for group in ('trumans','camera_wearer'):
        member=next(i for i in range(len(data.members[group])) if len(data.starts_for(group,i,'ego')))
        start=int(data.starts_for(group,member,'ego')[0])
        sample,identity=data.sample(group,'ego',member=member,start=start,yaw=.6)
        sample,scene,asset=builder.build(data,sample,identity,'ego',(15,))
        sample=cached_support(data,builder,sample,identity)
        items.append((sample,scene,asset,'ego'))
        identities.append({k:identity[k] for k in ('group','sequence_id','start','mode')})
    report['identities']=identities
    def run(size,threads,repeat,profile=False):
        torch.set_num_threads(threads);model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        batch=collate_observations(items*(size//2));known=model_inputs(config,batch)
        t=torch.full((size,),300,device='cuda');alpha=model.alphas[t];target=training_motion(config,batch,known)
        torch.manual_seed(config['seed']+73)
        noisy=alpha[:,None,None].sqrt()*target+(1-alpha[:,None,None]).sqrt()*torch.randn_like(target)
        model.train();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.monotonic()
        profiler=cProfile.Profile() if profile else None
        if profiler:profiler.enable()
        prediction,details=model(noisy,t,known,use_scene=True,return_details=True)
        torch.cuda.synchronize();fwd=time.monotonic()-began
        b=time.monotonic();loss,_=objective(prediction,batch['motion'],batch,alpha,details,config,weights);loss.backward();torch.cuda.synchronize();back=time.monotonic()-b
        if profiler:
            profiler.disable();profiler.dump_stats(str(ROOT/'fresh_full_training.pstats'))
            stats=pstats.Stats(profiler)
            report['profile_top_cumulative']=[dict(file=k[0],line=k[1],function=k[2],calls=v[1],own_s=v[2],cumulative_s=v[3]) for k,v in sorted(stats.stats.items(),key=lambda kv:-kv[1][3])[:20]]
            report['profile_top_own']=[dict(file=k[0],line=k[1],function=k[2],calls=v[1],own_s=v[2],cumulative_s=v[3]) for k,v in sorted(stats.stats.items(),key=lambda kv:-kv[1][2])[:20]]
        report['parameters_without_gradient_in_probe']={name:p.numel() for name,p in model.named_parameters() if p.grad is None}
        row=dict(batch_size=size,threads=threads,repeat=repeat,forward_s=fwd,loss_backward_s=back,total_s=fwd+back,seconds_per_sample=(fwd+back)/size,loss=float(loss.detach()),finite=bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,physical_accepted=float(details['physical_accepted_steps'].sum()))
        if not row['finite']:raise AssertionError('Nonfinite fresh backward')
        del batch,known,t,alpha,target,noisy,prediction,details,loss
        return row
    try:
        report['warmup']=run(2,4,-1)
        for size,threads in ((2,4),(2,2),(2,1),(4,4)):
            try:
                for repeat in range(2):
                    row=run(size,threads,repeat);report['rows'].append(row);print(json.dumps(row),flush=True)
                    write_json(ROOT/'FRESH_BACKBONE_BENCHMARK.json',report)
            except torch.cuda.OutOfMemoryError:
                report['rows'].append(dict(batch_size=size,threads=threads,status='OOM'));model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        report['profile_probe']=run(2,4,99,profile=True)
        if model_digest(model)!=initial:raise AssertionError('Benchmark changed model weights')
        report.update(status='passed',parameters_unchanged=True)
    finally:write_json(ROOT/'FRESH_BACKBONE_BENCHMARK.json',report)

if __name__=='__main__':main()
