"""Real TRAIN component timings, no parameter updates or model simplifications."""
import cProfile
import datetime as dt
import gc
import json
from pathlib import Path
import pstats
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import make_model, write_json
from experiments.offline_camera_retrain_v1.native30_overnight import scoped_data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_scene_support import attach_support_supervision
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, training_motion
from experiments.offline_camera_retrain_v1.native30_losses import objective
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
import experiments.offline_camera_retrain_v1.native30_physical_guidance as physics

ROOT = Path(__file__).resolve().parent
REPORT = dict(status='running', optimizer_steps=0, started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    scope='B2 L192 all model parameters trainable; head-only ego; t300; scene/physics/FP32 unchanged',
    limitations='Three fixed and two random TRAIN batches, not a corpus mean. Nested component timings overlap. Profiling and CUDA synchronization add overhead. No optimizer/clip/checkpoint/validation.', rows=[])


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2026101466)
    state = torch.load(HERE/'runs/native30_scene_relations_oct14/experiment/candidate/last.pt', map_location='cpu', weights_only=False)
    config = state['config'];config['ego_head_only_probability'] = 1.
    data = scoped_data(config, 'train')
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(), json.loads((HERE/config['mount_defaults']).read_text()), config['observation']['input_noise'])
    plan = json.loads((HERE/'runs/native30_overnight_ik_oct14/experiment/window_plan.json').read_text())['train']
    chosen = [next(v for v in plan if (v['group'], v['mode']) == p) for p in [('trumans','exo'),('rich','ego')]]
    model = make_model(config).cuda();model.load_state_dict(state['model']);del state
    for p in model.parameters():p.requires_grad_(True)
    moments = [torch.zeros_like(p) for p in model.parameters() for _ in range(2)]
    REPORT['resident_adam_bytes'] = sum(p.numel()*p.element_size() for p in moments)
    weights = json.loads((HERE/config['prepared_output']/'effective_config.json').read_text())['observation_coefficients']
    buckets = {}
    def wrap(name, fn):
        def measured(*args, **kwargs):
            torch.cuda.synchronize();began=time.monotonic()
            try:return fn(*args, **kwargs)
            finally:
                torch.cuda.synchronize();buckets[name]=buckets.get(name,0.)+time.monotonic()-began
        return measured
    for name in ('_core_pass','local_geometry','constrain_motion'):
        setattr(model,name,wrap(name,getattr(model,name)))
    physics.guide_motion=wrap('physical_guidance_inclusive',physics.guide_motion)
    full_plan=data.sampling_plan(config['ego_probability'])
    for repeat in range(5):
        buckets.clear();model.zero_grad(set_to_none=True);gc.collect();torch.cuda.reset_peak_memory_stats()
        rng=np.random.default_rng(2026101459+repeat);phases={};items=[];identities=[]
        profiler=cProfile.Profile() if repeat==2 else None
        if profiler:profiler.enable()
        started=time.monotonic()
        for entry in chosen:
            began=time.monotonic()
            if repeat<3:
                sample,identity=data.sample(entry['group'],entry['mode'],member=entry['member'],start=entry['start'])
            else:
                entry=full_plan[int(rng.choice(len(full_plan),p=[r['probability'] for r in full_plan]))]
                sample,identity=data.sample(entry['group'],entry['mode'])
            identities.append({k:identity[k] for k in ('group','mode','sequence_id','start')})
            phases['raw_sample']=phases.get('raw_sample',0.)+time.monotonic()-began
            began=time.monotonic()
            sample,scene,asset=builder.build(data,sample,identity,entry['mode'],None if entry['mode']=='exo' else (15,),noise_rng=rng)
            phases['observation_scene_build']=phases.get('observation_scene_build',0.)+time.monotonic()-began
            began=time.monotonic();sample=attach_support_supervision(sample,scene,asset)
            phases['support_labels']=phases.get('support_labels',0.)+time.monotonic()-began
            items.append((sample,scene,asset,entry['mode']))
        began=time.monotonic();batch=collate_observations(items);known=model_inputs(config,batch)
        times=torch.full((2,),300,device='cuda',dtype=torch.long);alpha=model.alphas[times]
        target=training_motion(config,batch,known)
        noisy=alpha[:,None,None].sqrt()*target+(1-alpha[:,None,None]).sqrt()*torch.randn_like(target)
        torch.cuda.synchronize();phases['collate_reference_noise']=time.monotonic()-began
        began=time.monotonic();model.train();prediction,details=model(noisy,times,known,use_scene=True,return_details=True)
        torch.cuda.synchronize();phases['forward']=time.monotonic()-began
        began=time.monotonic();loss,terms=objective(prediction,batch['motion'],batch,alpha,details,config,weights)
        torch.cuda.synchronize();phases['loss']=time.monotonic()-began
        began=time.monotonic();loss.backward();torch.cuda.synchronize();phases['backward']=time.monotonic()-began
        total=time.monotonic()-started
        if profiler:
            profiler.disable();profiler.dump_stats(str(ROOT/'training_components.pstats'))
            stats=pstats.Stats(profiler)
            REPORT['profile_top_own_time']=[dict(file=k[0],line=k[1],function=k[2],calls=v[1],own_s=v[2],inclusive_s=v[3])
                for k,v in sorted(stats.stats.items(),key=lambda kv:kv[1][2],reverse=True)[:25]]
        row=dict(repeat=repeat,profiled=bool(profiler),sampling='fixed' if repeat<3 else 'full_corpus_random',identities=identities,
            phases_s=phases,nested_forward_components_s=dict(buckets),total_s=total,
            loss=float(loss.detach()),finite=bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),
            peak_allocated_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
        REPORT['rows'].append(row);write_json(ROOT/'TRAIN_COMPONENTS.json',REPORT);print(json.dumps(row),flush=True)
        if not row['finite']:raise FloatingPointError('Nonfinite full-model probe')
        del items,batch,known,target,noisy,prediction,details,loss,terms,sample,scene,asset
    REPORT.update(status='passed',finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        REPORT.update(status='failed',traceback=traceback.format_exc());raise
    finally:write_json(ROOT/'TRAIN_COMPONENTS.json',REPORT)
