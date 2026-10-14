"""Same weights/config/windows/seeds; compare reference and exact query runtime."""
import contextlib
import cProfile
import datetime as dt
import gc
import json
from pathlib import Path
import pstats
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import time
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import make_model,write_json,code_hashes,sha256
from experiments.offline_camera_retrain_v1.native30_overnight import scoped_data,build_bank,augment,surface_metrics
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_geometry import Native30SurfaceWindow
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs,sample_motion
from experiments.offline_camera_retrain_v1.native30_validation import raw_metrics
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
import experiments.offline_camera_retrain_v1.native30_physical_guidance as physical

ROOT=Path(__file__).resolve().parent
RUN=HERE/'runs/native30_scene_relations_oct14/experiment'
REPORT=dict(status='building',optimizer_steps=0,config_changed=False,existing_trial_changed=False,rows=[],
    started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),code_sha256=code_hashes(),
    scope='Three predefined registered VAL windows (TRUMANS exo noisy, EgoBody master exo clean, RICH ego clean). Not full-corpus throughput.',
    prediction_protocol='same step4200, exact original noise seed, DDIM20, batch1, original observation/geometry/guidance configuration',
    timing='CUDA-synchronized complete sampling and separately scene metrics; current VAL runs concurrently; components are inclusive')
write_json(ROOT/'RUNTIME_BENCHMARK.json',REPORT)

@contextlib.contextmanager
def original_queries():
    names=('query_motion','query_wide_motion','query_nearest')
    saved={k:getattr(Native30SurfaceWindow,k) for k in names}
    def motion(self,v,r,device,neighbors=8):return self.query_reference(v,r,neighbors)
    def nearest(self,v,r):return self.query_reference(v,r)
    Native30SurfaceWindow.query_motion=motion;Native30SurfaceWindow.query_wide_motion=motion
    Native30SurfaceWindow.query_nearest=nearest
    try:yield
    finally:
        for k,v in saved.items():setattr(Native30SurfaceWindow,k,v)


def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.40)
    saved=torch.load(RUN/'candidate/last.pt',map_location='cpu',weights_only=False)
    config=saved['config'];REPORT['config']=config;REPORT['weight_sha256']=sha256(RUN/'candidate/last.pt')
    plan=json.loads((HERE/'runs/native30_overnight_ik_oct14/experiment/window_plan.json').read_text())['validation']
    selected=[]
    for group,mode,regime in (('trumans','exo','noisy'),('camera_wearer','exo','clean'),('rich','ego','clean')):
        index=next(i for i,v in enumerate(plan) if (v['group'],v['mode'])==(group,mode))
        selected.append((index,plan[index],regime))
    noise=dict(config['observation']['input_noise']);noise['clean_probability']=0.
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),json.loads((HERE/config['mount_defaults']).read_text()),noise)
    bank,identities=build_bank(scoped_data(config,'validation'),builder,[v[1] for v in selected])
    model=make_model(config).cuda();model.load_state_dict(saved['model']);model.eval();del saved;gc.collect()
    totals={}
    def timer(name,function):
        def call(*args,**kwargs):
            torch.cuda.synchronize();began=time.monotonic();result=function(*args,**kwargs);torch.cuda.synchronize()
            row=totals.setdefault(name,dict(seconds=0.,calls=0));row['seconds']+=time.monotonic()-began;row['calls']+=1
            return result
        return call
    model._core_pass=timer('two_pass_body_and_conditions',model._core_pass)
    model.local_geometry=timer('local_geometry_and_relations',model.local_geometry)
    model.constrain_motion=timer('observation_constraints',model.constrain_motion)
    physical.guide_motion=timer('physical_guidance_inclusive',physical.guide_motion)
    REPORT['status']='benchmarking';write_json(ROOT/'RUNTIME_BENCHMARK.json',REPORT)
    for (index,_,regime),item,identity in zip(selected,bank,identities):
        observed=augment(item,identity,builder,config,np.random.default_rng(2026101402+index),
                         noisy=regime=='noisy',yaw=False,head_only=True)
        batch=collate_observations([observed]);known=model_inputs(config,batch)
        results={};predictions={};metrics={}
        for name in ('reference','optimized'):
            manager=original_queries() if name=='reference' else contextlib.nullcontext()
            with manager,torch.no_grad():
                totals.clear();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                profile=cProfile.Profile();began=time.monotonic();profile.enable()
                prediction=sample_motion(model,known,config,steps=20,seed=2026101401+index,use_scene=True)
                torch.cuda.synchronize();profile.disable();elapsed=time.monotonic()-began
                predictions[name]=prediction.detach().cpu().numpy()
                component=json.loads(json.dumps(totals))
                begin=time.monotonic();metrics[name]=dict(raw=raw_metrics(prediction,batch,batch['observation_meta'][...,0]>0),surface=surface_metrics(prediction,batch));torch.cuda.synchronize()
                metric_s=time.monotonic()-begin
                summary=pstats.Stats(profile)
                hot=sorted([(v[3],str(k),v[0]) for k,v in summary.stats.items()],reverse=True)[:14]
                results[name]=dict(sampling_s=elapsed,scene_and_raw_metric_s=metric_s,components=component,
                    peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,profile_cumulative_top=hot)
                profile.dump_stats(str(ROOT/f'profile_{identity["group"]}_{identity["mode"]}_{name}.pstats'))
                print('BENCH',identity['group'],identity['mode'],name,round(elapsed,3),flush=True)
                del prediction
        difference=float(np.max(np.abs(predictions['reference']-predictions['optimized'])))
        reference_report=json.loads((RUN/'candidate/evaluation_latest.json').read_text())
        old=next(v for v in reference_report['rows'] if v['case']==index and v['regime']==regime and v['seed']==2026101401 and v['scene']=='full')
        row=dict(group=identity['group'],mode=identity['mode'],regime=regime,case=index,
            identity=identity,config_unchanged=True,prediction_bitwise_equal=bool(np.array_equal(predictions['reference'],predictions['optimized'])),
            maximum_absolute_channel_difference=difference,metrics_equal=metrics['reference']==metrics['optimized'],
            original_trial_mpjpe_mm=old['metrics']['mpjpe_mm']['value'],new_reference_mpjpe_mm=metrics['reference']['raw']['mpjpe_mm']['value'],
            timings=results,sampling_speedup=results['reference']['sampling_s']/results['optimized']['sampling_s'],metrics=metrics)
        REPORT['rows'].append(row);write_json(ROOT/'RUNTIME_BENCHMARK.json',REPORT)
        print('ROW',json.dumps({k:row[k] for k in ('group','mode','prediction_bitwise_equal','maximum_absolute_channel_difference','metrics_equal','sampling_speedup')}),flush=True)
        if difference>1e-6 or not row['metrics_equal']:raise AssertionError('Runtime optimization changed sampled motion or metrics')
        del batch,known,observed;gc.collect();torch.cuda.empty_cache()
    REPORT.update(status='passed',finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write_json(ROOT/'RUNTIME_BENCHMARK.json',REPORT)


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        REPORT.update(status='failed',traceback=traceback.format_exc());write_json(ROOT/'RUNTIME_BENCHMARK.json',REPORT);raise
