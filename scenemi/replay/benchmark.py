"""Discarded real-data replay optimizer probes with checked inputs and FP32."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import numpy as np
import torch
import yaml

from scenemi import native_joint_training as stage
from scenemi.config import ROOT,load_paths
from scenemi.data import native_repair
from scenemi.data.amass_dataset import AmassDataset,collate
from scenemi.data.amass_prepare import atomic_json,sha256
from scenemi.geometry.temporal import NativeJointMotionPrior
from scenemi.native_joint_data import SceneFreeEgoDataset
from scenemi.parallel_resources import configure_cuda
from scenemi.training import contract,deterministic,ensure_bounded,now,tensor_digest
from .data import Prefetch,ReplaySampler,requests
from .optimization import update

PROBE_CHECKPOINT='runs/scene_joint128_noscene/training/last.pt'
PROBE_SHA='b3091d9ccfca8e135e34b403efc866100c5d64fad9d5e45683208684eb99757a'


def setup():
    ensure_bounded();paths=load_paths(ROOT/'configs/paths.local.yaml')
    c=yaml.safe_load((ROOT/'configs/scene_joint128_noscene_batch32.yaml').read_text())
    checkpoint=ROOT/PROBE_CHECKPOINT
    if sha256(checkpoint)!=PROBE_SHA:raise ValueError('Static performance-probe checkpoint changed')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)['model']
    bases={kind:{key:state['base.native_body.bank.'+kind+'_'+key] for key in
        ('rest_template','rest_directions','eye_template','eye_directions')} for kind in ('smpl','smplx','grail_g1')}
    prior={k[5:]:v for k,v in state.items() if k.startswith('base.') and not k.startswith(('base.native_body.','base.pv_'))}
    local=ROOT/'configs/scene_paths.local.yaml';mount=json.loads(Path(yaml.safe_load(local.read_text())['pv_mount']).read_text())['camera_wearer']
    base=stage.NativeMotionPrior(paths['smplh_models']/'neutral/model.npz',mount,prior_state=prior,bases=bases)
    stage.configure_trainable(base,c);model=NativeJointMotionPrior(base,**c['model'])
    model.load_state_dict(state,strict=True)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    # Same immutable joint bases, avoiding repeated full licensed mesh loads.
    bank=copy.deepcopy(base.native_body.bank).eval();original=native_repair.NativeShapeBank
    print(json.dumps(dict(state='preparing_checked_native_train',updated_at_utc=now())),flush=True)
    try:
        native_repair.NativeShapeBank=lambda *a,**kw:bank
        native=dict(ego=SceneFreeEgoDataset(local,base.shape.rest_template[:1].numpy(),split='train',config=c,selection=ROOT/c['selection']),
            exo=stage.exo_stage.StageDataset(c,'train'))
    finally:native_repair.NativeShapeBank=original
    amass=AmassDataset(ROOT/'runs/amass128/index/complete.json',paths,'train')
    print(json.dumps(dict(state='datasets_ready',updated_at_utc=now())),flush=True)
    sampler=ReplaySampler(amass,native,c['frontend'],c['seed'])
    reference=yaml.safe_load((ROOT/c['reference_config']).read_text())
    return model,initial,sampler,c,reference,paths


def equal_data(sampler,batch,step):
    ordered=[sampler.sample(r) for r in requests(sampler.seed,step,batch)]
    serial=[collate([p[0] for p in ordered[k::2]],'cpu') for k in (0,1)]
    proof=Prefetch(sampler,batch,4)
    try:ego,exo,identities=proof.get(step)
    finally:proof.close()
    if identities!=[p[1] for p in ordered]:raise AssertionError('Parallel data identities differ')
    for expected,actual in zip(serial,(ego,exo)):
        if set(expected)!=set(actual) or any(not torch.equal(expected[k],actual[k]) for k in expected):
            raise AssertionError('Parallel data tensors differ')
    return dict(state='passed',batch=batch,seed=sampler.seed,step=step,workers=[1,4],
        all_tensors_bitwise_equal=True,identities=identities)


def benchmark(args):
    ensure_bounded()
    device='cuda';configure_cuda();deterministic(2026102801,device);torch.set_num_threads(1)
    model,initial,sampler,c,reference,paths=setup();model.to(device)
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    sources=contract(c,paths,ROOT/'runs/amass128/index/complete.json')['sources']
    sources.update({str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'scenemi/replay').glob('*.py')})
    result=dict(state='running',started_at_utc=now(),rows=[],precision='fp32_no_tf32',scene=False,test_used=False,
        benchmark_updates_discarded=True,formal_replay_training_started=False,
        checkpoint=PROBE_CHECKPOINT,checkpoint_sha256=PROBE_SHA,sources=sources,
        selection_sha256=sha256(ROOT/c['selection']),input_source='checked_train_only',
        allocator_gib=float(os.environ['SCENEMI_ALLOCATOR_GIB']),physical_reserve_gib=float(os.environ['SCENEMI_GPU_RESERVE_GIB']),
        device=torch.cuda.get_device_name(),device_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        cpu_affinity=sorted(os.sched_getaffinity(0)),workers=4)
    atomic_json(output/'DATA_PARITY.json',equal_data(sampler,32,731))
    optimizer=None
    # Forward and reverse rounds reduce cache/warm-up bias.
    modes=[('serial',32)]+[('prefetch',b) for b in args.batches]
    for pipeline,batch in modes+list(reversed(modes)):
        if optimizer is not None:del optimizer
        model.load_state_dict(initial,strict=True);model.zero_grad(set_to_none=True)
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        deterministic(c['seed'],device);torch.set_num_threads(1);optimizer=stage.optimizer_for(model,c)
        loader=Prefetch(sampler,batch,4) if pipeline=='prefetch' else None
        row=dict(pipeline=pipeline,batch=batch,mode_batch=batch//2,seconds=[],data_wait_seconds=[],
            round=1+sum(v['pipeline']==pipeline and v['batch']==batch for v in result['rows']))
        try:
            if loader:loader.submit(1001)
            for i in range(args.warmup+args.repeats):
                step=1001+i;torch.cuda.synchronize();tick=time.perf_counter()
                if loader:ego,exo,identities=loader.get(step,device,next_step=step+1)
                else:
                    pairs=[sampler.sample(r) for r in requests(sampler.seed,step,batch)]
                    ego,exo=[collate([p[0] for p in pairs[k::2]],device) for k in (0,1)]
                loaded=time.perf_counter();value=update(model,optimizer,ego,exo,c,reference,step)
                torch.cuda.synchronize();elapsed=time.perf_counter()-tick
                if i>=args.warmup:
                    row['seconds'].append(elapsed);row['data_wait_seconds'].append(loaded-tick)
            row.update(state='passed',mean_update_seconds=statistics.mean(row['seconds']),
                windows_per_second=batch/statistics.mean(row['seconds']),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),last_gradient_check=value,
                open_descriptors=len(list(Path('/proc/self/fd').iterdir())))
        except torch.OutOfMemoryError as error:
            row.update(state='allocator_limit',error=str(error).splitlines()[0]);model.zero_grad(set_to_none=True)
        except BaseException as error:
            row.update(state='failed',error=repr(error));result['rows'].append(row)
            result.update(state='failed',updated_at_utc=now());atomic_json(output/'THROUGHPUT.json',result)
            raise
        finally:
            if loader:loader.close()
        result['rows'].append(row);result['updated_at_utc']=now();atomic_json(output/'THROUGHPUT.json',result)
        print(json.dumps({k:v for k,v in row.items() if k not in ('seconds','data_wait_seconds','last_gradient_check')}),flush=True)
    model.load_state_dict(initial,strict=True)
    if tensor_digest(model.state_dict())!=tensor_digest(initial):raise AssertionError('Probe weights not restored')
    if sha256(ROOT/PROBE_CHECKPOINT)!=PROBE_SHA or any(sha256(ROOT/name)!=digest for name,digest in sources.items()):
        raise ValueError('Benchmark source or parent mutated')
    summary={}
    for pipeline,batch in modes:
        rows=[v for v in result['rows'] if v['pipeline']==pipeline and v['batch']==batch]
        if all(v['state']=='passed' for v in rows):
            timings=[t for v in rows for t in v['seconds']]
            summary[f'{pipeline}_{batch}']=dict(windows_per_second=batch/statistics.mean(timings),
                mean_update_seconds=statistics.mean(timings),peak_allocated_bytes=max(v['peak_allocated_bytes'] for v in rows))
    base=summary['serial_32']['windows_per_second']
    for value in summary.values():value['speedup_over_serial32']=value['windows_per_second']/base
    result.update(state='complete',summary=summary,initial_weights_restored=True,finished_at_utc=now())
    atomic_json(output/'THROUGHPUT.json',result);print(json.dumps(result['summary']),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['benchmark'])
    parser.add_argument('--output',default='runs/replay_benchmark')
    parser.add_argument('--batches',type=int,nargs='+',default=[32,64,128])
    parser.add_argument('--warmup',type=int,default=2);parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if any(b<8 or b%8 for b in args.batches) or args.warmup<1 or args.repeats<3:parser.error('Four equal sources, two modes, >=1 warm-up and >=3 measurements required')
    benchmark(args)
