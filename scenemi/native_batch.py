"""Measure real native EGO/EXO optimizer throughput before choosing a batch.

This uses a separate output directory and discards every benchmark update. The
sealed training engine, other running jobs, datasets and checkpoints are read-only.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
import yaml

from scenemi import native_joint_training as stage
from scenemi.config import ROOT
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.exo_initialization import CONDITIONS
from scenemi.native_joint_recovery import prepare
from scenemi.training import deterministic, now, tensor_digest


def batch_for(data, rng, config, device, forced=False):
    """Actually enlarge both forward batches, retaining the 1:1 mode weighting."""
    count = len(config['training']['sample_order'])//2
    values = {'ego': [], 'exo': []}; identities = []
    for i in range(count):
        for mode in ('ego', 'exo'):
            kwargs = {}
            if forced:
                corpus = ('egobody', 'trumans', 'grail')[i % 3]
                kwargs['member'] = min(j for choices in data[mode].groups[corpus].values() for j in choices)
            if mode == 'exo':
                kwargs['condition'] = CONDITIONS[i % len(CONDITIONS)] if forced else None
                pair = data[mode].sample(rng, config['frontend'], **kwargs)
            else:
                pair = data[mode].sample(rng, **kwargs)
            values[mode].append(pair[0]); identities.append(pair[1])
    return collate(values['ego'], device), collate(values['exo'], device), identities


def main():
    if len(sys.argv)>1 and sys.argv[1]=='run':
        from scenemi import parallel_resources
        from scenemi.native_joint_runtime import main as guarded
        parallel_resources.MODULES |= {'scenemi.native_batch', 'scenemi.native_joint_recovery'}
        guarded(); return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[8,16,32])
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=6)
    args=parser.parse_args()
    if any(v<8 or v%2 for v in args.batches) or args.warmup<1 or args.repeats<3:
        parser.error('Use even total batches >=8, >=1 warmup and >=3 measured updates')
    config=stage.configuration(args.config)
    model,data,cases,amass,amass_cases,reference,seal,out=prepare(config,args.config,'cuda')
    initial=torch.load(ROOT/config['recovery']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)['model']
    expected=tensor_digest(initial)
    result=dict(state='running',started_at_utc=now(),contract_sha256=seal['sha256'],
        config_sha256=sha256(args.config),device=torch.cuda.get_device_name(),
        total_device_bytes=torch.cuda.get_device_properties(0).total_memory,
        allocator_gib=config['runtime']['allocator_gib'],precision='float32_no_tf32',
        data='real_checked_train_windows_only',scene=False,test_used=False,
        other_running_jobs_unchanged=True,initial_model_sha256=expected,
        warmup_updates=args.warmup,measured_updates=args.repeats,rows=[],
        quality_claim='Throughput and finite gradients only; no claim of equal convergence.')
    status=out/'THROUGHPUT.json'
    original_batch=stage.batch_for
    optimizer=None
    for batch in args.batches + list(reversed(args.batches)):
        stage.batch_for=batch_for
        cfg=copy.deepcopy(config);cfg['training']['sample_order']=['ego','exo']*(batch//2)
        model.load_state_dict(initial,strict=True);model.zero_grad(set_to_none=True)
        if optimizer is not None: del optimizer
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        optimizer=stage.optimizer_for(model,cfg);deterministic(config['seed'],'cuda')
        rng=np.random.default_rng(config['seed']);timings=[];cached_times=[]
        row=dict(batch=batch,mode_batch=batch//2,round=1+sum(v['batch']==batch for v in result['rows']))
        try:
            for update in range(args.warmup+args.repeats):
                torch.cuda.synchronize();tick=time.perf_counter()
                value=stage.update(model,optimizer,data['train'],rng,cfg,reference,'cuda',update+1,forced=update==0)
                torch.cuda.synchronize();elapsed=time.perf_counter()-tick
                if update>=args.warmup:timings.append(elapsed)
                if any(not np.isfinite(v) or v<=0 for v in value['gradient_groups'].values()):
                    raise FloatingPointError('Invalid gradient group')
            fixed=batch_for(data['train'],rng,cfg,'cuda',forced=True)
            stage.batch_for=lambda *a,**kw:fixed
            for update in range(args.warmup+args.repeats):
                torch.cuda.synchronize();tick=time.perf_counter()
                stage.update(model,optimizer,data['train'],rng,cfg,reference,'cuda',update+1,forced=True)
                torch.cuda.synchronize();elapsed=time.perf_counter()-tick
                if update>=args.warmup:cached_times.append(elapsed)
            row.update(state='passed',end_to_end_seconds=timings,compute_seconds=cached_times,
                median_seconds=statistics.median(timings),
                mean_seconds=statistics.mean(timings),windows_per_second=batch/statistics.mean(timings),
                compute_windows_per_second=batch/statistics.mean(cached_times),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                identities=value['identities'],all_optimizer_groups_finite=True)
        except torch.OutOfMemoryError as error:
            row.update(state='allocator_limit',error=str(error).splitlines()[0])
            model.zero_grad(set_to_none=True)
        finally:
            stage.batch_for=original_batch
        result['rows'].append(row);result['updated_at_utc']=now()
        atomic_json(status,result)
        print(json.dumps({k:v for k,v in row.items() if k not in ('identities','end_to_end_seconds','compute_seconds')}),flush=True)
    model.load_state_dict(initial,strict=True)
    if tensor_digest(model.state_dict())!=expected:raise AssertionError('Benchmark weights not restored')
    summary={}
    for batch in args.batches:
        rows=[r for r in result['rows'] if r['batch']==batch]
        if all(r['state']=='passed' for r in rows):
            times=[v for r in rows for v in r['end_to_end_seconds']]
            summary[str(batch)]=dict(windows_per_second=batch/statistics.mean(times),
                mean_update_seconds=statistics.mean(times),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows))
    baseline=summary['8']['windows_per_second']
    for value in summary.values():value['speedup_over_8']=value['windows_per_second']/baseline
    result.update(state='complete',summary=summary,initial_weights_restored=True,finished_at_utc=now())
    atomic_json(status,result)
    print(json.dumps(dict(state='complete',summary=summary)),flush=True)


if __name__=='__main__':main()
