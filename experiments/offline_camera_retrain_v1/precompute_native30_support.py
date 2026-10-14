"""Resumable full TRAIN support/geometry cache, prioritizing the next stage.

Each worker owns its source mappings, body cache and trees. Only observations
enter geometry caches; label caches are separately TRAIN-scoped and excluded
from inference. Files publish atomically and cache keys reject changed inputs.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config,window_options,training_plan,write_json
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_support_cache import TrainingSupportCache
from experiments.offline_camera_retrain_v1.native30_sampling import StratifiedRecordingSampler


def run(config_path,worker,workers,priority_updates,limit=None):
    torch.set_num_threads(1)
    config=read_config(config_path)
    data=Native30Data(HERE/config['train_manifest'],'train',config['seed'],frames=config['frames'],
        memory_protocol=config['scene']['memory_protocol'],quality_config=config['gt_reliability'],
        exo_max_empty_frames=config.get('exo_max_empty_frames'),**window_options(config))
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(device='cpu'),
        json.loads((HERE/config['mount_defaults']).read_text()),config['observation']['input_noise'])
    cache=TrainingSupportCache(data,builder)
    sampler=StratifiedRecordingSampler(data,training_plan(data,config),config['sampler'],config['seed']+371)
    tasks=[];seen=set()
    for _ in range(priority_updates*config['batch_size']):
        row=sampler.draw();key=(row['group'],row['mode'],row['member'],row['start'])
        if key not in seen:tasks.append(key);seen.add(key)
    priority_count=len(tasks)
    for group,members in data.members.items():
        for member,_ in enumerate(members):
            for mode in ('ego','exo'):
                for start in data.starts_for(group,member,mode):
                    key=(group,mode,member,int(start))
                    if key not in seen:tasks.append(key);seen.add(key)
    tasks=tasks[worker::workers]
    if limit is not None:tasks=tasks[:limit]
    folder=HERE/config['prepared_output'];folder.mkdir(parents=True,exist_ok=True)
    out=folder/f'precompute_worker{worker}.json'
    began=time.monotonic()
    report=dict(status='running',split='train',worker=worker,workers=workers,planned_windows=len(tasks),
        total_train_windows=len(seen),priority_stage_windows=priority_count,completed_windows=0,
        optimizer_steps=0,cache_root=str(data.runtime_cache_root),started_pid=os.getpid())
    write_json(out,report)
    try:
        for index,(group,mode,member,start) in enumerate(tasks,1):
            if shutil.disk_usage(data.runtime_cache_root).free<20*2**30:raise RuntimeError('Runtime-cache volume has less than 20GiB free')
            _,identity=data.sample(group,mode,member=member,start=start,yaw=0)
            cache.get(identity)
            if index==1 or index%25==0 or index==len(tasks):
                report.update(completed_windows=index,elapsed_s=time.monotonic()-began,hits=cache.hits,misses=cache.misses,
                    mean_s_per_window=(time.monotonic()-began)/index,last_case=dict(group=group,mode=mode,start=start))
                write_json(out,report);print('PRECOMPUTE',worker,index,len(tasks),round(report['elapsed_s'],2),flush=True)
        report['status']='completed'
    except BaseException:
        import traceback
        report.update(status='failed',traceback=traceback.format_exc());raise
    finally:write_json(out,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--workers',type=int,default=1);parser.add_argument('--priority-updates',type=int,default=2000)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args();run(args.config,args.worker,args.workers,args.priority_updates,args.limit)
