"""Shared GPU admission with the user's physical 4 GiB free-memory reserve.

The retained launchers are unchanged. Larger allocator budgets replace the old
5.5/8 GiB ceiling, while unknown jobs, RAM limits and cgroups remain checked.
"""
import argparse
import json
import math
import os
import subprocess
import sys

from scenemi import parallel_resources
from experiments.offline_camera_retrain_v1.ops import resources as legacy


def budget(free_mib,reserve_gib=4.,cuda_gib=1.):
    return math.floor((free_mib/1024-reserve_gib-cuda_gib)*2)/2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','run'])
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--memory-gib',type=float,default=12.)
    parser.add_argument('--cpu-count',type=int,default=4)
    parser.add_argument('--cpu-list',default='14,15,16,17')
    parser.add_argument('--reserve-gib',type=float,default=4.)
    parser.add_argument('--allocator-gib',type=float)
    raw=sys.argv[1:];split=raw.index('--') if '--' in raw else len(raw)
    args=parser.parse_args(raw[:split]);command=raw[split+1:]
    cores=[int(x) for x in args.cpu_list.split(',')]
    if not 0<args.memory_gib<=16 or not 1<=args.cpu_count<=4 or len(set(cores))!=args.cpu_count:
        parser.error('Keep RAM <=16GiB and <=4 selected CPU cores')
    if not set(cores)<=os.sched_getaffinity(0):parser.error('Requested CPU affinity unavailable')
    if (args.action=='run')!=bool(command):parser.error('Only run takes a bounded command')
    modules={'scenemi.replay','scenemi.joint_training','scenemi.exo_continuation',
        'scenemi.native_joint_batched','scenemi.native_joint_recovery','scenemi.native_batch'}
    parallel_resources.MODULES|=modules
    cap=args.allocator_gib if args.allocator_gib is not None else 0.
    report=parallel_resources.admission(args.device,args.memory_gib,cap)
    report.update(allocator_gib=None,physical_free_reserve_gib=None,cpu_affinity=cores)
    print(json.dumps(report,indent=2),flush=True)
    if not report['admitted']:raise SystemExit('Resource admission refused; no job started')
    if command:
        os.sched_setaffinity(0,cores)
        env=dict(os.environ,SCENEMI_ALLOCATOR_GIB=str(cap),SCENEMI_GPU_RESERVE_GIB=str(args.reserve_gib))
        for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[key]='1'
        if args.device=='cpu':env['CUDA_VISIBLE_DEVICES']=''
        raise SystemExit(subprocess.call(legacy.scope_command(command,args.memory_gib,args.cpu_count),env=env))


if __name__=='__main__':main()
