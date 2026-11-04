"""Project default: user-managed GPU allocation, with CPU/RAM isolation.

No GPU occupancy, utilization, process-count or PID identity rejects a launch.
Old allocator/reserve arguments are accepted only for CLI compatibility and
do not impose GPU limits. Actual CUDA errors propagate from the workload.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import yaml

from scenemi.config import ROOT
from experiments.offline_camera_retrain_v1.ops import resources as legacy

POLICY=ROOT/'configs/runtime_policy.yaml'


def load_policy():
    value=yaml.safe_load(POLICY.read_text())
    expected=dict(allocator_limit_gib=None,minimum_free_gib=None,maximum_used_gib=None,
                  maximum_utilization_percent=None,concurrent_job_limit=None,enforce_process_identity=False)
    if (value.get('version')!='user-managed-gpu-v1' or value.get('gpu')!=expected
            or value.get('cpu_ram_isolation') is not True):
        raise ValueError('Unsupported project execution policy')
    return value


def configure_cuda(*args,**kwargs):
    load_policy()
    # Deliberately do not call torch.cuda.set_per_process_memory_fraction.
    return dict(gpu_limits_disabled=True,allocator_gib=None,pytorch_allocator_limit_bytes=None,
                memory_fraction=None,launch_time_reserve_gib=None,cuda_context_included=False)


def admission(device,memory_gib,*args,**kwargs):
    load_policy()
    report=legacy.admission('cpu',memory_gib)
    report.update(device=device,gpu_limits_disabled=True,
                  runtime_policy_sha256=hashlib.sha256(POLICY.read_bytes()).hexdigest())
    if device=='cuda':
        report['gpu']=dict(admitted=True,policy='User-managed GPU; occupancy and job-count gates disabled',
            allocator_gib=None,required_free_mib=0,minimum_free_enforced=False,
            process_count_checked=False,process_identity_checked=False)
    return report


def resource_main(argv=None):
    raw=list(sys.argv[1:] if argv is None else argv)
    cut=raw.index('--') if '--' in raw else len(raw)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','run','_exec'))
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--memory-gib',type=float,default=12)
    parser.add_argument('--cpu-count',type=int,default=4)
    parser.add_argument('--gpu-sharing',action='store_true',help='Compatibility option; GPU sharing is unrestricted')
    parser.add_argument('--allocator-gib',type=float,help='Compatibility option; no GPU allocator limit is installed')
    parser.add_argument('--config',type=Path,help='Compatibility option; GPU budgets in run configs are historical')
    args=parser.parse_args(raw[:cut]);command=raw[cut+1:]
    config=args.config
    if config is None and '--config' in command:
        config=Path(command[command.index('--config')+1])
    if config is not None and config.exists():
        runtime=(yaml.safe_load(config.read_text()) or {}).get('runtime',{})
        if '--memory-gib' not in raw[:cut]:args.memory_gib=runtime.get('memory_gib',args.memory_gib)
        if '--cpu-count' not in raw[:cut]:args.cpu_count=runtime.get('cpu_count',args.cpu_count)
    if (not math.isfinite(args.memory_gib) or not 0<args.memory_gib<=16 or not 1<=args.cpu_count<=4):
        parser.error('CPU/RAM isolation requires RAM <=16GiB and one to four CPU cores')
    if (args.action=='check')==bool(command):parser.error('run requires a command; check takes none')
    if args.action=='_exec':return legacy.bounded_exec(command,args.memory_gib,args.cpu_count)
    report=admission(args.device,args.memory_gib)
    print(json.dumps(report,indent=2),flush=True)
    if not report['admitted']:raise SystemExit('Host RAM admission refused; no worker started')
    if args.action=='check':return
    target=command[1:]
    if target[:1]==['-u']:target=target[1:]
    if args.config is not None and target[:3]==['-m','scenemi','amass'] and '--config' not in target:
        command+=['--config',str(args.config)]
    env=dict(os.environ,PYTHONUNBUFFERED='1')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        env[key]=str(args.cpu_count)
    if args.device=='cpu':env['CUDA_VISIBLE_DEVICES']=''
    raise SystemExit(subprocess.call(legacy.scope_command(command,args.memory_gib,args.cpu_count),env=env))
