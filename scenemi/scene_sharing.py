"""User-authorized bounded sharing launcher for the unified scene resume."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.offline_camera_retrain_v1.ops import resources

GIB = 2 ** 30


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('check', 'run'))
    p.add_argument('--memory-gib', type=float, default=12)
    p.add_argument('--cpu-count', type=int, default=4)
    p.add_argument('--config', type=Path, default=Path('configs/scene_joint128_conditioning.yaml'))
    raw = list(sys.argv[1:]); sep = raw.index('--') if '--' in raw else len(raw)
    a = p.parse_args(raw[:sep])
    command = raw[sep+1:]
    gpu = resources.inspect_gpu()
    free = resources.available_memory()
    required_gpu = 9 * GIB  # 4 GiB job allowance + 1 GiB CUDA + 4 GiB reserve.
    report = dict(device='cuda', available_memory_bytes=free,
                  memory_max_bytes=int(a.memory_gib*GIB),
                  shared_memory_reserve_bytes=4*GIB, reservation=False,
                  other_processes_modified=False, gpu=gpu)
    report['gpu']['user_authorized_external_pids'] = list(gpu['compute_pids'])
    report['gpu']['unknown_compute_pids'] = []
    report['gpu']['admitted'] = gpu['memory_used_mib']*1024**2 + required_gpu <= 24455*1024**2
    report['gpu']['policy'] = 'User-authorized scene resume sharing by live free memory'
    report['memory_admitted'] = free >= report['memory_max_bytes'] + 4*GIB
    report['admitted'] = report['memory_admitted'] and report['gpu']['admitted']
    print(json.dumps(report, indent=2), flush=True)
    if not report['admitted']:
        raise SystemExit('Scene resume sharing admission refused by live memory check')
    if a.action == 'check': return
    if not command: raise SystemExit('run requires command after --')
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = str(max(1, a.cpu_count))
    raise SystemExit(subprocess.call(resources.scope_command(command, a.memory_gib, a.cpu_count), env=env))


if __name__ == '__main__': main()
