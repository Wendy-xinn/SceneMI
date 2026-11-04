"""CPU/RAM-bounded launches using the project user-managed GPU policy."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from scenemi.config import ROOT
from experiments.offline_camera_retrain_v1.ops import resources as legacy

MODULES = {'scenemi.ego_training', 'scenemi.selection', 'scenemi.scene_training',
           'scenemi.geometry.native_shape', 'scenemi.native_ego4_training'}


def own_job(pid):
    try:
        proc = Path('/proc') / str(pid)
        words = proc.joinpath('cmdline').read_bytes().split(b'\0')
        words = [s.decode() for s in words if s]
        module = words[words.index('-m') + 1] if '-m' in words else None
        return (proc.stat().st_uid == os.getuid() and
                proc.joinpath('cwd').resolve() == ROOT and module in MODULES)
    except (OSError, ValueError, IndexError, UnicodeError):
        return False


def admission(device, memory_gib, allocator_gib):
    from scenemi.gpu_policy import admission as current
    return current(device,memory_gib)


def configure_cuda():
    from scenemi.gpu_policy import configure_cuda as current
    return current()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'run'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--memory-gib', type=float, default=6)
    parser.add_argument('--cpu-count', type=int, default=2)
    parser.add_argument('--allocator-gib', type=float, default=4)
    raw = sys.argv[1:]
    divider = raw.index('--') if '--' in raw else len(raw)
    args = parser.parse_args(raw[:divider])
    command = raw[divider+1:]
    if (not 0 < args.memory_gib <= 16 or
            not math.isfinite(args.memory_gib) or
            not 1 <= args.cpu_count <= 4):
        parser.error('Require RAM <=16GiB and <=4 CPU cores')
    if (args.action == 'run') != bool(command):
        parser.error('Only run takes a command after --')
    report = admission(args.device, args.memory_gib, args.allocator_gib)
    print(json.dumps(report, indent=2), flush=True)
    if not report['admitted']:
        raise SystemExit('Resource admission refused; no job was started')
    if command:
        env = dict(os.environ, SCENEMI_ALLOCATOR_GIB=str(args.allocator_gib))
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            env[key] = str(args.cpu_count)
        if args.device == 'cpu':
            env['CUDA_VISIBLE_DEVICES'] = ''
        raise SystemExit(subprocess.call(legacy.scope_command(command, args.memory_gib, args.cpu_count), env=env))


if __name__ == '__main__':
    main()
