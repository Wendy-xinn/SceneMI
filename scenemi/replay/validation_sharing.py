"""Bounded launcher for the explicitly authorized replay validation."""
import json
import os
from pathlib import Path
import subprocess
import sys

from scenemi.config import ROOT
from experiments.offline_camera_retrain_v1.ops import resources as legacy
from . import training
from .sharing import authorization


def main():
    config = training.CONFIG
    if '--config' in sys.argv:
        config = Path(sys.argv[sys.argv.index('--config') + 1])
    c = training.configuration(config)
    auth = authorization(ROOT/c['output']/'SHARING_AUTHORIZATION.json', config)
    cores = set(c['runtime']['cpu_affinity'])
    if not cores <= os.sched_getaffinity(0) or len(cores) > 4:
        raise RuntimeError('Configured validation CPU affinity is unavailable')
    os.sched_setaffinity(0, cores)
    # The child performs the live, user-authorized GPU check. This wrapper only
    # installs the same effective RAM/CPU cgroup used by formal training.
    command = [sys.executable, '-m', 'scenemi.replay.validation_runner', *sys.argv[1:]]
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = '1'
    return subprocess.call(legacy.scope_command(command, c['runtime']['memory_gib'],
                                                c['runtime']['cpu_count']), env=env)


if __name__ == '__main__':
    raise SystemExit(main())
