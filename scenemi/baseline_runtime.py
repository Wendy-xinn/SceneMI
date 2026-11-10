"""Bound the authorized pretrained-baseline comparison on the shared WSL host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
from experiments.offline_camera_retrain_v1.ops import resources

ROOT = Path(__file__).resolve().parents[1]


def process_identity(pid):
    p = Path('/proc') / str(pid)
    return dict(pid=pid, uid=p.stat().st_uid,
                start_ticks=p.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19],
                cwd=os.readlink(p / 'cwd'), executable=os.readlink(p / 'exe'),
                command_sha256=hashlib.sha256(p.joinpath('cmdline').read_bytes()).hexdigest())


def main():
    raw = sys.argv[1:]; cut = raw.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args(raw[:cut]); command = raw[cut + 1:]
    if command[1:3] != ['-m', 'scenemi.baseline_evaluation']:
        parser.error('Only the baseline evaluation module is authorized')
    c = yaml.safe_load(args.config.read_text()); r = c['runtime']
    report = resources.admission('cpu', r['memory_gib'])
    env = dict(os.environ)
    if args.device == 'cuda':
        # The user authorized sharing with this same, fully identified job.
        grant_path = ROOT.parent / 'egoallo/.local/SHARING_AUTHORIZATION.json'
        grant = json.loads(grant_path.read_text())
        gpu = resources.inspect_gpu()
        for pid in gpu['compute_pids']:
            if process_identity(pid) != grant['allowed_job']:
                raise SystemExit('GPU admission refused: process not covered by user sharing grant')
        total, free = map(int, subprocess.check_output([
            'nvidia-smi', '--id=0', '--query-gpu=memory.total,memory.free',
            '--format=csv,noheader,nounits'], text=True, timeout=10).strip().split(','))
        required = sum(r[k] for k in ('torch_allocator_gib', 'jax_allocator_gib',
                                     'cuda_allowance_gib', 'physical_free_reserve_gib')) * 1024
        report.update(gpu=gpu, free_mib=free, required_free_mib=required)
        report['admitted'] &= free >= required
        env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(r['jax_allocator_gib'] * 1024 / total)
        env['JAX_COMPILATION_CACHE_DIR'] = str(ROOT / c['output'] / 'jax-cache')
    else:
        env['CUDA_VISIBLE_DEVICES'] = ''; env['JAX_PLATFORMS'] = 'cpu'
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = str(r['cpu_count'])
    env['PYTHONUNBUFFERED'] = '1'
    report.update(device=args.device, other_processes_modified=False)
    print(json.dumps(report, indent=2), flush=True)
    if not report['admitted']:
        raise SystemExit('Resource admission refused; no baseline job started')
    raise SystemExit(subprocess.call(resources.scope_command(command, r['memory_gib'], r['cpu_count']), env=env))


if __name__ == '__main__':
    main()
