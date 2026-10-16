"""Bounded paired fine-tuning trial; preserves the original 30k checkpoint."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--init-from', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--profiles', nargs='+', default=['baseline', 'gait_v1'])
    parser.add_argument('--skeleton-profile', choices=('archived', 'trumans_male_v2'), default='archived')
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--precision', choices=('bf16', 'float32'), default='bf16')
    parser.add_argument('--low-noise-fraction', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--keep-inference-every', type=int, default=1000)
    parser.add_argument('--reference-checkpoint', type=Path,
                        help='If set, run a paired fresh-window, two-seed evaluation after training')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / 'source_snapshot'
    snapshot.mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source, snapshot / source.name)
    shutil.copy2(root / 'model/mdm_scene_unet.py', snapshot / 'mdm_scene_unet.py')
    state = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                 supervisor_pid=os.getpid(), runs=[])

    def save():
        temporary = args.output / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(args.output / 'status.json')

    for profile in args.profiles:
        output = args.output / profile
        output.mkdir()
        command = [sys.executable, '-u', '-m', 'experiments.offline_camera_retrain_v1.train',
                   '--output', str(output), '--init-from', str(args.init_from),
                   '--loss-profile', profile, '--steps', str(args.steps), '--batch-size', '4',
                   '--lr', str(args.lr), '--seed', str(args.seed), '--save-every', '1000',
                   '--ddim-every', '1000', '--ddim-head-only', '--validation-samples', '4']
        command += ['--skeleton-profile', args.skeleton_profile]
        command += ['--precision', args.precision, '--low-noise-fraction', str(args.low_noise_fraction)]
        command += ['--keep-inference-every', str(args.keep_inference_every)]
        with (output / 'console.log').open('w') as stream:
            process = subprocess.Popen(command, cwd=root, env=dict(os.environ, PYTHONPATH=str(root)),
                                       stdout=stream, stderr=subprocess.STDOUT)
            record = dict(profile=profile, command=command, pid=process.pid, status='running')
            state['runs'].append(record)
            save()
            code = process.wait()
        record.update(exit_code=code, status='completed' if code == 0 else 'failed')
        save()
        if code:
            state['status'] = 'failed'
            save()
            raise SystemExit(code)
    if args.reference_checkpoint:
        command = [sys.executable, '-u', '-m', 'experiments.offline_camera_retrain_v1.compare_feet',
                   '--checkpoint', f'original={args.reference_checkpoint}',
                   '--output', str(args.output / 'final_fresh_validation'),
                   '--data-seed', '20261005', '--sampling-seeds', '777', '778',
                   '--samples-per-group', '8']
        for profile in args.profiles:
            checkpoint = args.output / profile / f'evaluations/step_{args.steps:06d}/inference.pt'
            command += ['--checkpoint', f'{profile}={checkpoint}']
        state.update(status='evaluating', final_probe_command=command)
        save()
        with (args.output / 'final_probe.log').open('w') as stream:
            result = subprocess.run(command, cwd=root, env=dict(os.environ, PYTHONPATH=str(root)),
                                    stdout=stream, stderr=subprocess.STDOUT)
        state['final_probe_exit_code'] = result.returncode
        if result.returncode:
            state['status'] = 'evaluation_failed'
            save()
            raise SystemExit(result.returncode)
    state.update(status='completed', ended_utc=datetime.now(timezone.utc).isoformat())
    save()


if __name__ == '__main__':
    main()
