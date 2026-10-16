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
                   '--lr', '0.00003', '--seed', '20261002', '--save-every', '1000',
                   '--ddim-every', '1000', '--ddim-head-only', '--validation-samples', '4']
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
    state.update(status='completed', ended_utc=datetime.now(timezone.utc).isoformat())
    save()


if __name__ == '__main__':
    main()
