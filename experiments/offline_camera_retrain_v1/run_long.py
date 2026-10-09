"""Detached training supervisor with persistent status and reproducible command."""
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
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('status.json', 'training_log.jsonl')):
        raise FileExistsError(f'Run already exists: {output}')
    command = [sys.executable, '-u', '-m', 'experiments.offline_camera_retrain_v1.train',
               '--output', str(output), '--steps', str(args.steps),
               '--batch-size', str(args.batch_size), '--save-every', '1000',
               '--ddim-every', '1000', '--validation-samples', '4']
    source = output / 'source_snapshot'
    source.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path, source / path.name)
    shutil.copy2(root / 'model/mdm_scene_unet.py', source / 'mdm_scene_unet.py')
    state = dict(status='running', supervisor_pid=os.getpid(), command=command,
                 started_utc=datetime.now(timezone.utc).isoformat())
    status_path = output / 'status.json'

    def save():
        temporary = output / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(status_path)

    environment = dict(os.environ, PYTHONPATH=str(root))
    with (output / 'console.log').open('w') as stream:
        process = subprocess.Popen(command, cwd=root, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT)
        state['training_pid'] = process.pid
        save()
        print(json.dumps(state), flush=True)
        code = process.wait()
    state.update(status='completed' if code == 0 else 'failed', exit_code=code,
                 ended_utc=datetime.now(timezone.utc).isoformat())
    save()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
