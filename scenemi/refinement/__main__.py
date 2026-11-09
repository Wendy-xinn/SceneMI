"""Public Refiner entry with native-crash status reporting.

The small launcher stays CPU-only and never imports the training stack.
Heavy work runs in its child, inside the same resource scope.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]


def record_exit(argv, code):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('action', nargs='?')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/ego_scene_support_refinement.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path)
    args, _ = parser.parse_known_args(argv)
    if args.action not in ('train', 'preflight', 'evaluate'):
        return
    if args.device not in ('cpu', 'cuda'):
        return
    config = yaml.safe_load(args.config.read_text())
    out = (ROOT/config['output']).resolve()
    if not out.is_relative_to(ROOT/'runs'):
        return
    folder = out/'preflight'/args.device if args.action == 'preflight' else out
    if args.action == 'evaluate':
        if args.output is None or not args.output.resolve().is_relative_to(ROOT/'runs'):
            return
        folder = args.output.resolve()
    status = folder/('PROGRESS.json' if args.action == 'evaluate' else 'STATUS.json')
    folder.mkdir(parents=True, exist_ok=True)
    previous = json.loads(status.read_text()) if status.exists() else {}
    if previous.get('state') != 'failed':
        previous['error'] = f'Refiner process exited with code {code}'
    previous.update(state='failed', launcher_exit_code=code,
        native_signal=-code if code < 0 else None, process_alive=False,
        at=datetime.now(timezone.utc).isoformat())
    temporary = status.with_suffix('.partial')
    temporary.write_text(json.dumps(previous, indent=2)+'\n')
    temporary.replace(status)


def main():
    argv = sys.argv[1:]
    process = subprocess.Popen([sys.executable, '-m', 'scenemi.refinement.support_training', *argv],
        env=dict(os.environ, PYTHONFAULTHANDLER='1'))
    def forward(signum, _):
        if process.poll() is None:
            process.send_signal(signum)
    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    code = process.wait()
    if code:
        try:
            record_exit(argv, code)
        except (OSError, ValueError, KeyError) as error:
            print(f'Could not publish Refiner exit status: {error}', file=sys.stderr)
    raise SystemExit(code if code >= 0 else 128-code)


if __name__ == '__main__':
    main()
