"""Bounded manual LR/batch branches using the retained AMASS trainer.

Keep the completed trainer/launcher sources immutable. Reuse their config,
optimizer fork, exact-resume acceptance and training loop. Parent checkpoints
are deserialized from the same buffered bytes whose pinned hash was checked.
"""
import argparse
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import torch

from scenemi import amass_finetuning as branch
from scenemi import amass_extension as acceptance
from scenemi.config import ROOT

_parent_state = branch.parent_state


def verified_parent(path, expected):
    path = Path(path)
    before = path.stat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if not 0 < before.st_size <= 2 * 2**30:
        raise ValueError('Expected a parent checkpoint no larger than 2 GiB')
    with io.BytesIO() as buffer:
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                if buffer.tell() + len(block) > 2 * 2**30:
                    raise ValueError('Parent checkpoint grew beyond the read budget')
                buffer.write(block)
        if stamp(path.stat()) != stamp(before):
            raise ValueError('Parent checkpoint changed during read')
        if hashlib.sha256(buffer.getbuffer()).hexdigest() != expected:
            raise ValueError('Pinned parent checkpoint changed')
        with zipfile.ZipFile(buffer) as archive:
            if archive.testzip() is not None:
                raise ValueError('Parent checkpoint archive CRC failed')
        buffer.seek(0)
        state = torch.load(buffer, map_location='cpu', weights_only=False)
        if hashlib.sha256(buffer.getbuffer()).hexdigest() != expected:
            raise ValueError('Parent checkpoint buffer changed during load')
    return state


@contextmanager
def parent_reader(c):
    """Replace only this module's parent deserializer, restoring it on failure."""
    original = branch.torch
    checkpoint = (ROOT / c['continuation']['parent_checkpoint']).resolve()

    class Reader:
        def __getattr__(self, name):
            return getattr(original, name)

        def load(self, path, *, map_location, weights_only, mmap):
            if (Path(path).resolve() != checkpoint or map_location != 'cpu'
                    or weights_only is not False or mmap is not True):
                raise ValueError('Unexpected parent checkpoint load')
            return verified_parent(path, c['continuation']['parent_sha256'])

    branch.torch = Reader()
    try:
        yield
    finally:
        branch.torch = original


def parent_state(c, paths, seal):
    with parent_reader(c):
        state, best, provenance = _parent_state(c, paths, seal)
    provenance.update(parent_loading='pinned buffered snapshot; complete archive CRC; no mmap')
    return state, best, provenance


def run(c, paths_file, *, preflight_only=False):
    print(f"AMASS LR branch: {c['continuation']['parent_step']} -> "
          f"{c['training']['planned_updates']}; effective batch={c['training']['batch_size']}; "
          f"constant lr={c['optimizer']['lr']:.2e}; output={c['output']}", flush=True)
    phases = [('preflight', 'cpu'), ('preflight', 'cuda')]
    if not preflight_only:
        seal = branch.base.contract(c, branch.load_paths(paths_file))
        try:
            branch.base.require_preflights(c, seal)
        except (OSError, ValueError, KeyError):
            pass
        else:
            phases.clear()
            print('Reusing current CPU/CUDA exact-resume acceptance.', flush=True)
        phases.append(('train', 'cuda'))
    runtime = c['runtime']
    for action, device in phases:
        command = [sys.executable, '-m', 'scenemi', 'resources', 'run',
            '--device', device, '--memory-gib', str(runtime['memory_gib']),
            '--cpu-count', str(runtime['cpu_count']), '--', sys.executable, '-u',
            '-m', 'scenemi.amass_branch_runtime', action, '--config', c['_config_path'],
            '--paths', str(Path(paths_file).resolve()), '--device', device]
        if action == 'train' and (ROOT / c['output'] / 'training/last.pt').exists():
            command.append('--resume')
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            record = ROOT / c['output'] / (
                f'preflight/{device}/RESULT.json' if action == 'preflight' else 'training/STATUS.json')
            reason = ''
            if record.exists():
                value = json.loads(record.read_text())
                if value.get('state') == 'failed':
                    reason = value.get('error', '')
            raise SystemExit(f'{device} {action} failed (exit {result.returncode}): {reason}\n{record}')
    if preflight_only:
        print('CPU/CUDA acceptance passed. Formal training has not started.', flush=True)


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    # Bind the retained implementations without changing any sealed source.
    branch.load_config = acceptance.load_config
    branch.parent_state = parent_state
    branch.preflight = acceptance.preflight
    if raw[:1] in (['preflight'], ['train']):
        return branch.main(raw)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--paths', type=Path, default=ROOT / 'configs/amass_paths.local.yaml')
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument('--check', action='store_true')
    checks.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(raw)
    if args.check:
        return branch.main(['config', '--config', str(args.config), '--paths', str(args.paths)])
    return run(branch.load_config(args.config), args.paths, preflight_only=args.preflight_only)


if __name__ == '__main__':
    main()
