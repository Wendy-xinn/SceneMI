"""Supervise one detached resource-wrapped experiment without importing torch.

SIGSEGV and optionally verified, rebuildable cache faults have separate retry
limits. Optional source metadata restoration requires an exact original SHA256
copy. Other source-data, CUDA, numerical and Python errors stop the command.
The supplied command must already implement atomic checkpoint recovery.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone


def native_segfault(code):
    # systemd-run --scope can return the child's negative signal, subsequently
    # wrapped to unsigned 8-bit by sys.exit in the retained resource launcher.
    return code in (-signal.SIGSEGV, 128+signal.SIGSEGV, 256-signal.SIGSEGV)


def atomic_json(file, value):
    temporary = file.with_name(file.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(file)


def cache_fault(log, offset, cache_root, allow_pinned_metadata=False):
    """Recognize only the retained reader's fatal cache checksum error."""
    marker = 'ValueError: Prepared array checksum differs: '
    with Path(log).open('rb') as stream:
        stream.seek(max(offset, Path(log).stat().st_size-16384))
        lines = stream.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        if allow_pinned_metadata and line == 'ValueError: Selected source metadata changed':
            return 'pinned selected source metadata; restore requires original SHA256 copy'
        if line.startswith(marker):
            candidate = Path(line[len(marker):].strip()).resolve()
            if candidate.is_relative_to(Path(cache_root).resolve()) and candidate.suffix == '.npy':
                return str(candidate)
            return None
    return None


def supervise(command, output, max_native_retries=2, retry_seconds=10, *,
              cache_repair_command=None, cache_root=None, max_cache_repairs=0,
              allow_pinned_metadata=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stopped, child = False, None

    def stop(number, _):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, number)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    environment = dict(os.environ, PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1')
    status_file, history_file = output/'SUPERVISOR.json', output/'supervisor_events.jsonl'
    identity = dict(command=command, pid=os.getpid(), cwd=str(Path.cwd()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        max_native_retries=max_native_retries, max_cache_repairs=max_cache_repairs,
        retry_only='SIGSEGV or original-SHA-verified cache/metadata recovery',
        original_training_sources_unchanged=True, metadata_recovery_requires_original_sha256=True)

    def record(state, **extra):
        value = dict(identity, state=state, at_utc=datetime.now(timezone.utc).isoformat(), **extra)
        atomic_json(status_file, value)
        with history_file.open('a') as stream:
            stream.write(json.dumps(value, ensure_ascii=False)+'\n')
        print(json.dumps(value, ensure_ascii=False), flush=True)

    with (output/'SUPERVISOR.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt, native_retries, cache_repairs = 0, 0, 0
        while True:
            if stopped:
                record('interrupted', attempt=attempt)
                return 130
            with (output/'launcher.log').open('ab', buffering=0) as log:
                offset = log.tell()
                child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                record('running', attempt=attempt+1, child_pid=child.pid)
                code = child.wait()
            if stopped:
                record('interrupted', attempt=attempt+1, returncode=code)
                return 130
            if code == 0:
                record('complete', attempt=attempt+1, returncode=code)
                return 0
            # An uncatchable native failure cannot update the trainer's status.
            trainer_status = output/'STATUS.json'
            previous = json.loads(trainer_status.read_text()) if trainer_status.exists() else {}
            if previous.get('state') == 'running':
                atomic_json(trainer_status, dict(previous, state='failed', detected_at_utc=datetime.now(timezone.utc).isoformat(),
                    error=f'Training process exited with status {code}; see SUPERVISOR.json and launcher.log'))
            damaged = cache_fault(output/'launcher.log', offset, cache_root, allow_pinned_metadata) if cache_root else None
            if (code == 1 and damaged and cache_repair_command and cache_repairs < max_cache_repairs):
                cache_repairs += 1
                record('repairing_verified_cache', attempt=attempt+1, returncode=code,
                       cache_repair=cache_repairs, path=damaged, source_or_membership_changes_allowed=False)
                with (output/'cache_maintenance.log').open('ab', buffering=0) as log:
                    child = subprocess.Popen(cache_repair_command, env=environment, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    repair_code = child.wait()
                if repair_code or stopped:
                    record('failed' if not stopped else 'interrupted', attempt=attempt+1,
                           cache_repair_returncode=repair_code, dependent_stages_stopped=True)
                    return 1
                record('verified_cache_repaired', attempt=attempt+1, cache_repair=cache_repairs,
                       resume_from_atomic_checkpoint=True)
                attempt += 1
                continue
            if not native_segfault(code) or native_retries >= max_native_retries:
                record('failed', attempt=attempt+1, returncode=code, dependent_stages_stopped=True)
                return 1
            record('retrying_native_segfault', attempt=attempt+1, returncode=code,
                   resume_from_atomic_checkpoint=True, retry_seconds=retry_seconds)
            native_retries += 1
            attempt += 1
            deadline = time.monotonic()+retry_seconds
            while not stopped and time.monotonic() < deadline:
                time.sleep(min(.25, max(0, deadline-time.monotonic())))
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-native-retries', type=int, default=2)
    parser.add_argument('--retry-seconds', type=float, default=10)
    parser.add_argument('--cache-repair-config', type=Path)
    parser.add_argument('--max-cache-repairs', type=int, default=3)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.max_native_retries < 0 or args.retry_seconds < 0 or args.max_cache_repairs < 0:
        parser.error('A checkpoint-resumable command and nonnegative retry limits are required')
    repair_command, cache_root = None, None
    if args.cache_repair_config:
        import yaml
        c = yaml.safe_load(args.cache_repair_config.read_text())
        cache_root = Path(c['amass_cache']).resolve()
        repair_command = [sys.executable, '-m', 'scenemi', 'resources', 'run', '--device', 'cpu',
            '--memory-gib', '12', '--cpu-count', '4', '--', sys.executable, '-m', 'scenemi.data.cache_integrity',
            '--config', str(args.cache_repair_config), '--output', str(args.output/'recovery/automatic_cache_repair'),
            '--metadata-mirror', str(args.output/'recovery/verified_metadata'), '--repair']
    raise SystemExit(supervise(command, args.output, args.max_native_retries, args.retry_seconds,
        cache_repair_command=repair_command, cache_root=cache_root,
        max_cache_repairs=args.max_cache_repairs if repair_command else 0, allow_pinned_metadata=bool(repair_command)))


if __name__ == '__main__':
    main()
