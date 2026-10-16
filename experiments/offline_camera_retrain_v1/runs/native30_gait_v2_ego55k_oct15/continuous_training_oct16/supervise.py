"""Keep the original, unchanged-input-contract baseline training to step 55000.

This controller only resumes its existing full checkpoint. It never selects an
ablation, changes a schedule, or treats a review milestone as completion.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent
HORIZON = 55000
REQUIRED_STATE = ('optimizer', 'model', 'counts', 'data_rng', 'condition_rng',
                  'torch_rng', 'cuda_rng', 'numpy_rng', 'effective_config')


def now():
    return datetime.datetime.now().astimezone().isoformat()


def atomic_json(path, values):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(values, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_event(root, event, **values):
    with (Path(root) / 'EVENTS.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(at=now(), pid=os.getpid(), event=event,
                                     **values), ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_sources(protocol, root=ROOT):
    entries = protocol.get('expected_source_sha256')
    if not isinstance(entries, dict) or not entries:
        raise ValueError('Missing expected_source_sha256')
    root = Path(root)
    required = {root / 'supervise.py', root / 'worker.py',
                root.parent / 'run_training.py',
                root.parent.parent / 'native30_final_review_oct16/io_recovery.py'}
    if not {str(path.resolve()) for path in required}.issubset(entries):
        raise ValueError('Source manifest omits required execution sources')
    for filename, expected in entries.items():
        if not Path(filename).is_absolute() or sha256(filename) != expected:
            raise ValueError('Execution source SHA256 changed: ' + filename)


def validate_checkpoint(state, effective):
    step = state.get('step')
    if (isinstance(step, bool) or not isinstance(step, int) or step < 0
            or step > HORIZON or state.get('planned_steps') != HORIZON):
        raise ValueError('Checkpoint schedule differs from full 55000 baseline')
    if any(name not in state for name in REQUIRED_STATE):
        raise ValueError('Checkpoint is missing full optimizer/RNG state')
    if not state['optimizer'].get('state'):
        raise ValueError('Checkpoint Adam state is empty')
    if state['effective_config'] != effective:
        raise ValueError('Checkpoint effective configuration changed')
    return step


def checkpoint_step(baseline=BASELINE):
    import torch
    baseline = Path(baseline)
    effective = json.loads((baseline / 'prepared/effective_config.json').read_text())
    state = torch.load(baseline / 'training/last.pt', map_location='cpu',
                       mmap=True, weights_only=False)
    step = validate_checkpoint(state, effective)
    del state
    return step


def tail_json(path, max_bytes=262144):
    """Read a bounded tail, accepting only a complete JSON record."""
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            start = max(0, size - max_bytes)
            stream.seek(start)
            lines = stream.read(max_bytes).splitlines(keepends=True)
        if start and lines:
            lines = lines[1:]
        for line in reversed(lines):
            if not line.endswith(b'\n'):
                continue
            try:
                value = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(value, dict):
                return value
    except FileNotFoundError:
        pass
    return None


def quick_progress(baseline, verified_step):
    baseline = Path(baseline)
    trace = tail_json(baseline / 'training/trace.jsonl') or {}
    committed = verified_step
    try:
        protocol = json.loads((baseline / 'training/protocol.json').read_text())
        value = protocol.get('completed_steps')
        if isinstance(value, int) and not isinstance(value, bool):
            committed = max(committed, min(value, HORIZON))
    except (FileNotFoundError, ValueError):
        pass
    return dict(latest_trace_step=trace.get('step'),
                latest_committed_step=committed,
                latest_loss=trace.get('loss'),
                latest_gradient_norm=trace.get('gradient_norm'),
                latest_lr=trace.get('lr'))


def conflicting_processes(root=ROOT, proc_root=Path('/proc'), allowed_pids=()):
    root, proc_root = Path(root).resolve(), Path(proc_root)
    review = root.parent.parent / 'native30_final_review_oct16'
    scripts = {root / 'supervise.py', root / 'worker.py',
               root.parent / 'run_training.py'}
    scripts.update(review / name for name in
                   ('run_review.py', 'continue_review.py', 'recovered_worker.py',
                    'worker.py', 'resume_review.py'))
    ignored = {os.getpid(), *allowed_pids}
    conflicts = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit() or int(directory.name) in ignored:
            continue
        try:
            arguments = (directory / 'cmdline').read_bytes().split(b'\0')
            cwd = (directory / 'cwd').resolve()
            for argument in arguments[1:]:
                text = os.fsdecode(argument)
                if not text.endswith('.py'):
                    continue
                candidate = Path(text)
                candidate = candidate if candidate.is_absolute() else cwd / candidate
                if candidate.resolve() in scripts:
                    conflicts.append(dict(pid=int(directory.name), script=str(candidate.resolve())))
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return conflicts


def read_log_tail(path, offset):
    with Path(path).open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        stream.seek(max(offset, end - 262144))
        return stream.read().decode('utf-8', errors='replace')


def restartable_io_error(log):
    """Only recognized transient I/O failures may trigger bounded retries."""
    lowered = log.lower()
    fatal = ('nonfinite', 'non-finite', 'nan gradient', 'infinity',
             'sha256 mismatch', 'sha256 changed', 'source checksum mismatch',
             'source hashes differ', 'code/calibration changed',
             'native30 data changed', 'configuration changed',
             'resume requires identical', 'assertionerror')
    if any(term in lowered for term in fatal):
        return False
    transient = ('badzipfile', 'bad crc', 'failed crc', 'zlib.error',
                 'input/output error', 'stale file handle',
                 'transport endpoint is not connected',
                 'errno 5', 'errno 116', 'errno 107', 'errno 121')
    return any(term in lowered for term in transient)


def retry_count(previous_count, previous_step, current_step):
    """Progress resets the consecutive-failure budget; no progress consumes it."""
    return 1 if current_step > previous_step else previous_count + 1


def supervise(root=ROOT):
    root = Path(root).resolve()
    baseline = root.parent
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.supervise.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = json.loads((root / 'PROTOCOL.json').read_text())
        verify_sources(protocol, root)
        conflicts = conflicting_processes(root)
        if conflicts:
            raise RuntimeError('Conflicting training/review processes: ' + repr(conflicts))
        current_step = checkpoint_step(baseline)
        status = dict(status='starting', pid=os.getpid(), started_at=now(),
                      goal_step=HORIZON, initial_step=current_step,
                      automatic_ablation=False, optimizer_reset=False,
                      stop_at_review_milestone=False, worker_pid=None)
        stopping = [False]
        child = None
        previous_handlers = {}

        def request_stop(signum, frame):
            stopping[0] = True
            if child is not None and child.poll() is None:
                try:
                    child.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass

        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, request_stop)
        failures = 0
        failure_step = current_step
        limit = protocol.get('restart_limit', 3)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 0 <= limit <= 3:
            raise ValueError('restart_limit must be 0..3')
        interval = max(1., min(30., float(protocol.get('poll_seconds', 10))))
        try:
            while True:
                if current_step == HORIZON:
                    status.update(status='completed', completed_at=now(),
                                  latest_committed_step=current_step, worker_pid=None)
                    atomic_json(root / 'STATUS.json', status)
                    append_event(root, 'completed', step=current_step)
                    return
                if stopping[0]:
                    status.update(status='stopped_by_user', stopped_at=now(),
                                  latest_committed_step=current_step, worker_pid=None)
                    atomic_json(root / 'STATUS.json', status)
                    return
                verify_sources(protocol, root)
                conflicts = conflicting_processes(root)
                if conflicts:
                    raise RuntimeError('Conflicting training/review processes: ' + repr(conflicts))
                output_path = root / 'worker.log'
                offset = output_path.stat().st_size if output_path.exists() else 0
                with output_path.open('ab', buffering=0) as output:
                    child = subprocess.Popen([sys.executable, '-u', str(root / 'worker.py')],
                        cwd=str(baseline.parents[3]), stdout=output,
                        stderr=subprocess.STDOUT, start_new_session=True)
                status.update(status='training', worker_pid=child.pid,
                              worker_started_at=now(), consecutive_io_restarts=failures,
                              verified_resume_step=current_step)
                append_event(root, 'worker_started', worker_pid=child.pid,
                             checkpoint_step=current_step, io_restarts=failures)
                while child.poll() is None:
                    status.update(checked_at=now(), **quick_progress(baseline, current_step))
                    if stopping[0]:
                        status['status'] = 'stopping_waiting_for_safe_checkpoint'
                    atomic_json(root / 'STATUS.json', status)
                    time.sleep(interval)
                result = child.returncode
                status.update(worker_pid=None, last_worker_returncode=result,
                              last_worker_exited_at=now())
                current_step = checkpoint_step(baseline)
                status.update(**quick_progress(baseline, current_step))
                append_event(root, 'worker_exited', returncode=result,
                             checkpoint_step=current_step, user_stop=stopping[0])
                if stopping[0]:
                    status.update(status='stopped_by_user', stopped_at=now(),
                                  latest_committed_step=current_step)
                    atomic_json(root / 'STATUS.json', status)
                    return
                if current_step == HORIZON:
                    if result == 0:
                        continue
                    raise RuntimeError('Worker exited unsuccessfully after final checkpoint; '
                                       f'code={result}, step={current_step}')
                log = read_log_tail(output_path, offset)
                failures = retry_count(failures, failure_step, current_step)
                failure_step = current_step
                if result == 0 or failures > limit or not restartable_io_error(log):
                    raise RuntimeError('Worker stopped before verified completion; '
                        f'code={result}, step={current_step}, consecutive_failures={failures}')
                status.update(status='retrying_transient_io', consecutive_io_restarts=failures)
                atomic_json(root / 'STATUS.json', status)
                append_event(root, 'transient_io_restart', checkpoint_step=current_step,
                             consecutive_io_restarts=failures)
        finally:
            if child is not None and child.poll() is None:
                try:
                    child.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
                append_event(root, 'waiting_for_child_safe_stop', worker_pid=child.pid)
                try:
                    child.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    # Do not kill a process that may still be atomically saving.
                    append_event(root, 'child_save_pending_after_supervisor_failure',
                                 worker_pid=child.pid)
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def main():
    try:
        supervise()
    except BlockingIOError:
        # An already-running controller owns STATUS.json and its own child.
        raise
    except BaseException as failure:
        # Never report a dead or unverified child as successful training.
        record = dict(status='failed', pid=os.getpid(), failed_at=now(),
                      error=repr(failure), goal_step=HORIZON,
                      automatic_ablation=False, optimizer_reset=False)
        atomic_json(ROOT / 'STATUS.json', record)
        append_event(ROOT, 'supervisor_failed', error=repr(failure))
        raise


if __name__ == '__main__':
    main()
