"""Strict continuation of the original input-revised baseline through 55000.

Only verified input I/O and recovery of uncommitted log tails are added. The
original wrapper, model, objective, configuration and optimizer are unchanged.
"""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
REPO = BASE.parents[3]
sys.path.insert(0, str(REPO))


def now():
    return datetime.datetime.now().astimezone().isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def verify_protocol(root=ROOT):
    protocol = json.loads((Path(root) / 'PROTOCOL.json').read_text())
    sources = protocol['expected_source_sha256']
    if not sources:
        raise ValueError('Empty continuous-training source hashes')
    required = {str(Path(root) / name) for name in ('worker.py', 'supervise.py')}
    required.add(str(Path(root).parent / 'run_training.py'))
    adapter = Path(protocol['io_adapter'])
    required.add(str(adapter))
    if not adapter.is_absolute() or not required.issubset(sources):
        raise ValueError('Required continuous-training sources are not sealed')
    for path, expected in sources.items():
        if sha256(path) != expected:
            raise ValueError('Continuous-training source changed: ' + path)
    return protocol


def trim_uncommitted(path, step):
    """Keep committed rows; retain the exact discarded bytes for the audit."""
    path = Path(path)
    if not path.exists():
        return None
    kept, discarded = [], []
    with path.open('rb') as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                committed = isinstance(row.get('step'), int) and row['step'] <= step
            except (ValueError, TypeError, AttributeError):
                committed = False
            (kept if committed else discarded).append(line)
    if not discarded:
        return None
    suffix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    evidence = path.with_name(path.name + '.uncommitted.' + suffix)
    with evidence.open('wb') as stream:
        stream.writelines(discarded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary = path.with_name(path.name + '.trim.tmp')
    with temporary.open('wb') as stream:
        stream.writelines(kept)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return dict(path=str(path), committed_step=step, discarded_rows=len(discarded),
                discarded_bytes=sum(map(len, discarded)), retained_evidence=str(evidence))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_original(baseline, adapter, effective, log_path):
    # Install before main(): the original wrapper captures this adapted digest
    # and still checks the entire original TRAIN/VAL fingerprint dictionary.
    adapter.install(SimpleNamespace(trainer=baseline.trainer), effective, log_path)
    previous = sys.argv
    sys.argv = [str(BASE / 'run_training.py'), '--resume', '--review-step', '55000']
    try:
        baseline.main()
    finally:
        sys.argv = previous


def main():
    protocol = verify_protocol()
    import torch
    effective = json.loads((BASE / 'prepared/effective_config.json').read_text())
    state = torch.load(BASE / 'training/last.pt', map_location='cpu', mmap=True,
                       weights_only=False)
    required = ('model', 'optimizer', 'step', 'planned_steps', 'counts', 'data_rng',
                'condition_rng', 'torch_rng', 'cuda_rng', 'numpy_rng', 'effective_config')
    if (any(key not in state for key in required) or state['planned_steps'] != 55000
            or state['effective_config'] != effective):
        raise ValueError('Original baseline cannot resume with identical state and schedule')
    start = state['step']
    if not 0 < start <= 55000:
        raise ValueError('Invalid original baseline step')
    if start == 55000:
        write_json(ROOT / 'WORKER_STATE.json', dict(status='completed55000', step=start,
                                                  checked_at=now()))
        return
    optimizer_entries = len(state['optimizer']['state'])
    del state
    cleanup = []
    for name in ('trace.jsonl', 'validation.jsonl', 'foot_monitor.jsonl', 'scene_ablation.jsonl'):
        result = trim_uncommitted(BASE / 'training' / name, start)
        if result:
            cleanup.append(result)
    write_json(ROOT / 'WORKER_STATE.json', dict(status='starting', pid=os.getpid(),
        started_at=now(), start_step=start, target_step=55000,
        optimizer_state_entries=optimizer_entries, optimizer_reset=False,
        source_configuration_unchanged=True, io_adapter_only=True,
        uncommitted_log_recovery=cleanup))
    baseline = load_module('continuous_original_baseline', BASE / 'run_training.py')
    adapter = load_module('continuous_verified_input_io', protocol['io_adapter'])
    try:
        run_original(baseline, adapter, effective, ROOT / 'IO_RECOVERY.jsonl')
        verify_protocol()
        state = torch.load(BASE / 'training/last.pt', map_location='cpu', mmap=True,
                           weights_only=False)
        step = state['step']
        if state['planned_steps'] != 55000 or 'optimizer' not in state:
            raise ValueError('Baseline returned without a resumable checkpoint')
        del state
        write_json(ROOT / 'WORKER_STATE.json', dict(
            status='completed55000' if step == 55000 else 'stopped',
            pid=os.getpid(), start_step=start, completed_step=step, finished_at=now(),
            target_step=55000, optimizer_reset=False))
    except BaseException as exc:
        write_json(ROOT / 'WORKER_STATE.json', dict(status='failed', pid=os.getpid(),
            start_step=start, failed_at=now(), error=repr(exc), target_step=55000))
        raise


if __name__ == '__main__':
    main()
