import json
import sys
from scenemi.process_supervisor import supervise, native_segfault, cache_fault


def test_only_segfault_is_retryable():
    assert all(native_segfault(x) for x in (-11, 139, 245))
    assert not any(native_segfault(x) for x in (0, 1, 137, 247, -9, 134))


def test_native_retry_is_bounded(tmp_path):
    result = supervise([sys.executable, '-c', 'raise SystemExit(245)'], tmp_path, 1, 0)
    events = [json.loads(s) for s in (tmp_path/'supervisor_events.jsonl').read_text().splitlines()]
    assert result == 1
    assert [e['state'] for e in events] == ['running','retrying_native_segfault','running','failed']
    assert events[-1]['dependent_stages_stopped']


def test_python_error_stops_without_retry_and_repairs_stale_status(tmp_path):
    (tmp_path/'STATUS.json').write_text('{"state":"running"}')
    result = supervise([sys.executable, '-c', 'raise RuntimeError("failed data check")'], tmp_path, 2, 0)
    events = [json.loads(s) for s in (tmp_path/'supervisor_events.jsonl').read_text().splitlines()]
    assert result == 1 and [e['state'] for e in events] == ['running','failed']
    assert json.loads((tmp_path/'STATUS.json').read_text())['state'] == 'failed'


def test_success_does_not_retry(tmp_path):
    assert supervise([sys.executable, '-c', 'pass'], tmp_path, 2, 0) == 0
    assert json.loads((tmp_path/'SUPERVISOR.json').read_text())['state'] == 'complete'


def test_cache_retry_is_explicit_scoped_and_bounded(tmp_path):
    bad = tmp_path/'arrays'/'poses.npy'
    bad.parent.mkdir()
    command = [sys.executable, '-c', f'print("ValueError: Prepared array checksum differs: {bad}");raise SystemExit(1)']
    result = supervise(command, tmp_path, 0, 0, cache_root=bad.parent,
        cache_repair_command=[sys.executable,'-c','pass'], max_cache_repairs=1)
    assert result == 1
    events = [json.loads(s) for s in (tmp_path/'supervisor_events.jsonl').read_text().splitlines()]
    assert [e['state'] for e in events] == ['running','repairing_verified_cache','verified_cache_repaired','running','failed']


def test_cache_fault_rejects_other_paths_and_old_attempt_messages(tmp_path):
    log = tmp_path/'log'
    log.write_text('ValueError: Prepared array checksum differs: /elsewhere/poses.npy\n')
    assert cache_fault(log, 0, tmp_path/'arrays') is None
    log.write_text(f'ValueError: Prepared array checksum differs: {tmp_path}/arrays/poses.npy\n')
    assert cache_fault(log, log.stat().st_size, tmp_path/'arrays') is None


def test_cache_repair_resumes_the_same_command_after_success(tmp_path):
    bad = tmp_path/'arrays'/'poses.npy'
    bad.parent.mkdir()
    bad.write_text('bad')
    code = f'from pathlib import Path;p=Path({str(bad)!r});ok=p.read_text()=="verified";print("ok" if ok else "ValueError: Prepared array checksum differs: "+str(p));raise SystemExit(0 if ok else 1)'
    repair = f'from pathlib import Path;Path({str(bad)!r}).write_text("verified")'
    result = supervise([sys.executable,'-c',code], tmp_path, 0, 0, cache_root=bad.parent,
        cache_repair_command=[sys.executable,'-c',repair], max_cache_repairs=1)
    assert result == 0
    events = [json.loads(s) for s in (tmp_path/'supervisor_events.jsonl').read_text().splitlines()]
    assert [e['state'] for e in events] == ['running','repairing_verified_cache','verified_cache_repaired','running','complete']
