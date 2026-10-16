#!/usr/bin/env python3
"""Move only the reviewed, completed historical runs; preserve their contents."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUNS = HERE.parent
DEST = RUNS.parent / 'archive/historical_trials_oct16'


def main():
    plan = json.loads((HERE / 'ARCHIVE_PLAN.json').read_text())
    assert len(plan['directories']) == 29
    DEST.mkdir(parents=True, exist_ok=True)
    report_path = DEST / 'MOVED.json'
    assert not report_path.exists(), 'Inspect existing report before retrying.'
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=REPO).decode().split('\0')
    operations = []
    for entry in plan['directories']:
        source, target = REPO / entry['from'], REPO / entry['to']
        assert source.parent == RUNS and target.parent == DEST
        assert source.name not in plan['protected_runs']
        assert source.is_dir() and not source.is_symlink() and not target.exists()
        assert source.stat().st_dev == DEST.stat().st_dev, 'Use atomic same-filesystem rename.'
        for evidence in entry['completion_evidence']:
            record = json.loads((REPO / evidence['path']).read_text())
            value = record
            for key in evidence['field'].split('.'):
                value = value[key]
            assert value == evidence['value'], evidence
        mapping = [{'from': p, 'to': entry['to'] + p[len(entry['from']):]}
                   for p in tracked if p.startswith(entry['from'] + '/')]
        operations.append(dict(**entry, original_inode=source.stat().st_ino,
                               tracked_files=mapping))
    report = dict(status='moving', started_at=dt.datetime.now().astimezone().isoformat(),
                  reason='User requested fewer failed/redundant runs; preserve compact scientific evidence and required models.',
                  action='Atomic directory rename, no content or checkpoint-state changes.',
                  original_protocol_paths_unchanged=True, directories=[])

    def save():
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(report_path)

    save()
    for operation in operations:
        source, target = REPO / operation['from'], REPO / operation['to']
        os.rename(source, target)
        assert target.stat().st_ino == operation['original_inode'] and not source.exists()
        report['directories'].append(operation)
        save()
    report.update(status='complete', moved_count=len(operations),
                  finished_at=dt.datetime.now().astimezone().isoformat())
    save()
    print(json.dumps({k: v for k, v in report.items() if k != 'directories'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
