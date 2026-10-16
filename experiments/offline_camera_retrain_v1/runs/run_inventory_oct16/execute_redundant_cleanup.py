#!/usr/bin/env python3
"""Apply the reviewed, exact October 16 checkpoint deletion manifests once."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUNS = HERE.parent


def read(name):
    return json.loads((HERE / name).read_text())


def checked_file(relative):
    path = REPO / relative
    assert path.is_relative_to(RUNS), path
    assert not any(p.is_symlink() for p in [path, *path.parents]), path
    assert path.is_file(), path
    return path


def main():
    report_path = HERE / 'CLEANUP_EXECUTED.json'
    assert not report_path.exists(), 'Already executed or interrupted: inspect report before retry.'
    audit = read('RETAINED_MODELS_AUDIT.json')
    models = {x['retained_checkpoint']: x for x in audit['models']}
    covered = {s for x in audit['models'] for s in x['periodic_snapshots_covered']}
    assert all(x['model_tensors_finite'] and x['optimizer_present'] for x in audit['models'])
    deletions = []
    retained = set()
    periodic = read('PERIODIC_SNAPSHOT_DELETION.json')['files']
    for entry in periodic:
        path = checked_file(entry['path'])
        assert path.name == 'inference.pt' and path.parent.name.startswith('step_')
        assert path.parent.parent.name == 'evaluations'
        assert entry['path'] in covered and entry['retained_last'] in models
        checked_file(entry['retained_last'])
        assert not entry['optimizer_present']
        for record in entry['retained_evaluation_records']:
            checked_file(record)
        retained.add(entry['retained_last'])
        deletions.append(dict(path=entry['path'], bytes=entry['bytes'], kind='periodic_inference'))
    for entry in read('CANDIDATE_DUPLICATE_BEST.json')['files']:
        path = checked_file(entry['remove'])
        last = checked_file(entry['retained'])
        assert path.name == 'best.pt' and last == path.with_name('last.pt')
        assert entry['models_identical']
        assert entry['removed_model_sha256'] == entry['retained_model_sha256']
        retained.add(entry['retained'])
        deletions.append(dict(path=entry['remove'], bytes=entry['bytes'], kind='identical_best'))
    for entry in read('CORRUPT_CHECKPOINTS.json')['files']:
        path = checked_file(entry['path'])
        assert path.parent.name == 'orientation_v3_interrupted'
        assert 'failed finding central directory' in entry['load_error']
        deletions.append(dict(path=entry['path'], bytes=entry['bytes'], kind='unloadable_interrupted_zip'))
    assert len({x['path'] for x in deletions}) == len(deletions) == 85
    for entry in deletions:
        path = checked_file(entry['path'])
        assert path.stat().st_size == entry['bytes'], path
        assert entry['path'] not in retained
    before = shutil.disk_usage(RUNS)
    report = dict(started_at=dt.datetime.now().astimezone().isoformat(),
                  pre_deletion_audit_backup_commit='cc54e1e',
                  status='in_progress', disk_free_before=before.free,
                  plan_bytes=sum(x['bytes'] for x in deletions),
                  planned_files=deletions, deleted_files=[],
                  retained_checkpoints=sorted(retained),
                  retained_model_audit_sha256=hashlib.sha256((HERE/'RETAINED_MODELS_AUDIT.json').read_bytes()).hexdigest(),
                  note='Historical intermediate weights cannot be rerun after deletion; metrics, identities and final models retained.')

    def save():
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(report_path)

    save()
    for entry in deletions:
        path = checked_file(entry['path'])
        path.unlink()
        report['deleted_files'].append(entry)
        save()
    for relative in retained:
        checked_file(relative)
    report.update(status='complete', finished_at=dt.datetime.now().astimezone().isoformat(),
                  deleted_bytes=sum(x['bytes'] for x in report['deleted_files']),
                  deleted_count=len(report['deleted_files']),
                  disk_free_after=shutil.disk_usage(RUNS).free)
    save()
    print(json.dumps({k: v for k, v in report.items() if k not in ['planned_files', 'deleted_files', 'retained_checkpoints']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
