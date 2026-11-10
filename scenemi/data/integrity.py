"""Audit sealed prepared data and recreate exact originals with the CPU exporter.

Recovery writes a separate candidate only. Neither the source nor the manifest
is changed, and a candidate is accepted only if its complete original SHA256
matches. This utility has no training, test-set, or GPU entry point.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def audit(index_path, prepared_root, output):
    """Hash every eligible train/val variant, never open independent test data."""
    index_path, prepared_root, output = Path(index_path), Path(prepared_root), Path(output)
    index = json.loads(index_path.read_text())
    rows = [row for row in index['families'] if row['split'] in ('train', 'val')]
    variants = [(row['split'], v) for row in rows for v in row['variants']]
    issues, count, read_bytes = [], 0, 0
    started, tick = time.monotonic(), time.monotonic()
    status = dict(state='auditing', index_sha256=digest(index_path), splits=['train', 'val'],
                  independent_test_read=False, total_files=len(variants), issues=issues)
    def report(state):
        status.update(state=state, checked_files=count, checked_bytes=read_bytes,
                      elapsed_seconds=time.monotonic() - started,
                      updated_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(output, status)
        print(json.dumps({k: status[k] for k in ('state', 'checked_files', 'total_files',
              'checked_bytes', 'elapsed_seconds')}, ensure_ascii=False), flush=True)
    report('auditing')
    for split, variant in variants:
        path = prepared_root / variant['path']
        if not path.resolve().is_relative_to(prepared_root.resolve()):
            raise ValueError('Prepared path escapes root')
        try:
            size, actual = path.stat().st_size, digest(path)
            read_bytes += size
            if size != variant['bytes'] or actual != variant['sha256']:
                issues.append(dict(path=variant['path'], split=split, bytes=size,
                                   expected_bytes=variant['bytes'], sha256=actual,
                                   expected_sha256=variant['sha256']))
        except OSError as error:
            issues.append(dict(path=variant['path'], split=split, error=repr(error)))
        count += 1
        if time.monotonic() - tick >= 20:
            report('auditing')
            tick = time.monotonic()
    report('passed' if not issues else 'mismatches_found')
    return status


def reprocess(index_path, paths_path, relatives, output):
    """Recreate isolated train/val candidates using the original CPU exporter."""
    import os
    import torch
    from scenemi.config import ROOT, load_paths
    from scenemi.data import amass_prepare as original
    from scenemi.data.amass_floor import HeightDBSCAN

    relative_group = Path('/proc/self/cgroup').read_text().strip().split(':', 2)[2]
    group = Path('/sys/fs/cgroup') / relative_group.lstrip('/')
    maximum = (group / 'memory.max').read_text().strip()
    if maximum == 'max' or int(maximum) > 6 * 2**30 or len(os.sched_getaffinity(0)) > 4:
        raise ValueError('Use the resource wrapper with <=6 GiB RAM and <=4 CPU cores')
    paths = load_paths(Path(paths_path))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    job = json.loads((paths['prepared_amass'] / 'JOB.json').read_text())
    source_checks = []
    for name, expected in job['local_source_sha256'].items():
        actual = digest(ROOT / name)
        # The old root CLI is not executed by this isolated process_seq call.
        # Every original data-transform dependency is still required verbatim.
        required = name != 'scenemi/__main__.py'
        source_checks.append(dict(path=name, expected_sha256=expected, sha256=actual,
                                  matches=actual == expected, executed_dependency=required))
        if required and actual != expected:
            raise ValueError('Original preprocessing source changed: ' + name)
    for name, expected in job['upstream']['source_sha256'].items():
        if digest(paths['egoallo_repo'] / name) != expected:
            raise ValueError('Original upstream source changed: ' + name)
    for name, expected in job['assets_sha256'].items():
        if digest(name) != expected:
            raise ValueError('Original preprocessing asset changed: ' + name)
    module = original.load_preprocessor(paths['egoallo_repo'])
    if job.get('execution_adapter', {}).get('name') != 'linear_memory_1d_height_dbscan_v1':
        raise ValueError('Unexpected original preprocessing execution adapter')
    evidence = paths['output'] / 'preparation/memory_equivalence/RESULT.json'
    if digest(evidence) != job['execution_adapter']['equivalence_sha256']:
        raise ValueError('Original bounded-export equivalence evidence changed')
    module.DBSCAN = HeightDBSCAN
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    index = json.loads(Path(index_path).read_text())
    lookup = {v['path']: (row, v) for row in index['families']
              if row['split'] in ('train', 'val') for v in row['variants']}
    report = dict(state='reprocessing', source_files_unchanged=True, independent_test_read=False,
                  index_sha256=digest(index_path), original_local_source_checks=source_checks,
                  entrypoint='Direct original process_seq; historical root CLI not imported', candidates=[])
    for relative in relatives:
        row, variant = lookup[relative]
        raw = paths['amass_raw'] / row['source']
        if digest(raw) != row['raw_sha256']:
            raise ValueError('Original raw source is also damaged: ' + row['source'])
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError('Refuse to overwrite a recovery candidate: ' + str(target))
        with original.working_directory(paths['egoallo_repo']):
            module.process_seq(str(raw), str(target), str(paths['smplh_models']),
                               dev_id=0, beta_neutral=True, reflect=variant['reflected'],
                               split_frame_limit=256, overwrite=False)
        actual = digest(target)
        candidate = dict(path=relative, candidate=str(target), sha256=actual,
                         expected_sha256=variant['sha256'], exact_original=actual == variant['sha256'])
        report['candidates'].append(candidate)
        atomic_json(output / 'RESULT.json', report)
        print(json.dumps(candidate), flush=True)
        if actual != variant['sha256']:
            raise ValueError('Recreated file differs from sealed original; do not replace training data')
    report['state'] = 'exact_originals_recreated'
    atomic_json(output / 'RESULT.json', report)
    return report


def snapshot(index_path, prepared_root, destination, replacement_roots, output):
    """Copy exact train/val bytes to local storage without adopting mismatches."""
    from concurrent.futures import ThreadPoolExecutor
    import os
    import shutil
    index_path, prepared_root = Path(index_path), Path(prepared_root)
    destination, output = Path(destination), Path(output)
    index = json.loads(index_path.read_text())
    variants = [v for row in index['families'] if row['split'] in ('train', 'val')
                for v in row['variants']]
    destination.mkdir(parents=True, exist_ok=True)
    required = sum(v['bytes'] for v in variants)
    if shutil.disk_usage(destination).free < required + 10 * 2**30:
        raise ValueError('Require full snapshot size plus 10 GiB disk reserve')
    started = time.monotonic()
    report = dict(state='copying', index_sha256=digest(index_path), source=str(prepared_root),
                  destination=str(destination), independent_test_read=False,
                  total_files=len(variants), checked_files=0, checked_bytes=0,
                  replacements=[], issues=[], workers=4)

    def copy_one(variant):
        relative, expected = variant['path'], variant['sha256']
        target = destination / relative
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError('Snapshot target escapes destination')
        if target.exists() and digest(target) == expected:
            return dict(bytes=variant['bytes'])
        source = prepared_root / relative
        for replacement in replacement_roots:
            alternative = Path(replacement) / relative
            if alternative.exists():
                source = alternative
                break
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + '.snapshot.tmp')
        h = hashlib.sha256()
        try:
            with source.open('rb') as src, temp.open('wb') as dst:
                for block in iter(lambda: src.read(4 * 1024**2), b''):
                    h.update(block)
                    dst.write(block)
                dst.flush()
                os.fsync(dst.fileno())
            actual = h.hexdigest()
            if actual != expected:
                return dict(issue=dict(path=relative, source=str(source), sha256=actual,
                                       expected_sha256=expected))
            temp.replace(target)
            if digest(target) != expected:
                return dict(issue=dict(path=relative, error='Local snapshot reread failed SHA256'))
            return dict(bytes=variant['bytes'], replacement=dict(path=relative, source=str(source),
                        sha256=expected) if source != prepared_root / relative else None)
        except OSError as error:
            return dict(issue=dict(path=relative, error=repr(error)))
        finally:
            temp.unlink(missing_ok=True)

    tick = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(copy_one, variants):
            report['checked_files'] += 1
            report['checked_bytes'] += result.get('bytes', 0)
            if result.get('issue'):
                report['issues'].append(result['issue'])
            if result.get('replacement'):
                report['replacements'].append(result['replacement'])
            if time.monotonic() - tick >= 10:
                report.update(elapsed_seconds=time.monotonic()-started,
                              updated_at_utc=datetime.now(timezone.utc).isoformat())
                atomic_json(output, report)
                print(json.dumps(dict(state=report['state'], checked_files=report['checked_files'],
                                      total_files=len(variants), issues=len(report['issues']))), flush=True)
                tick = time.monotonic()
    if not report['issues']:
        for filename, key in [('JOB.json', 'job_sha256'), ('manifest.jsonl', 'manifest_sha256')]:
            source = prepared_root / filename
            if digest(source) != index[key]:
                raise ValueError('Original metadata changed: ' + filename)
            shutil.copyfile(source, destination / filename)
        # Preserve future explicit test access, but do not open or copy test arrays.
        test_corpora = {row['corpus'] for row in index['families'] if row['split'] == 'test'}
        for corpus in sorted(test_corpora):
            target = destination / 'neutral' / corpus
            original = prepared_root / 'neutral' / corpus
            if not target.exists():
                target.symlink_to(original, target_is_directory=True)
        report['independent_test_directory_links'] = sorted(test_corpora)
    report.update(state='passed' if not report['issues'] else 'mismatches_found',
                  elapsed_seconds=time.monotonic()-started,
                  updated_at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(output, report)
    print(json.dumps(dict(state=report['state'], checked_files=report['checked_files'],
                         total_files=len(variants), issues=len(report['issues']))), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    check = commands.add_parser('audit')
    check.add_argument('--index', type=Path, required=True)
    check.add_argument('--prepared-root', type=Path, required=True)
    check.add_argument('--output', type=Path, required=True)
    recreate = commands.add_parser('reprocess')
    recreate.add_argument('--index', type=Path, required=True)
    recreate.add_argument('--paths', type=Path, required=True)
    recreate.add_argument('--relative', action='append', required=True)
    recreate.add_argument('--output', type=Path, required=True)
    mirror = commands.add_parser('snapshot')
    mirror.add_argument('--index', type=Path, required=True)
    mirror.add_argument('--prepared-root', type=Path, required=True)
    mirror.add_argument('--destination', type=Path, required=True)
    mirror.add_argument('--replacement-root', type=Path, action='append', default=[])
    mirror.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'audit':
        result = audit(args.index, args.prepared_root, args.output)
        if result['state'] != 'passed':
            raise SystemExit(2)
    elif args.action == 'reprocess':
        reprocess(args.index, args.paths, args.relative, args.output)
    else:
        result = snapshot(args.index, args.prepared_root, args.destination, args.replacement_root, args.output)
        if result['state'] != 'passed':
            raise SystemExit(2)


if __name__ == '__main__':
    main()
