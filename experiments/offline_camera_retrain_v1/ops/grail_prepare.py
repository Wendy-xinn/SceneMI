"""Persistent serial GRAIL preparation with immutable provenance and recovery.

``--limit`` limits this invocation's inventory prefix, not the immutable job's
dataset identity: a four-record check can resume the same job without a limit.
SIGTERM/SIGINT finish the current atomic record and stop before the next one.
No training or CUDA work is started here.
"""
import argparse
import fcntl
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

from . import grail_export, grail_source
from experiments.offline_camera_retrain_v1.native30_cache_io import atomic_text

PROTOCOL = 'grail-persistent-cpu-preparation-v1'
MINIMUM_FREE_BYTES = 10 * 2 ** 30


class JobIdentityError(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def disk_inventory(output):
    points = {'output': Path(output).resolve(), 'linux_workspace': Path(__file__).resolve().parents[3]}
    return {name: dict(path=str(path), total_bytes=usage.total,
                       used_bytes=usage.used, free_bytes=usage.free)
            for name, path in points.items() for usage in [shutil.disk_usage(path)]}


def source_hashes(source_template):
    result = dict(grail_export.source_fingerprint())
    for path in (Path(__file__).resolve(), Path(__file__).with_name('grail_inventory.py').resolve()):
        result[str(path)] = grail_source.file_sha256(path)
    if source_template == 'grail_g1_baked_v1':
        assets = grail_source.g1_template_assets()
        for name, path in assets['paths'].items():
            result[str(Path(path).resolve())] = assets['sha256'][name]
    return result


def check_source_hashes(expected):
    for path, digest in expected.items():
        if grail_source.file_sha256(path) != digest:
            raise JobIdentityError('Preparation source/template changed: ' + path)


def load_inventory(path):
    text = Path(path).read_text()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = payload.get('rows') if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError('GRAIL preparation requires a nonempty complete inventory')
    identities = [row['sequence_id'] for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate GRAIL inventory source identities')
    return rows


def create_or_validate_job(inventory, output, options, rows, limit):
    """A published JOB is immutable, including when resume validation fails."""
    path = output / 'JOB.json'
    identity = dict(protocol=PROTOCOL, inventory_path=str(inventory.resolve()),
        inventory_sha256=grail_source.file_sha256(inventory), inventory_sequences=len(rows),
        source_hashes=source_hashes(options['source_template']), options=options,
        output=str(output.resolve()))
    if path.exists():
        job = json.loads(path.read_text())
        if any(job.get(key) != value for key, value in identity.items()):
            raise JobIdentityError('Existing JOB identity/options/inventory/source hashes changed; use a new job directory')
        return job
    if any((output / name).exists() for name in ('manifest.jsonl', 'review_manifest.jsonl')):
        raise JobIdentityError('Output already contains a manually prepared dataset without this immutable JOB; use a new directory')
    usage = disk_inventory(output)
    if usage['output']['free_bytes'] < MINIMUM_FREE_BYTES:
        raise OSError('GRAIL output has less than 10 GiB free')
    job = dict(identity, created_at_utc=utc_now(), created_pid=os.getpid(),
               first_invocation_limit=limit, disk_before=usage,
               execution_policy='Serial CPU; Torch uses 2 threads, Open3D uses 2 threads; no CUDA or optimizer',
               resume_policy='Immutable job; all completed records must pass source/output checksum verification before reuse')
    write_json(path, job)
    return job


def append_event(path, value):
    """One writer; fsync a complete event before announcing it in STATUS."""
    encoded = (json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    with Path(path).open('ab') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def recover_events(path):
    """Keep every complete JSONL event and record any interrupted final write."""
    path = Path(path)
    if not path.exists():
        return []
    events = []
    good = 0
    with path.open('rb') as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line.endswith(b'\n') or stream.read(1):
                    raise JobIdentityError('Corruption inside preparation event ledger: ' + str(path))
                break
            if not line.endswith(b'\n'):
                break
            events.append(value)
            good += len(line)
    size = path.stat().st_size
    if good < size:
        with path.open('rb') as stream:
            stream.seek(good)
            tail = stream.read()
        quarantine = path.with_name(path.name + '.interrupted.' + str(time.time_ns()))
        quarantine.write_bytes(tail)
        with path.open('r+b') as stream:
            stream.truncate(good)
            stream.flush()
            os.fsync(stream.fileno())
    return events


def output_bytes(row):
    folder = Path(row['path'])
    return sum(path.stat().st_size for path in folder.iterdir() if path.is_file())


def manifest_state(output):
    """Only compact references are resident; full body/visibility data stay local."""
    result = {}
    for name in ('manifest.jsonl', 'review_manifest.jsonl'):
        path = Path(output) / name
        entries = recover_events(path)
        for entry in entries:
            if set(entry) != {'path', 'metadata_sha256'}:
                raise JobIdentityError('Persistent preparation requires compact manifest references')
            key = str(Path(entry['path']).resolve())
            try:
                Path(key).relative_to(Path(output).resolve())
            except ValueError as exc:
                raise JobIdentityError('Bundle reference is outside this preparation job') from exc
            if key in result:
                raise JobIdentityError('Duplicate bundle reference in preparation manifests')
            if grail_source.file_sha256(Path(key) / 'metadata.json') != entry['metadata_sha256']:
                raise JobIdentityError('Published metadata checksum changed: ' + key)
            result[key] = (name, entry['metadata_sha256'])
    return result


def append_compact_manifest(output, row, existing):
    folder = str(Path(row['path']).resolve())
    name = 'manifest.jsonl' if row['trainable'] else 'review_manifest.jsonl'
    digest = grail_source.file_sha256(Path(folder) / 'metadata.json')
    if folder in existing:
        if existing[folder] != (name, digest):
            raise JobIdentityError('Previously published bundle metadata or trainability changed: ' + folder)
    else:
        append_event(Path(output) / name, dict(path=folder, metadata_sha256=digest))
        existing[folder] = (name, digest)
    return Path(output) / name


def check_pinned_input_hashes(row, exported):
    identities = row.get('source_identities', {})
    for inventory_key, output_key in (('recon', 'recon'), ('meta', 'meta'), ('usd', 'object_usd')):
        expected = identities.get(inventory_key, {}).get('sha256')
        if expected is not None and exported['source_hashes'].get(output_key) != expected:
            raise JobIdentityError('Source bytes differ from pinned download inventory: ' + row['sequence_id'] + '/' + inventory_key)


def _run(inventory, output, *, source_template='grail_g1_baked_v1', limit=None,
         body_template='auto', source_fps=24., scope='full'):
    inventory, output = Path(inventory).resolve(), Path(output).resolve()
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ValueError('Preparation --limit must be positive')
    if (source_template not in ('grail_g1_baked_v1', 'standard_smplx_scaled')
            or body_template not in ('auto', 'standard', 'g1') or scope not in ('pilot', 'full')
            or not math.isfinite(source_fps) or not 0 < source_fps <= 240):
        raise ValueError('Invalid immutable preparation options')
    output.mkdir(parents=True, exist_ok=True)
    rows = load_inventory(inventory)
    options = dict(source_template=source_template, body_template=body_template,
                   source_fps=float(source_fps), scope=scope)
    job = create_or_validate_job(inventory, output, options, rows, limit)
    manifests = manifest_state(output)
    ledger = output / 'PROGRESS.jsonl'
    events = recover_events(ledger)
    latest = {event['sequence_id']: event for event in events}
    failures = recover_events(output / 'FAILURES.jsonl')
    previous = json.loads((output / 'STATUS.json').read_text()) if (output / 'STATUS.json').exists() else {}
    previous_elapsed = float(previous.get('elapsed_seconds', 0))
    selected = rows if limit is None else rows[:limit]
    started, stopped = time.monotonic(), {'requested': False, 'signal': None}
    current = None
    invocation_attempts = 0
    originals = {}

    def request_stop(signum, _frame):
        stopped.update(requested=True, signal=signum)

    def status(state, **extra):
        completed = [event for event in latest.values() if event['outcome'] == 'complete']
        failed = [event for event in latest.values() if event['outcome'] == 'data_quality_failure']
        elapsed = previous_elapsed + time.monotonic() - started
        processed = len(completed) + len(failed)
        remaining = max(0, len(rows) - processed)
        value = dict(protocol=PROTOCOL, state=state, pid=os.getpid(), updated_at_utc=utc_now(),
            job_sha256=grail_source.file_sha256(output / 'JOB.json'),
            total_sequences=len(rows), completed=len(completed),
            trainable=sum(event['trainable'] for event in completed),
            quarantine=sum(not event['trainable'] for event in completed),
            failures=len(failed), failure_attempts=len(failures),
            processed=processed, current_sequence=current, invocation_attempts=invocation_attempts,
            invocation_limit=limit, elapsed_seconds=elapsed,
            eta_seconds=elapsed / processed * remaining if processed else None,
            output_bytes=sum(event['output_bytes'] for event in completed),
            completed_manifest=str(output / 'manifest.jsonl'),
            review_manifest=str(output / 'review_manifest.jsonl'),
            progress_ledger=str(ledger), failure_ledger=str(output / 'FAILURES.jsonl'),
            inventory_sha256=job['inventory_sha256'], optimizer_steps=0, no_gpu=True,
            stop_requested=stopped['requested'], **extra)
        write_json(output / 'STATUS.json', value)
        return value

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            originals[signum] = signal.signal(signum, request_stop)
        status('running')
        for index, row in enumerate(selected):
            if stopped['requested']:
                break
            check_source_hashes(job['source_hashes'])
            if index % 100 == 0:
                if grail_source.file_sha256(inventory) != job['inventory_sha256']:
                    raise JobIdentityError('GRAIL inventory changed during processing')
                if shutil.disk_usage(output).free < MINIMUM_FREE_BYTES:
                    raise OSError('GRAIL output has less than 10 GiB free; stopped before the next record')
            current = row['sequence_id']
            status('running', current_inventory_index=index)
            existing = output / grail_export.grail_data.family_split(
                grail_export.normalized_family(row)) / current
            try:
                exported = grail_export.export_record(row, output, **options)
            except grail_export.UnverifiedBodyTemplate:
                raise
            except ValueError as exc:
                check_source_hashes(job['source_hashes'])
                # Existing data checksum/options mismatches are not a reason
                # to skip a source and silently continue with a mixed dataset.
                if existing.exists():
                    raise JobIdentityError('Existing bundle failed resume verification: ' + current) from exc
                failure = dict(sequence_id=current, inventory_index=index, at_utc=utc_now(),
                    outcome='data_quality_failure', error_type=type(exc).__name__, message=str(exc),
                    inventory_sha256=job['inventory_sha256'])
                append_event(output / 'FAILURES.jsonl', failure)
                failures.append(failure)
                append_event(ledger, failure)
                latest[current] = failure
            else:
                check_pinned_input_hashes(row, exported)
                check_source_hashes(job['source_hashes'])
                # Exporter publishes each bundle before its manifest entry.
                # A crash here is recovered by export_record's verified resume.
                manifest = append_compact_manifest(output, exported, manifests)
                event = dict(sequence_id=current, inventory_index=index, at_utc=utc_now(),
                    outcome='complete', trainable=bool(exported['trainable']),
                    path=exported['path'], manifest=str(manifest),
                    metadata_sha256=grail_source.file_sha256(Path(exported['path']) / 'metadata.json'),
                    output_bytes=output_bytes(exported))
                append_event(ledger, event)
                latest[current] = event
            invocation_attempts += 1
            status('running', last_sequence=current)
        current = None
        state = 'stopped' if stopped['requested'] else 'complete' if limit is None or limit >= len(rows) else 'invocation_limit_reached'
        return status(state, disk_after=disk_inventory(output),
                      completed_at_utc=utc_now(), stop_signal=stopped['signal'])
    except BaseException as exc:
        status('stopped_fatal', error_type=type(exc).__name__, error=str(exc),
               disk_after=disk_inventory(output), traceback=traceback.format_exc(limit=8))
        raise
    finally:
        for signum, handler in originals.items():
            signal.signal(signum, handler)


def run(inventory, output, **options):
    """A process lock prevents two launchers from publishing into one job."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'PREPARE.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise JobIdentityError('Another preparation process already holds this output directory') from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(dict(pid=os.getpid(), started_at_utc=utc_now())) + '\n')
        handle.flush()
        try:
            return _run(inventory, output, **options)
        except BaseException as exc:
            # Startup identity rejection happens before _run's progress loop.
            # Preserve JOB while making the failed resume visible in STATUS.
            path = output / 'STATUS.json'
            old = json.loads(path.read_text()) if path.exists() else {}
            if old.get('state') != 'stopped_fatal' or old.get('pid') != os.getpid():
                write_json(path, dict(old, protocol=PROTOCOL, state='stopped_fatal',
                    pid=os.getpid(), updated_at_utc=utc_now(), error_type=type(exc).__name__,
                    error=str(exc), no_gpu=True, optimizer_steps=0))
            raise
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-template', choices=('grail_g1_baked_v1', 'standard_smplx_scaled'),
                        default='grail_g1_baked_v1')
    parser.add_argument('--body-template', choices=('auto', 'standard', 'g1'), default='auto')
    parser.add_argument('--source-fps', type=float, default=24.)
    parser.add_argument('--scope', choices=('pilot', 'full'), default='full')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    result = run(args.inventory, args.output, source_template=args.source_template,
                 body_template=args.body_template, source_fps=args.source_fps,
                 scope=args.scope, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
