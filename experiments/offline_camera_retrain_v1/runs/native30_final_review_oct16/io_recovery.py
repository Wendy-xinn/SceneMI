"""Runtime-only recovery for verified Native30 inputs and derived NPZ caches.

No original source or sealed preparation is modified. Source digests can only
be repaired to their already published value. A corrupt derived cache is kept
as evidence and recreated by the unchanged original cache implementation.
"""
import datetime
import functools
import hashlib
import io
import json
import mmap
import os
from pathlib import Path
import uuid
import zipfile
import zlib


class CorruptRuntimeCache(OSError):
    def __init__(self, path, attempts):
        self.path = Path(path)
        self.attempts = attempts
        super().__init__('Derived Native30 cache failed CRC: ' + str(path))


def timestamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record(log_path, **values):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(at=timestamp(), pid=os.getpid(), **values)
    with path.open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def runtime_path(path):
    """Limit recovery to three explicitly rebuildable Native30 cache families."""
    if not isinstance(path, (str, os.PathLike)):
        return False
    target = Path(path).resolve()
    return (target.suffix == '.npz' and target.parent.name in
            {'surfaces', 'support', 'static_maps'} and
            target.parent.parent.name == 'native30_runtime_cache_v2')


def checked_bytes(path):
    """Use independent reads; validate the entire ZIP before lazy np.load."""
    attempts = []
    for index in range(3):
        for method in ('buffered', 'mapped_refresh'):
            digest = None
            try:
                if method == 'buffered':
                    data = Path(path).read_bytes()
                else:
                    with Path(path).open('rb') as stream:
                        if hasattr(os, 'posix_fadvise'):
                            try:
                                os.posix_fadvise(stream.fileno(), 0, 0,
                                                 os.POSIX_FADV_DONTNEED)
                            except OSError:
                                pass
                        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
                            data = mapping[:]
                digest = hashlib.sha256(data).hexdigest()
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    invalid = archive.testzip()
                if invalid is not None:
                    raise zipfile.BadZipFile('Bad CRC for ' + invalid)
                return data, attempts
            except (OSError, ValueError, EOFError, zipfile.BadZipFile, zlib.error) as exc:
                attempts.append(dict(attempt=index, method=method,
                                     sha256=digest, error=repr(exc)))
    raise CorruptRuntimeCache(path, attempts)


class NumpyProxy:
    def __init__(self, original, log_path):
        self.original, self.log_path = original, Path(log_path)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def load(self, file, *args, **kwargs):
        if not runtime_path(file):
            return self.original.load(file, *args, **kwargs)
        data, attempts = checked_bytes(file)
        if attempts:
            record(self.log_path, event='runtime_read_retried', path=str(file),
                   accepted_sha256=hashlib.sha256(data).hexdigest(), attempts=attempts)
        return self.original.load(io.BytesIO(data), *args, **kwargs)


def cache_recovery(original, log_path):
    @functools.wraps(original)
    def recovered(*args, **kwargs):
        for retry in range(4):
            try:
                return original(*args, **kwargs)
            except CorruptRuntimeCache as exc:
                if retry == 3:
                    raise
                path = exc.path
                if not runtime_path(path):
                    raise
                quarantine = path.with_name(path.name + '.corrupt.' + uuid.uuid4().hex)
                previous = file_sha256(path)
                # Same-directory rename is atomic. No source arrays are removed.
                path.replace(quarantine)
                record(log_path, event='derived_cache_quarantined', path=str(path),
                       quarantine=str(quarantine), corrupt_sha256=previous,
                       bytes=quarantine.stat().st_size, attempts=exc.attempts)
                try:
                    result = original(*args, **kwargs)
                    if not path.exists():
                        raise RuntimeError('Original cache method did not recreate ' + str(path))
                    data, attempts = checked_bytes(path)
                    record(log_path, event='derived_cache_rebuilt', path=str(path),
                           quarantine=str(quarantine),
                           rebuilt_sha256=hashlib.sha256(data).hexdigest(),
                           bytes=len(data), attempts=attempts,
                           source_and_implementation='unchanged original cache method')
                    return result
                except CorruptRuntimeCache:
                    if retry == 2:
                        raise
                except BaseException as failure:
                    record(log_path, event='derived_cache_rebuild_failed', path=str(path),
                           quarantine=str(quarantine), error=repr(failure))
                    raise
        raise RuntimeError('Runtime cache recovery exhausted')
    return recovered


def audited_digest(original, expected, log_path, verifier):
    @functools.wraps(original)
    def checked(data):
        result = original(data)
        for path, actual in list(result.items()):
            if path not in expected:
                raise ValueError('Unexpected source fingerprint entry: ' + path)
            published = expected[path]
            if actual != published:
                # The independent verifier must return the original digest.
                accepted = verifier(Path(path), published)
                if accepted != published:
                    raise ValueError('Source verifier did not confirm original SHA: ' + path)
                record(log_path, event='source_digest_read_retried', path=path,
                       transient_read_sha256=actual, published_sha256=published,
                       accepted_sha256=accepted)
                result[path] = accepted
        return result
    return checked


def install(worker, effective, log_path):
    """Install after importing original worker, before calling its main()."""
    from experiments.offline_camera_retrain_v1 import native30_surface_cache as surfaces
    from experiments.offline_camera_retrain_v1 import native30_support_cache as support
    from experiments.offline_camera_retrain_v1 import native30_memory as memory
    from experiments.offline_camera_retrain_v1.native30_cache_io import verified_sha256
    for module in (surfaces, support, memory):
        if isinstance(module.np, NumpyProxy):
            raise RuntimeError('I/O recovery adapter already installed')
        module.np = NumpyProxy(module.np, log_path)
    surfaces.CanonicalSurfaceCache.get = cache_recovery(surfaces.CanonicalSurfaceCache.get, log_path)
    support.TrainingSupportCache.get = cache_recovery(support.TrainingSupportCache.get, log_path)
    memory.ObservedStaticMemory._map = cache_recovery(memory.ObservedStaticMemory._map, log_path)
    worker.trainer.dataset_digest = audited_digest(worker.trainer.dataset_digest,
        effective['data_sha256'], log_path, verified_sha256)
    record(log_path, event='adapter_installed', source_digest_policy='original SHA only',
           derived_cache_policy='CRC validation, independent retries, quarantine and original rebuild')
