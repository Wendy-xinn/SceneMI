"""Read mapped cache bytes and tolerate transient DrvFS metadata operations."""
import hashlib
import errno
import json
import mmap
import os
from pathlib import Path
import time
import uuid
import numpy as np


def buffered_sha256(path):
    with Path(path).open('rb') as stream:
        digest=hashlib.sha256()
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def verified_array(path, expected, mirror_root):
    """Map Linux bytes only: DrvFS mmap and buffered reads can disagree.

    A content-addressed mirror is published only after a full checksum match.
    Source files remain unchanged. The local mirror is a rebuildable read cache.
    """
    path=Path(path)
    if not str(path.resolve()).startswith('/mnt/'):
        verified_sha256(path,expected)
        return np.load(path,mmap_mode='r',allow_pickle=False)
    root=Path(mirror_root);root.mkdir(parents=True,exist_ok=True)
    destination=root/(expected+'.npy')
    if not destination.exists():
        import shutil
        if shutil.disk_usage(root).free-path.stat().st_size<10*2**30:
            raise OSError('Linux verified read cache would leave less than 10GiB free')
        temporary=root/(expected+'.'+uuid.uuid4().hex+'.tmp')
        try:
            matched=False
            for attempt in range(3):
                digest=hashlib.sha256()
                with path.open('rb') as source,temporary.open('wb') as target:
                    for block in iter(lambda:source.read(8*1024*1024),b''):
                        digest.update(block);target.write(block)
                    target.flush();os.fsync(target.fileno())
                if digest.hexdigest()==expected:
                    matched=True;break
            if not matched:raise ValueError(f'Buffered source checksum mismatch: {path}')
            temporary.replace(destination)
        finally:temporary.unlink(missing_ok=True)
    return np.load(destination,mmap_mode='r',allow_pickle=False)


def mapped_sha256(path,*,refresh=False):
    # DrvFS buffered reads have occasionally returned short/wrong blocks under
    # concurrent load. A read-only mapping also avoids copying a whole array.
    with Path(path).open('rb') as stream:
        if refresh and hasattr(os,'posix_fadvise'):
            try:os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            except OSError as exc:
                if exc.errno not in (errno.EINVAL,errno.ENOSYS,errno.EOPNOTSUPP):raise
        if os.fstat(stream.fileno()).st_size==0:return hashlib.sha256(b'').hexdigest()
        with mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
            return hashlib.sha256(mapping).hexdigest()


def verified_sha256(path,expected):
    """Retry stale cached reads; never accept bytes with a different hash."""
    for attempt in range(3):
        actual=mapped_sha256(path,refresh=attempt>0)
        if actual==expected:return actual
    # Opposite read paths have both failed in DrvFS. Accept only an independent
    # buffered read that exactly matches the published digest, never a new SHA.
    actual=buffered_sha256(path)
    if actual==expected:return actual
    raise ValueError(f'Cache SHA256 mismatch: {path}; expected {expected}, read {actual}')


def read_text_retry(path):
    for attempt in range(10):
        try:return Path(path).read_text()
        except (FileNotFoundError,PermissionError):
            if attempt==9:raise
            time.sleep(.1)


def read_json_retry(path):
    for attempt in range(10):
        try:return json.loads(read_text_retry(path))
        except json.JSONDecodeError:
            if attempt==9:raise
            time.sleep(.1)


def atomic_text(path,text):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    for attempt in range(10):
        try:
            with temporary.open('w') as stream:
                stream.write(text);stream.flush();os.fsync(stream.fileno())
            temporary.replace(path);return
        except (FileNotFoundError,PermissionError):
            if attempt==9:raise
            time.sleep(.1)
