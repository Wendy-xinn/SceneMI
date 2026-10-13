"""Read mapped cache bytes and tolerate transient DrvFS metadata operations."""
import hashlib
import errno
import json
import mmap
import os
from pathlib import Path
import time


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
