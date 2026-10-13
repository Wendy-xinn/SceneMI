"""Read mapped cache bytes and tolerate transient DrvFS metadata operations."""
import hashlib
import json
import mmap
import os
from pathlib import Path
import time


def mapped_sha256(path):
    # DrvFS buffered reads have occasionally returned short/wrong blocks under
    # concurrent load. A read-only mapping also avoids copying a whole array.
    with Path(path).open('rb') as stream:
        if os.fstat(stream.fileno()).st_size==0:return hashlib.sha256(b'').hexdigest()
        with mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
            return hashlib.sha256(mapping).hexdigest()


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
