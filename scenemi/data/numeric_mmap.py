"""Strict read-only numeric NPY mappings without Python AST construction.

Payloads, byte order and array layout are unchanged. Existing dataset checksum
checks remain authoritative. Non-numeric/structured/object files fail closed;
NPZ archives and ordinary loads retain NumPy's existing implementation.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import struct

import numpy as np


_KEY = re.compile(r"\s*(['\"])(descr|fortran_order|shape)\1\s*:\s*")
_DTYPE = re.compile(r"(['\"])([<>=|][biufc][0-9]+)\1")
_SHAPE = re.compile(r"\([^()]*\)")
_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)\Z")


def _fields(header):
    text = header.strip()
    if not text.startswith('{') or not text.endswith('}'):
        raise ValueError('Invalid numeric NPY header dictionary')
    values = {}; at = 1
    while text[at:].strip() != '}':
        key = _KEY.match(text, at)
        if key is None or key[2] in values:
            raise ValueError('Unexpected or duplicate numeric NPY header field')
        name = key[2]; at = key.end()
        if name == 'descr':
            value = _DTYPE.match(text, at)
            if value is None:
                raise ValueError('Only numeric, non-object NPY dtypes are supported')
            values[name] = value[2]; at = value.end()
        elif name == 'fortran_order':
            if text.startswith('True', at):
                values[name] = True; at += 4
            elif text.startswith('False', at):
                values[name] = False; at += 5
            else:
                raise ValueError('Invalid NPY array order')
        else:
            value = _SHAPE.match(text, at)
            if value is None:
                raise ValueError('Invalid NPY shape')
            body = value[0][1:-1].strip()
            parts = [s.strip() for s in body.split(',')] if body else []
            if parts and parts[-1] == '': parts.pop()
            if (len(parts) > 8 or any(_INTEGER.fullmatch(s) is None for s in parts)
                    or len(parts) == 1 and not body.endswith(',')):
                raise ValueError('Invalid NPY shape tuple')
            values[name] = tuple(int(s) for s in parts); at = value.end()
        while at < len(text) and text[at].isspace(): at += 1
        if at < len(text) and text[at] == ',': at += 1
        elif text[at:] != '}': raise ValueError('Invalid NPY header separator')
    if set(values) != {'descr', 'fortran_order', 'shape'}:
        raise ValueError('Missing numeric NPY header fields')
    return values


def map_numeric_npy(path):
    path = Path(path)
    with path.open('rb') as stream:
        magic = stream.read(8)
        if magic[:6] != b'\x93NUMPY' or magic[6:] not in (b'\x01\x00', b'\x02\x00', b'\x03\x00'):
            raise ValueError('Unsupported numeric NPY format')
        width = 2 if magic[6] == 1 else 4
        raw = stream.read(width)
        if len(raw) != width: raise ValueError('Truncated NPY header length')
        length = struct.unpack('<H' if width == 2 else '<I', raw)[0]
        if not 0 < length <= 65536: raise ValueError('Unexpected NPY header size')
        raw = stream.read(length)
        if len(raw) != length: raise ValueError('Truncated NPY header')
        values = _fields(raw.decode('utf-8' if magic[6] == 3 else 'latin1'))
        shape, order = values['shape'], values['fortran_order']
        dtype = np.dtype(values['descr'])
        if dtype.hasobject or dtype.fields or dtype.kind not in 'biufc':
            raise ValueError('Only numeric, non-object NPY arrays are supported')
        offset = stream.tell()
        if os.fstat(stream.fileno()).st_size != offset+math.prod(shape)*dtype.itemsize:
            raise ValueError('NPY payload length differs from its header')
        if math.prod(shape):
            # Map the same open file whose header and length were checked.
            return np.memmap(stream, dtype=dtype, mode='r', offset=offset,
                shape=shape, order='F' if order else 'C')
    result = np.empty(shape, dtype=dtype, order='F' if order else 'C')
    result.setflags(write=False)
    return result


@contextmanager
def numeric_mmap_backend(trace_file=None):
    original = np.load
    trace = Path(trace_file) if trace_file else None

    def load(file, *args, mmap_mode=None, allow_pickle=False, **kwargs):
        if (not args and mmap_mode == 'r' and not allow_pickle
                and isinstance(file, (str, os.PathLike)) and str(file).endswith('.npy')):
            if trace:
                temporary = trace.with_suffix('.partial')
                temporary.write_text(json.dumps(dict(path=str(file), backend='numeric_header_v1'))+'\n')
                temporary.replace(trace)
            return map_numeric_npy(file)
        return original(file, *args, mmap_mode=mmap_mode, allow_pickle=allow_pickle, **kwargs)

    np.load = load
    try: yield
    finally: np.load = original
