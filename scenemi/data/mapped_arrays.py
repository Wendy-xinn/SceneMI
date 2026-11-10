"""Read-only numeric NPY mappings without NumPy's legacy header tokenizer.

Keep the existing content-addressed mirror and checksum enforcement. Only the
numeric file-header decoding is replaced; source files and payload bytes never
change. Object/structured arrays are deliberately outside this training API.
"""
import ast
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import struct

import numpy as np


def map_numeric_npy(path):
    path=Path(path)
    with path.open('rb') as stream:
        magic=stream.read(8)
        if magic[:6]!=b'\x93NUMPY' or magic[6:] not in (b'\x01\x00',b'\x02\x00',b'\x03\x00'):
            raise ValueError('Expected a supported NPY file')
        width=2 if magic[6]==1 else 4
        length_bytes=stream.read(width)
        if len(length_bytes)!=width:raise ValueError('Truncated NPY header length')
        length=struct.unpack('<H' if width==2 else '<I',length_bytes)[0]
        if not 0<length<=65536:raise ValueError('Unexpected NPY header size')
        header=stream.read(length)
        if len(header)!=length:raise ValueError('Truncated NPY header')
        fields=ast.literal_eval(header.decode('utf-8' if magic[6]==3 else 'latin1').strip())
        if not isinstance(fields,dict) or set(fields)!={'descr','fortran_order','shape'}:
            raise ValueError('Unexpected NPY header fields')
        shape=fields['shape'];order=fields['fortran_order']
        if (not isinstance(shape,tuple) or len(shape)>8 or
            any(type(n) is not int or n<0 for n in shape) or type(order) is not bool):
            raise ValueError('Invalid NPY shape/order')
        dtype=np.dtype(fields['descr'])
        if dtype.hasobject or dtype.fields or dtype.kind not in 'biufc':
            raise ValueError('Only numeric, non-object training arrays are allowed')
        offset=stream.tell();size=os.fstat(stream.fileno()).st_size
        if size!=offset+math.prod(shape)*dtype.itemsize:raise ValueError('NPY payload length does not match header')
    if math.prod(shape)==0:
        result=np.empty(shape,dtype=dtype,order='F' if order else 'C');result.setflags(write=False)
        return result
    return np.memmap(path,dtype=dtype,mode='r',offset=offset,shape=shape,order='F' if order else 'C')


@contextmanager
def numeric_mmap_backend(trace_file=None):
    original=np.load
    trace_file=Path(trace_file) if trace_file else None

    def load(file,*args,mmap_mode=None,allow_pickle=False,**kwargs):
        if not args and mmap_mode=='r' and not allow_pickle and isinstance(file,(str,os.PathLike)) and str(file).endswith('.npy'):
            if trace_file:
                tmp=trace_file.with_suffix('.partial')
                tmp.write_text(json.dumps(dict(path=str(file),backend='numeric_npy_direct_mapping'))+'\n')
                tmp.replace(trace_file)
            return map_numeric_npy(file)
        return original(file,*args,mmap_mode=mmap_mode,allow_pickle=allow_pickle,**kwargs)

    np.load=load
    try:yield
    finally:np.load=original
