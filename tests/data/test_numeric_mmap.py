"""NPY reader equivalence and rejection without AST/parser execution."""
import ast
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest

from scenemi.data.numeric_mmap import map_numeric_npy, numeric_mmap_backend, _fields


@pytest.mark.parametrize('version', [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize('value', [np.arange(48, dtype=np.float32).reshape(2, 8, 3),
    np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4)),
    np.array([True, False]), np.arange(5, dtype='>i4'), np.empty((0, 3), np.float32),
    np.array(3.5, dtype=np.float32), np.array([1+2j], dtype=np.complex64)])
def test_numeric_mapping_preserves_bytes_layout_without_ast(tmp_path, monkeypatch, version, value):
    file = tmp_path/'array.npy'
    with file.open('wb') as stream: np.lib.format.write_array(stream, value, version=version)
    reference = np.load(file)
    def fail(*_a, **_k): raise AssertionError('AST parsing must not be used')
    monkeypatch.setattr(ast, 'parse', fail);monkeypatch.setattr(ast, 'literal_eval', fail)
    actual = map_numeric_npy(file)
    assert actual.dtype == reference.dtype and actual.shape == reference.shape
    assert np.array_equal(actual, reference) and actual.tobytes() == reference.tobytes()
    assert actual.strides == reference.strides or not actual.size
    assert not actual.flags.writeable
    if actual.size:
        with pytest.raises(ValueError): actual.flat[0] = 0
    with numeric_mmap_backend(): assert np.array_equal(np.load(file, mmap_mode='r'), value)


def test_numeric_backend_preserves_numpy_archives_and_restores_loader(tmp_path):
    file = tmp_path/'arrays.npz';np.savez(file, points=np.arange(12))
    original = np.load
    with numeric_mmap_backend():
        with np.load(file, allow_pickle=False) as arrays: assert np.array_equal(arrays['points'], np.arange(12))
    assert np.load is original


@pytest.mark.parametrize('header', [
    "{'descr': '<f4', 'shape': (2,), 'shape': (2,), 'fortran_order': False}",
    "{'descr': '<f4', 'shape': (-1,), 'fortran_order': False}",
    "{'descr': '<f4', 'shape': (2), 'fortran_order': False}",
    "{'descr': '<f4', 'shape': (2,), 'fortran_order': 0}",
    "{'descr': '|O8', 'shape': (2,), 'fortran_order': False}",
    "{'descr': [('a', '<f4')], 'shape': (2,), 'fortran_order': False}",
    "{'descr': '<f4', 'shape': (2,), 'fortran_order': False, 'extra': 1}",
    "{'descr': '<f4', 'shape': __import__('os'), 'fortran_order': False}",
])
def test_numeric_header_rejects_unsupported_or_executable_fields(header):
    with pytest.raises(ValueError): _fields(header)


def test_different_field_order_and_truncated_payload(tmp_path):
    assert _fields('{"shape": (2, 3), "fortran_order": True, "descr": ">f8", }')['shape'] == (2, 3)
    f = tmp_path/'bad.npy';np.save(f, np.arange(10));f.write_bytes(f.read_bytes()[:-1])
    with pytest.raises(ValueError, match='length'): map_numeric_npy(f)


def test_repeated_threaded_opening_and_mapping_lifetime(tmp_path):
    files = []
    for i in range(12):
        file = tmp_path/f'{i}.npy';np.save(file, np.arange(128, dtype=np.float32)+i);files.append(file)
    with ThreadPoolExecutor(max_workers=4) as pool:
        arrays = list(pool.map(map_numeric_npy, files*64))
    # Earlier mappings remain valid while later files are opened and evicted.
    for i, value in enumerate(arrays): assert np.array_equal(value, np.arange(128)+i % 12)
