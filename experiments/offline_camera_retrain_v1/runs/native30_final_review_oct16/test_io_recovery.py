"""Tests of data-integrity recovery, with real NPZ CRC corruption."""
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

import numpy as np

import io_recovery as recovery


class IntegrityRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = self.root/'native30_runtime_cache_v2/surfaces/a.npz'
        self.cache.parent.mkdir(parents=True)
        self.log = self.root/'recovery.jsonl'

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        np.savez(self.cache, static_normals=np.arange(12).reshape(4, 3))

    def corrupt(self):
        data = bytearray(self.cache.read_bytes())
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            info = archive.getinfo('static_normals.npy')
        name_length, extra_length = struct.unpack_from('<HH', data, info.header_offset + 26)
        start = info.header_offset + 30 + name_length + extra_length
        data[start + info.file_size - 1] ^= 1
        self.cache.write_bytes(data)

    def test_healthy_zip_uses_original_values(self):
        self.save()
        proxy = recovery.NumpyProxy(np, self.log)
        with proxy.load(self.cache, allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays['static_normals'], np.arange(12).reshape(4, 3))
        self.assertFalse(self.log.exists())

    def test_entire_crc_is_checked_before_lazy_array_read(self):
        self.save(); self.corrupt()
        with self.assertRaises(recovery.CorruptRuntimeCache) as result:
            recovery.NumpyProxy(np, self.log).load(self.cache, allow_pickle=False)
        self.assertEqual(len(result.exception.attempts), 6)

    def test_corrupt_derived_file_is_retained_and_rebuilt(self):
        self.save(); self.corrupt()
        bad_digest = recovery.file_sha256(self.cache)
        proxy = recovery.NumpyProxy(np, self.log)
        def original():
            if not self.cache.exists():
                self.save()
            with proxy.load(self.cache, allow_pickle=False) as arrays:
                return arrays['static_normals']
        result = recovery.cache_recovery(original, self.log)()
        np.testing.assert_array_equal(result, np.arange(12).reshape(4, 3))
        quarantines = list(self.cache.parent.glob('*.corrupt.*'))
        self.assertEqual(len(quarantines), 1)
        self.assertEqual(recovery.file_sha256(quarantines[0]), bad_digest)
        events = [json.loads(row)['event'] for row in self.log.read_text().splitlines()]
        self.assertEqual(events, ['derived_cache_quarantined', 'derived_cache_rebuilt'])

    def test_source_npz_outside_runtime_scope_is_not_repaired(self):
        self.save(); self.corrupt()
        outside = self.root/'source.npz'
        outside.write_bytes(self.cache.read_bytes())
        self.assertFalse(recovery.runtime_path(outside))
        with recovery.NumpyProxy(np, self.log).load(outside) as arrays:
            with self.assertRaises(zipfile.BadZipFile):
                arrays['static_normals']
        self.assertFalse(self.log.exists())

    def test_fingerprint_recheck_can_only_confirm_published_digest(self):
        expected = 'original'
        calls = []
        def verify(path, value):
            calls.append((str(path), value)); return value
        checked = recovery.audited_digest(lambda _: {'/source': 'transient'},
            {'/source': expected}, self.log, verify)
        self.assertEqual(checked(None), {'/source': expected})
        self.assertEqual(calls, [('/source', expected)])

    def test_wrong_fingerprint_remains_fatal(self):
        checked = recovery.audited_digest(lambda _: {'/source': 'bad'},
            {'/source': 'original'}, self.log, lambda *_: 'new')
        with self.assertRaises(ValueError):
            checked(None)

    def test_unexpected_source_entry_is_fatal(self):
        checked = recovery.audited_digest(lambda _: {'/unexpected': 'bad'},
            {'/source': 'original'}, self.log, lambda *_: 'original')
        with self.assertRaises(ValueError):
            checked(None)

    def test_rebuild_failure_is_not_swallowed(self):
        self.save(); self.corrupt()
        proxy = recovery.NumpyProxy(np, self.log)
        def original():
            if not self.cache.exists():
                raise RuntimeError('source reconstruction failed')
            return proxy.load(self.cache)
        with self.assertRaisesRegex(RuntimeError, 'source reconstruction failed'):
            recovery.cache_recovery(original, self.log)()
        self.assertEqual(len(list(self.cache.parent.glob('*.corrupt.*'))), 1)


if __name__ == '__main__':
    unittest.main()
