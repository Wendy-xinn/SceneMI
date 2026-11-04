import tempfile
from pathlib import Path
import unittest

from scenemi.data.source_cache import restore
from scenemi.native_visualization import sha256


class SourceCacheTests(unittest.TestCase):
    def test_bad_original_cannot_replace_a_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.npy';cache=root/'cache.npy'
            source.write_bytes(b'bad original');cache.write_bytes(b'old mirror')
            with self.assertRaises(ValueError):
                restore(source,cache,'0'*64,root/'evidence')
            self.assertEqual(cache.read_bytes(),b'old mirror')

    def test_restore_retains_corruption_evidence_and_exact_published_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.npy';cache=root/'cache.npy'
            source.write_bytes(b'published source');cache.write_bytes(b'corrupt mirror')
            expected=sha256(source);previous=sha256(cache)
            result=restore(source,cache,expected,root/'evidence')
            self.assertEqual(sha256(cache),expected)
            self.assertEqual((root/'evidence'/(previous+'.npy')).read_bytes(),b'corrupt mirror')
            self.assertEqual(result['operation'],'restored_corrupt')


if __name__=='__main__':
    unittest.main()
