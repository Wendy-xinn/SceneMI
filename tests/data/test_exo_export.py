"""Export admission and resume must preserve split and borrowed-source identities."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scenemi.data.exo_export import candidates, plan, verify_output, source_dependencies
from scenemi.native_visualization import sha256, ROOT


class ExoExportTests(unittest.TestCase):
    def test_source_seal_ignores_synthetic_relative_module_filenames(self):
        modules = dict(fake=SimpleNamespace(__file__='_classes.py'),
            real=SimpleNamespace(__file__=str(ROOT/'scenemi/data/exo_export.py')))
        with patch('scenemi.data.exo_export.sys.modules', modules):
            self.assertEqual(source_dependencies([]),[ROOT/'scenemi/data/exo_export.py'])

    def test_export_inventory_excludes_test_rich_and_diagnostic_sources(self):
        base = dict(dataset='trumans',split='train',allowed_modes=['ego','exo'],
                    diagnostic_only=False,sequence_id='a',frames=128)
        rows = [base,dict(base,split='validation',sequence_id='b'),
                dict(base,split='test',sequence_id='c'),dict(base,dataset='rich',sequence_id='d'),
                dict(base,diagnostic_only=True,sequence_id='e'),dict(base,allowed_modes=['ego'],sequence_id='f')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);manifest = root/'source.jsonl'
            manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            self.assertEqual([r['sequence_id'] for r in candidates(manifest)],['a','b'])
            result = plan(manifest,root/'does_not_exist'/'output')
            self.assertEqual(result['frames'],256)
            self.assertFalse(result['test_used'])
            manifest.write_text(manifest.read_text()+json.dumps(base)+'\n')
            with self.assertRaises(ValueError):
                candidates(manifest)

    def test_resume_verifies_both_overlay_and_borrowed_source_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);source = root/'source.npy';overlay = root/'overlay'
            overlay.mkdir();np.save(source,np.ones((2,3),np.float32))
            np.save(overlay/'new.npy',np.zeros((2,3),np.float32))
            meta = dict(path=str(overlay),array_sources={'borrowed.npy':str(source)},
                output_sha256={'borrowed.npy':sha256(source),'new.npy':sha256(overlay/'new.npy')})
            verify_output(meta)
            np.save(source,np.zeros((2,3),np.float32))
            with self.assertRaises(ValueError):
                verify_output(meta)


if __name__=='__main__':
    unittest.main()
