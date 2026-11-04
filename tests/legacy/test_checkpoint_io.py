import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
class CheckpointTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'last.pt';save_checkpoint({'step':1000,'tensor':torch.arange(5)},p)
            value=torch.load(p,weights_only=True);self.assertEqual(value['step'],1000)
            self.assertTrue(torch.equal(value['tensor'],torch.arange(5)))
    def test_failed_write_preserves_previous(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'last.pt';save_checkpoint({'step':10},p)
            def failure(value,stream):stream.write(b'partial');raise OSError('simulated interruption')
            with patch('experiments.offline_camera_retrain_v1.checkpoint_io.torch.save',side_effect=failure):
                with self.assertRaises(OSError):save_checkpoint({'step':20},p)
            self.assertEqual(torch.load(p,weights_only=True)['step'],10)
if __name__=='__main__':unittest.main()
