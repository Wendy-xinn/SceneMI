"""Fresh-run provenance and interruption behavior for expensive training."""
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest

import torch

from experiments.offline_camera_retrain_v1.prepare_native30 import read_config
from experiments.offline_camera_retrain_v1.train_native30 import model_digest,StopAtUpdateBoundary


class TrainingProtocolTests(unittest.TestCase):
    def test_fresh_config_rejects_checkpoint_lineage(self):
        config=json.loads(Path(__file__).with_name('native30_fresh_ego_config.json').read_text())
        config['warm_start']={'checkpoint':'old.pt'}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'config.json';path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'must not declare'):
                read_config(path)

    def test_digest_covers_parameters_and_persistent_buffers(self):
        model=torch.nn.Linear(2,3);model.register_buffer('scale',torch.tensor(1.))
        first=model_digest(model)
        with torch.no_grad():model.weight[0,0]+=1
        self.assertNotEqual(first,model_digest(model))
        second=model_digest(model);model.scale+=1
        self.assertNotEqual(second,model_digest(model))

    def test_sigterm_requests_checkpoint_boundary_and_restores_handler(self):
        previous=signal.getsignal(signal.SIGTERM)
        with StopAtUpdateBoundary() as stopping:
            self.assertFalse(stopping.requested)
            os.kill(os.getpid(),signal.SIGTERM)
            self.assertTrue(stopping.requested)
        self.assertEqual(signal.getsignal(signal.SIGTERM),previous)


if __name__=='__main__':unittest.main()
