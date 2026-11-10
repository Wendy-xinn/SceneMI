import json
from pathlib import Path
import tempfile
import unittest
import torch

from scenemi.amass_checkpoint_test import audit_predictions, check_selection, load_verified_checkpoint
from scenemi.baseline_evaluation import digest


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        run = self.root / 'run'
        (run / 'selection').mkdir(parents=True)
        checkpoint = run / 'selection/best.pt'
        checkpoint.touch()
        best = dict(step=130000, selection_score_mm=120.5, checkpoint=str(checkpoint))
        (run / 'selection/BEST.json').write_text(json.dumps(best))
        metrics = run / 'validation/step_130000_full/RESULT.json'
        metrics.parent.mkdir(parents=True)
        metrics.write_text(json.dumps(dict(test_used=False, selection_score_mm=120.5)))
        self.c = dict(ours_checkpoint=str(checkpoint), ours_step=130000,
            checkpoint_test=dict(validation_metrics_sha256=digest(metrics)))
        self.state = dict(step=130000, best=dict(step=130000, selection_score_mm=120.5),
            validation_pending=False, device='cuda')
        self.config = dict(output=str(run), validation=dict(test_used=False))

    def test_accepts_completed_validation_selected_checkpoint(self):
        check_selection(self.c, self.state, self.config)

    def test_rejects_wrong_or_unvalidated_checkpoint(self):
        for field, value in [('step', 135000), ('validation_pending', True), ('device', 'cpu'),
                ('best', dict(step=125000, selection_score_mm=121.))]:
            with self.subTest(field=field):
                state = dict(self.state, **{field: value})
                with self.assertRaisesRegex(ValueError, 'completed validation selection'):
                    check_selection(self.c, state, self.config)

    def test_rejects_test_based_selection(self):
        self.config['validation']['test_used'] = True
        with self.assertRaisesRegex(ValueError, 'without test selection'):
            check_selection(self.c, self.state, self.config)

    def test_reused_baseline_must_match_recorded_hashes(self):
        (self.root / 'predictions.npy').write_bytes(b'changed prediction')
        metrics = self.root / 'metrics.jsonl'
        metrics.write_text('')
        with self.assertRaisesRegex(ValueError, 'Retained predictions or metrics changed'):
            audit_predictions(self.root, [], None, expected={
                'predictions.npy': '0' * 64, 'metrics.jsonl': digest(metrics)})

    def test_deserializes_only_the_verified_buffer(self):
        path = self.root / 'checkpoint.pt'
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        torch.save(dict(model=dict(weight=tensor), step=130000), path)
        state = load_verified_checkpoint(path, digest(path))
        self.assertEqual(state['step'], 130000)
        self.assertTrue(torch.equal(state['model']['weight'], tensor))

    def test_rejects_bytes_with_different_pinned_hash(self):
        path = self.root / 'checkpoint.pt'
        torch.save(dict(model=dict(weight=torch.ones(4))), path)
        expected = digest(path)
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'predeclared test checkpoint changed'):
            load_verified_checkpoint(path, expected)
