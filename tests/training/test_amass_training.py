"""Real local window contracts and fail-closed formal training gates."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scenemi.config import ROOT, load_config, load_paths
from scenemi.data.amass_dataset import AmassDataset, build_index, collate, inference_payload
from scenemi.data.amass_prepare import source_indices
from scenemi.observations import INPUT_KEYS
from scenemi.training import preparation_progress, require_preflight


class ReadinessTests(unittest.TestCase):
    def test_live_resume_hash_verification_is_distinguished_from_dead_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'preparation').mkdir()
            status = dict(state='running', pid=123, updated_at_utc='2026-10-21T21:39:18+00:00',
                          total_raw_files=100, committed_sources=50, failures=0, counts={})
            (root/'preparation/STATUS.json').write_text(json.dumps(status))
            (root/'raw_inventory.summary.json').write_text(json.dumps(dict(upstream_filename_exclusions={})))
            (root/'preparation/RESUME_LAUNCH_001.json').write_text(json.dumps(dict(preparation_pid=456)))
            with patch('scenemi.training._preparation_process_alive', side_effect=[False, True]):
                result = preparation_progress(dict(output=root))
            self.assertEqual(result['state'], 'verifying_resume_outputs')
            self.assertIs(result['process_alive'], True)
            self.assertIsNone(result['estimated_remaining_seconds'])

    def test_resume_eta_uses_only_the_new_invocation_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'preparation').mkdir()
            status = dict(state='running', pid=123, updated_at_utc=datetime.now(timezone.utc).isoformat(),
                          total_raw_files=100, committed_sources=50, failures=0, counts={})
            (root/'preparation/STATUS.json').write_text(json.dumps(status))
            (root/'raw_inventory.summary.json').write_text(json.dumps(dict(upstream_filename_exclusions={})))
            rows = [dict(source='a', elapsed=100), dict(source='b', elapsed=105),
                    dict(source='c', elapsed=5), dict(source='d', elapsed=7)]
            (root/'preparation/prepare.log').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with patch('pathlib.Path.read_bytes', return_value=b'python\0-m\0scenemi\0amass\0prepare\0'):
                result = preparation_progress(dict(output=root))
            self.assertIs(result['process_alive'], True)
            self.assertEqual(result['estimated_remaining_seconds'], 100.)
            status['invocation_completed_sources'] = 0
            (root/'preparation/STATUS.json').write_text(json.dumps(status))
            with patch('scenemi.training._preparation_process_alive', return_value=True):
                self.assertIsNone(preparation_progress(dict(output=root))['estimated_remaining_seconds'])

    def test_killed_preprocessor_is_not_reported_as_running_with_an_eta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'preparation').mkdir()
            status = dict(state='running', pid=999999999, updated_at_utc='2026-10-21T21:39:18+00:00',
                          total_raw_files=100, committed_sources=50, failures=0, counts={})
            (root/'preparation/STATUS.json').write_text(json.dumps(status))
            (root/'raw_inventory.summary.json').write_text(json.dumps(dict(upstream_filename_exclusions={})))
            (root/'preparation/prepare.log').write_text(
                json.dumps(dict(source='a', elapsed=1))+'\n'+json.dumps(dict(source='b', elapsed=2))+'\n')
            result = preparation_progress(dict(output=root))
            self.assertEqual(result['state'], 'process_missing')
            self.assertIs(result['process_alive'], False)
            self.assertIsNone(result['estimated_remaining_seconds'])

    def test_partial_manifest_cannot_become_formal_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'preparation').mkdir()
            (root/'preparation/STATUS.json').write_text(json.dumps(dict(state='running')))
            with self.assertRaisesRegex(ValueError, 'not complete'):
                build_index(dict(output=root, prepared_amass=root))

    def test_stale_or_diagnostic_proof_cannot_authorize_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proof = root/'preflight/complete_cuda/RESULT.json'
            proof.parent.mkdir(parents=True)
            paths = dict(output=root)
            config = load_config()
            valid = dict(state='passed', diagnostic_only=False, contract=dict(sha256='current'),
                         resume_bitwise_exact=True, batch_size=config['training']['batch_size'])
            with patch('scenemi.training.contract', return_value=dict(sha256='current')):
                for key, value in (('state', 'failed'), ('diagnostic_only', True),
                                   ('contract', dict(sha256='old')), ('resume_bitwise_exact', False),
                                   ('batch_size', 2)):
                    row = copy.deepcopy(valid)
                    row[key] = value
                    proof.write_text(json.dumps(row))
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        require_preflight(paths, config, root/'index.json', 'cuda')


class LocalWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths_file = ROOT/'configs/paths.local.yaml'
        cls.index = ROOT/'runs/amass128/index/diagnostic.json'
        if not cls.index.is_file() or not paths_file.is_file():
            raise unittest.SkipTest('Optional prepared licensed AMASS pilot')
        cls.paths = load_paths(paths_file)

    def test_pilot_is_ineligible_for_formal_training(self):
        with self.assertRaisesRegex(ValueError, 'partial diagnostic'):
            AmassDataset(self.index, self.paths)

    def test_real_windows_input_labels_clock_and_split_separation(self):
        families = []
        rng = np.random.default_rng(71)
        for split in ('train', 'val', 'test'):
            data = AmassDataset(self.index, self.paths, split, diagnostic=True)
            families.append({row['family'] for row in data.members})
            for mode in ('ego', 'exo'):
                sample, identity = data.sample(rng, mode, member=0, start=0, reflected=False, yaw=0.)
                payload = inference_payload(collate([sample]))
                self.assertEqual(set(payload), set(INPUT_KEYS))
                self.assertNotIn('rest', payload)
                self.assertNotIn('target_betas', payload)
                self.assertEqual(sample['motion'].shape, (128, 201))
                self.assertTrue(all(np.isfinite(value).all() for value in sample.values()))
                row = data.members[0]
                clock = source_indices(row['raw_frames'], row['source_fps'])[1:]
                self.assertEqual(identity['source_first_frame'], int(clock[0]))
                self.assertEqual(identity['source_last_frame'], int(clock[127]))
                if mode == 'ego':
                    self.assertTrue((sample['trajectory'] == 0).all())
                    self.assertTrue((sample['observation_meta'][..., :4] == 0).all())
                    self.assertTrue((sample['camera_available'] == 1).all())
                else:
                    self.assertTrue((sample['camera_available'] == 0).all())
                    self.assertTrue((sample['trajectory'][..., 3:] == 0).all())
                    self.assertTrue((sample['observation_meta'][..., 1] == 0).all())
                    hidden = sample['observation_meta'][..., 0] == 0
                    self.assertTrue((sample['trajectory'][hidden] == 0).all())
        for a in range(3):
            for b in range(a):
                self.assertFalse(families[a] & families[b])


if __name__ == '__main__':
    unittest.main()
