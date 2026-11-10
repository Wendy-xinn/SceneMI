"""Clock, corruption and resume tests for the AMASS preparation contract."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scenemi.data.amass_prepare import inspect_raw, sha256, source_indices, verified_records


class PrepareTests(unittest.TestCase):
    def test_body_clock_crop_velocity_edges_and_relative_first_frame(self):
        # 1000 @ 120Hz -> middle 800, edge-trim to 798, downsample to 199.
        ids = source_indices(1000, 120.)
        self.assertEqual(len(ids), 199)
        self.assertEqual((ids[0], ids[-1]), (101, 898))
        self.assertEqual(len(ids[1:]), 198)
        self.assertTrue(np.all(np.diff(ids) > 0))
        self.assertEqual(len(source_indices(100, 120.)), 0)

    def test_raw_validation_corrections_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'BMLhandball_poses.npz'
            values = dict(poses=np.zeros((1000, 156)), trans=np.zeros((1000, 3)),
                          betas=np.zeros(16), mocap_framerate=120., gender='female')
            np.savez(path, **values)
            info = inspect_raw(path)
            self.assertEqual(info['source_fps'], 240.)
            self.assertEqual(len(info['body_source_frame_ids']), 99)
            values['poses'][31, 72] = np.nan
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                inspect_raw(path)
            values['poses'] = np.zeros((1000, 156))
            values['betas'] = np.zeros(10)
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, 'shape/gender'):
                inspect_raw(path)

    def test_resume_checks_committed_bytes_and_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'output.npz'
            target.write_bytes(b'committed content')
            ledger = root / 'manifest.jsonl'
            inventory = dict(source='CMU/a_poses.npz', bytes=123, mtime_ns=17,
                             split='train', family='CMU/a')
            record = dict(source=inventory['source'], inventory={k: inventory[k] for k in
                          ('bytes', 'mtime_ns', 'split', 'family')},
                          variants=[dict(path='output.npz', sha256=sha256(target))])
            ledger.write_text(json.dumps(record) + '\n')
            self.assertEqual(len(verified_records(ledger, root, [inventory])), 1)
            target.write_bytes(b'corrupted content')
            with self.assertRaisesRegex(ValueError, 'changed or missing'):
                verified_records(ledger, root, [inventory])
            with self.assertRaisesRegex(ValueError, 'inventory'):
                verified_records(ledger, root, [dict(inventory, split='test')])


class LocalPilotTests(unittest.TestCase):
    def test_real_body_clock_neutral_shape_and_reflected_rotations(self):
        from scenemi.config import ROOT, load_paths
        pilot_path = ROOT / 'runs/amass128/preparation/PILOT.json'
        paths_file = ROOT / 'configs/paths.local.yaml'
        if not pilot_path.is_file() or not paths_file.is_file():
            self.skipTest('Optional licensed local AMASS preprocessing pilot')
        paths = load_paths(paths_file)
        pilot = json.loads(pilot_path.read_text())
        rows = [json.loads(line) for line in Path(pilot['manifest']).read_text().splitlines()
                ][:pilot['committed_sources']]
        required = [paths['prepared_amass'] / variant['path'] for row in rows
                    if row['state'] == 'prepared' for variant in row['variants']]
        if any(not path.is_file() for path in required):
            self.skipTest('Historical local pilot arrays are unavailable; current index tests remain enabled')
        swap = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20]
        covered = set()
        for row in rows:
            if row['state'] != 'prepared':
                continue
            covered.add(row['corpus'])
            raw_path = paths['amass_raw'] / row['source']
            info = inspect_raw(raw_path)
            ids = info['body_source_frame_ids']
            with np.load(raw_path, allow_pickle=False) as raw:
                expected_pose = raw['poses'][ids, :66].reshape(-1, 22, 3).copy()
                trans = raw['trans'][ids].copy()
                conversion = paths['egoallo_repo'] / f"data/smplh_gender_conversion/{info['raw_gender']}_to_neutral.npz"
                with np.load(conversion, allow_pickle=False) as mapping:
                    betas = mapping['A'] @ raw['betas'] + mapping['b']
            for variant in row['variants']:
                with self.subTest(source=row['source'], reflected=variant['reflected']):
                    with np.load(paths['prepared_amass'] / variant['path'], allow_pickle=False) as processed:
                        actual = np.concatenate((processed['root_orient'], processed['pose_body']), -1).reshape(-1, 22, 3)
                        expected = expected_pose[:, swap] * [1, -1, -1] if variant['reflected'] else expected_pose
                        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
                        np.testing.assert_allclose(processed['betas'][0], betas, atol=1e-5, rtol=1e-6)
                        if not variant['reflected']:
                            recovered_trans = processed['trans'].astype(np.float64)
                            recovered_trans[:, 2] += float(processed['floor_height'])
                            np.testing.assert_allclose(recovered_trans, trans, atol=5e-5, rtol=1e-6)
        self.assertEqual(len(covered), 19)


if __name__ == '__main__':
    unittest.main()
