"""Meaningful identity/recording/role constraints for expanded VAL selection."""
from collections import Counter
from pathlib import Path
import tempfile
import unittest

from experiments.offline_camera_retrain_v1.ops.validation_cohort import (
    common_members, recording_representatives_first, select_recording_balanced, sha)


class ValidationCohortTest(unittest.TestCase):
    def entries(self):
        return [dict(recording=f'record{i}', path=f'record{i}/{role}', group=role, start=start,
                     source_first=start) for i in range(6) for role in ('camera_wearer', 'interactee')
                for start in range(0, 2048, 64)]

    def test_first_round_independent_then_nonoverlapping_role_balanced(self):
        chosen = select_recording_balanced(self.entries(), 24, 192, 17)
        self.assertEqual(len({row['recording'] for row in chosen[:6]}), 6)
        self.assertEqual(Counter(row['group'] for row in chosen), {'camera_wearer': 12, 'interactee': 12})
        self.assertFalse(any(row['overlaps_selected_same_recording'] for row in chosen))
        for recording in {row['recording'] for row in chosen}:
            times = sorted(row['source_first'] for row in chosen if row['recording'] == recording)
            self.assertTrue(all(b-a >= 192 for a, b in zip(times, times[1:])))

    def test_order_independent_reproducible_and_no_duplicate_source_start(self):
        entries = self.entries()
        first = select_recording_balanced(entries, 30, 192, 123)
        self.assertEqual(first, select_recording_balanced(list(reversed(entries)), 30, 192, 123))
        self.assertEqual(len({(row['path'], row['start']) for row in first}), len(first))

    def test_no_fake_independence_and_explicit_overlap_fallback(self):
        entries = [dict(recording='only_record', path='one', group='trumans', start=s, source_first=s)
                   for s in (0, 64, 128)]
        chosen = select_recording_balanced(entries, 3, 192, 9)
        self.assertEqual(len({row['recording'] for row in chosen}), 1)
        self.assertEqual(sum(row['overlaps_selected_same_recording'] for row in chosen), 2)
        with self.assertRaises(ValueError):
            select_recording_balanced(entries, 4, 192, 9)

    def test_representatives_first_preserves_all_source_identities(self):
        entries = [dict(recording=r, path=r, start=s, source_first=s)
                   for r in ('a', 'b') for s in (0, 64, 128, 192)]
        result = recording_representatives_first(entries)
        self.assertEqual([(row['recording'], row['start']) for row in result[:2]], [('a', 128), ('b', 128)])
        self.assertEqual({(row['path'], row['start']) for row in result},
                         {(row['path'], row['start']) for row in entries})
        self.assertEqual(len(result), len(entries))

    def test_member_order_includes_exo_only_sources_and_excludes_invalid_clock(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            for ordinal, (mode, valid) in enumerate((('exo', True), ('ego', False), ('ego', True))):
                folder = Path(directory)/str(ordinal)
                folder.mkdir()
                path = folder/'observation_valid.npy'
                np.save(path, np.full(256, valid, dtype=bool))
                rows.append(dict(path=str(folder), frames=256, split='validation', group='same',
                    allowed_modes=[mode], output_sha256={'observation_valid.npy': sha(path)}))
            members, _ = common_members(rows, 192, 64)
            self.assertEqual(members, {0: 0, 2: 1})
            rows[0]['output_sha256']['observation_valid.npy'] = 'wrong'
            with self.assertRaises(ValueError):
                common_members(rows, 192, 64)


if __name__ == '__main__':
    unittest.main()
