"""Fail closed on unintended changes in the fixed-compute EGO ablation."""
import copy
from pathlib import Path
import tempfile
import unittest

import yaml

from scenemi.config import load_config
from scenemi.ego_training import load_ego_config
from scenemi.data.grail_selection import select


class EgoConfigurationTests(unittest.TestCase):
    def test_only_observation_mixture_can_change(self):
        config = copy.deepcopy(load_config())
        config['observation']['mode_probabilities'] = dict(ego=1., exo=0.)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.yaml'
            path.write_text(yaml.safe_dump(config))
            self.assertEqual(load_ego_config(path), config)
            for key, value in [('batch_size', 16), ('planned_updates', 1000)]:
                changed = copy.deepcopy(config)
                changed['training'][key] = value
                path.write_text(yaml.safe_dump(changed))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_ego_config(path)


class ChairSelectionTests(unittest.TestCase):
    def rows(self):
        return [dict(sequence_id=f'{family}-{i}', object_family_id=family, split='train',
                     category='sitting', condition_provenance='proposed_synthetic_chair_observation',
                     eligible_windows=8, eligible_start_ranges=[[2, 9]])
                for family in ('a', 'b', 'c') for i in range(4)]

    def test_reproducible_family_coverage_independent_of_manifest_order(self):
        rows = self.rows()
        result = select(rows, 7, 77)
        self.assertEqual(result, select(list(reversed(rows)), 7, 77))
        self.assertEqual(len({r['sequence_id'] for r in result}), 7)
        self.assertEqual({r['object_family_id'] for r in result[:3]}, {'a', 'b', 'c'})
        self.assertTrue(all(r['window_start'] == 2 for r in result))

    def test_rejects_test_sources_duplicates_and_oversubscription(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            select(rows, 13, 77)
        with self.assertRaises(ValueError):
            select(rows + [rows[0]], 7, 77)
        rows[0]['split'] = 'test'
        with self.assertRaises(ValueError):
            select(rows, 7, 77)


if __name__ == '__main__':
    unittest.main()
