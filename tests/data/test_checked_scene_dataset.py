"""Selection guards must reject old validation, wrong modes and test leakage."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scenemi.data.checked_scene_dataset import SceneSelection, checked_ranges


class CurrentSceneSelectionTest(unittest.TestCase):
    def row(self, **changes):
        return dict(dict(split='validation', frames=301, mode='shared_motion_quality',
            protocol='grail-heldout-contact-support-v1', visibility_used_for_selection=False,
            eligible_start_ranges=[[0, 22]]), **changes)

    def test_only_accepted_starts_are_exposed(self):
        row = self.row(eligible_start_ranges=[[0, 22]], bad_frame_ranges=[[150, 179]], eligible_windows=23)
        self.assertEqual(checked_ranges(row, 'validation', mode='ego').tolist(), list(range(23)))
        self.assertEqual(checked_ranges(row, 'validation', mode='exo').tolist(), list(range(23)))

    def test_unfiltered_ranges_and_wrong_split_or_mode_are_rejected(self):
        for changes in [dict(eligible_start_ranges=[[0, 173]], bad_frame_ranges=[[150, 179]]),
                        dict(mode='ego'), dict(protocol='old-unfiltered'), dict(split='test'),
                        dict(eligible_start_ranges=[[0, 174]]),
                        dict(eligible_start_ranges=[[0, 22]], eligible_windows=24)]:
            with self.assertRaises(ValueError):
                checked_ranges(self.row(**changes), 'validation', mode='ego')
        with self.assertRaises(ValueError):
            checked_ranges(self.row(split='train', mode='exo', eligible_start_ranges=[[0, 22]]),
                           'train', mode='ego')

    def test_changed_manifest_fails_before_dataset_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'source';source.write_text('original')
            paths = root/'paths.yaml';paths.write_text('source: '+str(source)+'\n')
            config = root/'selection.json'
            config.write_text(json.dumps(dict(version='scene-data-quality-v2',frames=128,fps=30,
                test_requires_explicit_opt_in=True, paths_config=str(paths),
                sources={'native_manifest':dict(path_key='source',sha256=hashlib.sha256(b'original').hexdigest())})))
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Pinned scene-data source changed'):
                SceneSelection(config)

    def test_training_cannot_open_test_or_substitute_validation(self):
        selection = object.__new__(SceneSelection)
        with self.assertRaisesRegex(ValueError, 'Test is forbidden'):
            selection.records('ego', 'test')
        with self.assertRaisesRegex(ValueError, 'preparation is incomplete'):
            selection.records('exo', 'test', allow_test=True)


if __name__ == '__main__':
    unittest.main()
