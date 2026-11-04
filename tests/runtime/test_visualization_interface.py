"""Shared playback must verify prepared assets before opening a server."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scenemi.visualization import main, prepared_playback, ego_prediction_viewer, grail_review_specs, grail_review_viewer


class VisualizationInterfaceTests(unittest.TestCase):
    def test_review_selection_verifies_sources_and_keeps_excluded_controls_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root / 'metadata.json'; meta.write_text('{"split":"validation","frames":256}')
            row = dict(flags=[dict(type='both_soles_clearance_over_3cm')], in_current_selection=True,
                path=directory, source_metadata_sha256=hashlib.sha256(meta.read_bytes()).hexdigest(),
                start=0, split='validation', sequence_id='stairs-one', category='stair',
                current_use='validation', viewer_case_index=1)
            path = root / 'annotations.jsonl'
            path.write_text(json.dumps(row) + '\n' + json.dumps(dict(row,
                in_current_selection=False, current_use='not_selected_at_this_window', viewer_case_index=2)))
            self.assertEqual(len(grail_review_specs(path)), 1)
            self.assertEqual(len(grail_review_specs(path, True)), 2)
            blind = grail_review_specs(path, True, True)
            self.assertTrue(all(s['status'] == 'blind_review' for s in blind))
            self.assertTrue(all('验证' not in s['label'] and '已排除' not in s['label'] for s in blind))
            path.write_text(json.dumps(dict(row, flags=[])))
            self.assertEqual(len(grail_review_specs(path, True, True)), 1)
            path.write_text(json.dumps(row))
            meta.write_text('{"split":"train","frames":256}')
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                grail_review_specs(path)

    def test_review_check_only_never_constructs_a_server(self):
        with patch('scenemi.visualization.grail_review_specs', return_value=[{}]), \
             patch('scenemi.visualization.grail_review_viewer') as constructor, patch('builtins.print'):
            main(['grail-review', '--annotations', '/tmp/annotations.jsonl', '--check-only'])
        constructor.assert_not_called()

    def test_review_navigation_jumps_between_distinct_marked_intervals(self):
        import numpy as np
        import threading
        viewer_class = grail_review_viewer()
        viewer = viewer_class.__new__(viewer_class)
        viewer.lock = threading.RLock(); viewer.frame = SimpleNamespace(value=3)
        mask = np.zeros(128, bool); mask[2:6] = True; mask[20:23] = True
        viewer.runtime = dict(ego=dict(data=dict(review_mask=mask)))
        viewer.next_mark(); self.assertEqual(viewer.frame.value, 20)
        viewer.next_mark(); self.assertEqual(viewer.frame.value, 2)

    def test_seat_review_localizes_support_gap_without_changing_skin(self):
        import numpy as np
        from scenemi.data.grail_alignment import AlignmentViewer
        for height, flagged in ((.4, True), (.53, False)):
            vertices = np.zeros((128, 1, 3), np.float32); vertices[..., 1] = height
            original = vertices.copy()
            joints = np.zeros((128, 22, 3), np.float32)
            joints[:, 4, 0] = 1.; joints[:, 5, 0] = -1.
            data = dict(truth_vertices=vertices, truth_joints=joints,
                scene_vertices=np.array([[-1,.5,-1],[1,.5,-1],[1,.5,1],[-1,.5,1]], np.float32),
                scene_faces=np.array([[0,1,2],[0,2,3]], np.int32))
            viewer_class = grail_review_viewer(); viewer = viewer_class.__new__(viewer_class)
            viewer.review_body = SimpleNamespace(seat_ids=np.array([0]))
            viewer.specs = [dict(start=0, row=dict(category='sitting'), annotation=dict(flags=[]))]
            with self.subTest(height=height), patch.object(AlignmentViewer, 'read_case', return_value=[dict(data=data)]):
                result = viewer.read_case(0)[0]['data']
            self.assertEqual(bool(result['review_mask'].any()), flagged)
            np.testing.assert_array_equal(vertices, original)

    def test_scene_preview_description_matches_inputs_without_starting_a_server(self):
        import viser
        for scene in (False, True):
            messages = []
            def markdown(api, content, *args, **kwargs):
                messages.append(content)
            def initialize(viewer, cases, port):
                viser.GuiApi.add_markdown(None, '输入参考人体是测试')
            with self.subTest(scene=scene), \
                 patch.object(viser.GuiApi, 'add_markdown', markdown), \
                 patch('scenemi.visualization.MotionViewer.__init__', initialize):
                ego_prediction_viewer()([dict(scene_fed_to_model=scene)], 8840)
            self.assertEqual('离线非因果' in messages[0], scene)
            self.assertEqual('仅相机轨迹输入模型' in messages[0], not scene)

    def test_relative_cache_paths_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'case.npz'
            artifact.write_bytes(b'prepared geometry')
            manifest = root / 'PREPARED_CASES.json'
            manifest.write_text(json.dumps(dict(protocol='ego-prediction-preview-v1', cases=[dict(
                models=[dict(case=dict(cache='case.npz', cache_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()))])])) )
            cases, _ = prepared_playback(manifest)
            self.assertEqual(cases[0]['models'][0]['case']['cache'], str(artifact))
            artifact.write_bytes(b'changed geometry')
            with self.assertRaisesRegex(ValueError, 'playback changed'):
                prepared_playback(manifest)

    def test_check_only_does_not_start_server(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text('{}')
            with patch('scenemi.visualization.prepared_playback', return_value=([{}], object)) as validate, \
                 patch('builtins.print'):
                main(['serve', '--manifest', str(path), '--check-only'])
            validate.assert_called_once_with(path)
            self.assertFalse((path.parent / 'SERVER.json').exists())

    def test_unknown_manifest_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text('{"protocol": "unknown", "cases": [{}]}')
            with self.assertRaisesRegex(ValueError, 'Unsupported prepared'):
                prepared_playback(path)


if __name__ == '__main__':
    unittest.main()
