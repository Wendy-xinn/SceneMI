"""CPU-only GRAIL export guards and a tiny real first-hit integration export."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from . import grail_export as export


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))
        self.faces = np.array([[0, 1, 2]], np.int64)
        self.lbs_weights = torch.zeros(3, 55)


class ExportTests(unittest.TestCase):
    def test_auto_template_gate_and_explicit_review_policy(self):
        record = dict(quality=dict(body_template_verified=False))
        with self.assertRaises(export.UnverifiedBodyTemplate):
            export.body_policy(record)
        for template in ('standard', 'g1'):
            self.assertFalse(export.body_policy(record, template)['trainable'])
        record['quality'].update(body_template_verified=True, body_template='g1')
        with self.assertRaisesRegex(export.UnverifiedBodyTemplate, 'native body'):
            export.body_policy(record)
        record['quality']['native_body_assets_verified'] = True
        self.assertTrue(export.body_policy(record)['trainable'])

    def test_head_mount_roundtrip_for_primary_and_wide_cameras(self):
        head = np.array([[0, 1.6, 0], [.2, 1.3, .1], [.4, 1., .2]], np.float32)
        rotations = Rotation.from_euler('xyz', [[0, 0, 0], [5, 30, 10], [-20, 60, 20]], degrees=True).as_matrix()
        for options in (export.PRIMARY_CAMERA, export.AUDIT_CAMERA):
            position, optical, mount, intr = export.camera_from_head(head, rotations, [.01, .04, .08], options)
            result = export._verify_mount(position, optical, head, rotations, mount)
            self.assertLess(result['max_position_error_m'], 2e-6)
            self.assertEqual(intr.shape, (4,))
        # Camera-local +Z is pitched down in world for an upright head.
        _, optical, _, _ = export.camera_from_head(head[:1], np.eye(3)[None], [0, 0, 0], export.AUDIT_CAMERA)
        self.assertLess(optical[0, 1, 2], 0)

    def test_stair_family_and_ground_are_explicit(self):
        first = dict(object_family_id='stair_p1__beige_steps')
        second = dict(object_family_id='stair_p2__beige_steps')
        self.assertEqual(export.normalized_family(first), export.normalized_family(second))
        vertices, faces = export.ground_mesh()
        np.testing.assert_array_equal(vertices[:, 1], 0)
        self.assertEqual(float(np.max(np.abs(vertices))), 20.)
        normal = np.cross(vertices[faces[0, 1]] - vertices[faces[0, 0]],
                          vertices[faces[0, 2]] - vertices[faces[0, 0]])
        self.assertGreater(normal[1], 0)

    def test_external_camera_depends_on_initial_root_only(self):
        root = np.array([2., .9, 4.])
        camera = export.external_observer(root)
        self.assertAlmostEqual(np.linalg.norm((camera['position'] - root)[[0, 2]]), 3.)
        self.assertEqual(camera['position'][1], 1.6)
        self.assertFalse(camera['benchmark_camera_verified'])
        self.assertEqual(camera['kind'], 'synthetic_training_observer')

    def _record(self, root):
        source = root / 'released_source.pkl'
        source.write_bytes(b'fixture source bytes, never executed')
        n = 4
        params = {key: np.zeros((n, width), np.float32) for key, width in (
            ('global_orient', 3), ('body_pose', 63), ('transl', 3), ('betas', 10),
            ('left_hand_pose', 45), ('right_hand_pose', 45), ('jaw_pose', 3),
            ('leye_pose', 3), ('reye_pose', 3), ('expression', 10))}
        return dict(params=params, quality=dict(body_template_verified=False, object_static=True),
            human_scale=1., world_rotation=np.eye(3), world_offset=np.zeros(3),
            object_vertices=np.array([[-.5, 0, 2], [.5, 0, 2], [0, 1., 2]], np.float32),
            object_faces=np.array([[0, 1, 2]], np.int64),
            object_rotations=np.broadcast_to(np.eye(3), (n, 3, 3)).copy(),
            object_positions=np.zeros((n, 3), np.float32),
            output_timestamps_s=np.arange(n) / 30, source_timestamps_s=np.arange(n) / 24,
            source_fps=24., source_frames=n,
            input_sha256='1' * 64, source_files=dict(recon=str(source)),
            source_hashes=dict(recon=export.grail_source.file_sha256(source)), source_provenance={})

    @staticmethod
    def _forward(model, params, lo, hi, transform):
        n = hi - lo
        vertices = np.broadcast_to(np.array([[-.1, 1.2, .2], [.1, 1.2, .2], [0, 1.5, .2]], np.float32), (n, 3, 3)).copy()
        joints = np.zeros((n, 22, 3), np.float32)
        joints[:, 0, 1] = .9
        joints[:, 15, 1] = 1.6
        return vertices, joints

    def test_default_refuses_before_body_model_or_rendering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = dict(sequence_id='sitting__chair__000', category='sitting', object_family_id='sitting__chair')
            record = self._record(root)
            with patch.object(export.grail_source, 'load_record', return_value=record), \
                    patch.object(export, 'resolve_body_model') as factory:
                with self.assertRaises(export.UnverifiedBodyTemplate):
                    export.export_record(row, root / 'out')
                factory.assert_not_called()
            self.assertFalse((root / 'out').exists())

    def test_tiny_review_export_real_raycast_and_resumable_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = dict(sequence_id='sitting__chair__000', category='sitting', object_family_id='sitting__chair')
            record = self._record(root)
            model = ToyModel()
            with patch.object(export.grail_source, 'load_record', return_value=record), \
                    patch.object(export, '_rest_geometry', return_value=(np.zeros((22, 3), np.float32), np.array([0, .04, .08]))), \
                    patch.object(export, 'forward_chunk', side_effect=self._forward), \
                    patch.object(export, 'regional_faces', return_value=np.zeros(1, np.int64)), \
                    patch.object(export, 'wearer_faces', return_value=(model.faces, {})):
                result = export.export_record(row, root / 'out', body_template='standard', model=model, chunk_size=2)
            self.assertFalse(result['trainable'])
            self.assertFalse(result['quality']['body_template_verified'])
            self.assertEqual(result['frames'], 4)
            folder = Path(result['path'])
            for key, spec in export.grail_data.array_specs(4).items():
                array = np.load(folder / (key + '.npy'))
                self.assertIn(list(array.shape), spec['shapes'])
                self.assertEqual(array.dtype, np.dtype(spec['dtype']))
            np.testing.assert_array_equal(np.load(folder / 'source_frame_ids.npy'), np.arange(4))
            self.assertEqual(len(result['primary_visibility_by_frame']), 4)
            self.assertLess(result['mount_roundtrip']['max_position_error_m'], 2e-5)
            self.assertTrue((folder / 'source_pose_body_pose.npy').exists())
            self.assertTrue((folder / 'scene_ground_vertices.npy').exists())
            with np.load(folder / 'selected_raycast_audit.npz') as audit:
                self.assertIn('wide_owner_0', audit)
                self.assertIn('exo_owner_0', audit)
            manifest = export.publish_manifest(root / 'out', result)
            self.assertEqual(manifest.name, 'review_manifest.jsonl')
            self.assertFalse((root / 'out/manifest.jsonl').exists())
            # Existing complete export is checksum-verified without running a model.
            resumed = export.export_record(row, root / 'out', body_template='standard')
            self.assertEqual(resumed, result)
            np.save(folder / 'rest.npy', np.ones((22, 3), np.float32))
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                export.export_record(row, root / 'out', body_template='standard')

    def test_manifest_updates_one_identity_without_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            row = dict(sequence_id='same', trainable=False, metric=1)
            path = export.publish_manifest(temp, row)
            export.publish_manifest(temp, dict(row, metric=2))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows, [dict(row, metric=2)])

    def test_below_ground_objects_are_quarantined_without_changing_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = dict(sequence_id='stair__descending__000', category='stair_p1', object_family_id='stair__descending')
            record = self._record(root)
            record['object_vertices'][:, 1] -= .2
            record['quality'].update(body_template_verified=True, body_template='standard')
            model = ToyModel()
            with patch.object(export.grail_source, 'load_record', return_value=record), \
                    patch('experiments.offline_camera_retrain_v1.ops.grail_assets.validate_body_assets',
                          return_value=dict(native_body_assets_verified=True)), \
                    patch.object(export, '_rest_geometry', return_value=(np.zeros((22, 3), np.float32), np.array([0, .04, .08]))), \
                    patch.object(export, 'forward_chunk', side_effect=self._forward), \
                    patch.object(export, 'regional_faces', return_value=np.zeros(1, np.int64)), \
                    patch.object(export, 'wearer_faces', return_value=(model.faces, {})):
                result = export.export_record(row, root / 'out', model=model, chunk_size=2)
            self.assertFalse(result['trainable'])
            self.assertTrue(result['quality']['ground_plane_quarantined'])
            saved = np.load(Path(result['path']) / 'scene_object_vertices.npy')
            self.assertAlmostEqual(float(saved[:, 1].min()), -.2, places=6)
            self.assertEqual(export.publish_manifest(root / 'out', result).name, 'review_manifest.jsonl')


if __name__ == '__main__':
    unittest.main()
