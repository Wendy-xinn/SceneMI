import unittest

import numpy as np
import torch

from scenemi.geometry.hybrid import bounded_planes, hybrid_geometry
from scenemi.geometry.observed_relations import local_planes
from scenemi.geometry.scene_features import relation_features
from scenemi.data.amass_dataset import collate


class HybridGeometryTest(unittest.TestCase):
    def test_planes_keep_degenerate_and_mixed_owner_rejection(self):
        rng = np.random.default_rng(37)
        points = torch.tensor(rng.normal(size=(2, 8, 16, 3)), dtype=torch.float32)
        points[..., 2] = 0
        owners = torch.zeros((2, 8, 16), dtype=torch.long)
        valid = torch.ones_like(owners, dtype=torch.bool)
        valid[0, 0] = False
        owners[0, 1, 1:] = 1
        points[1, 0] = 0
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            p, o, v = [x.to(device) for x in (points, owners, valid)]
            expected = local_planes(p, o, v)
            actual = bounded_planes(p, o, v, chunk=3)
            for key in ('valid', 'neighbours', 'owner'):
                self.assertTrue(torch.equal(actual[key], expected[key]))
            torch.testing.assert_close(actual['centre'], expected['centre'], rtol=0, atol=0)
            usable = expected['valid']
            dot = (actual['normal']*expected['normal']).sum(-1).abs()
            self.assertTrue(torch.all(dot[usable] > .99999))

    def test_cpu_execution_preserves_features_and_hooks(self):
        from types import SimpleNamespace
        rng = np.random.default_rng(43)
        samples = []
        for _ in range(2):
            samples.append(dict(local_history_points=rng.normal(size=(32, 3)).astype('f'),
                local_history_valid=np.ones(32, bool),
                local_frame_points=rng.normal(size=(4, 16, 3)).astype('f'),
                local_frame_owner=np.zeros((4, 16), np.int64),
                camera=np.zeros((4, 9), np.float32), camera_available=np.ones((4, 1), np.float32),
                camera_confidence=np.ones((4, 1), np.float32)))
        anchors = torch.tensor(rng.normal(size=(2, 4, 3, 3)), dtype=torch.float32)
        rotations = torch.eye(3).expand(2, 4, 3, 3, 3)
        policy = dict(neighbours=8, query_radius_m=.6, plane_radius_m=.15,
            minimum_points=6, maximum_planarity_ratio=.03, minimum_spread_m=.005,
            reliability_distance_m=.3, ray_tube_m=.015, ray_endpoint_margin_m=.05,
            ray_minimum_depth_m=.1)
        batch = collate(samples)
        expected = relation_features(anchors, rotations, batch, policy)
        trainer = SimpleNamespace(collate=collate)
        from scenemi import scene_adapter
        before = scene_adapter.relation_features
        with hybrid_geometry(trainer):
            batch = trainer.collate(samples)
            actual = scene_adapter.relation_features(anchors, rotations, batch, policy)
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a, b))
        self.assertIs(trainer.collate, collate)
        self.assertIs(scene_adapter.relation_features, before)
