import unittest
from unittest.mock import patch

import torch

from scenemi import scene_adapter, scene_joint_replay
from scenemi.replay import comparison_geometry as hybrid


class HybridGeometryTests(unittest.TestCase):
    def test_cache_preserves_observations_but_refreshes_predicted_anchors(self):
        original = scene_adapter.relation_features
        batch = {key:torch.zeros(1) for key in hybrid.GEOMETRY_INPUTS}
        cache = [object()]
        calls = []

        def geometry(anchors, rotations, payload, policy, neighborhoods):
            calls.append((anchors.clone(), payload, neighborhoods))
            return anchors, anchors.sum(-1), anchors.mean(-1)

        with patch.object(hybrid, 'relation_features', geometry):
            with self.assertRaisesRegex(RuntimeError, 'leave'):
                with hybrid.cpu_observed_geometry():
                    for value in (1., 2.):
                        anchors = torch.full((1, 2, 3), value)
                        result = scene_adapter.relation_features(anchors, anchors, batch, {}, cache)
                        self.assertTrue(torch.equal(result[0], anchors))
                    raise RuntimeError('leave')
        self.assertIs(calls[0][1], calls[1][1])
        self.assertIs(calls[0][2], cache)
        self.assertFalse(torch.equal(calls[0][0], calls[1][0]))
        self.assertIs(scene_adapter.relation_features, original)
        self.assertIs(scene_joint_replay.relation_features, original)


if __name__=='__main__':unittest.main()
