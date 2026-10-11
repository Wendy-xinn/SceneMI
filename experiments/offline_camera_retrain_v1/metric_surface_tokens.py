"""Untrained scene-geometry prototype preserving fixed native anchor identity.

Queries carry metric displacement, oriented normals and their reliability.
Unknown space is masked. No global occupancy/SDF or semantic label is inferred.
This feature extractor changes no existing model or checkpoint interface.
"""
import numpy as np
from scipy.spatial import cKDTree

from experiments.offline_camera_retrain_v1.observed_body_surfaces import oriented_planes

SCALES = (.08, .25, .60)
FEATURES = ('delta_x_m', 'delta_y_m', 'delta_z_m', 'distance_m',
            'normal_x', 'normal_y', 'normal_z', 'signed_patch_m',
            'normal_reliable', 'current_dynamic')


class MetricSurfaceTokens:
    def __init__(self, points, observing_camera, dynamic=None, cameras=None, neighbors=8):
        points = np.asarray(points, np.float32).reshape(-1, 3)
        observing_camera = np.broadcast_to(observing_camera, points.shape)
        assert np.isfinite(points).all() and np.isfinite(observing_camera).all()
        _, normals, reliable = oriented_planes(points, observing_camera)
        self.static = self._cloud(points, normals, reliable, False)
        self.dynamic, self.cameras = dynamic, cameras
        self.neighbors = neighbors

    @staticmethod
    def _cloud(points, normals, reliable, moving):
        return (points, normals, reliable, np.full(len(points), moving, bool), cKDTree(points) if len(points) else None)

    def query(self, vertices, rotations):
        vertices, rotations = np.asarray(vertices), np.asarray(rotations)
        frames, anchors = vertices.shape[:2]
        assert rotations.shape == (frames, anchors, 3, 3)
        tokens = np.zeros((frames, anchors, len(SCALES), self.neighbors, len(FEATURES)), np.float32)
        valid = np.zeros(tokens.shape[:-1], bool)
        reliable_mask = np.zeros_like(valid)
        for t in range(frames):
            p, normals, reliable, moving, tree = self.static
            if self.dynamic is not None and len(self.dynamic[t]):
                dp = np.asarray(self.dynamic[t], np.float32)
                _, dn, dr = oriented_planes(dp, np.broadcast_to(self.cameras[t], dp.shape), neighbors=8)
                p = np.concatenate((p, dp)); normals = np.concatenate((normals, dn))
                reliable = np.r_[reliable, dr]; moving = np.r_[moving, np.ones(len(dp), bool)]
                tree = cKDTree(p)
            if tree is None:
                continue
            for scale_ix, radius in enumerate(SCALES):
                neighborhoods = tree.query_ball_point(vertices[t], radius)
                for anchor, neighborhood in enumerate(neighborhoods):
                    if not neighborhood:
                        continue
                    index = np.asarray(neighborhood)
                    delta = p[index]-vertices[t, anchor]
                    # Start at nearest surface, then cover the entire radius
                    # patch. Repeating the same eight nearest points at three
                    # thresholds would discard the wider geometric context.
                    chosen = [int(np.argmin((delta**2).sum(-1)))]
                    spread = np.full(len(index), np.inf)
                    for _ in range(min(self.neighbors, len(index))-1):
                        spread = np.minimum(spread, ((delta-delta[chosen[-1]])**2).sum(-1))
                        spread[chosen] = -1
                        chosen.append(int(spread.argmax()))
                    selected = index[chosen]
                    local = (p[selected]-vertices[t, anchor])@rotations[t, anchor]
                    reliable_here = reliable[selected]
                    normal = (normals[selected]@rotations[t, anchor])*reliable_here[:, None]
                    distance = np.linalg.norm(local, axis=-1)
                    features = np.c_[local, distance, normal, -(local*normal).sum(-1), reliable_here, moving[selected]]
                    count = len(selected)
                    tokens[t, anchor, scale_ix, :count] = features
                    valid[t, anchor, scale_ix, :count] = True
                    reliable_mask[t, anchor, scale_ix, :count] = reliable_here
        return dict(tokens=tokens, valid=valid, normal_reliable=reliable_mask,
                    anchor_indices=np.arange(anchors), scales_m=SCALES, feature_names=FEATURES)


def functional_checks():
    grid = np.linspace(-.15, .15, 13)
    points = np.asarray([[x, 0, z] for x in grid for z in grid], np.float32)
    # Break artificial grid ties so equivariance checks compare the same
    # physically selected neighbours, rather than KD-tree tie order.
    points[:, [0, 2]] += np.random.default_rng(431).normal(0, .0001, (len(points), 2))
    camera = np.asarray([0., 1., 0.], np.float32)
    vertices = np.asarray([[[0., .03, 0.], [3., 2., 1.]]], np.float32)
    rotations = np.broadcast_to(np.eye(3), (1, 2, 3, 3)).copy()
    a = MetricSurfaceTokens(points, camera).query(vertices, rotations)
    assert a['valid'][0, 0, 0].all() and not a['valid'][0, 1].any()
    assert np.all(a['tokens'][0, 0, 0, :, 7] > .02)
    assert a['tokens'][0, 0, 1, :, 3].max() > 2*a['tokens'][0, 0, 0, :, 3].max()
    # Rigid-coordinate equivariance; metric and local features must agree.
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_rotvec([.2, .6, -.1]).as_matrix()
    offset = np.array([3., -1., 2.])
    b = MetricSurfaceTokens(points@rotation.T+offset, camera@rotation.T+offset).query(
        vertices@rotation.T+offset, rotation[None, None]@rotations)
    assert np.array_equal(a['valid'], b['valid'])
    # Equal-distance neighbour identities can be permuted. Compare sorted
    # intrinsic distances/signed distances instead of imposing arbitrary IDs.
    for feature in (3, 7):
        assert np.allclose(np.sort(a['tokens'][..., feature], axis=-1), np.sort(b['tokens'][..., feature], axis=-1), atol=1e-5)
    empty = MetricSurfaceTokens(np.empty((0, 3)), camera).query(vertices, rotations)
    assert not empty['valid'].any() and not empty['tokens'].any()
    dynamic = [points+np.array([3., 1.98, 1.])]
    dynamic_result = MetricSurfaceTokens(np.empty((0, 3)), camera, dynamic=dynamic,
                                        cameras=np.array([[3., 3., 1.]])).query(vertices, rotations)
    assert dynamic_result['valid'][0, 1].any() and not dynamic_result['valid'][0, 0].any()
    assert dynamic_result['tokens'][0, 1, :, :, 9][dynamic_result['valid'][0, 1]].all()
    return dict(empty_scene_unknown=True, local_metric_rigid_equivariance=True,
                observed_plane_oriented_sign=True, current_dynamic_separate=True,
                larger_radius_samples_wider_patch=True, trained=False, model_performance_claim=False)
