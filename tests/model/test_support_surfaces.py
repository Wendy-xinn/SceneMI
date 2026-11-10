"""Observed support layers, missing geometry and fit-free temporal lookup."""
import copy

import numpy as np
import pytest
import yaml

from scenemi.config import ROOT
from scenemi.geometry.spatial_scene import ObservedSceneIndex
from scenemi.geometry.support_surfaces import (RELIABLE, UNCERTAIN, UNKNOWN,
    SupportSurfaceAtlas, _trajectory_selection, extract_support_surfaces, validate_policy)


def policy():
    return yaml.safe_load((ROOT/'configs/ego_scene_support.yaml').read_text())


def plane(y=0., lo=-.6, hi=.6, step=.025):
    x, z = np.meshgrid(np.arange(lo, hi+step/2, step), np.arange(lo, hi+step/2, step))
    return np.column_stack((x.ravel(), np.full(x.size, y), z.ravel())).astype(np.float32)


def index_for(points):
    return ObservedSceneIndex(points, np.tile([0., 1., 0.], (len(points), 1)),
                              np.ones(len(points), bool))


def atlas_for(points):
    p = policy()
    index = index_for(points)
    return SupportSurfaceAtlas(extract_support_surfaces(index, p, seed=123), points, p)


def test_stacked_floor_and_seat_remain_candidates_and_select_plausible_height():
    atlas = atlas_for(np.concatenate((plane(), plane(.35))))
    result = atlas.query(np.array([[[0., .37, 0.]]]*4, np.float32))
    np.testing.assert_allclose(result['height'], .35, atol=1e-6)
    assert result['reliable'].all() and (result['state'] == RELIABLE).all()
    for heights, valid in zip(result['candidate_height'].reshape(-1, 4), result['candidate_valid'].reshape(-1, 4)):
        np.testing.assert_allclose(np.sort(heights[valid]), [0., .35], atol=1e-6)


def test_equally_plausible_levels_are_features_without_a_physical_support_claim():
    atlas = atlas_for(np.concatenate((plane(), plane(.24))))
    result = atlas.query(np.array([[[0., .12, 0.]]], np.float32))
    assert result['valid'].all() and result['ambiguous'].all()
    assert not result['reliable'].any() and (result['state'] == UNCERTAIN).all()


def test_unobserved_holes_and_outside_footprints_are_not_fabricated_floor():
    points = plane()
    keep = np.linalg.norm(points[:, [0, 2]], axis=-1) > .35
    atlas = atlas_for(points[keep])
    result = atlas.query(np.array([[0., .05, 0.], [2., .05, 0.]], np.float32))
    assert not result['valid'].any() and (result['state'] == UNKNOWN).all()
    keep = np.linalg.norm(points[:, [0, 2]], axis=-1) > .11
    uncertain = atlas_for(points[keep]).query(np.array([[0., .05, 0.]], np.float32))
    assert uncertain['valid'].all() and not uncertain['reliable'].any()


def test_unknown_map_and_vertical_wall_produce_no_support():
    for points in (np.empty((0, 3), np.float32), plane()[:, [1, 0, 2]]):
        index = ObservedSceneIndex(points, np.tile([1., 0., 0.], (len(points), 1)), np.ones(len(points), bool))
        p = policy()
        atlas = SupportSurfaceAtlas(extract_support_surfaces(index, p), points, p)
        result = atlas.query(np.zeros((3, 2, 3), np.float32))
        assert not result['valid'].any() and not result['candidate_valid'].any()


def test_query_does_not_fit_or_construct_convex_hulls(monkeypatch):
    atlas = atlas_for(plane())
    def fail(*args, **kwargs):
        raise AssertionError('Online query attempted geometry fitting')
    monkeypatch.setattr(np.linalg, 'lstsq', fail)
    monkeypatch.setattr('scenemi.geometry.support_surfaces.ConvexHull', fail)
    import open3d as o3d
    monkeypatch.setattr(o3d.geometry.PointCloud, 'segment_plane', fail)
    result = atlas.query(np.array([[[0., .03, 0.]]]*5, np.float32))
    assert result['reliable'].all()


def test_temporal_selection_reduces_layer_flicker_and_resets_at_unknown_frames():
    height = np.broadcast_to([0., .2], (6, 1, 2)).copy()
    cost = np.array([[[.1, .101]], [[.101, .1]], [[.1, .101]],
                     [[np.inf, np.inf]], [[.3, .01]], [[.301, .01]]])
    chosen = _trajectory_selection(cost, height, .2)
    assert chosen[:3, 0].tolist() in ([0, 0, 0], [1, 1, 1])
    assert chosen[4:, 0].tolist() == [1, 1]
    cost[:3, 0] = [.3, .01]
    cost[4:, 0] = [.01, .3]
    chosen = _trajectory_selection(cost, height, .2)
    assert chosen[:3, 0].tolist() == [1, 1, 1]
    assert chosen[4:, 0].tolist() == [0, 0]


def test_public_gravity_is_required_and_never_inferred_from_gt_body():
    for change in (dict(source='gt_body'), dict(up_world=[0., 0., 1.])):
        p = copy.deepcopy(policy())
        p['gravity'].update(change)
        with pytest.raises(ValueError, match='public calibrated Y-up'):
            validate_policy(p)


def test_small_stair_treads_remain_separate_planes():
    lower = plane(lo=-.3, hi=.3)
    upper = plane(.18, lo=-.3, hi=.3)
    lower[:, 0] -= .32
    upper[:, 0] += .32
    result = atlas_for(np.concatenate((lower, upper))).query(
        np.array([[[-.32, .03, 0.], [.32, .21, 0.]]], np.float32))
    np.testing.assert_allclose(result['height'][0], [0., .18], atol=1e-6)
    assert result['reliable'].all()
