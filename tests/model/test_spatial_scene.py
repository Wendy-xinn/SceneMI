"""Full-map lookup, unknown geometry and bounded observed support evidence."""
import copy

import numpy as np
import yaml

from scenemi.config import ROOT
from scenemi.data.video_scene import frame_ranges, observer_key, voxel_observations
from scenemi.geometry.spatial_scene import ObservedSceneIndex, SceneView


def policy():
    return yaml.safe_load((ROOT/'configs/ego_scene_spatial_query.yaml').read_text())


def plane(y=0., lo=-.5, hi=.5, step=.04):
    x, z = np.meshgrid(np.arange(lo, hi+step/2, step), np.arange(lo, hi+step/2, step))
    points = np.column_stack((x.ravel(), np.full(x.size, y), z.ravel())).astype(np.float32)
    normals = np.tile([0., 1., 0.], (len(points), 1))
    return points, normals


def test_full_map_preserves_point_lost_by_historical_global_budget():
    points = np.column_stack((np.arange(20000), np.ones(20000), np.zeros(20000))).astype(np.float32)
    kept = np.linspace(0, len(points)-1, 8192).astype(int)
    missed = next(i for i in range(len(points)) if i not in set(kept))
    anchor = points[missed:missed+1]+[0, .1, 0]
    assert ObservedSceneIndex(points[kept]).nearest(anchor)[0] > .2
    query = ObservedSceneIndex(points).query(anchor, .2, 8)
    assert query['valid'].sum() == 1
    np.testing.assert_allclose(query['distance'][0, 0], .1, atol=1e-7)
    assert query['points'].shape == (1, 8, 3)


def test_context_and_density_survive_near_radius_boundary():
    index = ObservedSceneIndex(np.array([[.19, 0, 0], [.21, 0, 0], [.3, 0, 0]], np.float32))
    anchors = np.zeros((2, 3), np.float32)
    near, context = index.query(anchors, .2, 1), index.query(anchors, .8, 1)
    assert near['neighbour_count'].tolist() == [1, 1]
    assert context['neighbour_count'].tolist() == [3, 3]
    anchors[0, 0] = -.03
    assert not index.query(anchors, .2)['valid'][0].any()
    assert index.query(anchors, .8)['valid'][0].any()


def test_empty_map_has_explicit_unknown_masks_and_no_fabricated_ground():
    index = ObservedSceneIndex(np.empty((0, 3), np.float32))
    anchors = np.zeros((2, 4, 3), np.float32)
    query = index.query(anchors, .2)
    assert not query['valid'].any() and np.isinf(query['nearest_distance']).all()
    assert np.isfinite(query['points']).all()
    ground = index.support_planes(anchors, policy()['ground'])
    assert not ground['valid'].any() and not ground['confidence'].any()


def test_ground_can_interpolate_a_small_observation_hole_but_not_large_unknown_area():
    points, normals = plane()
    keep = np.linalg.norm(points[:, [0, 2]], axis=-1) > .12
    index = ObservedSceneIndex(points[keep], normals[keep], np.ones(keep.sum(), bool))
    ground = index.support_planes(np.array([[0., .1, 0], [1., .1, 0]]), policy()['ground'])
    assert ground['valid'].tolist() == [True, False]
    np.testing.assert_allclose(ground['height'][0], 0., atol=1e-8)
    keep = np.linalg.norm(points[:, [0, 2]], axis=-1) > .35
    index = ObservedSceneIndex(points[keep], normals[keep], np.ones(keep.sum(), bool))
    assert not index.support_planes(np.array([[0., .1, 0]]), policy()['ground'])['valid'].any()


def test_multiple_support_levels_remain_ambiguous_not_one_invented_floor():
    lower, normal = plane(); upper, _ = plane(y=.3)
    points = np.concatenate((lower, upper));normals = np.concatenate((normal, normal))
    index = ObservedSceneIndex(points, normals, np.ones(len(points), bool))
    ground = index.support_planes(np.array([[0., .35, 0]]), policy()['ground'])
    assert ground['ambiguous'][0] and not ground['valid'][0]
    assert ground['confidence'][0] <= .25
    assert min(abs(ground['height'][0]), abs(ground['height'][0]-.3)) < 1e-6


def test_window_transform_roundtrips_without_changing_metric_neighbours():
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    origin = np.array([5., 0., -8.])
    index = ObservedSceneIndex(np.array([[5., .05, -8.]]))
    view = SceneView(index, origin, rotation)
    local = np.array([[0., .1, 0.]])
    np.testing.assert_allclose(view.to_local(view.to_world(local)), local)
    np.testing.assert_allclose(index.nearest(view.to_world(local)), [.05])


def test_voxel_union_counts_distinct_frames_not_duplicate_pixels_or_clips():
    points = np.array([[0., 0, 0], [.001, 0, 0], [.002, 0, 0], [1, 0, 0]], np.float32)
    selected, cameras, counts = voxel_observations(points, [3, 3, 4, 3], np.ones_like(points), .025)
    np.testing.assert_array_equal(selected, points[[0, 3]])
    assert counts.tolist() == [2, 1]
    assert frame_ranges([2, 3, 7, 8, 8]) == [[2, 3], [7, 8]]
    assert frame_ranges({2, 3, 7, 8}) == [[2, 3], [7, 8]]


def test_video_group_does_not_pool_other_people_world_frames_splits_or_camera_mounts():
    meta = dict(dataset='egobody', allowed_modes=['ego'], recording='video', group='camera_wearer',
        world_frame_id='world', split='validation', scene_id='scene', actual_virtual_mount=None)
    key, _ = observer_key(meta)
    for change in (dict(group='interactee'), dict(world_frame_id='other'), dict(split='train')):
        assert observer_key(dict(meta, **change))[0] != key
    virtual = dict(meta, actual_virtual_mount=dict(translation_m=[0., .1, .1]), native_body=dict(betas=[0.]*10))
    changed = copy.deepcopy(virtual);changed['actual_virtual_mount']['translation_m'][1] += .01
    assert observer_key(virtual)[0] != key and observer_key(changed)[0] != observer_key(virtual)[0]
