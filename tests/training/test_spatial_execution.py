"""Process queries preserve geometry, ordering, gradients and failure semantics."""
import copy

import numpy as np
import pytest
import torch

from scenemi.geometry.spatial_execution import SceneIndexCache, SpatialQueryExecutor
from scenemi.geometry.spatial_scene import query_view
from scenemi.geometry.spatial_scene import ObservedSceneIndex
from scenemi.refinement.spatial_losses import observed_patches
from tests.training.test_spatial_refinement import fixture


def equal_tree(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal_tree(a[k], b[k])
    else:
        assert np.array_equal(a, b)


@pytest.mark.parametrize('array_backend', ['numpy', 'numeric_header_v1'])
def test_process_queries_and_training_labels_match_serial_and_propagate_failures(tmp_path, array_backend):
    model, batch, points, _ = fixture()
    view = batch['scene_views'][0];policy = model.spatial_policy
    state = copy.deepcopy(np.random.get_state())
    anchors = points[0].numpy()
    serial = query_view(view, anchors, policy)
    labels = observed_patches(model, points, batch, ground=True)
    with SpatialQueryExecutor(2, tmp_path/'indices', array_backend=array_backend) as pool:
        futures = [pool.submit('features', view, anchors, policy) for _ in range(3)]
        for future in futures: equal_tree(serial, future.result(timeout=60))
        model.query_executor = pool
        actual = observed_patches(model, points, batch, ground=True)
        for k in labels: assert torch.equal(labels[k], actual[k])
        with pytest.raises(ValueError, match='Unknown spatial action'):
            pool.submit('invalid', view, anchors, policy).result(timeout=60)
    assert pool.closed
    with pytest.raises(RuntimeError, match='closed'):
        pool.submit('features', view, anchors, policy)
    current = np.random.get_state()
    assert state[0] == current[0] and np.array_equal(state[1], current[1]) and state[2:] == current[2:]


def test_complete_map_cache_reuses_exact_normals_and_rejects_modified_arrays(tmp_path):
    model, batch, _, _ = fixture();index = batch['scene_views'][0].index
    cache = SceneIndexCache(tmp_path)
    a = cache.get({'source_sha256': 'test', 'split': 'train'}, lambda: index)
    def fail(): raise AssertionError('Cached full-map normals were rebuilt')
    b = cache.get({'source_sha256': 'test', 'split': 'train'}, fail)
    assert a.execution_reference == b.execution_reference
    for key in ('points', 'normals', 'normal_valid', 'observations'):
        assert np.array_equal(getattr(index, key), getattr(b, key))
    from pathlib import Path
    path = Path(a.execution_reference[0])/'normals.npy'
    with path.open('r+b') as stream:
        stream.seek(-1, 2);value=stream.read(1);stream.seek(-1, 2);stream.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match='checksum differs'):
        SceneIndexCache(tmp_path).get({'source_sha256': 'test', 'split': 'train'}, fail)


def test_parallel_patches_keep_current_object_frame_and_exclude_body_owners(tmp_path):
    model, batch, points, _ = fixture(ObservedSceneIndex(np.empty((0, 3))))
    batch['frame_scene_points'] = torch.zeros(1, 8, 2, 3)
    batch['frame_scene_points'][..., 0, 1] = .05
    batch['frame_scene_points'][:, 4:, 0, 0] = 2.
    batch['frame_scene_points'][..., 1, 1] = .001
    batch['frame_scene_owner'] = torch.tensor([1, 100])[None, None].expand(1, 8, 2)
    batch['frame_normals'] = torch.tensor([0., 1., 0.])[None, None, None].expand(1, 8, 2, 3)
    batch['frame_normal_valid'] = torch.ones(1, 8, 2, dtype=torch.bool)
    serial = observed_patches(model, points, batch)
    with SpatialQueryExecutor(2, tmp_path) as pool:
        model.query_executor = pool
        actual = observed_patches(model, points, batch)
    for k in serial: assert torch.equal(serial[k], actual[k])
    assert actual['valid'][:, :4].all() and not actual['valid'][:, 4:].any()
    assert torch.all(actual['centre'][:, :4, :, 1] == .05)


def test_execution_configuration_preserves_model_data_loss_and_optimizer():
    from scenemi.config import ROOT
    from scenemi.refinement.spatial_training import load_config
    serial = load_config(ROOT/'configs/ego_scene_spatial_refinement.yaml')
    parallel = load_config(ROOT/'configs/ego_scene_spatial_refinement_parallel.yaml')
    assert parallel.pop('spatial_execution')['processes'] == 4
    for name in ('_config_path', 'output'):
        serial.pop(name);parallel.pop(name)
    assert serial == parallel


def test_stable_execution_preserves_science_and_only_disables_data_thread():
    from scenemi.config import ROOT
    from scenemi.refinement.spatial_training import load_config, execution_config_matches
    old = load_config(ROOT/'configs/ego_scene_spatial_refinement_overnight.yaml')
    new = load_config(ROOT/'configs/ego_scene_spatial_refinement_stable.yaml')
    assert new['prepared_data']['prefetch_batches'] == 0
    assert new['spatial_execution']['processes'] == 4
    assert execution_config_matches(old, new)
    new['spatial_loss']['weights']['geometry'] *= 2
    assert not execution_config_matches(old, new)
