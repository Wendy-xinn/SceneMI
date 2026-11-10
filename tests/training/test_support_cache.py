"""Cached support provenance, process execution and physical-confidence masks."""
from pathlib import Path

import numpy as np
import pytest

from scenemi.data.support_cache import (PreparedSupportIndex, SupportCache,
    SupportQueryExecutor, support_patches)
from scenemi.geometry.spatial_execution import SceneIndexCache
from scenemi.geometry.spatial_scene import SceneView, query_view
from tests.model.test_support_surfaces import index_for, plane, policy
from tests.model.test_spatial_scene import policy as spatial_policy
from tests.training.test_spatial_execution import equal_tree


def prepared(tmp_path, points):
    index = SceneIndexCache(tmp_path/'indices').get({'source': 'observed'}, lambda: index_for(points))
    cache = SupportCache(tmp_path/'support', policy())
    row = cache.prepare(index.execution_reference, index)
    return index, cache, row


def test_preparation_is_reused_and_modified_arrays_are_rejected(tmp_path, monkeypatch):
    index, cache, row = prepared(tmp_path, plane())
    def fail(*args, **kwargs):
        raise AssertionError('Prepared sequence was fitted again')
    monkeypatch.setattr('scenemi.data.support_cache.extract_support_surfaces', fail)
    assert cache.prepare(index.execution_reference, index) == row
    file = Path(row['path'])/'coefficients.npy'
    with file.open('r+b') as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match='checksum differs'):
        SupportCache(tmp_path/'support', policy()).load(index.execution_reference, index)


def test_unprepared_training_query_cannot_trigger_plane_extraction(tmp_path):
    index = index_for(plane())
    with pytest.raises(FileNotFoundError, match='unprepared'):
        SupportCache(tmp_path, policy()).load(('unused', '0'*64), index)


def test_query_tuning_reuses_extraction_but_changed_fit_policy_requires_preparation(tmp_path):
    import copy
    index, cache, row = prepared(tmp_path, plane())
    changed = copy.deepcopy(policy())
    changed['query']['temporal_weight'] = .3
    reader = SupportCache(tmp_path/'support', changed)
    assert reader.location(index.execution_reference)[0] == Path(row['path'])
    assert len(reader.load(index.execution_reference, index).arrays['coefficients']) > 0
    changed['extraction']['radius_m'] = .5
    with pytest.raises(FileNotFoundError, match='unprepared'):
        SupportCache(tmp_path/'support', changed).load(index.execution_reference, index)


def test_uncertain_support_never_overrides_measured_patch_with_physical_force(tmp_path):
    points = plane()
    points = points[np.linalg.norm(points[:, [0, 2]], axis=-1) > .11]
    index, cache, _ = prepared(tmp_path, points)
    view = SceneView(PreparedSupportIndex(index, cache.load(index.execution_reference, index)),
                     np.zeros(3), np.eye(3))
    anchors = np.zeros((4, 2, 3), np.float32)
    anchors[..., 1] = .03
    result = support_patches(view, anchors, spatial_policy(), ground_mask=np.ones(2, bool))
    assert not result['valid'].any() and not result['confidence'].any()


def test_process_and_serial_queries_share_geometry_and_never_prepare_missing_maps(tmp_path):
    index, cache, _ = prepared(tmp_path, np.concatenate((plane(), plane(.35))))
    view = SceneView(PreparedSupportIndex(index, cache.load(index.execution_reference, index)),
                     np.zeros(3), np.eye(3))
    anchors = np.zeros((4, 2, 3), np.float32)
    anchors[..., 1] = .37
    p = spatial_policy()
    expected = query_view(view, anchors, p, ground_mask=np.ones(2, bool))
    patches = support_patches(view, anchors, p, ground_mask=np.ones(2, bool))
    with SupportQueryExecutor(2, tmp_path/'indices', tmp_path/'support', policy()) as executor:
        actual = executor.submit('features', view, anchors, p, ground_mask=np.ones(2, bool)).result(timeout=60)
        equal_tree(expected, actual)
        actual = executor.submit('patches', view, anchors, p, ground_mask=np.ones(2, bool)).result(timeout=60)
        equal_tree(patches, actual)
        with pytest.raises(ValueError, match='Unknown support query'):
            executor.submit('invalid', view, anchors, p).result(timeout=60)


def test_dead_query_worker_is_replaced_and_identical_task_replayed_once(tmp_path):
    index, cache, _ = prepared(tmp_path, plane())
    view = SceneView(PreparedSupportIndex(index, cache.load(index.execution_reference, index)),
                     np.zeros(3), np.eye(3))
    anchors = np.zeros((4, 2, 3), np.float32)
    anchors[..., 1] = .03
    expected = query_view(view, anchors, spatial_policy())
    logs = tmp_path/'logs'
    with SupportQueryExecutor(1, tmp_path/'indices', tmp_path/'support', policy(), log_root=logs) as executor:
        process = executor.workers[0][1]
        process.kill()
        process.wait()
        actual = executor.submit('features', view, anchors, spatial_policy()).result(timeout=60)
        equal_tree(expected, actual)
        assert len(executor.workers) == 2
    import json
    records = list(logs.glob('failure_*/FAILURE.json'))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record['observed_exit_code'] == -9 and record['replay_scheduled']
    with np.load(records[0].parent/'QUERY.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['anchors'], anchors)


def test_repeated_worker_death_has_a_retry_limit(tmp_path, monkeypatch):
    index, cache, _ = prepared(tmp_path, plane())
    view = SceneView(PreparedSupportIndex(index, cache.load(index.execution_reference, index)),
                     np.zeros(3), np.eye(3))
    logs = tmp_path/'logs'
    with SupportQueryExecutor(1, tmp_path/'indices', tmp_path/'support', policy(), log_root=logs) as executor:
        process = executor.workers[0][1]
        process.kill()
        process.wait()
        spawn = executor._spawn_worker
        def killed_worker():
            worker = spawn()
            worker[1].kill()
            worker[1].wait()
            return worker
        monkeypatch.setattr(executor, '_spawn_worker', killed_worker)
        with pytest.raises(RuntimeError, match='bounded replay'):
            executor.submit('features', view, np.zeros((2, 2, 3), np.float32), spatial_policy()).result(timeout=60)
    assert len(list(logs.glob('failure_*/FAILURE.json'))) == 2


def test_bad_worker_input_is_not_retried(tmp_path):
    index, _, row = prepared(tmp_path, plane())
    file = Path(row['path'])/'coefficients.npy'
    with file.open('r+b') as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 1]))
    view = SceneView(index, np.zeros(3), np.eye(3))
    logs = tmp_path/'logs'
    with SupportQueryExecutor(1, tmp_path/'indices', tmp_path/'support', policy(), log_root=logs) as executor:
        with pytest.raises(ValueError, match='checksum differs'):
            executor.submit('features', view, np.zeros((2, 2, 3), np.float32), spatial_policy()).result(timeout=60)
        assert len(executor.workers) == 1
    assert not list(logs.glob('failure_*'))


def test_evaluation_resumes_committed_batches_and_rejects_changed_weights(tmp_path):
    import copy
    from types import SimpleNamespace
    import torch
    from scenemi.config import ROOT
    from scenemi.refinement.support_training import evaluate, load_config
    config = copy.deepcopy(load_config(ROOT/'configs/ego_scene_support_refinement.yaml'))
    config['validation']['eval_batch_size'] = 4
    model = torch.nn.Module()
    model.refiner = torch.nn.Linear(1, 1)
    cases = [dict(case_id=i, dataset=source) for i, source in enumerate(
        config['data']['datasets']+config['data']['datasets'])]
    called = []
    def batch(model, data, selected, config, device):
        called.append(selected[0]['case_id'])
        rows = [dict(case=case, identity=dict(dataset=case['dataset'], family='family'),
            metrics=dict(mpjpe_mm=float(case['case_id']), head_error_mm=1., jerk_m_s3=2.)) for case in selected]
        return rows, copy.deepcopy(rows)
    def interrupted(*args):
        if args[2][0]['case_id'] == 4:
            raise RuntimeError('injected worker failure')
        return batch(*args)
    with pytest.raises(RuntimeError, match='injected'):
        evaluate(model, SimpleNamespace(), cases, config, 'cpu', tmp_path/'resume', batch_evaluator=interrupted)
    assert not (tmp_path/'resume/RESULT.json').exists()
    assert called == [0]
    result = evaluate(model, SimpleNamespace(), cases, config, 'cpu', tmp_path/'resume', batch_evaluator=batch)
    assert called == [0, 4]
    fresh = evaluate(model, SimpleNamespace(), cases, config, 'cpu', tmp_path/'fresh', batch_evaluator=batch)
    assert result['dataset_macro_mean'] == fresh['dataset_macro_mean']
    assert result['baseline'] == fresh['baseline']
    with torch.no_grad():
        model.refiner.weight.add_(1.)
    with pytest.raises(ValueError, match='different weights'):
        evaluate(model, SimpleNamespace(), cases, config, 'cpu', tmp_path/'resume', batch_evaluator=batch)


def test_configuration_keeps_network_sampling_and_optimizer_schedule():
    from scenemi.refinement.support_training import load_config
    from scenemi.refinement.spatial_training import load_config as old_config
    from scenemi.config import ROOT
    current = load_config(ROOT/'configs/ego_scene_support_refinement.yaml')
    old = old_config(ROOT/current['base_config'])
    for key in ('seed', 'data', 'training', 'optimizer', 'refiner', 'sampling', 'spatial_loss'):
        assert current[key] == old[key]
    assert current['output'] != old['output']
    assert 'execution_continuation' not in current


def test_preparation_addresses_cached_maps_without_constructing_parent_indices(tmp_path):
    from types import SimpleNamespace
    from scenemi.refinement.support_training import observed_reference
    identity = dict(version='complete-observed-index-v1', split='train', map_sha256='observed-map')
    cache = SceneIndexCache(tmp_path)
    index = cache.get(identity, lambda: index_for(plane()))
    def fail(row):
        raise AssertionError('Preparation rebuilt an existing full-map index')
    data = SimpleNamespace(split='train', index_cache=cache, source_index=fail,
        video_maps=SimpleNamespace(by_path={'source': {'sha256': 'observed-map'}}))
    assert observed_reference(data, {'dataset': 'egobody', 'path': 'source'}) == index.execution_reference


@pytest.mark.parametrize('action', ['train', 'preflight'])
def test_public_launcher_records_native_exit_and_preserves_python_error(tmp_path, monkeypatch, action):
    import json
    import yaml
    from scenemi.refinement import __main__ as entry
    monkeypatch.setattr(entry, 'ROOT', tmp_path)
    out = tmp_path/'runs/example'
    folder = out/'preflight/cuda' if action == 'preflight' else out
    folder.mkdir(parents=True)
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump({'output': 'runs/example'}))
    status = folder/'STATUS.json'
    status.write_text(json.dumps({'state': 'running', 'step': 206}))
    args = [action, '--config', str(config), '--device', 'cuda']
    entry.record_exit(args, -11)
    result = json.loads(status.read_text())
    assert result['state'] == 'failed' and result['native_signal'] == 11
    assert result['step'] == 206 and result['process_alive'] is False
    result['error'] = 'original Python failure'
    status.write_text(json.dumps(result))
    entry.record_exit(args, 1)
    assert json.loads(status.read_text())['error'] == 'original Python failure'
    status.unlink()
    folder.rmdir()
    entry.record_exit(args, -11)
    assert json.loads(status.read_text())['native_signal'] == 11
