import numpy as np
import pytest
import torch

from scenemi.geometry.head_anchor import correction_operator, translation_grid_metrics, translate_motion, camera_residual
from scenemi.geometry.alignment import place_ego_motion
from scenemi.metrics_batch import metric_tensors
from tests.model.test_ego_alignment import EgoAlignmentTests
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def quadratic(acc=.06**4, jerk=.06**6):
    return dict(kind='quadratic', data_weight=1., acceleration_weight=acc, jerk_weight=jerk)


def test_quadratic_solves_the_stated_objective_and_preserves_affine_corrections():
    n = 128; op = correction_operator(quadratic(), n, 30)
    rng = np.random.default_rng(4); residual = rng.normal(size=(n, 3)); result = op @ residual
    d2 = np.diff(np.eye(n), n=2, axis=0) * 30**2
    d3 = np.diff(np.eye(n), n=3, axis=0) * 30**3
    gradient = result - residual + .06**4 * d2.T @ (d2 @ result) + .06**6 * d3.T @ (d3 @ result)
    np.testing.assert_allclose(gradient, 0, atol=1e-10)
    affine = np.stack((np.ones(n), np.arange(n)), 1)
    np.testing.assert_allclose(op @ affine, affine, atol=1e-9)
    np.testing.assert_allclose(correction_operator(quadratic(0, 0), n, 30), np.eye(n))


def test_smoothing_suppresses_high_frequency_correction_without_filtering_the_body():
    n = 128; residual = .05 * (-1.)**np.arange(n)
    for kind in ('gaussian', 'savgol', 'quadratic'):
        method = quadratic() if kind == 'quadratic' else dict(kind=kind, window=15, strength=1., polyorder=2)
        op = correction_operator(method, n, 30)
        assert np.linalg.norm(np.diff(op @ residual, n=3)) < .15 * np.linalg.norm(np.diff(residual, n=3))
        np.testing.assert_allclose(op @ np.ones(n), 1., atol=1e-10)


def test_translation_keeps_every_rotation_and_matches_existing_hard_anchor():
    fixture = EgoAlignmentTests(); fixture.setUp()
    motion, rest, camera, mount = fixture.motion, fixture.rest, fixture.camera, fixture.mount
    correction = camera_residual(motion, rest, mount, camera)
    shifted = translate_motion(motion, rest, correction)
    hard = place_ego_motion(motion, rest, mount, camera, rotation=False)
    torch.testing.assert_close(shifted, hard, atol=1e-12, rtol=1e-12)
    assert torch.equal(shifted[..., 3:135], motion[..., 3:135])
    before, after = [forward_kinematics(x, rest) for x in (motion, shifted)]
    torch.testing.assert_close(after - before, correction[:, :, None].expand_as(after), atol=1e-12, rtol=1e-12)


def test_grid_metrics_match_original_metrics_for_time_varying_translation():
    rng = torch.Generator().manual_seed(3)
    pred, gt = [torch.randn(2, 16, 22, 3, generator=rng, dtype=torch.float64) for _ in range(2)]
    residual = torch.randn(2, 16, 3, generator=rng, dtype=torch.float64)
    op = torch.as_tensor(np.stack([correction_operator(m, 16, 30) for m in
        (dict(kind='raw'), dict(kind='hard'), quadratic())]))
    grid = translation_grid_metrics(pred, gt, residual, op)
    for i in range(len(op)):
        shifted = pred + torch.einsum('tq,bqc->btc', op[i], residual)[:, :, None]
        reference = metric_tensors(shifted, gt)
        for key in reference.keys() & grid.keys():
            torch.testing.assert_close(grid[key][i], reference[key], atol=1e-8, rtol=1e-10)


def test_invalid_smoothing_parameters_are_rejected():
    with pytest.raises(ValueError): correction_operator(dict(kind='gaussian', window=8, strength=1.), 128, 30)
    with pytest.raises(ValueError): correction_operator(quadratic(-1, 0), 128, 30)
