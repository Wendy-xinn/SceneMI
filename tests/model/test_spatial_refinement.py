"""The new scene inputs preserve the frozen prior and trainable SO(3) path."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scenemi.config import ROOT
from scenemi.geometry.spatial_scene import ObservedSceneIndex, SceneView, query_view
from scenemi.refinement.spatial_model import SpatialSceneRefinementPipeline

spec = importlib.util.spec_from_file_location('spatial_base_fixture', Path(__file__).with_name('test_scene_refinement.py'))
base_fixture = importlib.util.module_from_spec(spec);spec.loader.exec_module(base_fixture)


def fixture():
    old, raw, details, public, _, _ = base_fixture.fixture()
    policy = yaml.safe_load((ROOT/'configs/ego_scene_spatial_query.yaml').read_text())
    refiner = base_fixture.policies()[0]
    model = SpatialSceneRefinementPipeline(old, refiner, policy).double()
    # Context without any 20cm neighbours, and no claimed support plane.
    index = ObservedSceneIndex(np.array([[.4, 1., 0.]], np.float32))
    view = SceneView(index, np.zeros(3), np.eye(3))
    public = dict(public, scene_views=(view, view))
    return model, raw, details, public, policy


def test_context_only_geometry_does_not_disappear_at_the_20cm_boundary():
    model, raw, details, public, _ = fixture()
    final, extra = model.refine(raw, details, public)
    assert extra['anchor_hit'].any() and not extra['anchor_near_hit'].any()
    assert torch.equal(final, model.locate(raw, details))
    torch.nn.init.normal_(model.refiner.rotation.weight, std=.02)
    final, extra = model.refine(raw, details, public)
    assert extra['rotation_residual'].abs().max() > 0
    assert torch.isfinite(final).all()


def test_unknown_disabled_and_per_example_dropout_restore_exact_baseline_after_learning():
    model, raw, details, public, _ = fixture()
    torch.nn.init.normal_(model.refiner.rotation.weight, std=.02)
    baseline = model.locate(raw, details)
    empty = SceneView(ObservedSceneIndex(np.empty((0, 3))), np.zeros(3), np.eye(3))
    for inputs, enabled in ((dict(public, scene_views=(empty, empty)), True),
                            (public, False), (dict(public, scene_views=(None, None)), True)):
        output, _ = model.refine(raw, details, inputs, use_scene=enabled)
        assert torch.equal(output, baseline)
    output, _ = model.refine(raw, details, public, use_scene=torch.tensor([True, False]))
    assert torch.equal(output[1], baseline[1])


def test_point_encoder_and_rotation_head_receive_finite_gradients_while_prior_stays_frozen():
    model, raw, details, public, _ = fixture()
    torch.nn.init.normal_(model.refiner.rotation.weight, std=.02)
    final, extra = model.refine(raw, details, public)
    loss = final[..., 3:135].square().mean()+extra['rotation_residual'].square().mean()
    loss.backward()
    for parameter in (model.refiner.point_encoder[0].weight, model.refiner.rotation.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().max() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.base.parameters())
    with pytest.raises(ValueError, match='Labels'):
        model.refine(raw, details, dict(public, target_rest=details['rest']))


def test_dynamic_context_is_current_frame_only_and_never_added_to_static_map():
    _, _, _, _, policy = fixture()
    index = ObservedSceneIndex(np.empty((0, 3)))
    view = SceneView(index, np.zeros(3), np.eye(3))
    anchors = np.zeros((2, 1, 3), np.float32)
    dynamic = (np.array([[[.1, 0, 0]], [[.1, 0, 0]]], np.float32), np.array([[1], [100]]),
        np.zeros((2, 1, 3), np.float32), np.zeros((2, 1), bool))
    result = query_view(view, anchors, policy, dynamic=dynamic)
    assert result['near']['valid'].any(-1).ravel().tolist() == [True, False]
    assert len(index.points) == 0
