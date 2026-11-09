"""Check that the causal interventions do not change the frozen motion input."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from scenemi.refinement.diagnostics import (donor_cases, replace_scene, SCENE_FIELDS,
    oracle_scene_features, from_features, paired_statistics, coverage, PRIMARY)
from scenemi.refinement.losses import contact_targets

# Share the established synthetic body/scene fixture without another body model.
spec = importlib.util.spec_from_file_location('refiner_fixture',
    Path(__file__).parents[1]/'model/test_scene_refinement.py')
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


def test_scene_intervention_preserves_camera_and_does_not_mutate_inputs():
    public = {key:torch.tensor([1.]) for key in SCENE_FIELDS}
    public.update(camera=torch.tensor([2.]), body_format=torch.tensor([1]))
    donor = {key:torch.tensor([3.]) for key in SCENE_FIELDS}
    changed = replace_scene(public, donor)
    assert changed['camera'] is public['camera']
    assert changed['body_format'] is public['body_format']
    assert all(changed[key] is donor[key] and public[key].item()==1 for key in SCENE_FIELDS)


def test_mismatch_is_deterministic_same_source_and_different_recording():
    cases = [dict(dataset=source, family=f'rec{i}', case_id=i, start=0)
             for source in ('egobody','trumans','grail') for i in range(3)]
    cases += [dict(dataset='amass',family='a',case_id=10,start=0)]
    donors = donor_cases(cases, 123)
    assert donors == donor_cases(cases, 123)
    for (source, family), donor in donors.items():
        assert donor['dataset']==source and donor['family']!=family
    with pytest.raises(ValueError, match='at least two'):
        donor_cases([dict(dataset='egobody',family='only')], 123)


def test_oracle_with_identical_anchors_matches_trained_features_and_readout():
    model, raw, details, public, batch, _ = fixture_module.fixture()
    torch.nn.init.normal_(model.refiner.rotation.weight, std=.02)
    placed = model.locate(raw, details)
    motion, trained, hit = model.features(placed, details, public, True)
    anchors, _ = model.skin_points(placed, details, public['body_format'])
    oracle, oracle_hit, _ = oracle_scene_features(model, placed, details, public, anchors, anchors)
    torch.testing.assert_close(oracle, trained, atol=1e-12, rtol=1e-12)
    assert torch.equal(hit, oracle_hit)
    expected, expected_extra = model.refine(raw, details, public)
    actual, actual_extra = from_features(model, raw, details, placed, motion, trained, hit)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_extra['rotation_residual'], expected_extra['rotation_residual'])


def test_lookup_oracle_selects_gt_patch_but_encodes_predicted_body_distance():
    model, raw, details, public, _, _ = fixture_module.fixture()
    placed = model.locate(raw, details)
    anchors, _ = model.skin_points(placed, details, public['body_format'])
    lookup = anchors.clone()
    lookup[..., 0] += .1
    features, hit, patch = oracle_scene_features(model, placed, details, public, anchors, lookup)
    distance = (patch['centre']-anchors).norm(dim=-1)/model.geometry_policy['query_radius_m']
    torch.testing.assert_close(features[..., 7][hit], distance[hit])
    assert torch.equal(anchors, model.skin_points(placed, details, public['body_format'])[0])


def test_zero_correction_audit_distinguishes_missing_neighbours_from_head_output():
    model, raw, details, public, batch, policy = fixture_module.fixture()
    placed = model.locate(raw, details)
    anchors, _ = model.skin_points(placed, details, public['body_format'])
    label, weight = contact_targets(model, batch, policy['contact'])
    final, extra = model.refine(raw, details, public)
    audit = coverage(model, placed, raw, details, batch, public, extra, final,
                     label, weight, anchors, anchors)
    assert audit[0]['zero_reason']=='zero_rotation_head_output'
    assert sum(audit[0]['contact_probability_histogram'])>0
    far = dict(public, static_points=public['static_points']+10)
    final, extra = model.refine(raw, details, far)
    audit = coverage(model, placed, raw, details, batch, far, extra, final,
                     label, weight, anchors, anchors)
    assert audit[0]['zero_reason']=='no_anchor_neighbour_hits'
    assert audit[0]['motion_exactly_unchanged']


def test_paired_summary_bootstraps_recordings_instead_of_counting_windows():
    rows = []
    for family, count, gain in [('long', 10, 1.), ('short', 1, 9.)]:
        for _ in range(count):
            variants = {name:{metric:100. for metric in PRIMARY}
                        for name in ('baseline','correct','empty','mismatched','query_oracle')}
            variants['correct']['mpjpe_mm'] -= gain
            rows.append(dict(identity=dict(dataset='trumans',family=family,path=family),variants=variants))
    result = paired_statistics(rows, 11, bootstrap=100)
    score = result['trumans']['comparisons']['correct_minus_baseline']['mpjpe_mm']
    assert score['recordings']==2 and score['delta']==-5.
    assert score['ci95'][0] <= -5 <= score['ci95'][1]
    assert result == paired_statistics(copy.deepcopy(rows), 11, bootstrap=100)


def test_recording_bootstrap_groups_camera_wearer_clips_from_same_recording():
    rows = []
    for family, clip, gain in [('recording_A','clip_1',1.), ('recording_A','clip_2',3.),
                               ('recording_B','clip_3',10.)]:
        variants = {name:{metric:100. for metric in PRIMARY}
                    for name in ('baseline','correct','empty','mismatched','query_oracle')}
        variants['correct']['mpjpe_mm'] -= gain
        rows.append(dict(identity=dict(dataset='egobody', family=family, path=clip), variants=variants))
    score = paired_statistics(rows, 11, bootstrap=100)['egobody']['comparisons']['correct_minus_baseline']['mpjpe_mm']
    assert score['recordings']==2 and score['delta']==-6.
