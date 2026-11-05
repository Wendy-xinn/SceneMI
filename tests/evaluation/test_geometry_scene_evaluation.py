import pytest
import torch

from scenemi.geometry_scene_evaluation import SCENE_INPUTS, condition_payload


def scene():
    return dict(occupancy=torch.ones(1, 2, 2, 2),
        frame_scene_points=torch.ones(1, 2, 3, 3), frame_scene_owner=torch.zeros(1, 2, 3, dtype=torch.long),
        static_points=torch.ones(1, 4, 3), static_normals=torch.ones(1, 4, 3),
        static_valid=torch.ones(1, 4, dtype=torch.bool), static_normal_valid=torch.ones(1, 4, dtype=torch.bool),
        frame_normals=torch.ones(1, 2, 3, 3), frame_normal_valid=torch.ones(1, 2, 3, dtype=torch.bool))


def test_interventions_replace_every_geometry_channel_and_preserve_camera_body():
    public = dict(scene(), camera=torch.randn(1, 2, 9), body_format=torch.tensor([2]))
    donor = {k: (torch.zeros_like(v) if v.dtype == torch.bool else v + 2) for k, v in scene().items()}
    original = {k: v.clone() for k, v in public.items()}
    for condition in ('matched', 'disabled', 'empty', 'mismatched'):
        result = condition_payload(public, condition, donor)
        assert result['camera'] is public['camera'] and result['body_format'] is public['body_format']
        for k in SCENE_INPUTS:
            if condition == 'disabled': assert k not in result
            elif condition == 'mismatched': assert result[k] is donor[k]
            elif condition == 'empty':
                assert torch.all(result[k] == (-1 if k == 'frame_scene_owner' else 0))
            else: assert result[k] is public[k]
    assert all(torch.equal(public[k], v) for k, v in original.items())


def test_partial_or_incompatible_donor_is_rejected():
    public = scene()
    with pytest.raises(ValueError): condition_payload(public, 'mismatched')
    donor = scene(); donor['static_valid'] = torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match='static_valid'): condition_payload(public, 'mismatched', donor)


def test_unknown_condition_cannot_silently_keep_the_matched_scene():
    with pytest.raises(ValueError): condition_payload(scene(), 'shuffled')
