import torch
from scenemi.metrics_temporal import temporal_metrics


def test_cubic_trajectory_has_known_physical_jerk():
    time = torch.arange(128, dtype=torch.float64) / 30
    points = torch.zeros(1, 128, 22, 3, dtype=torch.float64)
    points[..., 0] = time[None, :, None] ** 3
    value = temporal_metrics(points, points)
    torch.testing.assert_close(value['jitter_100m_s3'], torch.tensor([.06], dtype=torch.float64))
    torch.testing.assert_close(value['gt_jitter_100m_s3'], value['jitter_100m_s3'])


def test_static_truth_and_one_millimeter_sliding_distinguish_reductions():
    truth = torch.zeros(1, 128, 22, 3, dtype=torch.float64)
    pred = truth.clone()
    pred[:, :, [7, 8, 10, 11], 0] = torch.arange(128, dtype=torch.float64)[None, :, None] * .001
    value = temporal_metrics(pred, truth)
    torch.testing.assert_close(value['foot_sliding_avatarjlm_cm_frame'], torch.tensor([.1 / 3], dtype=torch.float64))
    torch.testing.assert_close(value['foot_sliding_horizontal_mm_contact_frame'], torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(value['egoallo_cumulative_foot_skate_m'], torch.tensor([.508], dtype=torch.float64))
    assert value['gt_foot_contact_transitions'].item() == 127 * 4


def test_no_contact_is_explicit_and_does_not_produce_nan():
    truth = torch.zeros(1, 128, 22, 3, dtype=torch.float64)
    truth[..., 0] = torch.arange(128, dtype=torch.float64)[None, :, None]
    value = temporal_metrics(truth, truth)
    assert value['foot_sliding_horizontal_has_contact'].item() == 0
    assert value['foot_sliding_horizontal_mm_contact_frame'].item() == 0
    assert all(torch.isfinite(v).all() for v in value.values())
