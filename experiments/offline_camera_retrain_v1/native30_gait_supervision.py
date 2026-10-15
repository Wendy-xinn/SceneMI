"""Opt-in TRAIN-only gait restoration; physical-time lags and GT reliability.

Ports the old55k balanced stance/swing velocity, pelvis-relative feet, stride
and limb coordination/amplitude to30Hz. It adds no inference inputs or floor.
GT height/envelope proxies are deliberately not added on top of scene physics.
"""
import math
import torch
from experiments.offline_camera_retrain_v1.native30_gt_reliability import derivative_valid

PROTOCOL = 'native30-gait-restore-v1'
DEFAULT = dict(enabled=True, protocol=PROTOCOL, fps=30, stance_speed_m_s=.2,
    stride_lags_s=[.2, .4], coordination_window_s=1.6,
    coefficients=dict(stance_velocity=.10, swing_velocity=.05,
        relative_position=1., stride_displacement=2., coordination_amplitude=5., coordination_relation=.05))
ENDS = (15, 9, 18, 19, 20, 21, 4, 5, 7, 8, 10, 11)
BASES = (0, 0, 16, 17, 16, 17, 1, 2, 1, 2, 1, 2)


def validate_config(options):
    if options.get('enabled') is not True or options.get('protocol') != PROTOCOL or options.get('fps') != 30:
        raise ValueError('Unknown native30 gait supervision')
    if options.get('stride_lags_s') != [.2, .4] or options.get('coordination_window_s') != 1.6:
        raise ValueError('Gait physical-time windows changed')
    if not math.isfinite(options['stance_speed_m_s']) or options['stance_speed_m_s'] <= 0:
        raise ValueError('Invalid stance speed')
    values = options.get('coefficients', {})
    if set(values) != set(DEFAULT['coefficients']) or any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError('Invalid gait coefficients')
    return options


def gait_terms(joints, target, quality, signal, options):
    validate_config(options)
    if joints.shape != target.shape or joints.ndim != 4 or joints.shape[2:] != (22, 3):
        raise ValueError('Expected matching [B,T,22,3] joints')
    if quality.shape != joints.shape[:3] or joints.shape[1] < 2:
        raise ValueError('Expected per-joint GT reliability and at least two frames')
    target = target.detach()
    gate = signal.detach().reshape(len(joints))  # Old55k alpha_bar, not alpha_bar squared.

    def reduce(value, mask):
        mask = mask.to(value)
        per = (value * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)
        return (per * gate).mean()

    feet, truth_feet = joints[:, :, (10, 11)], target[:, :, (10, 11)]
    velocity = feet.diff(dim=1) * options['fps']
    truth_velocity = truth_feet.diff(dim=1) * options['fps']
    reliable = derivative_valid(quality[:, :, (10, 11)])
    stance = truth_velocity.norm(dim=-1) < options['stance_speed_m_s']
    velocity_error = (velocity - truth_velocity).square().sum(-1)
    relative, true_relative = feet - joints[:, :, :1], truth_feet - target[:, :, :1]
    relative_quality = torch.minimum(quality[:, :, (10, 11)], quality[:, :, :1])
    # Keep the full denominator for soft point reliability, as in base pose loss.
    position_error = (relative - true_relative).square().sum(-1) * relative_quality
    position = (position_error.flatten(1).mean(1) * gate).mean()
    stride = joints.sum() * 0
    lags = [round(s * options['fps']) for s in options['stride_lags_s']]
    for lag in lags:
        if joints.shape[1] <= lag:
            continue
        error = (relative[:, lag:] - relative[:, :-lag]) - (true_relative[:, lag:] - true_relative[:, :-lag])
        stride = stride + reduce(error.square().sum(-1), derivative_valid(relative_quality, lag)) / len(lags)

    width = min(round(options['coordination_window_s'] * options['fps']), joints.shape[1])
    stride_frames = max(1, width // 2)

    def windows(x):
        vectors = x[:, :, ENDS] - x[:, :, BASES]
        result = vectors.unfold(1, width, stride_frames).permute(0, 1, 4, 2, 3)
        return result - result.mean(2, keepdim=True)

    pred, truth = windows(joints), windows(target)
    q = torch.minimum(quality[:, :, ENDS], quality[:, :, BASES])
    valid = q.unfold(1, width, stride_frames).amin(-1) >= 1
    pred_scale = (pred.square().sum(-1).mean(2) + .03**2).sqrt()
    true_scale = (truth.square().sum(-1).mean(2) + .03**2).sqrt()
    amplitude = reduce((pred_scale - true_scale).square(), valid)
    relation = joints.sum() * 0
    relation_lags = [lag for lag in (0, *lags) if lag < width]
    pair_mask = valid[..., :, None] & valid[..., None, :]
    pair_mask = pair_mask & ~torch.eye(len(ENDS), device=joints.device, dtype=torch.bool)
    for lag in relation_lags:
        n = width - lag

        def correlation(x, scale):
            covariance = torch.einsum('bwtjc,bwtkc->bwjk', x[:, :, :n], x[:, :, lag:]) / n
            return covariance / (scale.unsqueeze(-1) * scale.unsqueeze(-2))

        error = (correlation(pred, pred_scale) - correlation(truth, true_scale)).square()
        relation = relation + reduce(error, pair_mask) / len(relation_lags)
    terms = dict(stance_velocity=reduce(velocity_error, reliable * stance),
        swing_velocity=reduce(velocity_error, reliable * ~stance), relative_position=position,
        stride_displacement=stride, coordination_amplitude=amplitude, coordination_relation=relation)
    terms['total'] = sum(options['coefficients'][k] * v for k, v in terms.items())
    return terms
