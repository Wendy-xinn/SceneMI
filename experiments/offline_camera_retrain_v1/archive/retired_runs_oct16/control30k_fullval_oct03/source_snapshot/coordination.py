"""Action-agnostic, training-only short-window whole-body relations.

No gait label, preferred limb phase, or held-out information is used at inference.
These are reference-relative diagnostics, NOT a universal naturalness score.
"""
import torch


# Head/chest, upper/lower arms, hands, upper/lower legs, feet. Long segments
# retain end-effector coordination that bone rotations alone can obscure.
ENDS = (15, 9, 18, 19, 20, 21, 4, 5, 7, 8, 10, 11)
BASES = (0, 0, 16, 17, 16, 17, 1, 2, 1, 2, 1, 2)


def coordination_losses(joints, target, signal_weight=None):
    """Match local lagged relations and amplitude, not an absolute gait phase.

    Centering removes static shape; segment vectors remove body translation.
    Dot products are invariant to a common fixed 3-D rotation. 3-cm energy
    regularization prevents tiny/noisy stationary signals dominating a ratio.
    Low-signal diffusion examples are downweighted by the caller's alpha_bar.
    Window statistics tolerate phase changes but are not strictly phase-invariant.
    """
    if joints.shape != target.shape or joints.ndim != 4 or joints.shape[2:] != (22, 3):
        raise ValueError('Expected matching [B,T,22,3] tensors')
    if joints.shape[1] < 2:
        raise ValueError('Coordination requires at least two frames')
    width = min(32, joints.shape[1])
    stride = max(1, width // 2)

    def windows(p):
        x = p[:, :, ENDS] - p[:, :, BASES]
        # [B,W,J,3,T] -> [B,W,T,J,3]
        x = x.unfold(1, width, stride).permute(0, 1, 4, 2, 3)
        return x - x.mean(dim=2, keepdim=True)

    pred, truth = windows(joints.float()), windows(target.detach().float())
    pred_energy = pred.square().sum(-1).mean(2)
    true_energy = truth.square().sum(-1).mean(2)
    floor = .03 ** 2
    pred_scale, true_scale = (pred_energy + floor).sqrt(), (true_energy + floor).sqrt()
    weight = (joints.new_ones(len(joints)) if signal_weight is None
              else signal_weight.detach().to(joints).reshape(len(joints)))

    def reduce(x):
        return (x.flatten(1).mean(1) * weight).mean()

    amplitude = reduce((pred_scale - true_scale).square())
    relation = joints.new_zeros(())
    lags = [lag for lag in (0, 4, 8) if lag < width]
    off_diagonal = ~torch.eye(len(ENDS), device=joints.device, dtype=torch.bool)
    for lag in lags:
        def correlation(x, scale):
            n = width - lag
            covariance = torch.einsum('bwtjc,bwtkc->bwjk', x[:, :, :n], x[:, :, lag:]) / n
            return covariance / (scale.unsqueeze(-1) * scale.unsqueeze(-2))
        error = (correlation(pred, pred_scale) - correlation(truth, true_scale)).square()
        relation = relation + reduce(error[..., off_diagonal]) / len(lags)
    return dict(coordination_relation_mse=relation, coordination_amplitude_m2=amplitude)
