"""Motion metrics in meters, with explicitly named alignment conventions.

Body arrays include pelvis and follow native SMPL-H's first 22 joints.
The compatibility metric preserves a released EgoAllo/EgoGVAE implementation
quirk; it must never be presented as standard PA-MPJPE.
"""
import torch


def similarity_align(source, target):
    """Least-squares proper similarity transform, independently per frame."""
    if source.shape != target.shape or source.shape[-2:] != (22, 3):
        raise ValueError('Expected matching [...,22,3] joint positions')
    source_mean, target_mean = source.mean(-2, keepdim=True), target.mean(-2, keepdim=True)
    x, y = source-source_mean, target-target_mean
    u, singular, vh = torch.linalg.svd(y.transpose(-1, -2) @ x / 22)
    sign = torch.ones_like(singular)
    sign[..., -1] = torch.where(torch.linalg.det(u @ vh) < 0, -1., 1.)
    rotation = (u*sign.unsqueeze(-2)) @ vh
    variance = x.square().sum((-1, -2))/22
    scale = (singular*sign).sum(-1)/variance.clamp_min(1e-12)
    return scale[..., None, None]*(x @ rotation.transpose(-1, -2)) + target_mean


def motion_metrics(prediction, target, *, fps=30):
    """One [T,22,3] window; no root, floor or head alignment before MPJPE."""
    if prediction.ndim != 3 or prediction.shape != target.shape or prediction.shape[1:] != (22, 3):
        raise ValueError('Expected matching [T,22,3] windows')
    if len(target) < 4 or not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('Finite motion windows with at least four frames required')
    # Double precision makes the SVD parity check meaningful.
    pred, gt = prediction.double(), target.double()
    error = (pred-gt).norm(dim=-1)
    aligned = similarity_align(pred, gt)
    # Released compute_mpjpe aligns label -> prediction, then compares that
    # aligned label with the unaligned label (rather than with prediction).
    released_aligned_label = similarity_align(gt, pred)
    feet = [7, 8, 10, 11]
    thresholds = pred.new_tensor([.08, .08, .04, .04])
    foot_height = pred[:, feet, 1]
    values = dict(
        mpjpe_mm=error.mean()*1000,
        pa_mpjpe_mm=(aligned-gt).norm(dim=-1).mean()*1000,
        released_pa_compat_mm=(released_aligned_label-gt).norm(dim=-1).mean()*1000,
        root_error_mm=error[:, 0].mean()*1000,
        head_error_mm=error[:, 15].mean()*1000,
        root_relative_mpjpe_mm=((pred-pred[:, :1])-(gt-gt[:, :1])).norm(dim=-1).mean()*1000,
        ground_lowest22_mm=(pred[..., 1].amin(-1)-gt[..., 1].amin(-1)).abs().mean()*1000,
        egoallo_gnd_any_contact=(foot_height < thresholds).any().double(),
        mpjve_mm_s=(torch.diff(pred, dim=0)-torch.diff(gt, dim=0)).norm(dim=-1).mean()*fps*1000,
        jerk_m_s3=torch.diff(pred, n=3, dim=0).norm(dim=-1).mean()*fps**3,
        gt_jerk_m_s3=torch.diff(gt, n=3, dim=0).norm(dim=-1).mean()*fps**3,
    )
    return {key: float(value) for key, value in values.items()}
