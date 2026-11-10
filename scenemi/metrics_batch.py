"""The retained motion metrics, vectorized over independent windows.

Alignment remains per frame in float64. Reductions and contact semantics remain
per window; a single host transfer replaces repeated scalar CUDA synchronizations.
"""
import torch
from scenemi.metrics import similarity_align


def metric_tensors(prediction, target, *, fps=30):
    if prediction.ndim!=4 or prediction.shape!=target.shape or prediction.shape[-2:]!=(22,3) or prediction.shape[1]<4:
        raise ValueError('Expected matching [B,T>=4,22,3] windows')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('Finite motion windows required')
    pred,gt=prediction.double(),target.double();error=(pred-gt).norm(dim=-1)
    aligned=similarity_align(pred,gt);released=similarity_align(gt,pred)
    mean=lambda x:x.flatten(1).mean(1)
    feet=[7,8,10,11];thresholds=pred.new_tensor([.08,.08,.04,.04])
    return dict(mpjpe_mm=mean(error)*1000,
        pa_mpjpe_mm=mean((aligned-gt).norm(dim=-1))*1000,
        released_pa_compat_mm=mean((released-gt).norm(dim=-1))*1000,
        root_error_mm=error[:,:,0].mean(1)*1000,head_error_mm=error[:,:,15].mean(1)*1000,
        root_relative_mpjpe_mm=mean(((pred-pred[:,:,:1])-(gt-gt[:,:,:1])).norm(dim=-1))*1000,
        ground_lowest22_mm=(pred[...,1].amin(-1)-gt[...,1].amin(-1)).abs().mean(1)*1000,
        egoallo_gnd_any_contact=(pred[:,:,feet,1]<thresholds).flatten(1).any(1).double(),
        mpjve_mm_s=mean((torch.diff(pred,dim=1)-torch.diff(gt,dim=1)).norm(dim=-1))*fps*1000,
        jerk_m_s3=mean(torch.diff(pred,n=3,dim=1).norm(dim=-1))*fps**3,
        gt_jerk_m_s3=mean(torch.diff(gt,n=3,dim=1).norm(dim=-1))*fps**3)


def metric_rows(prediction,target,*,extra=None):
    values=metric_tensors(prediction,target)
    if extra:values.update(extra)
    names=list(values);matrix=torch.stack([values[n].double() for n in names],1).cpu().tolist()
    return [dict(zip(names,row)) for row in matrix]


def repair_rows(points,truth,initial,visible,condition):
    error=(points-truth).norm(dim=-1)*1000
    region=~visible if condition in ('occlusion','gap') else torch.ones_like(visible)
    def masked(mask):
        count=mask.flatten(1).sum(1)
        if (count==0).any():raise ValueError('A repair metric has an empty evaluation region')
        return (error*mask).flatten(1).sum(1)/count
    return metric_rows(points,truth,extra=dict(repair_region_mpjpe_mm=masked(region),
        tracked_visible_mpjpe_mm=masked(visible),
        max_change_from_initial_mm=(points-initial).norm(dim=-1).flatten(1).amax(1)*1000))
