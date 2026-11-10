"""Temporal proposals using actual anchors around each separate tracking outage."""
import torch
from torch import nn
from torch.nn import functional as F
from pytorch3d.transforms import matrix_to_axis_angle

from scenemi.exo_basis import TemporalBasisExoPrior
from scenemi.joint_model import JointMotionPrior
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d


def missing_intervals(available):
    missing = (available <= .5).to(torch.int8)
    edges = F.pad(missing, (1, 1)).diff()
    return list(zip(torch.where(edges == 1)[0].tolist(), torch.where(edges == -1)[0].tolist()))


@torch.no_grad()
def continuation_proposal(initial, available):
    """Interpolate every outage separately; estimated hidden frames are not anchors."""
    result = initial.clone(); frames = initial.shape[1]
    for b in range(len(initial)):
        known = torch.where(available[b] > .5)[0]
        if len(known) < 2:
            raise ValueError('Completion requires at least two actual estimated anchors')
        x = initial[b, :, :135]
        for start, end in missing_intervals(available[b]):
            left, right = start-1, end
            if left >= 0:
                candidates = known[known <= left]
                before = int(candidates[(candidates-(left-6)).abs().argmin()])
                v0 = (x[left]-x[before])/max(left-before, 1)
            if right < frames:
                candidates = known[known >= right]
                after = int(candidates[(candidates-(right+6)).abs().argmin()])
                v1 = (x[after]-x[right])/max(after-right, 1)
            query = torch.arange(start, end, device=x.device, dtype=x.dtype)
            if left < 0:
                result[b, start:end, :135] = x[right]+(query-right)[:, None]*v1
            elif right >= frames:
                result[b, start:end, :135] = x[left]+(query-left)[:, None]*v0
            else:
                span = right-left; u = (query-left)/span
                result[b, start:end, :135] = ((2*u**3-3*u**2+1)[:, None]*x[left]
                    +(u**3-2*u**2+u)[:, None]*span*v0+(-2*u**3+3*u**2)[:, None]*x[right]
                    +(u**3-u**2)[:, None]*span*v1)
    return result


class MultiGapTemporalBasis(TemporalBasisExoPrior):
    @torch.no_grad()
    def proposals(self, payload):
        initial, rest = payload['initial_motion'], payload['initial_rest']
        cache = self._proposal_cache
        if cache is not None and cache[0] is initial and cache[1] is payload['estimate_available']:
            return cache[2], cache[3]
        values = initial[:, :, :135].transpose(1, 2); motions = [initial]
        for width in self.widths:
            axis = torch.arange(width, device=initial.device, dtype=initial.dtype)
            kernel = width//2+1-(axis-width//2).abs(); kernel /= kernel.sum()
            smooth = F.conv1d(F.pad(values, (width//2, width//2), mode='replicate'),
                kernel.reshape(1, 1, width).expand(135, 1, width), groups=135).transpose(1, 2)
            proposal = initial.clone(); proposal[:, :, :135] = smooth
            motions.append(reconcile_motion(proposal, rest))
        motions.append(reconcile_motion(continuation_proposal(initial, payload['estimate_available']), rest))
        rotations = rotation_from_6d(initial[:, :, 3:135].reshape(len(initial), 128, 22, 6))
        roots = torch.stack([(motion[:, :, :3]-initial[:, :, :3])*2 for motion in motions], 2)
        angles = torch.stack([matrix_to_axis_angle(rotation_from_6d(
            motion[:, :, 3:135].reshape(len(initial), 128, 22, 6)) @ rotations.transpose(-1, -2))
            for motion in motions], 3)
        roots[:, :, 0] = 0.; angles[:, :, :, 0] = 0.
        self._proposal_cache = (initial, payload['estimate_available'], roots, angles)
        return roots, angles


class NativeJointMotionPrior(JointMotionPrior):
    def __init__(self, base, **settings):
        super().__init__(base, **settings)
        previous = self.geometry
        geometry = MultiGapTemporalBasis(nn.Identity(), hidden=settings.get('hidden', 128),
            root_limit_m=.25, rotation_limit_rad=.6).requires_grad_(False)
        geometry.load_state_dict(previous.state_dict(), strict=True)
        self.geometry = geometry.eval()
