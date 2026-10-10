"""Offline camera adapter around SceneMI's original AdaGN U-Net and ViT."""
import torch
from torch import nn

from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from model.mdm_scene_unet import MDM_Scene_UNET


class OfflineSceneMI(nn.Module):
    def __init__(self, latent_dim=256, dim_mults=(1, 2, 4), body_conditioning=False, contact_prediction=False):
        super().__init__()
        self.contact_prediction=contact_prediction
        if contact_prediction:self.contact_head=nn.Sequential(nn.Linear(402,128),nn.SiLU(),nn.Linear(128,22))
        self.body_conditioning = body_conditioning
        if body_conditioning:
            self.body_encoder = nn.Sequential(nn.Linear(69,128),nn.SiLU(),nn.Linear(128,latent_dim))
        self.core = MDM_Scene_UNET(
            modeltype='', njoints=201, nfeats=1, num_actions=1,
            translation=True, pose_rep='rot6d', glob=True, glob_rot=True,
            latent_dim=latent_dim, dim_mults=dim_mults,
            attention=False, adagn=True, zero=True, arch='unet',
            data_rep='smpl_glo', dataset='trumans',
            light_bps=True, sub_bps=0, beta=False,
            body_abstract='part_height', scene_type='occ_map24',
            scene_size=48, free_p=0., wo_frame_feature=False,
            wo_scene_feature=False, offline_camera_condition=False,
            offline_sparse_control=True)

    def forward(self, noisy_motion, timestep, batch, *, control_mask=None,
                use_scene=True, use_control=True):
        batch_size, frames, channels = noisy_motion.shape
        if channels != 201 or frames not in (64, 128, 192):
            raise ValueError('Expected 64/128/192-frame [B,T,201] motion')
        noisy = noisy_motion.permute(0, 2, 1)[:, :, None]
        # Masked body tracks are soft conditions; no clean body pose is hard
        # imputed into the diffusion state.
        observed = torch.zeros_like(noisy)
        observed_mask = torch.zeros_like(noisy, dtype=torch.bool)
        bps = batch['bps'].permute(0, 2, 3, 1).contiguous()
        if control_mask is None:
            control_mask = fixed_control_mask(batch_size, frames, 'head', noisy_motion.device)
        if control_mask.shape != (batch_size, frames, 22):
            raise ValueError(f'Expected control mask {(batch_size, frames, 22)}')
        if not use_control:
            control_mask = torch.zeros_like(control_mask)
        sparse_control = torch.cat(
            (batch['trajectory'] * control_mask[..., None], control_mask[..., None].float()), dim=-1)
        y = {'occ_map': batch['occupancy'], 'bps_sbj': bps,
             'sparse_control': sparse_control,
             # In evaluation, disable the training-time random global-scene
             # dropout. ``uncond`` explicitly masks the global scene branch.
             'sampling': not self.training,
             'uncond': not use_scene}
        if 'body_local_embedding' in batch:y['body_local_embedding']=batch['body_local_embedding']
        if self.body_conditioning:
            relative_rest=batch['rest']-batch['rest'][:,:1]
            body_features=torch.cat((relative_rest.flatten(1),batch['body_type'],batch['body_scale']),dim=-1)
            y['native_body_embedding']=self.body_encoder(body_features)
        bps_mask = torch.ones_like(bps) if use_scene else torch.zeros_like(bps)
        if use_scene and 'bps_valid' in batch:
            bps_mask = bps_mask * batch['bps_valid'].permute(0,2,1)[:,:,None,:].to(bps.dtype)
        prediction = self.core(noisy, timestep, y=y, obs_x0=observed,
                               obs_mask=observed_mask, bps_sbj_mask=bps_mask)
        return prediction[:, :, 0].permute(0, 2, 1)

    def contact_logits(self,motion,batch,*,use_scene=True):
        if not self.contact_prediction:raise ValueError('Contact prediction disabled')
        bps=batch['bps'].flatten(2)
        if 'bps_valid' in batch:bps=(batch['bps']*batch['bps_valid'][...,None]).flatten(2)
        return self.contact_head(torch.cat((motion,bps if use_scene else torch.zeros_like(bps)),dim=-1))

def cosine_alphas(steps=1000):
    import math
    t = torch.arange(steps + 1, dtype=torch.float64) / steps
    alpha = torch.cos((t + .008) / 1.008 * math.pi / 2).square()
    alpha = alpha / alpha[0]
    beta = (1 - alpha[1:] / alpha[:-1]).clamp(.0001, .999)
    return torch.cumprod(1 - beta, dim=0).float()


@torch.no_grad()
def ddim_sample(model, batch, frames, *, steps=50, seed=777,
                control_mask=None, use_scene=True, use_control=True):
    """Deterministic DDIM sampling for this experiment's x0 predictor."""
    if steps < 2 or steps > 1000:
        raise ValueError('DDIM steps must be between 2 and 1000')
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    sample = torch.randn((len(batch['camera']), frames, 201), device=device,
                         generator=generator)
    alphas = cosine_alphas().to(device)
    schedule = torch.linspace(999, 0, steps, device=device).round().long().unique_consecutive()
    was_training = model.training
    model.eval()
    for index, timestep_value in enumerate(schedule):
        timestep = torch.full((len(sample),), int(timestep_value), device=device, dtype=torch.long)
        predicted_x0 = model(sample, timestep, batch, control_mask=control_mask,
                             use_scene=use_scene, use_control=use_control)
        alpha = alphas[timestep_value]
        epsilon = (sample - alpha.sqrt() * predicted_x0) / (1 - alpha).sqrt().clamp_min(1e-6)
        if index + 1 == len(schedule):
            sample = predicted_x0
        else:
            next_alpha = alphas[schedule[index + 1]]
            sample = next_alpha.sqrt() * predicted_x0 + (1 - next_alpha).sqrt() * epsilon
    model.train(was_training)
    return sample
