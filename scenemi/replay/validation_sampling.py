"""Pack original logical evaluation batches without changing their DDIM noise."""
import torch

from scenemi.exo_denoise import channel_reliability
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas


def reference_noise(start, count, total, logical_batch, *, seed=777,
                    device='cuda', dtype=torch.float32):
    if not 0 <= start < start+count <= total or logical_batch < 1:
        raise ValueError('Invalid logical validation slice')
    parts, position = [], start
    while position < start+count:
        group = position//logical_batch*logical_batch
        size = min(logical_batch, total-group)
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn((size, 128, 201), generator=generator, device=device, dtype=dtype)
        offset = position-group
        length = min(size-offset, start+count-position)
        parts.append(noise[offset:offset+length])
        position += length
    return torch.cat(parts)


@torch.no_grad()
def sample_ego(model, payload, noise, *, steps=20):
    was_training = model.training
    model.eval()
    try:
        state = noise.clone()
        alphas = cosine_alphas().to(state.device)
        schedule = torch.linspace(999, 0, steps, device=state.device).round().long().unique_consecutive()
        for index, value in enumerate(schedule):
            times = torch.full((len(state),), int(value), device=state.device, dtype=torch.long)
            prediction = model(state, times, payload, control_mask=None,
                               use_scene=False, use_control=True)
            epsilon = (state-alphas[value].sqrt()*prediction)/(1-alphas[value]).sqrt().clamp_min(1e-6)
            if index+1 == len(schedule):
                state = prediction
            else:
                following = alphas[schedule[index+1]]
                state = following.sqrt()*prediction+(1-following).sqrt()*epsilon
        return state
    finally:
        model.train(was_training)


@torch.no_grad()
def sample_exo(model, payload, noise, *, steps=8, start_timestep=200):
    initial = payload['initial_motion']
    if ((payload['initial_root_reliability'] >= 1).all() and
            (payload['initial_reliability'] >= 1).all()):
        return initial
    anchor = model.anchor(payload)
    schedule = torch.linspace(start_timestep, 0, steps, device=initial.device).round().long()
    alpha = model.alphas[start_timestep]
    state = alpha.sqrt()*anchor+(1-alpha).sqrt()*noise
    confidence = channel_reliability(payload)
    was_training = model.training
    model.eval()
    try:
        for index, value in enumerate(schedule):
            prediction = model(state, value.expand(len(initial)), payload)
            if index == len(schedule)-1:
                return prediction
            following = model.alphas[schedule[index+1]]
            epsilon = (state-model.alphas[value].sqrt()*prediction)/(1-model.alphas[value]).sqrt().clamp_min(1e-6)
            state = following.sqrt()*prediction+(1-following).sqrt()*epsilon
            observed = following.sqrt()*anchor+(1-following).sqrt()*noise
            state = torch.lerp(state, observed, confidence)
    finally:
        model.train(was_training)
