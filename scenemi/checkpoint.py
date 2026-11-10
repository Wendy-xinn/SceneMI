"""Atomic complete checkpoints and strict inference/resume loading."""
from pathlib import Path
import torch
from scenemi.config import ROOT, load_paths
from scenemi.pipeline import MotionPipeline
from scenemi.training import rng_state, restore_rng
from scenemi.data.amass_prepare import sha256


def continuation_identity(config):
    """Verify the explicit parent; never relax same-run resume contracts."""
    parent = config['continuation']
    for key in ('checkpoint', 'validation_manifest', 'metrics'):
        path = ROOT / parent[key]
        if sha256(path) != parent[key + '_sha256']:
            raise ValueError(f'Continuation {key} identity differs')
    output = (ROOT / config['output']).resolve()
    checkpoint_path = (ROOT / parent['checkpoint']).resolve()
    if checkpoint_path.is_relative_to(output):
        raise ValueError('A continuation must use a new output directory')
    return parent


def initialize_continuation(model, optimizer, rng, device, config, *, diagnostic=False):
    """Restore full state into an explicit new protocol, with a pinned parent.

    CPU diagnostic checks may load a CUDA parent but never claim an exact
    cross-device resume. Formal continuation retains the parent's device.
    """
    parent = continuation_identity(config)
    state = torch.load(ROOT / parent['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    old = state['contract']['config']
    for key in ('data', 'selection', 'scene_paths', 'amass_index', 'reference_loss_config', 'seed'):
        if old[key] != config[key]:
            raise ValueError(f'Continuation changed parent {key}')
    for key in ('latent_dim', 'dim_mults', 'scene_modules_trainable'):
        if old['model'][key] != config['model'][key]:
            raise ValueError(f'Continuation changed model {key}')
    for key in ('seed', 'ddim_steps', 'eval_batch_size', 'test_used'):
        if old['validation'][key] != config['validation'][key]:
            raise ValueError(f'Continuation changed validation {key}')
    for key in ('weight_decay', 'clip_norm'):
        if old['optimizer'][key] != config['optimizer'][key]:
            raise ValueError(f'Continuation changed optimizer {key}')
    if state['step'] != parent['step']:
        raise ValueError('Continuation checkpoint step differs')
    if state['device'] != device and not diagnostic:
        raise ValueError('Formal continuation must retain the parent device')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'], rng, device)
    return state['step']

def checkpoint(model, optimizer, rng, device, step, seal, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        rng=rng_state(rng, device), contract=seal, device=device), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_trained_model(checkpoint_path, paths=None):
    """Load existing weights strictly, using their saved config and body bases.

    Inference does not require the optimizer's source contract to match today's
    trainer. Exact training resume remains protected by load_checkpoint().
    """
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    c = state['contract']['config']
    weights = state['model']
    bases = {kind: {key: weights[f'native_body.bank.{kind}_{key}']
                   for key in ('rest_template', 'rest_directions', 'eye_template', 'eye_directions')}
             for kind in ('smpl', 'smplx', 'grail_g1')}
    paths = load_paths(ROOT / 'configs/paths.local.yaml') if paths is None else paths
    mount = dict(rotation=weights['pv_rotation'].tolist(), translation_m=weights['pv_translation'].tolist(),
                 position_p90_m=float(weights['pv_uncertainty'][0]),
                 rotation_p90_rad=float(weights['pv_uncertainty'][1]))
    model = MotionPipeline(paths['smplh_models'] / 'neutral/model.npz', mount, bases=bases,
                           latent_dim=c['model']['latent_dim'], dim_mults=tuple(c['model']['dim_mults']))
    model.load_state_dict(weights, strict=True)
    return model.requires_grad_(False).eval(), c, state['step']


def load_checkpoint(model, optimizer, rng, device, seal, path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['contract'] != seal or state['device'] != device:
        raise ValueError('Checkpoint contract/device differs')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'], rng, device)
    return state['step']


