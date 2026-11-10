"""EGO scene-stage components used by the shared train/evaluate interfaces."""
import json
import math
import numpy as np
import torch

from scenemi.config import ROOT
from scenemi.pipeline import MotionPipeline, load_camera_mount
from scenemi.contact_loss import reliable_native_motion_loss
from scenemi.checkpoint import continuation_identity
from scenemi.data.scene_ego import collate, payload
from scenemi.training import restore_rng, tensor_digest

VERSION = 'ego-scene-adaptation-v1'
SCENE_MODULES = ('patch_projection', 'patch_position', 'motion_query', 'region_embedding',
                 'patch_attention', 'patch_condition', 'owner_embedding', 'dynamic_embedding',
                 'dynamic_attention', 'dynamic_condition', 'core.scene_embedding')


def scene_parameter(name):
    return any(name == prefix or name.startswith(prefix + '.') for prefix in SCENE_MODULES)


class EgoScenePrior(MotionPipeline):
    """Keep the frozen prior in eval mode while training only scene modules."""
    def train(self, mode=True):
        super().train(False)
        self.training = mode
        for name in SCENE_MODULES:
            self.get_submodule(name).train(mode)
        return self


def build_model(c, paths):
    mount, _ = load_camera_mount(paths, ROOT / c['scene_paths'])
    model = EgoScenePrior(paths['smplh_models'] / 'neutral/model.npz', mount,
                         latent_dim=c['model']['latent_dim'], dim_mults=tuple(c['model']['dim_mults']))
    model.requires_grad_(False)
    for name in SCENE_MODULES:
        model.get_submodule(name).requires_grad_(True)
    model.patch_enabled = True
    model.local_enabled = model.feedback_enabled = False
    return model


def optimizer_for(model, c):
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not scene_parameter(name) for name, _ in trainable):
        raise ValueError('Scene stage must freeze the complete motion/shape/camera prior')
    return torch.optim.AdamW([p for _, p in trainable], lr=c['optimizer']['lr'],
                             weight_decay=c['optimizer']['weight_decay'])


def initialize(model, optimizer, rng, device, c, *, diagnostic=False):
    parent = continuation_identity(c)
    state = torch.load(ROOT / parent['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    old = state['contract']['config']
    for key in ('selection', 'scene_paths', 'amass_index', 'reference_loss_config', 'seed'):
        if old[key] != c[key]:
            raise ValueError(f'Scene initialization changed parent {key}')
    for key in ('datasets', 'excluded_datasets', 'frames', 'fps'):
        if old['data'][key] != c['data'][key]:
            raise ValueError(f'Scene initialization changed data {key}')
    for key in ('latent_dim', 'dim_mults'):
        if old['model'][key] != c['model'][key]:
            raise ValueError('Scene initialization changed the backbone architecture')
    for key in ('seed', 'ddim_steps', 'eval_batch_size', 'test_used'):
        if old['validation'][key] != c['validation'][key]:
            raise ValueError('Scene initialization changed validation conditions')
    if state['step'] != parent['step'] or (state['device'] != device and not diagnostic):
        raise ValueError('Scene initializer step/device differs')
    model.load_state_dict(state['model'], strict=True)
    for name in ('patch_condition.weight', 'dynamic_condition.weight',
                 'core.scene_embedding.mlp_head.weight', 'core.scene_embedding.mlp_head.bias'):
        if torch.count_nonzero(model.state_dict()[name]):
            raise ValueError('Expected the scene-free parent with zero-ended scene paths')
    if optimizer.state:
        raise ValueError('The new scene optimizer must start fresh')
    restore_rng(state['rng'], rng, device)
    return state['step']


def frozen_digest(model):
    return tensor_digest({name: p for name, p in model.named_parameters() if not p.requires_grad})


def update(model, optimizer, data, rng, c, device, step):
    pairs = [data.sample(source, rng) for source in c['data']['datasets']
             for _ in range(c['data']['source_quota_per_update'])]
    model.train(); optimizer.zero_grad(set_to_none=True)
    total = 0.; terms_sum = {}; micro = c['training']['microbatch_size']
    import yaml
    loss_config = yaml.safe_load((ROOT / c['reference_loss_config']).read_text())
    for first in range(0, len(pairs), micro):
        part = pairs[first:first + micro]
        batch = collate([p[0] for p in part], device)
        t = torch.randint(0, 1000, (len(part),), device=device)
        alpha = model.alphas[t]
        noisy = alpha[:, None, None].sqrt() * batch['motion'] + (1 - alpha[:, None, None]).sqrt() * torch.randn_like(batch['motion'])
        has_scene = any(identity['dataset'] != 'amass' for _, identity in part)
        prediction, details = model(noisy, t, payload(batch), use_scene=has_scene, return_details=True)
        terms = reliable_native_motion_loss(prediction, batch['motion'], batch, alpha.sqrt(),
            details=details, config=loss_config, contact_weight=batch['contact_weight'])
        if not all(torch.isfinite(value).all() for value in terms.values()):
            raise FloatingPointError('Nonfinite scene objective; no update applied')
        fraction = len(part) / len(pairs)
        if terms['total'].requires_grad:
            (terms['total'] * fraction).backward()
        total += float(terms['total'].detach()) * fraction
        for key, value in terms.items():
            terms_sum[key] = terms_sum.get(key, 0.) + float(value.detach()) * fraction
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                         c['optimizer']['clip_norm'], error_if_nonfinite=True)
    local = step - c['continuation']['step']
    budget = c['training']['planned_updates'] - c['continuation']['step']
    factor = min(1., local / c['optimizer']['warmup_updates'])
    floor = c['optimizer']['lr_floor_fraction']
    factor *= floor + (1 - floor) * .5 * (1 + math.cos(math.pi * local / budget))
    for group in optimizer.param_groups:
        group['lr'] = c['optimizer']['lr'] * factor
    gradients = {name: float(p.grad.norm()) for name, p in model.named_parameters()
                 if name in ('patch_condition.weight', 'dynamic_condition.weight',
                             'core.scene_embedding.mlp_head.weight') and p.grad is not None}
    optimizer.step()
    grail = [p[0]['contact_weight'] for p in pairs if p[1]['dataset'] == 'grail']
    return dict(step=step, total=total, terms=terms_sum, gradient_norm=float(norm),
                lr=optimizer.param_groups[0]['lr'], scene_gradient_groups=gradients,
                grail_contact_weight_mean=float(np.mean(grail)),
                grail_contact_weight_zero_fraction=float((np.stack(grail) == 0).mean()),
                identities=[identity for _, identity in pairs])


def selection_record(result, c, step, checkpoint_path, metrics_path):
    baseline = json.loads((ROOT / c['continuation']['metrics']).read_text())
    checks = {name: values['mpjpe_mm'] <= baseline['by_dataset'][name]['mpjpe_mm'] +
              (1e-4 if name == 'amass' else c['validation']['clean_dataset_regression_mm'])
              for name, values in result['by_dataset'].items()}
    checks['macro'] = result['selection_score_mm'] <= baseline['selection_score_mm'] + c['validation']['clean_macro_regression_mm']
    fallback = result.get('no_scene_fallback')
    checks['frozen_fallback'] = fallback is not None and all(
        abs(fallback['by_dataset'][name]['mpjpe_mm'] - baseline['by_dataset'][name]['mpjpe_mm']) < 1e-4
        for name in baseline['by_dataset'])
    return dict(step=step, selection_score_mm=result['selection_score_mm'],
                checkpoint=str(checkpoint_path), metrics=str(metrics_path), clean_guards=checks,
                objective='scene MPJPE with per-dataset guards and unchanged no-scene fallback'), all(checks.values())


@torch.no_grad()
def preflight_checks(model, data, c, device):
    rng = np.random.default_rng(c['seed'])
    pairs = [data.sample(source, rng) for source in c['data']['datasets']]
    batch = collate([p[0] for p in pairs], device)
    public = payload(batch)
    model.eval(); noisy = torch.zeros_like(batch['motion'])
    times = torch.full((len(pairs),), 500, device=device, dtype=torch.long)
    fallback = model(noisy, times, public, use_scene=False)
    scene = model(noisy, times, public, use_scene=True)
    if not torch.equal(scene, fallback):
        raise AssertionError('Zero-ended scene branches changed the 14k initializer')
    poisoned = dict(batch, motion=torch.full_like(batch['motion'], float('nan')),
                    contact_weight=torch.full_like(batch['contact_weight'], float('nan')),
                    target_rest=torch.full_like(batch['target_rest'], float('nan')))
    if not torch.equal(scene, model(noisy, times, payload(poisoned), use_scene=True)):
        raise AssertionError('Private contact/motion labels entered scene inference')
    for sample, identity in pairs:
        if identity['dataset'] != 'amass':
            from scenemi.native_joint_data import SceneFreeEgoDataset
            native_plain, _ = SceneFreeEgoDataset.sample(data.native, rng,
                member=identity['member'], start=identity['start'], yaw=0.)
            for key in ('motion', 'camera', 'target_rest'):
                if not np.array_equal(sample[key], native_plain[key]):
                    raise AssertionError(f'Scene loader changed {key}')
    return dict(zero_scene_exact=True, private_labels_isolated=True, motion_camera_targets_unchanged=True,
                frozen_digest=frozen_digest(model), no_scene_probe=fallback.cpu(),
                probe_payload={k: v.cpu() for k, v in public.items()})
