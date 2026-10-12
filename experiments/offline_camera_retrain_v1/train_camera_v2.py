"""Explicit future training entry; never invoked by configuration preparation."""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.camera_v2_config import (
    CONFIG_PATH, read_config, effective_sampling_plan, sample_item,
    optimizer_groups, observation_losses)
from experiments.offline_camera_retrain_v1.camera_v2_model import CameraAwareSceneMI
from experiments.offline_camera_retrain_v1.camera_v2_observations import CameraObservationBuilder, inference_payload_v2
from experiments.offline_camera_retrain_v1.audit_unified_inputs import dataset
from experiments.offline_camera_retrain_v1.unified_geometry import SkinAnchorCache
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1.supervision import supervised_losses
from experiments.offline_camera_retrain_v1.contact_supervision import calibrated_contact_loss
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.prepare_camera_v2 import sha256, write_json

CODE_FILES = ('camera_v2_config.py', 'camera_v2_model.py', 'camera_v2_observations.py',
              'prepare_camera_v2.py', 'train_camera_v2.py', 'unified_model.py',
              'unified_losses.py', 'unified_observations.py', 'unified_geometry.py',
              'scene_model.py', 'supervision.py', 'native_body_data.py', 'data.py',
              'contact_supervision.py', 'training_contract_v2.py', 'checkpoint_io.py',
              'frozen_exo_inputs.py', 'orientation_supervision.py')


def code_hashes():
    return {name: sha256(HERE / name) for name in CODE_FILES} | {
        'model/mdm_scene_unet.py': sha256(HERE.parents[1] / 'model/mdm_scene_unet.py')}


def train(config_path, steps, output=None, resume=False):
    if steps <= 0:
        raise ValueError('Steps must be positive')
    config = read_config(config_path)
    prepared = HERE / config['prepared_output']
    checks = json.loads((prepared / 'input_checks.json').read_text())
    effective = json.loads((prepared / 'effective_config.json').read_text())
    hashes = code_hashes()
    if checks['status'] != 'passed' or effective['config'] != config:
        raise ValueError('Prepare and inspect this exact configuration first')
    if effective.get('code_sha256') != hashes:
        raise ValueError('Code changed since preparation; rerun the zero-step checks')
    if sha256(HERE / config['mount_defaults']) != effective['mount_defaults_sha256']:
        raise ValueError('Mount calibrations changed since preparation')
    folder = Path(output or prepared / 'training')
    if folder.resolve() in {prepared.resolve(), (HERE / config['initialization']).parent.resolve()}:
        raise ValueError('Training needs its own directory; retain prepared records and original weights')
    if (folder / 'last.pt').exists() and not resume:
        raise ValueError('Checkpoint exists; resume explicitly')
    if resume and not (folder / 'last.pt').exists():
        raise ValueError('No checkpoint to resume')
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(config['seed']); np.random.seed(config['seed'])
    inherited = effective['inherited_dataset_configuration']
    data = dataset(inherited, 'train', config['seed'])
    builder = CameraObservationBuilder(SkinAnchorCache(),
                json.loads((HERE / config['mount_defaults']).read_text()))
    plan = effective_sampling_plan(data, config)
    if plan != effective['sampling_plan']:
        raise ValueError('Dataset duration/sampling changed since preparation')
    model = CameraAwareSceneMI(inherited['latent_dim'], tuple(inherited['dim_mults']),
                              body_conditioning=True, contact_prediction=True).cuda()
    model.surface_bps_weight = config['scene']['surface_bps_weight']
    optimizer = torch.optim.AdamW(optimizer_groups(model, config),
                                  weight_decay=config['optimizer']['weight_decay'])
    rng = np.random.default_rng(config['seed'] + 991)
    weights = effective['observation_coefficients']
    start = 0; counts = {}
    if resume:
        state = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        if state['effective_config'] != effective or state['code_sha256'] != hashes or state['planned_steps'] != steps:
            raise ValueError('Resume requires the same effective configuration, code and schedule')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        for entry in optimizer.state.values():
            for key, value in entry.items():
                if torch.is_tensor(value):
                    entry[key] = value.cuda()
        data.load_sampler_state(state['data_rng_state']); rng.bit_generator.state = state['condition_rng_state']
        torch.set_rng_state(state['torch_rng_state']); torch.cuda.set_rng_state(state['cuda_rng_state'])
        np.random.set_state(state['numpy_rng_state'])
        start, counts = state['step'], state['counts']; del state
    else:
        source = HERE / config['initialization']
        if sha256(source) != effective['source_sha256']:
            raise ValueError('Source checkpoint changed since preparation')
        state = torch.load(source, map_location='cpu', weights_only=False)
        model.load_legacy_state_dict(state['model']); del state
    started = time.monotonic()
    protocol = dict(status='training', config=config, effective_config=effective, code_sha256=hashes,
                    planned_steps=steps, completed_steps=start, counts=counts,
                    free_disk_before=shutil.disk_usage(HERE).free,
                    checkpoint_policy='one atomic last.pt with optimizer and all sampler/RNG states')
    write_json(folder / 'protocol.json', protocol)

    def checkpoint(step):
        save_checkpoint(dict(step=step, planned_steps=steps, model=model.state_dict(),
                             optimizer=optimizer.state_dict(), effective_config=effective,
                             code_sha256=hashes, counts=counts, torch_rng_state=torch.get_rng_state(),
                             cuda_rng_state=torch.cuda.get_rng_state(), numpy_rng_state=np.random.get_state(),
                             data_rng_state=data.sampler_state(), condition_rng_state=rng.bit_generator.state),
                        folder / 'last.pt')

    alphas = cosine_alphas(config['noise']['steps']).cuda()
    with (folder / 'trace.jsonl').open('a' if resume else 'w') as trace:
        for step in range(start + 1, steps + 1):
            pairs = [sample_item(data, builder, config, plan, rng) for _ in range(config['batch_size'])]
            items, identities = zip(*pairs)
            batch = collate_observations(items); known = inference_payload_v2(batch)
            t = torch.randint(config['noise']['steps'], (len(items),), device='cuda')
            low = torch.rand(len(items), device='cuda') < config['noise']['extra_low_noise_probability']
            t = torch.where(low, torch.randint(config['noise']['low_noise_steps'], t.shape, device='cuda'), t)
            alpha = alphas[t]
            xt = alpha[:, None, None].sqrt() * batch['motion'] + (1 - alpha[:, None, None]).sqrt() * torch.randn_like(batch['motion'])
            scene_on = rng.random() >= config['scene']['dropout_probability']
            model.train(); optimizer.zero_grad(set_to_none=True)
            prediction = model(xt, t, known, control_mask=batch['observation_meta'][..., :2].amax(-1) > 0,
                               use_scene=scene_on)
            base = supervised_losses(prediction, batch['motion'], batch,
                                      profile=config['supervision']['base_profile'], signal_weight=alpha)
            terms = observation_losses(prediction, batch['motion'], batch, alpha)
            contact = calibrated_contact_loss(model.contact_logits(prediction, known, use_scene=scene_on), batch, alpha)
            total = base['total'] + sum(weights[k] * v for k, v in terms.items()) + config['supervision']['contact_coefficient'] * contact
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite loss')
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip_norm'], error_if_nonfinite=True)
            schedule = min(1., step / config['optimizer']['warmup_steps']) * max(
                config['optimizer']['floor_fraction'], .5 * (1 + math.cos(math.pi * step / steps)))
            for group in optimizer.param_groups:
                group['lr'] = config['optimizer']['existing_lr' if group['name'] == 'existing' else 'new_lr'] * schedule
            optimizer.step()
            for identity, item in zip(identities, items):
                branch = f"{identity['group']}/{item[3]}"
                counts[branch] = counts.get(branch, 0) + 1
            row = dict(step=step, identities=identities, modes=batch['mode'], timesteps=t.tolist(),
                       scene=bool(scene_on), loss=float(total.detach()), gradient_norm=float(norm),
                       observations={k: float(v.detach()) for k, v in terms.items()},
                       learning_rates={group['name']: group['lr'] for group in optimizer.param_groups})
            trace.write(json.dumps(row) + '\n'); trace.flush()
            if step == 1 or step % 25 == 0:
                print(step, row['modes'], row['loss'], flush=True)
            if step % 100 == 0 or step == steps:
                checkpoint(step)
                protocol.update(status='training_completed' if step == steps else 'training',
                                completed_steps=step, counts=counts,
                                elapsed_s=time.monotonic() - started, free_disk_after=shutil.disk_usage(HERE).free)
                write_json(folder / 'protocol.json', protocol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); train(args.config, args.steps, args.output, args.resume)
