"""Overfit a fixed mixed batch before committing to a long SceneMI run."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import OfflineSceneMIData, collate
from experiments.offline_camera_retrain_v1.control import HEAD, PELVIS, WRISTS
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI, cosine_alphas, ddim_sample
from experiments.offline_camera_retrain_v1.supervision import supervised_losses


def to_cuda(batch):
    return {key: value.cuda(non_blocking=True) for key, value in batch.items()}


def scalar_losses(losses):
    return {key: float(value.detach()) for key, value in losses.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--length', type=int, default=64, choices=(64, 128, 192))
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ddim-steps', type=int, default=20)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / 'overfit_diagnostic.json')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    torch.manual_seed(2026)
    np.random.seed(2026)
    dataset = OfflineSceneMIData('train', seed=2026)
    samples = [dataset.sample(args.length, group)[0]
               for group in ('trumans', 'camera_wearer', 'interactee')]
    batch = to_cuda(collate(samples))
    clean = batch['motion']
    control_mask = torch.zeros((len(clean), args.length, 22), dtype=torch.bool, device='cuda')
    control_mask[0, :, HEAD] = True
    control_mask[1, :, PELVIS] = True
    control_mask[2, ::5, list(WRISTS)] = True
    model = OfflineSceneMI().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    alphas = cosine_alphas().cuda()
    probe_timestep = torch.full((len(clean),), 750, device='cuda', dtype=torch.long)
    probe_generator = torch.Generator(device='cuda').manual_seed(777)
    probe_noise = torch.randn(clean.shape, device='cuda', generator=probe_generator)
    probe_alpha = alphas[probe_timestep][:, None, None]
    probe_noisy = probe_alpha.sqrt() * clean + (1 - probe_alpha).sqrt() * probe_noise

    @torch.no_grad()
    def probe():
        model.eval()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(probe_noisy, probe_timestep, batch, control_mask=control_mask)
            no_scene = model(probe_noisy, probe_timestep, batch, control_mask=control_mask,
                             use_scene=False)
            no_control = model(probe_noisy, probe_timestep, batch, control_mask=control_mask,
                               use_control=False)
        losses = scalar_losses(supervised_losses(prediction.float(), clean, batch))
        losses['scene_effect_l1'] = float((prediction - no_scene).abs().mean())
        losses['control_effect_l1'] = float((prediction - no_control).abs().mean())
        return losses

    report = {'configuration': vars(args), 'initial_probe': probe(), 'progress': []}
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        timestep = torch.randint(len(alphas), (len(clean),), device='cuda')
        alpha = alphas[timestep][:, None, None]
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * torch.randn_like(clean)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(noisy, timestep, batch, control_mask=control_mask)
        losses = supervised_losses(prediction.float(), clean, batch)
        losses['total'].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f'Nonfinite gradient at step {step}')
        optimizer.step()
        if step == 1 or step % 20 == 0 or step == args.steps:
            row = {'step': step, 'elapsed_s': time.time() - started,
                   'training_total': float(losses['total'].detach()),
                   'probe': probe()}
            report['progress'].append(row)
            print(json.dumps(row), flush=True)
            model.train()

    generated = ddim_sample(model, batch, args.length, steps=args.ddim_steps,
                            control_mask=control_mask)
    report['ddim'] = scalar_losses(supervised_losses(generated.float(), clean, batch))
    report['elapsed_s'] = time.time() - started
    args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({'ddim': report['ddim'], 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
