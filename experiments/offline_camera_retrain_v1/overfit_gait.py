"""Training-only capacity check: not a generalization result or production model."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import OfflineSceneMIData, collate
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI, cosine_alphas, ddim_sample
from experiments.offline_camera_retrain_v1.supervision import supervised_losses
from experiments.offline_camera_retrain_v1.compare_feet import metrics
from experiments.offline_camera_retrain_v1.gallery import write_gallery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--skeleton-profile', choices=('archived', 'trumans_male_v2'), default='archived')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(19283)
    dataset = OfflineSceneMIData('train', seed=19283, skeleton_profile=args.skeleton_profile)
    examples, identities = [], []
    for attempt in range(100):
        sample, identity = dataset.sample(128, 'trumans')
        root = sample['joints'][:, 0] * 2
        if np.linalg.norm(root[-1, [0, 2]] - root[0, [0, 2]]) > 2:
            examples.append(sample)
            identities.append(identity)
        if len(examples) == 2:
            break
    if len(examples) != 2:
        raise RuntimeError('Did not find two walking training clips')
    batch = {k: v.cuda() for k, v in collate(examples).items()}
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = OfflineSceneMI(config['latent_dim'], tuple(config['dim_mults'])).cuda()
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    alphas = cosine_alphas().cuda()
    mask = fixed_control_mask(2, 128, 'head', 'cuda')
    cases = [dict(group='train_only_trumans', identity=identity,
                  camera=(batch['camera'][i, :, :3] * 2).cpu().tolist(),
                  tracks={'GT': (batch['joints'][i] * 2).cpu().numpy().round(4).tolist()}, metrics={})
             for i, identity in enumerate(identities)]
    rows = []

    @torch.no_grad()
    def evaluate(step):
        generated = ddim_sample(model, batch, 128, steps=20, seed=3737, control_mask=mask)
        for i, case in enumerate(cases):
            values, joints = metrics(generated[i:i+1], {k: v[i:i+1] for k, v in batch.items()})
            case['tracks'][f'overfit_{step}'] = joints[0].cpu().numpy().round(4).tolist()
            case['metrics'][f'overfit_{step}'] = values
            rows.append(dict(step=step, index=i, metrics=values))
            print(json.dumps(rows[-1]), flush=True)

    evaluate(0)
    model.train()
    for step in range(1, args.steps + 1):
        t = torch.randint(1000, (2,), device='cuda')
        alpha = alphas[t][:, None, None]
        noisy = alpha.sqrt() * batch['motion'] + (1-alpha).sqrt() * torch.randn_like(batch['motion'])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            predicted = model(noisy, t, batch, control_mask=mask)
        loss = supervised_losses(predicted.float(), batch['motion'], batch,
                                 profile='gait_v1', signal_weight=alpha.flatten())['total']
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite overfit loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            evaluate(step)
    (args.output / 'metrics.json').write_text(json.dumps(dict(
        note='Only two TRAINING clips, intentional overfit. Not evidence of held-out quality.',
        skeleton_profile=args.skeleton_profile, identities=identities, rows=rows), indent=2) + '\n')
    write_gallery(args.output / 'gallery.html', cases, 'TRAINING ONLY capacity check')


if __name__ == '__main__':
    main()
