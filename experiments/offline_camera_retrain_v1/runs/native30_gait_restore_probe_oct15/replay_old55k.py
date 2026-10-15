"""Replay historical55k development cases without training or touching sources."""
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE, GROUPS, collate
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI, ddim_sample
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.supervision import supervised_losses
from experiments.offline_camera_retrain_v1.evaluate_sampling import predicted_motion_metrics

ROOT = Path(__file__).resolve().parent


def main():
    began = time.monotonic()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.24)
    old = HERE / 'runs/native_dynamic_scene20_contact_55k_oct07'
    checkpoint = torch.load(old / 'last.pt', map_location='cpu', weights_only=False, mmap=True)
    config = checkpoint['config']
    model = OfflineSceneMI(config['latent_dim'], tuple(config['dim_mults']),
        body_conditioning=config['body_conditioning'], contact_prediction=config['contact_prediction']).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    historical = json.loads((old / 'history_reset_verification.json').read_text())
    data = NativeBodyData('validation', seed=777, skeleton_profile=config['skeleton_profile'],
        rich_source=config['rich_source'], trumans_scene_manifest=config['trumans_scene_manifest'],
        trumans_window_protocol=config['trumans_window_protocol'], contact_root=config['rich_contact_root'],
        temporal_scene_manifest=config['temporal_scene_manifest'])
    rows, outputs = [], {}
    with torch.no_grad():
        for gi, group in enumerate(GROUPS):
            for si in range(4):
                sample, identity = data.sample(128, group)
                reference = historical['groups'][group]['cases'][si]
                for key in ('sequence_id', 'source_start_30fps'):
                    assert identity[key] == reference['identity'][key], (group, si, key)
                if si:
                    continue  # Preserve old data RNG draws, infer first case per group only.
                batch = {k: v.cuda() for k, v in collate([sample]).items()}
                pred = ddim_sample(model, batch, 128, steps=20, seed=777 + gi * 1000,
                    control_mask=fixed_control_mask(1, 128, 'head', 'cuda'), use_scene=True)
                losses = supervised_losses(pred, batch['motion'], batch)
                proxy = predicted_motion_metrics(pred, batch)
                now = dict(fk_mpjpe_cm=float(losses['fk_mpjpe_m'] * 100),
                    truth_stance_foot_slide_cm_per_frame=float(losses['foot_sliding_m_per_frame'] * 100),
                    foot_relative_speed_ratio=proxy['foot_relative_speed_ratio'])
                then = reference['variants']['head_full_scene']
                delta = {k: abs(v - then[k]) for k, v in now.items()}
                row = dict(group=group, sequence_id=identity['sequence_id'], source_start_30fps=identity['source_start_30fps'],
                    historical={k: then[k] for k in now}, reproduced=now, absolute_difference=delta)
                rows.append(row)
                outputs[group + '_prediction'] = pred[0].cpu().numpy()
                outputs[group + '_truth'] = sample['motion']
                print(json.dumps(row), flush=True)
    np.savez_compressed(ROOT / 'old55k_replay_outputs.npz', **outputs)
    report = dict(created_at=datetime.datetime.now().astimezone().isoformat(),
        status='passed' if max(v for r in rows for v in r['absolute_difference'].values()) < 1e-3 else 'historical metric mismatch; investigate before reproduction claim',
        checkpoint_step=checkpoint['step'], strict_model_load=True, optimizer_steps=0,
        source_checkpoint=str((old / 'last.pt').resolve()), source_checkpoint_bytes=(old / 'last.pt').stat().st_size,
        config_sha256=hashlib.sha256((old / 'config.json').read_bytes()).hexdigest(),
        launch_command=json.loads((old / 'launch_command.json').read_text()), cases=rows,
        elapsed_s=time.monotonic() - began,
        limits=['Four historical internal-VAL cases, not official benchmark comparison.',
            'Original metadata/static-reference archive was reconstructed; no bitwise55000-step retraining claim.',
            'Historical TRAIN includes12 official EgoBodyTEST recordings; development reference only.',
            'Historical20Hz128frames and present30Hz192frames have different input, split and aggregation protocols.'])
    (ROOT / 'OLD55K_REPRODUCTION_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(report['status'], flush=True)


if __name__ == '__main__':
    main()
