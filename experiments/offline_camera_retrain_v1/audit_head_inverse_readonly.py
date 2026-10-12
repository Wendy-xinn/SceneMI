"""Compare a frozen camera-to-head inverse with native labels, without training.

The labels are used after conversion for auditing only. No mount is fitted,
no network is instantiated, and no data cache or checkpoint is rewritten.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from experiments.offline_camera_retrain_v1.audit_unified_inputs import dataset
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=HERE / 'runs/ego_exo_retrain_design_oct12/READONLY_HEAD_AUDIT.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    effective_path = HERE / 'runs/ego_exo_camera_config_oct12/effective_config.json'
    mount_path = HERE / 'runs/ego_exo_surface_continue_oct11/TRAIN_camera_mounts.json'
    config = json.loads(effective_path.read_text())['inherited_dataset_configuration']
    mounts = json.loads(mount_path.read_text())
    report = dict(optimizer_steps=0, network_forward=False, mount_fit_performed=False,
                  fps=20, seed=2026101261,
                  window_policy='First valid 128-frame window of sequence index 0 in each group/split; small panel, not a population estimate',
                  head_inverse_uses='Camera pose and previously frozen TRAIN mount only',
                  GT_use='Comparison only; no validation mount fitting',
                  input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (effective_path, mount_path, Path(__file__))}, rows=[])
    for split in ('train', 'validation'):
        data = dataset(config, split, report['seed'])
        for group in ('trumans', 'camera_wearer', 'interactee', 'rich'):
            sample, identity = data.sample(128, group, sequence_index=0, start_index=0)
            camera = np.asarray(sample['camera'], np.float64)
            column0, column1 = camera[:, 3:6], camera[:, 6:9]
            rc = Rotation.from_matrix(np.stack(
                (column0, column1, np.cross(column0, column1)), axis=-1)).as_matrix()
            calibration = mounts[group]
            mount_r = np.asarray(calibration['rotation'])
            mount_t = np.asarray(calibration['translation_m'])
            rh = rc @ mount_r.T
            ph = camera[:, :3] * 2 - np.einsum('tij,j->ti', rh, mount_t)
            # GT is first consulted here, after the camera-only inverse.
            truth_r = global_rotations(torch.from_numpy(sample['motion'][None]))[0, :, 15].numpy()
            pos_error = np.linalg.norm(ph - sample['joints'][:, 15] * 2, axis=-1) * 100
            rot_error = np.rad2deg(Rotation.from_matrix(rh @ truth_r.transpose(0, 2, 1)).magnitude())
            pc_back = ph + np.einsum('tij,j->ti', rh, mount_t)
            rc_back = rh @ mount_r
            row = dict(split=split, group=group, sequence_id=identity['sequence_id'],
                       source_start_30fps=float(identity['source_start_30fps']),
                       position_mean_cm=float(pos_error.mean()),
                       position_p90_cm=float(np.quantile(pos_error, .9)),
                       rotation_mean_deg=float(rot_error.mean()),
                       rotation_p90_deg=float(np.quantile(rot_error, .9)),
                       roundtrip_position_max_m=float(np.abs(pc_back - camera[:, :3] * 2).max()),
                       roundtrip_rotation_max=float(np.abs(rc_back - rc).max()))
            if row['roundtrip_position_max_m'] > 1e-10 or row['roundtrip_rotation_max'] > 1e-10:
                raise ValueError(f'Camera/head round trip failed: {row}')
            report['rows'].append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
