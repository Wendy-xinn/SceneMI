"""Verify masked-scene independence and exact replay of the factual full sweep."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import GROUPS, OfflineSceneMIData, collate
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI, ddim_sample
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--factual', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads((args.factual / 'manifest.json').read_text())
    checkpoint = torch.load(manifest['checkpoint'], map_location='cpu', weights_only=False)
    config = checkpoint['config']
    dataset = OfflineSceneMIData('validation', skeleton_profile=config['skeleton_profile'],
                                 rich_source=config.get('rich_source', 'legacy5interp'))
    model = OfflineSceneMI(config['latent_dim'], tuple(config['dim_mults'])).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    results = []
    for group in GROUPS:
        window = next(w for w in manifest['windows'] if w['group'] == group)
        sample, identity = dataset.sample(manifest['length'], group,
                                         sequence_index=window['sequence_index'], start_index=window['start_index'])
        batch = {k: v.cuda() for k, v in collate([sample]).items()}
        opts = dict(steps=manifest['ddim_steps'], seed=manifest['seed'])
        on = ddim_sample(model, batch, manifest['length'], use_scene=True, **opts)
        joints = forward_kinematics(on, batch['rest'])[0].cpu().numpy()
        with np.load(args.factual / f'motions/{window["index"]:06d}.npz') as saved:
            np.testing.assert_allclose(joints, saved['prediction'], atol=1e-6, rtol=1e-6)
            np.testing.assert_array_equal((batch['joints'][0] * 2).cpu().numpy(), saved['truth'])
            np.testing.assert_array_equal((batch['camera'][0, :, :3] * 2).cpu().numpy(), saved['camera'])
            replay_error = float(np.abs(joints - saved['prediction']).max())
        off = ddim_sample(model, batch, manifest['length'], use_scene=False, **opts)
        altered = dict(batch, occupancy=1 - batch['occupancy'], bps=-batch['bps'] + .3)
        off_altered = ddim_sample(model, altered, manifest['length'], use_scene=False, **opts)
        torch.testing.assert_close(off, off_altered, atol=0, rtol=0)
        if torch.equal(on, off):
            raise AssertionError('Scene intervention has no effect on this probe')
        results.append(dict(group=group, identity=identity, factual_replay_max_error_m=replay_error,
                            masked_scene_perturbation_max_error=float((off - off_altered).abs().max()),
                            factual_vs_masked_feature_l1=float((on - off).abs().mean())))
    with args.output.open('x') as stream:
        json.dump(dict(passed=True, checks=results), stream, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
