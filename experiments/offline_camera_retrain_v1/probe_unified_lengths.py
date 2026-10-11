"""Generation interface probe; short DDIM schedule is not a quality test."""
import json
from pathlib import Path
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.audit_unified_inputs import dataset
from experiments.offline_camera_retrain_v1.scene_model import ddim_sample
from experiments.offline_camera_retrain_v1.unified_model import UnifiedSceneMI
from experiments.offline_camera_retrain_v1.unified_geometry import SkinAnchorCache
from experiments.offline_camera_retrain_v1.unified_observations import ObservationBuilder, collate_observations, inference_payload


def main():
    torch.set_num_threads(4)
    output = HERE / 'runs/ego_exo_surface_continue_oct11'
    state = torch.load(output / 'integration/last.pt', map_location='cpu', weights_only=False)
    c = state['config']
    model = UnifiedSceneMI(c['latent_dim'], tuple(c['dim_mults']), body_conditioning=True, contact_prediction=True).cuda().eval()
    model.load_state_dict(state['model']); model.surface_bps_weight = c['surface_bps_weight']; del state
    data = dataset(c, 'validation'); builder = ObservationBuilder(SkinAnchorCache()); rows = []
    for length in (64, 128, 192):
        sample, identity = data.sample(length, 'camera_wearer', sequence_index=0, start_index=0)
        for mode in ('ego', 'exo'):
            observed, scene, asset = builder.build(data, sample, identity, mode,
                                                   joints=(15,) if mode == 'ego' else (15, 20, 21))
            batch = collate_observations([(observed, scene, asset, mode)])
            motion = ddim_sample(model, inference_payload(batch), length, steps=4, seed=2026101151,
                                 control_mask=batch['observation_meta'][..., :2].amax(-1) > 0)
            assert motion.shape == (1, length, 201) and torch.isfinite(motion).all()
            rows.append(dict(mode=mode, frames=length, shape=list(motion.shape), finite=True, identity=identity))
            print('PROBE', mode, length, flush=True)
    (output / 'length_interface_checks.json').write_text(json.dumps(dict(status='passed', seed=2026101151,
             ddim_steps=4, scope='generation interface only; quality and arbitrary lengths unverified', rows=rows), indent=2) + '\n')


if __name__ == '__main__': main()
