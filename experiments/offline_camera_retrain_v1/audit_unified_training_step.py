"""One real EXO optimization step; plumbing evidence, never a trained result."""
import json
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.audit_unified_inputs import dataset
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas
from experiments.offline_camera_retrain_v1.supervision import supervised_losses
from experiments.offline_camera_retrain_v1.unified_geometry import SkinAnchorCache
from experiments.offline_camera_retrain_v1.unified_model import UnifiedSceneMI
from experiments.offline_camera_retrain_v1.unified_observations import ObservationBuilder, collate_observations, inference_payload
from experiments.offline_camera_retrain_v1.unified_losses import track_losses


def main():
    torch.set_num_threads(4); torch.manual_seed(2026101153)
    output = HERE / 'runs/ego_exo_surface_continue_oct11'
    cp = torch.load(HERE / 'runs/native_dynamic_scene20_contact_55k_oct07/last.pt', map_location='cpu', weights_only=False)
    c = cp['config']; model = UnifiedSceneMI(c['latent_dim'], tuple(c['dim_mults']), body_conditioning=True, contact_prediction=True).cuda()
    model.load_state_dict(cp['model']); del cp
    data = dataset(c, 'train', seed=2026101153); builder = ObservationBuilder(SkinAnchorCache()); items, identities = [], []
    for group in ('camera_wearer', 'interactee'):
        sample, identity = data.sample(128, group, sequence_index=0, start_index=0)
        observed, scene, asset = builder.build(data, sample, identity, 'exo', joints=(15, 20, 21))
        items.append((observed, scene, asset, 'exo')); identities.append(identity)
    batch = collate_observations(items); known = inference_payload(batch)
    t = torch.tensor([30, 500], device='cuda'); alpha = cosine_alphas().cuda()[t][:, None, None]
    noisy = alpha.sqrt() * batch['motion'] + (1 - alpha).sqrt() * torch.randn_like(batch['motion'])
    mask = batch['observation_meta'][..., 0] > 0
    pred = model(noisy, t, known, control_mask=mask)
    base = supervised_losses(pred, batch['motion'], batch, profile='coordination_v1', signal_weight=alpha.flatten())
    mounts = json.loads((output / 'TRAIN_camera_mounts.json').read_text())
    weights = {k: v['coefficient'] for k, v in json.loads((output / 'TRAIN_loss_scale_audit.json').read_text())['terms'].items()}
    terms = track_losses(pred, batch['motion'], batch, alpha.flatten(), mounts, ['camera_wearer', 'interactee'])
    total = base['total'] + sum(weights[k] * v for k, v in terms.items()); assert torch.isfinite(total)
    total.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5); optimizer.step()
    result = dict(status='passed', scope='one TRAIN exo optimization step from55k; model discarded, not a performance result',
                  identities=identities, seed=2026101153, timesteps=t.tolist(), loss=float(total.detach()),
                  gradient_norm=float(norm), observed_position_entries=int(mask.sum()),
                  rotation_available_entries=int(batch['observation_meta'][..., 1].sum()),
                  terms={k: float(v.detach()) for k, v in terms.items()})
    assert result['rotation_available_entries'] == 0 and result['observed_position_entries'] > 0
    (output / 'exo_training_step_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
