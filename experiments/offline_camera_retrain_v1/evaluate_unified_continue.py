"""Paired random-noise window generation; observations frozen before scoring."""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.audit_unified_inputs import dataset
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI, ddim_sample
from experiments.offline_camera_retrain_v1.unified_model import UnifiedSceneMI
from experiments.offline_camera_retrain_v1.unified_geometry import SkinAnchorCache
from experiments.offline_camera_retrain_v1.unified_observations import ObservationBuilder, collate_observations, inference_payload
from experiments.offline_camera_retrain_v1.unified_metrics import score, metric_checks


@torch.no_grad()
def evaluate(folder, windows_per_group=3, ddim_steps=20):
    torch.set_num_threads(4)
    output = folder / 'evaluation'; output.mkdir(exist_ok=True)
    source = HERE / 'runs/native_dynamic_scene20_contact_55k_oct07/last.pt'
    state = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
    original = torch.load(source, map_location='cpu', weights_only=False)
    c = state['config']; models = {}
    for name, cp, cls in (('original55k', original, OfflineSceneMI), ('unified', state, UnifiedSceneMI)):
        model = cls(c['latent_dim'], tuple(c['dim_mults']), body_conditioning=True, contact_prediction=True).cuda().eval()
        model.load_state_dict(cp['model'])
        if name == 'unified': model.surface_bps_weight = c['surface_bps_weight']
        models[name] = model
    step = state['step']; del state, original
    data = dataset(c, 'validation', seed=2026101149)
    builder = ObservationBuilder(SkinAnchorCache()); selected = []
    for group in ('trumans', 'camera_wearer', 'interactee', 'rich'):
        source_members = data.rich_members if group == 'rich' else data.base.groups[group]
        count = sum(bool(starts[128]) for _, starts in source_members)
        indices = sorted(set(np.linspace(0, count - 1, windows_per_group, dtype=int).tolist()))
        for index in indices:
            selected.append(dict(group=group, sequence_index=index, start_index=0, frames=128))
    protocol = dict(status='registered_before_scoring', selected=selected, seed=2026101149, ddim_steps=ddim_steps,
                    checkpoint_step=step, checkpoint_sha256=hashlib.sha256((folder / 'last.pt').read_bytes()).hexdigest(),
                    frame_clock='20Hz', input='same frozen own-view observations and native body calibration; no prefix',
                    baseline_exo='original55k zero-shot position-only sparse transport; no added GT rotations',
                    comparison='ego across three domains; ego/exo mode comparison only matched EgoBody person/time windows',
                    holdout='validation development panel; input checks used first windows; not a pristine test set',
                    metric_checks=metric_checks())
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    rows = []; started = time.monotonic()
    for number, selected_window in enumerate(selected):
        w = selected_window; sample, identity = data.sample(128, w['group'], sequence_index=w['sequence_index'], start_index=0)
        modes = ['ego', 'exo'] if w['group'] in ('camera_wearer', 'interactee') else ['ego']
        for mode in modes:
            observed, scene, asset = builder.build(data, sample, identity, mode, joints=(15,) if mode == 'ego' else (15, 20, 21))
            batch = collate_observations([(observed, scene, asset, mode)])
            known = inference_payload(batch); mask = batch['observation_meta'][..., :2].amax(-1) > 0
            for label, model in models.items():
                motion = ddim_sample(model, known, 128, steps=ddim_steps, seed=2026101149,
                                     control_mask=mask, use_scene=True)
                values, joints = score(motion, batch)
                rows.append(dict(window=selected_window, identity=identity, mode=mode, model=label, metrics=values))
                np.savez_compressed(output / f'{number:03d}_{mode}_{label}.npz', motion=motion[0].cpu().numpy(), joints=joints,
                                    observation_meta=observed['observation_meta'], trajectory=observed['trajectory'])
            (output / 'rows.json').write_text(json.dumps(rows, indent=2) + '\n')
            print('EVAL', number, mode, {row['model']: round(row['metrics']['world_unaligned_mpjpe_mm'], 2) for row in rows[-2:]}, flush=True)
    grouped = defaultdict(list)
    for row in rows: grouped[f"{row['model']}/{row['identity']['group']}/{row['mode']}"].append(row['metrics'])
    summary = {}
    for key, values in grouped.items():
        summary[key] = {name: float(np.mean([r[name] for r in values if r[name] is not None]))
                        if any(r[name] is not None for r in values) else None for name in values[0]}
        summary[key]['windows'] = len(values)
        summary[key]['turn_denominator'] = sum(r['turn_eligible'] for r in values)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    protocol.update(status='completed', elapsed_s=time.monotonic() - started)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--folder', type=Path, default=HERE / 'runs/ego_exo_surface_continue_oct11')
    p.add_argument('--windows-per-group', type=int, default=3); p.add_argument('--ddim-steps', type=int, default=20)
    args = p.parse_args(); evaluate(args.folder, args.windows_per_group, args.ddim_steps)
