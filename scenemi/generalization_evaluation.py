"""Shared camera-only A/B/C evaluation, always with temporal scheme B."""
import json
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from scenemi.config import ROOT
from scenemi.data.ego import collate, payload
from scenemi.data.amass_prepare import atomic_json
from scenemi.evaluate import aggregate
from scenemi.body_diagnostic_metrics import diagnostics
from scenemi.generalization_losses import LOWER, bone_lengths, skeleton_height
from scenemi.geometry.temporal_head_anchor import TemporalHeadAnchor
from scenemi.observations import public_observations
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


@torch.inference_mode()
def evaluate(model, data, cases, c, device, output, *, shape_only=False, rich=False):
    model.eval()
    anchor = TemporalHeadAnchor(c['anchor']).to(device)
    rows = []
    size = c['evaluation']['rich_batch_size' if rich else 'batch_size']
    for offset in tqdm(range(0, len(cases), size), desc='Shape validation' if shape_only else 'B-anchor validation', unit='batch', mininterval=2):
        chunk = cases[offset:offset+size]
        pairs = [data.sample_case(case) for case in chunk]
        batch = collate([x[0] for x in pairs], device)
        public = payload(batch)
        if shape_only:
            details = model.predicted_body(public_observations(public), batch['body_format'])
            metrics = dict(bone_length_mae_mm=(bone_lengths(details['rest'])-bone_lengths(batch['target_rest'])).abs().mean(-1)*1000,
                skeleton_height_error_mm=(skeleton_height(details['rest'])-skeleton_height(batch['target_rest'])).abs()*1000,
                rest_mpjpe_mm=(details['rest']-batch['target_rest']).norm(dim=-1).mean(-1)*1000)
            converted = [{k:float(v[i]) for k,v in metrics.items()} for i in range(len(chunk))]
            for item in converted:
                # Same aggregate implementation; selection has a dedicated,
                # predeclared shape score rather than motion MPJPE.
                item['mpjpe_mm'] = item['rest_mpjpe_mm']
        else:
            raw = sample_native30(model, public, steps=c['evaluation']['ddim_steps'],
                                  seed=c['evaluation']['seed'], use_scene=False)
            _, details = model(raw, torch.zeros(len(chunk), device=device, dtype=torch.long), public,
                               use_scene=False, return_details=True)
            obs = details['observations']
            motion = anchor(raw, details['rest'], details['head_camera_translation'], obs['camera'], obs['camera_available'][..., 0])
            p = forward_kinematics(motion, details['rest'])
            g = forward_kinematics(batch['motion'], batch['target_rest'])
            standard = model.shape.rest_template[None].expand(len(chunk), -1, -1)
            diagnostic = diagnostics(p, motion, details['rest'], g, batch['motion'], batch['target_rest'], standard)
            pr, gr = p[:, :, LOWER]-p[:, :, :1], g[:, :, LOWER]-g[:, :, :1]
            amp = lambda x: ((x-x.mean(1, keepdim=True)).square().sum(-1).mean((1, 2))).sqrt()*1000
            pa, ga = amp(pr), amp(gr)
            converted = []
            for i, item in enumerate(diagnostic):
                values = {k:v for k,v in item['metrics'].items() if 'support_proxy' not in k}
                values.update(lower_pred_amplitude_mm=float(pa[i]), lower_gt_amplitude_mm=float(ga[i]),
                              lower_amplitude_ratio=float(pa[i]/ga[i].clamp_min(1e-6)))
                converted.append(values)
        rows.extend(dict(identity=pairs[i][1], case=case, metrics=converted[i]) for i,case in enumerate(chunk))
    if rich:
        result = dict(windows=len(rows), mean={k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']},
                      aggregation='equal window; independent RICH diagnostic, never selection')
    else:
        result = aggregate(rows)
        if shape_only:
            m = result['dataset_macro_mean']
            result['selection_score_mm'] = m['rest_mpjpe_mm'] + m['bone_length_mae_mm'] + .25*m['skeleton_height_error_mm']
            result['selection_objective'] = 'rest_mpjpe + bone_length_mae + 0.25*skeleton_height_error, all mm'
    result.update(state='complete', shape_only=shape_only, scene=False, camera_anchor='temporal_B_l2', test_used=rich)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'metrics.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    atomic_json(output/'RESULT.json', result)
    return result
