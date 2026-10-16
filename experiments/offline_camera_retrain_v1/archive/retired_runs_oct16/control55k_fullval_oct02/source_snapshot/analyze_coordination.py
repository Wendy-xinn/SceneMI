"""Audit generated galleries without running inference or modifying motions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.coordination import coordination_losses


def body_signals(p):
    lateral = p[:, 2] - p[:, 1]
    lateral[:, 1] = 0
    lateral /= np.linalg.norm(lateral, axis=-1, keepdims=True).clip(1e-8)
    forward = np.cross(lateral, np.array([0., 1., 0.]))
    arms = (p[:, 20] - p[:, 16]) - (p[:, 21] - p[:, 17])
    legs = (p[:, 10] - p[:, 1]) - (p[:, 11] - p[:, 2])
    return (arms * forward).sum(-1), (legs * forward).sum(-1)


def describe(p):
    arm, leg = body_signals(p)
    arm_sd, leg_sd = arm.std(), leg.std()
    correlation = (float(np.corrcoef(arm, leg)[0, 1])
                   if min(arm_sd, leg_sd) > 1e-6 else None)
    wrist = p[:, (20, 21)] - p[:, (16, 17)]
    return dict(arm_leg_correlation=correlation,
                arm_differential_rms_cm=float(arm_sd * 100),
                leg_differential_rms_cm=float(leg_sd * 100),
                wrist_relative_speed_cm_frame=float(np.linalg.norm(np.diff(wrist, axis=0), axis=-1).mean() * 100))


def analyze(path):
    page = Path(path).read_text()
    data = json.loads(page.split('const data=', 1)[1].split(';\nconst $=', 1)[0])
    rows = []
    for i, case in enumerate(data['cases']):
        truth = np.asarray(case['tracks']['GT'])
        gt = describe(truth)
        separation = (truth[:, 10] - truth[:, 11])[:, (0, 2)]
        path_length = np.linalg.norm(np.diff(truth[:, 0, (0, 2)], axis=0), axis=-1).sum()
        walking_proxy = bool(path_length > 1. and np.sqrt(((separation - separation.mean(0)) ** 2).sum(-1).mean()) > .1)
        for variant, track in case['tracks'].items():
            if variant == 'GT':
                continue
            p = np.asarray(track)
            values = describe(p)
            values.update({k: float(v) for k, v in coordination_losses(
                torch.tensor(p[None], dtype=torch.float32),
                torch.tensor(truth[None], dtype=torch.float32)).items()})
            values['arm_leg_correlation_error'] = (
                abs(values['arm_leg_correlation'] - gt['arm_leg_correlation'])
                if values['arm_leg_correlation'] is not None and gt['arm_leg_correlation'] is not None else None)
            # Correlation sign alone is not a naturalness criterion: GT can
            # legitimately be in-phase (e.g. interaction or object handling).
            values['same_sign_as_gt'] = (float(values['arm_leg_correlation'] * gt['arm_leg_correlation'] > 0)
                                       if values['arm_leg_correlation'] is not None and gt['arm_leg_correlation'] is not None else None)
            rows.append(dict(case=i, identity=case['identity'], group=case['group'],
                             variant=variant, walking_proxy=walking_proxy,
                             gt=gt, values=values))
    summaries = {}
    for variant in dict.fromkeys(r['variant'] for r in rows):
        group = [r for r in rows if r['variant'] == variant]
        summaries[variant] = {}
        for name, selected in [('all', group), ('walking_proxy', [r for r in group if r['walking_proxy']]),
                               ('other_windows', [r for r in group if not r['walking_proxy']])]:
            summary = {'n': len(selected)}
            for key in group[0]['values']:
                values = [r['values'][key] for r in selected if r['values'][key] is not None]
                summary[key] = float(np.mean(values)) if values else None
            summaries[variant][name] = summary
    return dict(gallery=str(path), notes='Reference-relative proxies; not a universal realism score. Other windows are not action-labelled non-walking. Small-amplitude correlation can be unstable.',
                summaries=summaries, rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gallery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = analyze(args.gallery)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['summaries'], indent=2))


if __name__ == '__main__':
    main()
