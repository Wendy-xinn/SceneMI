"""Verify the completed ego stage and summarize unchanged fixed VAL panels."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from diagnose_articulation import measures

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    training = ROOT / 'training'
    checkpoint = torch.load(training / 'last.pt', map_location='cpu', weights_only=False, mmap=True)
    protocol = json.loads((training / 'protocol.json').read_text())
    assert checkpoint['step'] == protocol['completed_steps'] == protocol['planned_steps'] == 10000
    assert protocol['status'] == 'completed'
    model_finite = all(torch.isfinite(v).all().item() for v in checkpoint['model'].values() if torch.is_floating_point(v))
    adam_finite = all(torch.isfinite(v).all().item() for state in checkpoint['optimizer']['state'].values()
        for v in state.values() if isinstance(v, torch.Tensor) and torch.is_floating_point(v))
    changed = [p for p, h in checkpoint['effective_config']['code_sha256'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
    assert model_finite and adam_finite and not changed
    validation = [json.loads(line) for line in (training / 'validation.jsonl').read_text().splitlines()]
    assert validation[-1]['step'] == 10000
    fixed = [{**{'step': row['step']}, **{k: v for k, v in row['summary'].items() if k.startswith('fixed/ego/')}} for row in validation]
    feet = json.loads((ROOT / 'FOOT_TREND.json').read_text())
    assert feet['latest_val_step'] == 10000 and feet['status'] == 'completed'
    articulation = []
    with np.load(training / 'validation_latest.npz') as final, np.load(ROOT / 'visual_review_step8000/validation_outputs.npz') as earlier:
        assert int(final['step']) == 10000
        for case in validation[-1]['cases']:
            if case['mode'] != 'ego' or case['panel'] != 'fixed' or case['regime'] != 'clean' or case.get('scene_variant', 'scene_on') != 'scene_on':
                continue
            i = case['case']
            assert np.array_equal(final[f'{i}_truth'], earlier[f'{i}_truth'])
            truth, _, _ = measures(final[f'{i}_truth'], final[f'{i}_rest'])
            for regime in ('clean', 'noisy'):
                for suffix in ('observations', 'observation_meta'):
                    assert np.array_equal(final[f'{i}_{regime}_{suffix}'], earlier[f'{i}_{regime}_{suffix}'])
                pred, _, _ = measures(final[f'{i}_{regime}_prediction'], final[f'{i}_rest'])
                articulation.append(dict(dataset=case['dataset'], sequence_id=case['sequence_id'], regime=regime,
                    body_local_foot_temporal_RMS_ratio=(np.array(pred['body_local_foot_temporal_RMS_m']) / np.array(truth['body_local_foot_temporal_RMS_m'])).tolist()))
    report = dict(observed_at=datetime.datetime.now().astimezone().isoformat(), status='completed registered ego10000 stage; quality not accepted',
        completed_checkpoint_step=checkpoint['step'], checkpoint_saved_at=datetime.datetime.fromtimestamp((training / 'last.pt').stat().st_mtime).astimezone().isoformat(),
        model_all_finite=model_finite, optimizer_all_finite=adam_finite, changed_sealed_sources=changed,
        initialization=checkpoint['initialization'], sample_slots=protocol['counts'], fixed_panel_GT_and_observations_match_step8000=True,
        fixed_ego_trends=fixed, latest_foot_metrics=[x for x in feet['entries'] if x['step'] == 10000],
        latest_fixed_articulation=articulation,
        conclusion='Fixed clean/noisy MPJPE regresses at10000 vs8000; noisy stance foot speed worsens. SceneON improves overall MPJPE vsOFF at10000 but not all cases/foot metrics. Three fixed windows are monitoring, not full official VAL or a proof of overfitting.',
        next_stage='Do not blindly extend unchanged objective or automatically start exo. Preserve clean10000 checkpoint with Adam/RNG; prioritize matched articulation-loss and projection diagnostics described in master4.14. Old55k is development-only due officialTEST overlap.',
        viewers='Existing ports8788/8789 still show actual8000 outputs, not10000.', optimizer_steps_in_review=0,
        disk_free_GiB={p: shutil.disk_usage(p).free / 2**30 for p in ('/', '/mnt/e')})
    (ROOT / 'FINAL_EGO_STAGE_REVIEW.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for regime, color in (('clean', '#246ed0'), ('noisy', '#bc187e')):
        for ax, key, label in ((axes[0], 'mpjpe_mm', 'MPJPE (mm)'), (axes[1], 'foot_speed_during_GT_stance_m_s', 'Foot speed during GT stance (m/s)')):
            ax.plot([r['step'] for r in fixed], [r[f'fixed/ego/{regime}/scene_on'][key]['value'] for r in fixed], 'o-', label=regime, color=color)
            ax.set(xlabel='Optimizer updates', ylabel=label)
            ax.grid(alpha=.25)
            ax.legend()
    fig.suptitle('Completed ego stage: same three fixed official VAL windows / same seeds\nShort monitoring panel; not full benchmark')
    fig.tight_layout()
    fig.savefig(ROOT / 'FINAL_EGO_STAGE_TRENDS.png', dpi=140)
    print(json.dumps(dict(status=report['status'], articulation=articulation), ensure_ascii=False))


if __name__ == '__main__':
    main()
