"""Exhaustive eligible-window evaluation; resumable, no training or test-set access."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.compare_feet import metrics
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.data import GROUPS, OfflineSceneMIData, collate
from experiments.offline_camera_retrain_v1.gallery import write_gallery
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI, ddim_sample


def enumerate_windows(dataset, length):
    rows = []
    for group in GROUPS:
        eligible = [(meta, valid) for meta, valid in dataset.base.groups[group] if valid[length]]
        for si, (meta, valid) in enumerate(eligible):
            for wi, (_, source_start) in enumerate(valid[length]):
                rows.append(dict(index=len(rows), group=group, sequence_index=si, start_index=wi,
                                 sequence_id=meta['sequence_id'], source_start_30fps=int(source_start)))
    keys = [(r['group'], r['sequence_id'], r['source_start_30fps']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate windows in audited start manifest')
    expected = sum(dataset.summary()[g][str(length)]['starts'] for g in GROUPS)
    if len(rows) != expected:
        raise ValueError('Window enumeration does not match dataset summary')
    return rows


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def summarize(rows):
    if not rows:
        return dict(windows=0, sequences=0)
    keys = list(rows[0]['metrics'])
    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[(row['group'], row['identity']['sequence_id'])].append(row)
    def mean(items):
        return {k: float(np.mean([r['metrics'][k] for r in items])) for k in keys}
    sequence_means = [mean(items) for items in by_sequence.values()]
    return dict(windows=len(rows), sequences=len(by_sequence), window_mean=mean(rows),
                sequence_mean={k: float(np.mean([s[k] for s in sequence_means])) for k in keys},
                percentiles={k: dict(zip(('p50', 'p90', 'p95'),
                                         np.percentile([r['metrics'][k] for r in rows], [50, 90, 95]).tolist())) for k in keys})


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--length', type=int, choices=(64, 128, 192), default=128)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--ddim-steps', type=int, default=20)
    parser.add_argument('--scene', choices=('on', 'off'), default='on',
                        help='Inference ablation of BOTH occupancy embedding and local BPS; weights unchanged')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-windows', type=int, help='Smoke-test only; incomplete coverage is labelled explicitly')
    args = parser.parse_args()
    if args.max_windows is not None and args.max_windows < 1:
        parser.error('max-windows must be positive')
    torch.set_num_threads(4)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    if config.get('hard_head_rotation'):
        raise ValueError('Hard-head checkpoint requires calibrated projection; use evaluate_turn_balance.py')
    dataset = OfflineSceneMIData('validation', seed=args.seed,
                                 skeleton_profile=config.get('skeleton_profile', 'archived'),
                                 rich_source=config.get('rich_source', 'legacy5interp'))
    windows = enumerate_windows(dataset, args.length)
    full_count = len(windows)
    if args.max_windows is not None:
        windows = windows[:args.max_windows]
    manifest = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
                    checkpoint_step=checkpoint['step'],
                    cumulative_step=config.get('initial_total_steps', 0) + checkpoint['step'],
                    loss_profile=config.get('loss_profile', 'baseline'),
                    skeleton_profile=config.get('skeleton_profile', 'archived'),
                    split='validation', length=args.length, fps=20, control='head',
                    ddim_steps=args.ddim_steps, seed=args.seed,
                    use_scene=args.scene == 'on',
                    seed_policy='same initial noise seed for every window; single-seed full sweep',
                    eligible_windows=full_count, planned_windows=len(windows),
                    dataset_summary=dataset.summary(), windows=windows,
                    notes='All audited candidate starts, not every possible video frame. Overlapping windows are not independent. Height metrics are GT proxies, not scene collisions. Activity strata are GT-derived proxies, not action labels.')
    if args.resume:
        stored_manifest = json.loads((args.output / 'manifest.json').read_text())
        stored_manifest.setdefault('use_scene', True)  # Original evaluator always used the scene.
        if stored_manifest != manifest:
            raise ValueError('Resume protocol/checkpoint/window manifest mismatch')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output / 'manifest.json', manifest)
        snapshot = args.output / 'source_snapshot'
        snapshot.mkdir()
        for p in Path(__file__).parent.glob('*.py'):
            shutil.copy2(p, snapshot / p.name)
        shutil.copy2(Path(__file__).resolve().parents[2] / 'model/mdm_scene_unet.py', snapshot / 'mdm_scene_unet.py')
    (args.output / 'motions').mkdir(exist_ok=True)
    label = f"{manifest['cumulative_step'] // 1000}k · 场景{'开启' if args.scene == 'on' else '关闭'}"
    variant_label = f"checkpoint_{manifest['cumulative_step']}_scene_{args.scene}"
    index_template = '''<!doctype html><meta charset="utf-8"><title>RUN_LABEL 全量验证</title>
<style>body{font:17px system-ui;max-width:1000px;margin:40px auto}pre{white-space:pre-wrap}a{color:#176ac2}</style>
<h1>RUN_LABEL · 全量验证</h1><p>PROTOCOL_LABEL。遍历审计清单内全部候选窗口，不是随机抽样。这里只评估，不训练。</p>
<p><a href="metrics.json">聚合指标（含序列平均、来源和活动分层）</a> · <a href="rows.jsonl">逐窗口指标</a> · <a href="manifest.json">评估清单</a></p>
<p>完成后：<a href="representative.html">固定代表片段</a> · <a href="worst_cases.html">误差极端片段（非代表性样本）</a></p><pre id="status">正在读取进度…</pre>
<script>async function refresh(){try{const r=await fetch('status.json',{cache:'no-store'});document.getElementById('status').textContent=JSON.stringify(await r.json(),null,2)}catch(e){document.getElementById('status').textContent=String(e)}}refresh();setInterval(refresh,10000)</script>'''
    (args.output / 'index.html').write_text(index_template.replace('RUN_LABEL', label).replace(
        'PROTOCOL_LABEL', f'{args.length}帧 / 头部条件 / DDIM{args.ddim_steps} / seed{args.seed}'))
    rows = []
    row_path = args.output / 'rows.jsonl'
    if args.resume and row_path.exists():
        rows = [json.loads(line) for line in row_path.read_text().splitlines() if line]
    for i, row in enumerate(rows):
        if row['window'] != windows[i] or not (args.output / row['motion_file']).is_file():
            raise ValueError('Resume rows/motion archive mismatch')
    model = OfflineSceneMI(config['latent_dim'], tuple(config['dim_mults'])).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    start = time.monotonic()
    initial_count = len(rows)

    def save(status, error=None):
        elapsed = time.monotonic() - start
        rate = (len(rows) - initial_count) / max(elapsed, 1e-6)
        progress = dict(status=status, completed_windows=len(rows), planned_windows=len(windows),
                        eligible_windows=full_count, full_coverage=len(rows) == full_count,
                        completed_sequences=len({(r['group'], r['identity']['sequence_id']) for r in rows}),
                        elapsed_this_session_s=round(elapsed, 1),
                        eta_seconds=round((len(windows) - len(rows)) / rate) if rate else None,
                        pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat())
        if error:
            progress['error'] = error
        report = dict(status=status, full_coverage=progress['full_coverage'], overall=summarize(rows),
                      groups={g: summarize([r for r in rows if r['group'] == g]) for g in GROUPS},
                      activity_strata={s: summarize([r for r in rows if r['activity'][s]]) for s in
                                       ('walking_proxy', 'low_activity_proxy', 'upper_body_active_proxy', 'other_proxy')},
                      notes=manifest['notes'] + ' Activity strata may overlap; upper-body activity is not a semantic action label.')
        atomic_json(args.output / 'metrics.json', report)
        atomic_json(args.output / 'status.json', progress)
        print(json.dumps(progress), flush=True)

    save('running')
    try:
        with row_path.open('a') as stream:
            for window in windows[initial_count:]:
                sample, identity = dataset.sample(args.length, window['group'],
                    sequence_index=window['sequence_index'], start_index=window['start_index'])
                if identity['sequence_id'] != window['sequence_id'] or identity['source_start_30fps'] != window['source_start_30fps']:
                    raise ValueError('Loaded window differs from enumerated identity')
                batch = {k: v.cuda() for k, v in collate([sample]).items()}
                motion = ddim_sample(model, batch, args.length, steps=args.ddim_steps, seed=args.seed,
                                     control_mask=fixed_control_mask(1, args.length, 'head', 'cuda'),
                                     use_scene=args.scene == 'on')
                values, joints = metrics(motion, batch)
                if not all(np.isfinite(v) for v in values.values()) or not torch.isfinite(joints).all():
                    raise ValueError('Nonfinite sampled motion/metrics')
                gt = batch['joints'] * 2
                wrist = gt[:, :, (20, 21)] - gt[:, :, (16, 17)]
                wrist_speed = float(torch.diff(wrist, dim=1).norm(dim=-1).mean())
                joint_speed = float(torch.diff(gt, dim=1).norm(dim=-1).mean())
                values.update(gt_wrist_relative_speed_m_frame=wrist_speed, gt_joint_speed_m_frame=joint_speed)
                activity = dict(walking_proxy=values['gt_root_path_length_m'] > 1 and values['gt_foot_separation_rms_cm'] > 10,
                                low_activity_proxy=values['gt_root_path_length_m'] < .25 and joint_speed < .002,
                                upper_body_active_proxy=values['gt_root_path_length_m'] < 1 and wrist_speed > .005)
                activity['other_proxy'] = not any(activity.values())
                motion_file = f'motions/{window["index"]:06d}.npz'
                np.savez_compressed(args.output / motion_file,
                                    prediction=joints[0].cpu().numpy(), truth=gt[0].cpu().numpy(),
                                    camera=(batch['camera'][0, :, :3] * 2).cpu().numpy())
                row = dict(window=window, group=window['group'], identity=identity,
                           activity=activity, metrics=values, motion_file=motion_file)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                rows.append(row)
                if len(rows) == 1 or len(rows) % 25 == 0:
                    save('running')
        # Keep galleries small; all generated joint tracks remain available as npz.
        selected = []
        for key in ('head_cm', 'slowest_foot_horizontal_cm_per_frame', 'coordination_relation_mse', 'support_floating_m'):
            selected.extend(sorted(rows, key=lambda r: r['metrics'][key], reverse=True)[:3])
        worst = list({r['window']['index']: r for r in selected}.values())
        representative = []
        for group in GROUPS:
            members = [r for r in rows if r['group'] == group]
            if members:
                ids = np.linspace(0, len(members) - 1, min(4, len(members))).astype(int)
                representative.extend(members[i] for i in ids)
        for name, selection in [('worst_cases', worst), ('representative', representative)]:
            cases = []
            for row in selection:
                with np.load(args.output / row['motion_file']) as arrays:
                    cases.append(dict(group=row['group'], identity=row['identity'], camera=arrays['camera'].round(4).tolist(),
                                      tracks={'GT': arrays['truth'].round(4).tolist(), variant_label: arrays['prediction'].round(4).tolist()},
                                      metrics={variant_label: row['metrics']}))
            write_gallery(args.output / f'{name}.html', cases, f'{label} / {name}')
        save('completed' if len(rows) == full_count else 'smoke_completed')
    except Exception as exc:
        save('failed', repr(exc))
        raise


if __name__ == '__main__':
    main()
