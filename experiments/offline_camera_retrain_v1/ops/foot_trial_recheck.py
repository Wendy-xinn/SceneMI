"""CPU-only independent verification of saved paired foot-trial evaluations."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from .motion_ablation import BASE, FULL, load_module, now, read, require, sha, surface_metrics, write

DEFAULT_OUT = BASE / 'runs/native30_live_monitor_oct16/paired_foot_trial/evaluation'


def array_digest(arrays, keys):
    digest = hashlib.sha256()
    for key in keys:
        value = np.ascontiguousarray(arrays[key])
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def independent_metrics(prediction, truth, rest, observed, meta):
    """Direct NumPy formulas after unchanged native FK, including timing checks."""
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

    p, g = (torch.from_numpy(np.array(a, dtype=np.float32, copy=True))[None] for a in (prediction, truth))
    r = torch.from_numpy(np.array(rest, dtype=np.float32, copy=True))[None]
    with torch.no_grad():
        jp, jg = (forward_kinematics(a, r)[0].numpy().astype(np.float64) for a in (p, g))
        rp, rg = (global_rotations(a)[0, :, 0].numpy().astype(np.float64) for a in (p, g))
    frames = len(jp)
    valid = (meta[:, 15, 0] > 0) & (meta[:, 15, 2] > 0)
    vp, vg = (np.diff(j[:, [10, 11]], axis=0) * 30 for j in (jp, jg))
    low = np.linalg.norm(vg, axis=-1) < .2
    mean = lambda a, mask=None: float((a if mask is None else a[mask]).mean()) if (a if mask is None else a[mask]).size else None
    pa_errors = []
    for x, y in zip(jp, jg):
        xc, yc = x - x.mean(0), y - y.mean(0)
        u, singular, vt = np.linalg.svd(xc.T @ yc)
        sign = np.ones(3)
        sign[-1] = np.linalg.det(u @ vt)
        rotation = (u * sign[None]) @ vt
        scale = (singular * sign).sum() / max(float((xc * xc).sum()), 1e-12)
        pa_errors.extend(np.linalg.norm(scale * (xc @ rotation) + y.mean(0) - y, axis=-1))
    foot_y = jp[:, [10, 11], 1] - jg[:, [10, 11], 1]
    root_y = jp[:, 0, 1] - jg[:, 0, 1]
    extent_p, extent_g = (j[..., 1].max(-1) - j[..., 1].min(-1) for j in (jp, jg))
    span_p, span_g = (j[:, 15, 1] - j[:, [10, 11], 1].min(-1) for j in (jp, jg))
    local_p, local_g = (np.stack([rotation.T @ (j[10] - j[11]) for rotation, j in zip(rotations, joints)])
                        for rotations, joints in ((rp, jp), (rg, jg)))
    rms = lambda a: float(np.sqrt(np.mean(np.sum((a - a.mean(0)) ** 2, axis=-1))))
    rms_p, rms_g = rms(local_p), rms(local_g)
    dp, dg = ([a[lag:] - a[:-lag] for lag in (6, 12)] for a in (local_p, local_g))
    changes_p, changes_g = ([np.linalg.norm(a, axis=-1) for a in values] for values in (dp, dg))
    metrics = dict(w_mpjpe_mm=float(np.linalg.norm(jp - jg, axis=-1).mean() * 1000),
        pa_mpjpe_mm=float(np.mean(pa_errors) * 1000),
        observed_head_position_error_mm=mean(np.linalg.norm(jp[valid, 15] - observed[valid, 15, :3] * 2, axis=-1) * 1000),
        foot_joint_horizontal_speed_gt_low_m_s=mean(np.linalg.norm(vp[..., [0, 2]], axis=-1), low),
        foot_joint_vertical_speed_gt_low_m_s=mean(np.abs(vp[..., 1]), low),
        gt_foot_joint_horizontal_speed_gt_low_m_s=mean(np.linalg.norm(vg[..., [0, 2]], axis=-1), low),
        gt_foot_joint_vertical_speed_gt_low_m_s=mean(np.abs(vg[..., 1]), low),
        foot_joint_y_abs_error_gt_low_mm=mean(np.abs(foot_y[1:]) * 1000, low),
        foot_joint_y_signed_error_gt_low_mm=mean(foot_y[1:] * 1000, low),
        root_y_signed_error_mm=mean(root_y * 1000), root_y_abs_error_mm=mean(np.abs(root_y) * 1000),
        predicted_root_y_mean_m=mean(jp[:, 0, 1]), gt_root_y_mean_m=mean(jg[:, 0, 1]),
        predicted_body_joint_vertical_extent_mean_m=mean(extent_p), gt_body_joint_vertical_extent_mean_m=mean(extent_g),
        body_joint_vertical_extent_abs_error_mm=mean(np.abs(extent_p - extent_g) * 1000),
        predicted_head_to_foot_vertical_span_mean_m=mean(span_p), gt_head_to_foot_vertical_span_mean_m=mean(span_g),
        head_to_foot_vertical_span_abs_error_mm=mean(np.abs(span_p - span_g) * 1000),
        bilateral_motion_rms_m=rms_p, gt_bilateral_motion_rms_m=rms_g,
        bilateral_motion_rms_error_mm=abs(rms_p - rms_g) * 1000,
        bilateral_motion_rms_ratio=rms_p / rms_g if rms_g > .01 else None,
        bilateral_stride_change_mean_m=float(np.mean([a.mean() for a in changes_p])),
        gt_bilateral_stride_change_mean_m=float(np.mean([a.mean() for a in changes_g])),
        bilateral_stride_change_abs_error_mm=float(np.mean([np.abs(a - b).mean() for a, b in zip(changes_p, changes_g)]) * 1000))
    predicted_low = np.linalg.norm(vp, axis=-1) < .2
    confusion = dict(tp=int((predicted_low & low).sum()), fp=int((predicted_low & ~low).sum()),
                     fn=int((~predicted_low & low).sum()), tn=int((~predicted_low & ~low).sum()))
    articulated = rms_g > .01
    sagittal_p, sagittal_g = local_p[:, 2] - local_p[:, 2].mean(), local_g[:, 2] - local_g[:, 2].mean()
    correlation = float(np.corrcoef(sagittal_p, sagittal_g)[0, 1]) if articulated and sagittal_p.std() > 1e-8 and sagittal_g.std() > .01 else None
    drifts, gt_drifts, drift_errors = [], [], []
    for side in range(2):
        changes = np.diff(np.r_[False, low[:, side], False].astype(np.int8))
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            if end - start >= 3:
                displacement = (jp[end, [10, 11][side]] - jp[start, [10, 11][side]])[[0, 2]]
                gt_displacement = (jg[end, [10, 11][side]] - jg[start, [10, 11][side]])[[0, 2]]
                drifts.append(float(np.linalg.norm(displacement) * 1000))
                gt_drifts.append(float(np.linalg.norm(gt_displacement) * 1000))
                drift_errors.append(float(np.linalg.norm(displacement - gt_displacement) * 1000))
    supplemental = dict(low_speed_proxy_confusion=confusion,
        low_speed_proxy_accuracy=float((predicted_low == low).mean()),
        bilateral_sagittal_temporal_correlation=correlation,
        bilateral_stride_vector_error_mm=float(np.mean([np.linalg.norm(a - b, axis=-1).mean() for a, b in zip(dp, dg)]) * 1000),
        gt_articulated_window=articulated, gt_low_speed_runs_min_3_intervals=len(drifts),
        predicted_horizontal_drift_over_GT_low_runs_mean_mm=float(np.mean(drifts)) if drifts else None,
        gt_horizontal_drift_over_low_runs_mean_mm=float(np.mean(gt_drifts)) if gt_drifts else None,
        horizontal_displacement_error_over_GT_low_runs_mean_mm=float(np.mean(drift_errors)) if drift_errors else None)
    denominators = dict(frames=frames, body_joint_samples=frames * 22, observed_head_frames=int(valid.sum()),
        gt_low_speed_foot_samples=int(low.sum()), left_gt_low_speed_foot_samples=int(low[:, 0].sum()),
        right_gt_low_speed_foot_samples=int(low[:, 1].sum()), foot_velocity_samples=int(low.size),
        stride_pairs_by_lag_frames={'6': frames - 6, '12': frames - 12},
        stride_aggregation='Arithmetic mean of the two per-lag means; not pooled pair mean.',
        bilateral_rms_windows=1, bilateral_ratio_valid=int(rms_g > .01),
        PA_alignment='Per-frame proper similarity transform across all 22 joints.')
    return metrics, supplemental, denominators


def run(out, labels):
    from experiments.offline_camera_retrain_v1.prepare_native30 import window_options
    from experiments.offline_camera_retrain_v1.native30_data import Native30Data
    from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
    from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
    from experiments.offline_camera_retrain_v1.native30_surface_cache import CanonicalSurfaceCache
    from .support_metrics import support_metrics
    from .foot_trial_evaluation import extended_metrics

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), 'CPU verifier must not initialize CUDA')
    plan, cohort = read(out / 'PLAN.json'), read(out / 'COHORT.json')
    require(sha(out / 'COHORT.json') == plan['cohort_sha256'], 'Cohort changed')
    require(all(label in plan['models'] for label in labels), 'Unknown requested model')
    source_matches = {path: sha(path) == digest for path, digest in plan['dependency_sha256'].items()}
    require(all(source_matches.values()), 'Sealed evaluation dependency changed')
    all_rows = [json.loads(line) for line in (out / 'rows.jsonl').read_text().splitlines()]
    rows = [row for row in all_rows if row['model'] in labels]
    expected = {(label, entry['case'], regime, guidance) for label in labels for entry in cohort['entries']
                for regime in cohort['regimes'] for guidance in ('off', 'on')
                if guidance == 'off' or entry['case'] in cohort['secondary']['cases']}
    found = {(row['model'], row['case'], row['regime'], row['guidance']) for row in rows}
    require(len(rows) == len(found) and found == expected, 'Requested phase is not exactly complete')
    for label in labels:
        checkpoint = read(out / 'checkpoints' / (label + '.json'))
        require(checkpoint['declaration'] == plan['models'][label], 'Checkpoint declaration differs')
        require(sha(checkpoint['checkpoint']) == checkpoint['checkpoint_sha256'], 'Pinned checkpoint changed')
        require(all(row['checkpoint_sha256'] == checkpoint['checkpoint_sha256'] for row in rows if row['model'] == label), 'Mixed checkpoint rows')
    helper = load_module('foot_recheck_source_guard', FULL / 'runner.py')
    effective = helper.verify_sources()
    cfg = effective['config']
    data = Native30Data(BASE / cfg['train_manifest'], 'validation', cfg['seed'] + 1, frames=cfg['frames'],
        memory_protocol=cfg['scene']['memory_protocol'], quality_config=cfg['gt_reliability'],
        exo_max_empty_frames=cfg['exo_max_empty_frames'], **window_options(cfg))
    data.runtime_cache_root = out / 'native30_runtime_cache_v2'
    cache_path = data.scene_memory._cache_path
    data.scene_memory._cache_path = lambda *args: (path if (path := cache_path(*args)) is not None and path.exists() else None)
    data.surface_cache = CanonicalSurfaceCache(data, limit=1)
    data.scene_memory.cached_streams = 1
    skin = Native30SkinAnchorCache(device='cpu', limit=4)
    clean = Native30ObservationBuilder(skin, read(BASE / cfg['mount_defaults']), cfg['observation']['input_noise'])
    noisy = Native30ObservationBuilder(skin, clean.mounts, dict(clean.noise_config, clean_probability=0.))
    maximum_errors, direct_errors = collections.defaultdict(float), collections.defaultdict(float)
    checks, stable = [], {}
    aggregate_rows = []
    for entry in cohort['entries']:
        members = [i for i, (meta, _) in enumerate(data.members[entry['group']]) if meta['path'] == entry['path']]
        require(len(members) == 1, 'Nonunique source recording')
        sample, identity = data.sample(entry['group'], 'ego', member=members[0], start=entry['start'], yaw=0)
        require(identity['source_start_30fps'] == entry['source_first'], 'Source frame changed')
        for regime in cohort['regimes']:
            observed, scene, asset = (clean if regime == 'clean' else noisy).build(data, sample, identity, 'ego', (15,),
                noise_rng=None if regime == 'clean' else np.random.default_rng(entry['case_seed']))
            gt_surface = surface_metrics(sample['motion'], sample['motion'], asset, scene)
            subset = [row for row in rows if row['case'] == entry['case'] and row['regime'] == regime]
            for row in subset:
                require(all(row[key] == entry[key] for key in ('dataset', 'sequence_id', 'start', 'source_first', 'case_seed')),
                        'Saved row cohort identity differs')
                require(row['checkpoint_step'] == plan['models'][row['model']]['expected_step'], 'Saved checkpoint step differs')
                require(sha(row['artifact']) == row['artifact_sha256'], 'Saved motion artifact changed')
                with np.load(row['artifact'], allow_pickle=False) as saved:
                    arrays = {key: saved[key] for key in saved.files}
                saved_identity = json.loads(str(arrays['identity_json']))
                require(all(saved_identity[key] == identity[key] for key in ('path', 'start', 'sequence_id', 'source_start_30fps')),
                        'Artifact source identity differs')
                require(str(arrays['model']) == row['model'] and str(arrays['guidance']) == row['guidance'], 'Artifact arm differs')
                require(str(arrays['checkpoint_sha256']) == row['checkpoint_sha256'] and str(arrays['input_sha256']) == row['input_sha256'], 'Artifact provenance differs')
                for key, actual in [('truth', sample['motion']), ('rest', sample['rest']),
                                    ('observations', observed['trajectory']), ('observation_meta', observed['observation_meta'])]:
                    require(np.array_equal(arrays[key], actual), 'Saved source GT/input changed: ' + key)
                digest = array_digest(arrays, ['truth', 'rest', 'observations', 'observation_meta'])
                key = entry['case'], regime
                require(key not in stable or stable[key] == (digest, row['input_sha256']), 'Same-window GT/input differs across models/modes')
                stable[key] = digest, row['input_sha256']
                prediction, truth, rest = (arrays[key] for key in ('prediction', 'truth', 'rest'))
                regenerated = support_metrics(prediction, truth, rest, arrays['observations'], arrays['observation_meta'], asset)
                regenerated.update(extended_metrics(prediction, truth, rest))
                regenerated.update(surface_metrics(prediction, truth, asset, scene))
                require(set(regenerated) == set(row['metrics']), 'Saved metric schema differs')
                for metric, actual in regenerated.items():
                    saved_value = row['metrics'][metric]
                    require((actual is None) == (saved_value is None), 'Metric valid coverage differs: ' + metric)
                    if actual is not None:
                        error = abs(actual - saved_value)
                        maximum_errors[metric] = max(maximum_errors[metric], error)
                        require(error < 1e-3, 'CPU metric differs: ' + metric)
                direct, supplementary, denominators = independent_metrics(prediction, truth, rest, arrays['observations'], arrays['observation_meta'])
                for metric, actual in direct.items():
                    saved_value = row['metrics'][metric]
                    require((actual is None) == (saved_value is None), 'Independent metric coverage differs')
                    if actual is not None:
                        error = abs(actual - saved_value)
                        direct_errors[metric] = max(direct_errors[metric], error)
                        require(error < 1e-3, 'Independent metric differs: ' + metric)
                denominators['support_anchor_samples'] = regenerated.get('support_evaluation_anchor_count', 0)
                checks.append(dict(model=row['model'], case=entry['case'], regime=regime, guidance=row['guidance'],
                    artifact_sha256=row['artifact_sha256'], gt_and_observed_array_sha256=digest,
                    denominators=denominators, supplementary=supplementary))
                extra = dict(support_gt_float_over_50mm_fraction=gt_surface.get('support_float_over_50mm_fraction'),
                    support_gt_penetrate_below_20mm_fraction=gt_surface.get('support_penetrate_below_20mm_fraction'))
                checks[-1]['sparse_support_proxies'] = {key: value for key, value in regenerated.items() if key.startswith('support_')}
                checks[-1]['sparse_support_proxies'].update(extra)
                aggregate_rows.append(dict(model=row['model'], case=entry['case'], dataset=row['dataset'], regime=regime,
                    guidance=row['guidance'], metrics=dict(regenerated, **extra)))
            del observed, scene, asset
        data.arrays.clear()
    by_case = collections.defaultdict(list)
    for check in checks:
        by_case[check['case'], check['regime']].append(check['denominators'])
    require(all(all(item == values[0] for item in values) for values in by_case.values()), 'Same-window denominators differ')
    require(not torch.cuda.is_initialized(), 'Verifier unexpectedly initialized CUDA')
    groups = {}
    group_keys = ('w_mpjpe_mm', 'pa_mpjpe_mm', 'observed_head_position_error_mm',
        'foot_joint_horizontal_speed_gt_low_m_s', 'foot_joint_vertical_speed_gt_low_m_s',
        'foot_joint_y_abs_error_gt_low_mm', 'root_y_signed_error_mm', 'root_y_abs_error_mm',
        'body_joint_vertical_extent_abs_error_mm', 'bilateral_motion_rms_ratio',
        'bilateral_motion_rms_error_mm', 'bilateral_stride_change_abs_error_mm',
        'support_calibrated_absolute_error_mm', 'support_signed_height_mm', 'support_gt_signed_height_mm',
        'support_float_over_50mm_fraction', 'support_penetrate_below_20mm_fraction',
        'support_gt_float_over_50mm_fraction', 'support_gt_penetrate_below_20mm_fraction')
    for label in labels:
        for guidance in ('off', 'on'):
            for dataset in ('all', 'rich', 'nonrich', 'trumans', 'egobody'):
                for regime in cohort['regimes']:
                    subset = [row['metrics'] for row in aggregate_rows if row['model'] == label and
                        row['guidance'] == guidance and row['regime'] == regime and
                        (dataset == 'all' or row['dataset'] == dataset or dataset == 'nonrich' and row['dataset'] != 'rich')]
                    if not subset:
                        continue
                    valid = [item for item in subset if item.get('support_evaluation_anchor_count', 0) > 0]
                    total = sum(item['support_evaluation_anchor_count'] for item in valid)
                    means = {key: statistics.mean(values) if (values := [item[key] for item in subset if item.get(key) is not None]) else None for key in group_keys}
                    pooled = {key: sum(item[key] * item['support_evaluation_anchor_count'] for item in valid) / total if total else None
                        for key in ('support_float_over_50mm_fraction', 'support_penetrate_below_20mm_fraction',
                                    'support_gt_float_over_50mm_fraction', 'support_gt_penetrate_below_20mm_fraction')}
                    groups['/'.join((label, guidance, dataset, regime))] = dict(windows=len(subset),
                        valid_sparse_surface_windows=len(valid), sparse_surface_anchor_samples=total,
                        window_means=means, anchor_pooled_fractions=pooled)
    indexed = {(row['model'], row['case'], row['regime'], row['guidance']): row for row in aggregate_rows}
    physics_paired = {}
    for label in labels:
        for dataset in ('all', 'rich', 'nonrich', 'trumans', 'egobody'):
            for regime in cohort['regimes']:
                pairs = [(indexed[label, row['case'], regime, 'off']['metrics'], row['metrics'])
                    for row in aggregate_rows if row['model'] == label and row['guidance'] == 'on' and row['regime'] == regime and
                    (dataset == 'all' or row['dataset'] == dataset or dataset == 'nonrich' and row['dataset'] != 'rich')]
                if pairs:
                    physics_paired['/'.join((label, dataset, regime))] = dict(windows=len(pairs),
                        delta_on_minus_off={key: statistics.mean(values) if (values := [b[key] - a[key] for a, b in pairs
                            if a.get(key) is not None and b.get(key) is not None]) else None for key in group_keys})
    text = Path(__file__).read_text()
    result = dict(status='independent_cpu_passed', verified_at=now(), models=labels, predictions=len(rows),
        CUDA_initialized=False, optimizer_updates=0, exact_expected_phase_coverage=True,
        source_dependency_sha256_matches=source_matches, plan_sha256=sha(out / 'PLAN.json'),
        maximum_cpu_recomputed_metric_errors=dict(maximum_errors), maximum_independent_formula_errors=dict(direct_errors),
        gt_rest_and_observed_arrays_byte_identical_to_regenerated_sources=True,
        same_window_inputs_and_evaluation_denominators_identical=True, checks=checks,
        groups=groups, within_model_physics_paired_deltas=physics_paired,
        metric_method='Reopened every committed NPZ, independently rebuilt source/observations/native CPU skin assets/visible scene. Recomputed all saved metrics using existing pure evaluators; separately implemented World/PA/foot speed/height/root height/pose extent/bilateral RMS/stride formulas with NumPy after unchanged FK.',
        supplementary_scope='GT low-speed is a speed proxy, not measured contact. Run drift uses >=3 consecutive low-speed intervals, including truncated window-boundary runs. Root-local temporal correlation and stride-vector error diagnose timing; stationary/low-amplitude correlation is null. No new training objective or inference was run.',
        sparse_surface_scope='Float >50mm and penetrate below -20mm are fractions of sparse foot/ankle skin anchors against identical GT-associated reliable visible upward surface patches. Their signs include scan/body fitting bias; GT reference fractions are recomputed and reported. These are not full-mesh floating, collision, or self-collision rates. Window means and anchor-count pooled fractions are separately labeled.',
        input_scope='Saved observed trajectory/meta/rest and full tensor input fingerprints are checked across compared rows. CPU verifier does not regenerate GPU reference tensors or claim byte identity for unsaved GPU intermediates.',
        verifier_script_sha256=hashlib.sha256(text.encode()).hexdigest(), verifier_script=text)
    suffix = labels[0] if len(labels) == 1 else 'all'
    write(out / ('CPU_RECHECK_' + suffix + '.json'), result)
    print(json.dumps({key: value for key, value in result.items() if key not in ('verifier_script', 'checks', 'groups')}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--models', nargs='+', default=['parent'])
    args = parser.parse_args()
    run(args.out.resolve(), args.models)


if __name__ == '__main__':
    main()
