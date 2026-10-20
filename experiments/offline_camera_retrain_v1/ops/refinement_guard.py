"""CPU diagnostics for limiting EXO local-rotation changes from its reference.

This is an inference diagnostic, not a trained model or a guarantee of lower
error. It uses no truth, contact label, or scene information. An optional
observation mask protects every measured joint position and global rotation.
The native motion layout is root translation / 2 (3), identity-centred local
rotation columns (22 * 6), and global FK joints / 2 (22 * 3). Units for rest
joints and FK output are metres, as in ``supervision.forward_kinematics``.

Only non-root local rotations outside the angular limit are changed. Prediction
root channels are preserved exactly. When something changes, FK rebuilds all
position channels with the same rest skeleton; otherwise the prediction is
copied verbatim. Existing sealed training/sampling modules do not import this.
"""
import numbers

import numpy as np
from scipy.spatial.transform import Rotation


PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
           16, 17, 18, 19)
IDENTITY_6D = np.array([1., 0., 0., 0., 1., 0.])


def _motion_array(value, name):
    result = np.asarray(value)
    if (result.ndim not in (2, 3) or result.shape[-1] != 201
            or any(size == 0 for size in result.shape[:-1])):
        raise ValueError(f'{name} must have shape [T,201] or [B,T,201] with nonempty axes')
    if not np.issubdtype(result.dtype, np.floating) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain finite floating-point values')
    return result


def review_saved(run, output, *, step=8000, panels=('fixed', 'rotating'), caps=(30., 60., 90.),
                 scope='all'):
    """Paired CPU review of completed saved VAL predictions, never training.

    Truth is used only for reporting. The projection itself never receives it.
    This output-only diagnostic does not replay DDIM or observation IK.
    """
    import hashlib
    import json
    import resource
    from datetime import datetime, timezone
    from pathlib import Path

    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Review output must be new or empty')
    if not panels or set(panels) - {'fixed', 'rotating'} or len(set(panels)) != len(panels):
        raise ValueError('Choose unique fixed/rotating panels')
    if scope not in ('all', 'unobserved'):
        raise ValueError('scope must be all or unobserved')
    caps = tuple(float(v) for v in caps)
    if not caps or len(set(caps)) != len(caps) or any(not 0 <= v <= 180 for v in caps):
        raise ValueError('Choose unique finite caps in [0,180]')

    def digest(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()

    inputs = {}
    manifest = run / 'prepared/panels.json'
    inputs[str(manifest)] = digest(manifest)
    sources = {}
    for panel in panels:
        folder = run / (panel + '_validation') / f'step_{step:06d}'
        metrics, arrays = folder / 'validation.jsonl', folder / 'validation_latest.npz'
        for path in (metrics, arrays):
            inputs[str(path)] = digest(path)
        content = json.loads(metrics.read_text().splitlines()[-1])
        if content['step'] != step:
            raise ValueError('Saved metric stage differs')
        selected = [row for row in content['cases']
                    if row['mode'] == 'exo' and row['panel'] == panel]
        if len(selected) != 200 or len({(r['case'], r['regime']) for r in selected}) != 200:
            raise ValueError('Require a complete official 100-window clean/noisy panel')
        sources[panel] = arrays, selected
    output.mkdir(parents=True, exist_ok=True)
    rows, maximum_recompute_difference = [], 0.

    def evaluate(joints, truth, visible):
        error = np.linalg.norm(joints - truth, axis=-1) * 1000
        speed = np.linalg.norm(np.diff(joints[:, [10, 11]], axis=0) * 30, axis=-1)
        stance = np.linalg.norm(np.diff(truth[:, [10, 11]], axis=0) * 30, axis=-1) < .2
        relative = joints - joints[:, :1] - truth + truth[:, :1]
        hidden = ~visible
        residual = joints - truth
        return dict(hidden_mm=float(error[hidden].mean()) if hidden.any() else None,
                    visible_mm=float(error[visible].mean()) if visible.any() else None,
                    all_mm=float(error.mean()), hidden_points=int(hidden.sum()),
                    root_relative_hidden_mm=float(np.linalg.norm(relative, axis=-1)[hidden].mean() * 1000)
                    if hidden.any() else None,
                    mpjve_mm_s=float(np.linalg.norm(np.diff(residual, axis=0) * 30, axis=-1).mean() * 1000)
                    if len(joints) > 1 else None,
                    acceleration_error_mm_s2=float(np.linalg.norm(np.diff(residual, n=2, axis=0) * 900,
                                                                  axis=-1).mean() * 1000)
                    if len(joints) > 2 else None,
                    stance_foot_speed_m_s=float(speed[stance].mean()) if stance.any() else None)

    for panel, (archive, selected) in sources.items():
        with np.load(archive, allow_pickle=False) as saved:
            if int(saved['step']) != step:
                raise ValueError('Saved array stage differs')
            for row in selected:
                index, regime = row['case'], row['regime']
                prefix = f'{index}_{regime}_'
                pred, ref = saved[prefix + 'prediction'], saved[prefix + 'reference_motion']
                truth, rest = saved[f'{index}_truth'], saved[f'{index}_rest']
                observation_meta = saved[prefix + 'observation_meta']
                visible = observation_meta[..., 0] > 0
                eligible = (unobserved_rotation_mask(observation_meta) if scope == 'unobserved'
                            else np.broadcast_to(np.arange(22) != 0, (*pred.shape[:-1], 22)))
                gt = forward_kinematics_numpy(truth, rest)
                base = evaluate(forward_kinematics_numpy(pred, rest), gt, visible)
                ref_metrics = evaluate(forward_kinematics_numpy(ref, rest), gt, visible)
                for name, computed in (('unconditioned_mpjpe_mm', 'hidden_mm'),
                                       ('conditioned_mpjpe_mm', 'visible_mm')):
                    expected = row['metrics'][name]['value']
                    if (expected is None) != (base[computed] is None):
                        raise ValueError('Saved metric missingness differs')
                    if expected is not None:
                        difference = abs(expected - base[computed])
                        maximum_recompute_difference = max(maximum_recompute_difference, difference)
                        if difference > .01:
                            raise ValueError('Saved metric recomputation differs by >0.01mm')
                gt_local, ref_local = motion_rotations(truth), motion_rotations(ref)
                relative = np.swapaxes(ref_local[..., 1:, :, :], -1, -2) @ gt_local[..., 1:, :, :]
                required = np.rad2deg(Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude())
                identity = {k: row[k] for k in ('case', 'dataset', 'recording', 'sequence_id',
                                               'source_first', 'start', 'case_seed', 'regime', 'panel')}
                for cap in (None, *caps):
                    candidate = project_local_rotations(ref, pred, rest, cap,
                        observation_meta=observation_meta if scope == 'unobserved' else None)
                    if not np.array_equal(candidate[..., :9], pred[..., :9]):
                        raise ValueError('Projection changed root channels')
                    values = base if cap is None else evaluate(forward_kinematics_numpy(candidate, rest), gt, visible)
                    rows.append(dict(identity, cap_deg=cap, scope=scope, metrics=values, baseline=base,
                        eligible_nonroot_joint_frames=int(eligible[..., 1:].sum()),
                        total_nonroot_joint_frames=int(eligible[..., 1:].size),
                        reference=ref_metrics,
                        required_large_rotation_fraction=float((required > cap).mean()) if cap is not None else None))
    groups = []
    for panel in panels:
        for dataset in ('rich', 'egobody', 'trumans'):
            for regime in ('clean', 'noisy'):
                for cap in (None, *caps):
                    items = [r for r in rows if (r['panel'], r['dataset'], r['regime'], r['cap_deg'])
                             == (panel, dataset, regime, cap)]
                    group = dict(panel=panel, dataset=dataset, regime=regime, cap_deg=cap, scope=scope,
                                 windows=len(items))
                    for metric in ('hidden_mm', 'visible_mm', 'all_mm', 'root_relative_hidden_mm',
                                   'stance_foot_speed_m_s', 'mpjve_mm_s', 'acceleration_error_mm_s2'):
                        valid = [r for r in items if r['metrics'][metric] is not None]
                        values = [r['metrics'][metric] for r in valid]
                        group[metric] = dict(mean=float(np.mean(values)) if values else None,
                            median=float(np.median(values)) if values else None,
                            p90=float(np.quantile(values, .9)) if values else None, valid_windows=len(valid))
                    hidden = [r for r in items if r['metrics']['hidden_mm'] is not None]
                    group['hidden_worse_than_reference'] = sum(r['metrics']['hidden_mm'] > r['reference']['hidden_mm'] for r in hidden)
                    group['hidden_worse_than_reference_over50mm'] = sum(r['metrics']['hidden_mm'] > r['reference']['hidden_mm'] + 50 for r in hidden)
                    group['hidden_worse_than_baseline_over50mm'] = sum(r['metrics']['hidden_mm'] > r['baseline']['hidden_mm'] + 50 for r in hidden)
                    large = [r for r in hidden if cap is not None and r['required_large_rotation_fraction'] >= .05]
                    group['large_GT_correction_subset'] = dict(windows=len(large), definition='At least5% of nonroot joint-frames require reference-to-GT rotation larger than cap; GT is used only in reporting.',
                        hidden_mean_mm=float(np.mean([r['metrics']['hidden_mm'] for r in large])) if large else None,
                        baseline_hidden_mean_mm=float(np.mean([r['baseline']['hidden_mm'] for r in large])) if large else None,
                        worsened_windows=sum(r['metrics']['hidden_mm'] > r['baseline']['hidden_mm'] for r in large))
                    for metric, absolute, proportion in (('visible_mm', 1., .02), ('stance_foot_speed_m_s', .01, .05)):
                        valid = [r for r in items if r['metrics'][metric] is not None]
                        old = float(np.mean([r['baseline'][metric] for r in valid])) if valid else None
                        new = group[metric]['mean']
                        group[metric + '_guard_pass'] = None if old is None else new <= old + max(absolute, old * proportion)
                    groups.append(group)
    for path, expected in inputs.items():
        if digest(path) != expected:
            raise ValueError('Review input changed during computation: ' + path)
    result = dict(protocol='saved-exo-reference-SO3-cap-review-v1', status='completed', scope=scope,
        recorded_at=datetime.now(timezone.utc).isoformat(), step=step, panels=list(panels), caps_deg=list(caps),
        eligibility=('All nonroot local joints.' if scope == 'all' else
            'Only local joints with no effective observed position in strict descendants and no effective '
            'observed global orientation in inclusive subtree; availability and confidence must be finite and >0.'),
        source_sha256={str(Path(__file__).resolve()): digest(__file__)}, inputs_sha256=inputs,
        optimizer_updates=0, inference_calls=0, cuda_initialized=False,
        peak_cpu_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        maximum_saved_metric_recompute_difference_mm=maximum_recompute_difference,
        aggregation='Equal valid-window means; panel/dataset/regime separate. P90 is the90th percentile of window means.',
        limitations=['Saved-output projection, not DDIM-internal constraint or a newly trained model.',
                     'Root unchanged; observed joint IK is not rerun. Visible-joint and foot guards must be assessed.',
                     'All caps predeclared for diagnosis; no automatic model promotion or checkpoint selection.',
                     'Per-frame eligibility may switch over time and create temporal discontinuities; velocity and acceleration errors are reported.',
                     'EGO excluded and unchanged by design; this review modifies EXO outputs only.'],
        groups=groups, rows=rows)
    (output / 'REVIEW.json').write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    lines = ['# EXO 参考旋转修正范围 CPU 对照', '',
             f'阶段{step}；范围{scope}；仅处理已保存的 EXO 输出；不训练、不重跑 DDIM/IK，不自动选最终方案。', '',
             '| 面板 | 数据集 | 输入 | 上限° | hidden mm | visible mm | visible守卫 | foot守卫 |',
             '|---|---|---|---:|---:|---:|---|---|']
    for group in groups:
        lines.append('| ' + ' | '.join((group['panel'], group['dataset'], group['regime'],
            '原输出' if group['cap_deg'] is None else str(group['cap_deg']),
            f"{group['hidden_mm']['mean']:.2f}", f"{group['visible_mm']['mean']:.2f}",
            str(group['visible_mm_guard_pass']), str(group['stance_foot_speed_m_s_guard_pass']))) + ' |')
    lines.extend(['', '真实需要大幅修正的子集、P90、逐窗退化数、源身份与哈希见 REVIEW.json。',
                  '范围相对于参考局部旋转，不是限制膝盖的绝对弯曲角。GT仅用于指标与分组，投影函数不读取GT。',
                  'unobserved仅使用输入观测元数据保护可见位置和全局旋转；按帧mask切换可能引入跳动，速度和加速度误差已记录。', ''])
    (output / 'REVIEW_zh.md').write_text('\n'.join(lines))
    return result


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--step', type=int, default=8000)
    parser.add_argument('--panels', nargs='+', choices=('fixed', 'rotating'), default=('fixed', 'rotating'))
    parser.add_argument('--caps', nargs='+', type=float, default=(30., 60., 90.))
    parser.add_argument('--scope', choices=('all', 'unobserved'), default='all')
    args = parser.parse_args(argv)
    result = review_saved(args.run, args.output, step=args.step, panels=args.panels, caps=args.caps,
                          scope=args.scope)
    print(json.dumps({key: result[key] for key in ('status', 'step', 'peak_cpu_rss_mib',
                                                  'maximum_saved_metric_recompute_difference_mm')}))


def _rest_array(rest, motion):
    result = np.asarray(rest)
    expected = (22, 3) if motion.ndim == 2 else (motion.shape[0], 22, 3)
    if result.shape not in ((22, 3), expected):
        raise ValueError(f'rest must have shape [22,3] or {expected}')
    if (not np.issubdtype(result.dtype, np.number)
            or np.issubdtype(result.dtype, np.complexfloating)
            or not np.isfinite(result).all()):
        raise ValueError('rest must contain finite real values')
    if motion.ndim == 3 and result.ndim == 2:
        result = np.broadcast_to(result, expected)
    return result.astype(np.float64, copy=False)


def motion_rotations(motion):
    """Decode native identity-centred rotation columns into [...,22,3,3].

    Finite but degenerate 6D rotations are rejected instead of silently
    replacing them by identity. All numerical geometry is computed in float64.
    """
    motion = _motion_array(motion, 'motion')
    six = motion[..., 3:135].astype(np.float64).reshape(*motion.shape[:-1], 22, 6)
    six = six + IDENTITY_6D
    first_length = np.linalg.norm(six[..., :3], axis=-1, keepdims=True)
    if np.any(first_length <= 1e-6):
        raise ValueError('motion contains degenerate first rotation columns')
    first = six[..., :3] / first_length
    second = six[..., 3:] - np.sum(first * six[..., 3:], axis=-1, keepdims=True) * first
    second_length = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_length <= 1e-6):
        raise ValueError('motion contains degenerate second rotation columns')
    second = second / second_length
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def forward_kinematics_numpy(motion, rest):
    """Native 22-joint FK in metres, accepting [T,201] or [B,T,201]."""
    motion = _motion_array(motion, 'motion')
    rest = _rest_array(rest, motion)
    local = motion_rotations(motion)
    # Insert the time axis for batched rest joints, and broadcast a shared rest
    # skeleton across all frames for unbatched motion.
    temporal_rest = rest if motion.ndim == 2 else rest[:, None]
    joints = np.empty((*motion.shape[:-1], 22, 3), dtype=np.float64)
    rotations = np.empty_like(local)
    rotations[..., 0, :, :] = local[..., 0, :, :]
    joints[..., 0, :] = motion[..., :3].astype(np.float64) * 2 + temporal_rest[..., 0, :]
    for joint in range(1, 22):
        parent = PARENTS[joint]
        offset = temporal_rest[..., joint, :] - temporal_rest[..., parent, :]
        parent_rotation = rotations[..., parent, :, :]
        joints[..., joint, :] = (joints[..., parent, :]
                                + np.einsum('...ij,...j->...i', parent_rotation, offset))
        rotations[..., joint, :, :] = parent_rotation @ local[..., joint, :, :]
    return joints


def unobserved_rotation_mask(observation_meta):
    """Find local rotations that cannot change any effective observation.

    Accept [T,22,C] or [B,T,22,C], C >= 4. Metadata channels are position
    availability, rotation availability, position confidence, and rotation
    confidence. A measurement is effective only when both its availability
    and confidence are finite and strictly positive. This matches the input
    metadata contract, without consulting truth or a reference's confidence.

    A local joint's rotation affects positions in its *strict descendants*,
    but global rotations throughout its *inclusive subtree*. Self-position
    alone therefore does not block a local rotation. Root is always blocked.
    This guarantee concerns FK from motion rotations and rest joints, within
    numerical precision; it does not assert consistency of saved joint channels.
    Per-frame changes in eligibility can cause temporal discontinuities.
    """
    meta = np.asarray(observation_meta)
    if (meta.ndim not in (3, 4) or meta.shape[-2] != 22 or meta.shape[-1] < 4
            or any(size == 0 for size in meta.shape[:-2])):
        raise ValueError('observation_meta must have shape [T,22,C] or [B,T,22,C], C>=4')
    if (not np.issubdtype(meta.dtype, np.number)
            or np.issubdtype(meta.dtype, np.complexfloating)):
        raise ValueError('observation_meta must contain real numeric values')
    pm, rm, pc, rc = (meta[..., channel] for channel in range(4))
    position = np.isfinite(pm) & np.isfinite(pc) & (pm > 0) & (pc > 0)
    orientation = np.isfinite(rm) & np.isfinite(rc) & (rm > 0) & (rc > 0)
    descendants = np.zeros((22, 22), dtype=bool)
    for joint in range(1, 22):
        parent = PARENTS[joint]
        while parent >= 0:
            descendants[parent, joint] = True
            parent = PARENTS[parent]
    inclusive = descendants | np.eye(22, dtype=bool)
    blocked = (np.any(position[..., None, :] & descendants, axis=-1)
               | np.any(orientation[..., None, :] & inclusive, axis=-1))
    eligible = ~blocked
    eligible[..., 0] = False
    return eligible


def project_local_rotations(reference, prediction, rest, max_angle_deg, *, observation_meta=None):
    """Project non-root local rotations to a reference-centred SO(3) ball.

    For each local joint, let D = R_reference.T @ R_prediction. The output is
    R_reference @ Exp(min(1, limit / |Log(D)|) * Log(D)), using the shortest
    SO(3) geodesic. At exactly 180 degrees the shortest axis is not unique.

    ``max_angle_deg`` is None (disabled), or a finite real scalar in [0,180].
    Disabled and already-within-limit inputs return an independent, bitwise
    identical copy of prediction, including its position channels. If any
    joint is projected, its 6D columns are replaced and position channels are
    regenerated by FK. Prediction dtype, translation, root 6D rotation, and
    all unprojected 6D channels are preserved. Inputs are never modified.

    If observation_meta is supplied, only local rotations returned by
    unobserved_rotation_mask are eligible. This preserves all effective
    observed FK joint positions and global rotations relative to prediction.
    It does not enforce agreement with the observed values themselves.

    Zero permits reference local rotations with prediction root pose; it
    does not lock any joint to ground or prevent root drift. A fixed limit
    may suppress a necessary large correction, so evaluate both improvements
    and regressions rather than selecting a limit on individual bad cases.
    """
    reference = _motion_array(reference, 'reference')
    prediction = _motion_array(prediction, 'prediction')
    if reference.shape != prediction.shape:
        raise ValueError('reference and prediction shapes must match')
    rest = _rest_array(rest, prediction)
    eligible = None
    if observation_meta is not None:
        eligible = unobserved_rotation_mask(observation_meta)
        if eligible.shape != (*prediction.shape[:-1], 22):
            raise ValueError('observation_meta leading axes must match prediction')
    if max_angle_deg is not None:
        if (isinstance(max_angle_deg, (bool, np.bool_))
                or not isinstance(max_angle_deg, numbers.Real)
                or not np.isfinite(max_angle_deg) or not 0 <= max_angle_deg <= 180):
            raise ValueError('max_angle_deg must be None or a finite real scalar in [0,180]')
    result = prediction.copy()
    if max_angle_deg is None:
        return result

    reference_rotations = motion_rotations(reference)[..., 1:, :, :]
    prediction_rotations = motion_rotations(prediction)[..., 1:, :, :]
    relative = np.swapaxes(reference_rotations, -1, -2) @ prediction_rotations
    vectors = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec()
    vectors = vectors.reshape(*prediction.shape[:-1], 21, 3)
    angles = np.linalg.norm(vectors, axis=-1)
    limit = np.deg2rad(float(max_angle_deg))
    # A tiny numerical equality tolerance avoids reconstructing identical
    # rotations at cap=0. The limit tolerance is below float32 output accuracy.
    projected = angles > limit + 1e-12
    if eligible is not None:
        projected &= eligible[..., 1:]
    if not projected.any():
        return result

    if limit == 0:
        new_six = reference[..., 3:135].reshape(*reference.shape[:-1], 22, 6)[..., 1:, :]
    else:
        bounded = vectors[projected] * (limit / angles[projected, None])
        relative_bounded = Rotation.from_rotvec(bounded).as_matrix()
        matrices = reference_rotations[projected] @ relative_bounded
        new_six = np.concatenate((matrices[..., :, 0], matrices[..., :, 1]), axis=-1) - IDENTITY_6D
    output_six = result[..., 3:135].reshape(*result.shape[:-1], 22, 6)[..., 1:, :]
    output_six[projected] = new_six[projected] if limit == 0 else new_six
    result[..., 135:] = (forward_kinematics_numpy(result, rest) / 2).reshape(*result.shape[:-1], 66)
    return result


if __name__ == '__main__':
    main()
