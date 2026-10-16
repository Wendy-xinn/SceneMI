"""Manifest-driven, resumable frozen-weight motion constraint experiments.

Interventions affect this evaluator's model instance only. Training sources,
weights, inputs, DDIM seeds and evaluation masks remain fixed across arms.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
FULL = BASE / 'runs/native30_live_monitor_oct16/rich_full_validation_oct16'
DEFAULT_OUT = BASE / 'runs/native30_live_monitor_oct16/support_ablation'
sys.path.insert(0, str(REPO))


def now():
    return dt.datetime.now().astimezone().isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(out):
    manifest = BASE / 'runs/native30_live_monitor_oct16/visuals/viser/PREPARED_CASES.json'
    preview = read(manifest)
    validations = [json.loads(line) for line in (BASE / 'runs/native30_gait_v2_ego55k_oct15/training/validation.jsonl').read_text().splitlines()]
    entries = []
    for index in (0, 4, 2):
        item = preview['cases'][index]
        identity = read(item['cache_manifest'])['identity']
        matches = [c for row in validations if row['step'] == item['step'] for c in row['cases']
                   if c['sequence_id'] == identity['sequence_id'] and c['start'] == identity['start']]
        seed = matches[0]['case_seed'] if matches else 2026101650 + len(entries)
        entries.append(dict(case=len(entries), label=item['label'], dataset=identity['dataset'],
            recording=identity.get('recording', identity['sequence_id']), group=identity['group'],
            sequence_id=identity['sequence_id'], path=identity['path'], start=identity['start'],
            source_first=identity['source_start_30fps'], case_seed=seed,
            source_cache=item['cache'], source_cache_manifest=item['cache_manifest'],
            source_manifest_sha256=sha(item['cache_manifest'])))
    full = read(FULL / 'PLAN.json')
    for name in ('ParkingLot1_004_pushup1', 'Pavallion_000_yoga1'):
        source = next(e for e in full['plan'] if e['sequence_id'] == name and e['start'] == 0)
        entries.append(dict(source, case=len(entries), label=name))
    zero = dict(gravity_weight=0.)
    weak = dict(gravity_weight=.15, gravity_deadzone_rad=.17453292519943295,
                gravity_step_limit_rad=.03490658503988659, minimum_signal=.5)
    arms = dict(
        baseline=dict(observation_anchor_gain=.9, constraints=zero),
        ik_only=dict(observation_anchor_gain=0., constraints=zero),
        gravity=dict(observation_anchor_gain=.9, constraints=weak),
        ik_gravity=dict(observation_anchor_gain=0., constraints=weak))
    dependencies = [Path(__file__), Path(__file__).with_name('support_constraints.py'),
                    Path(__file__).with_name('support_metrics.py'), FULL / 'FROZEN_CHECKPOINT.json',
                    FULL / 'runner.py', BASE / 'runs/native30_final_review_oct16/io_recovery.py']
    plan = dict(protocol='frozen-motion-constraints-v1', created_at=now(),
        checkpoint=str(FULL / 'checkpoint_eval.pt'), checkpoint_sha256=read(FULL / 'FROZEN_CHECKPOINT.json')['evaluation_checkpoint_sha256'],
        checkpoint_step=15100, fps=30, frames=192, ddim_steps=20,
        regimes=['clean', 'noisy'], entries=entries, arms=arms, planned_predictions=len(entries) * 2 * len(arms),
        selection='Three reported failure/walking windows, plus two preselected non-upright RICH counterexamples. Diagnostic VAL selection, not unbiased benchmark.',
        intervention_order='Optional weak root gravity alignment, then original constrain_motion (same configured six IK iterations); scene, learned feedback and physical guidance unchanged.',
        gravity_scope='Root pitch/roll alignment proxy; not Newtonian dynamics or physical contact simulation. No inferred common floor.',
        targets='GT and GT low-speed masks are evaluation-only, excluded from model inference payload and constraint function.',
        dependency_sha256={str(path): sha(path) for path in dependencies},
        source_manifest=str(manifest), source_manifest_sha256=sha(manifest))
    if (out / 'PLAN.json').exists():
        old = read(out / 'PLAN.json')
        plan['created_at'] = old['created_at']
        require(plan == old, 'Existing immutable plan differs')
    else:
        write(out / 'PLAN.json', plan)
    return plan


def summarize(out, rows, plan, complete=False):
    keys = ['mpjpe_mm', 'observed_head_position_error_mm', 'foot_joint_horizontal_speed_gt_low_m_s',
            'foot_joint_y_abs_error_gt_low_mm', 'root_tilt_abs_error_deg',
            'support_calibrated_absolute_error_mm', 'support_float_over_50mm_fraction']
    summary = dict(status='complete' if complete else 'partial', checked_at=now(), rows=len(rows),
                   planned_predictions=plan['planned_predictions'], checkpoint_step=plan['checkpoint_step'], regimes={})
    lines = ['# 头部锚定与重力方向：冻结权重配对实验', '',
        '**' + ('完整诊断队列已完成。' if complete else '部分结果，尚不可据此选择最终方案。') + '**', '',
        '同一15100步权重、相同输入/seed/DDIM20。baseline原设置；ik_only去整身头平移；gravity加弱root重力对齐；ik_gravity同时两项。所有组保留原场景路径和物理修正。', '',
        '选取已知失败/走动窗口与非直立反例，属于开发诊断，不能当总体评估。GT只用于指标；低速是代理，未验证楼梯。', '']
    for regime in plan['regimes']:
        values = {}
        lines += ['## ' + regime, '', '| 方案 | 窗数 | MPJPE mm | 头对输入 mm | GT低速期水平足速 m/s |', '|---|---:|---:|---:|---:|']
        for arm in plan['arms']:
            subset = [r for r in rows if r['regime'] == regime and r['arm'] == arm]
            metrics = {k: statistics.mean(v) if (v := [r['metrics'][k] for r in subset if r['metrics'].get(k) is not None]) else None for k in keys}
            values[arm] = dict(windows=len(subset), metrics=metrics)
            display = ['—' if metrics[k] is None else f'{metrics[k]:.3f}' for k in keys[:3]]
            lines.append('| ' + arm + ' | ' + str(len(subset)) + ' | ' + ' | '.join(display) + ' |')
        summary['regimes'][regime] = values
        lines.append('')
    write(out / 'SUMMARY.json', summary)
    (out / 'REPORT_zh.md').write_text('\n'.join(lines) + '\n')


def surface_metrics(prediction, truth, asset, scene):
    """Common GT-associated local patches used only after generation."""
    import numpy as np
    import torch
    from scipy.spatial import cKDTree
    from experiments.offline_camera_retrain_v1.unified_geometry import skin_points
    while not hasattr(scene, 'static') and hasattr(scene, 'base'):
        scene = scene.base
    static = scene.static
    valid = static['reliable'] & (static['normals'][:, 1] > .9) & (static['owners'] == 0)
    cloud, normals = static['points'][valid], static['normals'][valid]
    labels = asset['labels'].detach().cpu().numpy()
    feet = np.isin(labels, [7, 8, 10, 11])
    cpu_asset = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in asset.items()}
    with torch.no_grad():
        p = skin_points(torch.from_numpy(prediction), cpu_asset).numpy()[:, feet]
        g = skin_points(torch.from_numpy(truth), cpu_asset).numpy()[:, feet]
    if not len(cloud):
        return dict(support_evaluation_anchor_count=0)
    distance, ids = cKDTree(cloud).query(g)
    point, normal = cloud[ids], normals[ids]
    gd = g - point
    signed_gt = (gd * normal).sum(-1)
    tangent = np.linalg.norm(gd - signed_gt[..., None] * normal, axis=-1)
    common = (distance < .25) & (tangent < .12) & (np.abs(signed_gt) < .2)
    ps = ((p - point) * normal).sum(-1)
    if not common.any():
        return dict(support_evaluation_anchor_count=0)
    return dict(support_evaluation_anchor_count=int(common.sum()),
        support_calibrated_absolute_error_mm=float(np.abs(ps - signed_gt)[common].mean() * 1000),
        support_signed_height_mm=float(ps[common].mean() * 1000),
        support_float_over_50mm_fraction=float((ps[common] > .05).mean()),
        support_penetrate_below_20mm_fraction=float((ps[common] < -.02).mean()),
        support_gt_signed_height_mm=float(signed_gt[common].mean() * 1000))


def run(out, plan, memory_fraction):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model, window_options
    from experiments.offline_camera_retrain_v1.native30_data import Native30Data
    from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
    from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, sample_motion
    from experiments.offline_camera_retrain_v1 import native30_surface_cache as surfaces, native30_memory as memory
    from experiments.offline_camera_retrain_v1.native30_surface_cache import CanonicalSurfaceCache
    from .support_constraints import apply_support_constraints
    from .support_metrics import support_metrics

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    helper = load_module('constraint_original_eval_helpers', FULL / 'runner.py')
    effective = helper.verify_sources()
    cfg = effective['config']
    for path, digest in plan['dependency_sha256'].items():
        require(sha(path) == digest, 'Experiment source changed: ' + path)
    require(sha(plan['checkpoint']) == plan['checkpoint_sha256'], 'Checkpoint file changed')
    state = torch.load(plan['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    require(state['step'] == plan['checkpoint_step'] and state['effective_config'] == effective, 'Checkpoint identity differs')
    before = helper.model_tensor_digest(state['model'])
    model = make_model(cfg).cuda().eval()
    model.load_state_dict(state['model'], strict=True)
    del state
    original_constrain, original_forward = model.constrain_motion, model.forward
    active = {}
    diagnostics = collections.Counter()

    def constrained(motion, timestep, batch):
        motion = apply_support_constraints(motion, batch, model.alphas[timestep], active['constraints'])
        return original_constrain(motion, timestep, batch)

    def monitored(*args, **kwargs):
        requested = kwargs.pop('return_details', False)
        result, details = original_forward(*args, return_details=True, **kwargs)
        diagnostics['ddim_forward_calls'] += 1
        for key in ('physical_accepted_steps', 'physical_energy_before', 'physical_energy_after'):
            if key in details:
                diagnostics[key] += float(details[key].detach().sum())
        if 'observation_root_delta_m' in details:
            diagnostics['head_anchor_root_delta_norm_sum_m'] += float(details['observation_root_delta_m'].detach().norm(dim=-1).sum())
        return (result, details) if requested else result

    model.constrain_motion, model.forward = constrained, monitored
    io = load_module('constraint_crc_reads', BASE / 'runs/native30_final_review_oct16/io_recovery.py')
    surfaces.np = io.NumpyProxy(surfaces.np, out / 'IO_READS.jsonl')
    memory.np = io.NumpyProxy(memory.np, out / 'IO_READS.jsonl')
    data = Native30Data(BASE / cfg['train_manifest'], 'validation', cfg['seed'] + 1, frames=cfg['frames'],
        memory_protocol=cfg['scene']['memory_protocol'], quality_config=cfg['gt_reliability'],
        exo_max_empty_frames=cfg['exo_max_empty_frames'], **window_options(cfg))
    data.runtime_cache_root = out / 'native30_runtime_cache_v2'
    original_cache_path = data.scene_memory._cache_path
    data.scene_memory._cache_path = lambda *args: (p if (p := original_cache_path(*args)) is not None and p.exists() else None)
    data.surface_cache = CanonicalSurfaceCache(data, limit=1)
    data.scene_memory.cached_streams = 1
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(), read(BASE / cfg['mount_defaults']), cfg['observation']['input_noise'])
    noisy = Native30ObservationBuilder(builder.skin_cache, builder.mounts, dict(builder.noise_config, clean_probability=0.))
    protocol = dict(status='running', pid=os.getpid(), started_at=now(), plan_sha256=sha(out / 'PLAN.json'),
        checkpoint_step=plan['checkpoint_step'], model_tensor_digest_before=before,
        training_modified=False, optimizer_updates=0, resources_before=helper.resources())
    write(out / 'PROTOCOL.json', protocol)
    rows = [json.loads(line) for line in (out / 'rows.jsonl').read_text().splitlines()] if (out / 'rows.jsonl').exists() else []
    done = {(r['case'], r['regime'], r['arm']): r for r in rows}
    require(len(done) == len(rows), 'Duplicate saved rows')
    for row in rows:
        require(sha(row['artifact']) == row['artifact_sha256'], 'Saved artifact changed')
    (out / 'motions').mkdir(exist_ok=True)
    for entry in plan['entries']:
        matches = [i for i, (meta, _) in enumerate(data.members[entry['group']]) if meta['path'] == entry['path']]
        require(len(matches) == 1, 'Nonunique dataset source')
        sample, identity = data.sample(entry['group'], 'ego', member=matches[0], start=entry['start'], yaw=0)
        require(identity['sequence_id'] == entry['sequence_id'] and identity['source_start_30fps'] == entry['source_first'], 'Sample identity differs')
        if 'source_cache' in entry:
            with np.load(entry['source_cache'], allow_pickle=False) as source:
                require(np.array_equal(source['truth'], sample['motion']) and np.array_equal(source['rest'], sample['rest']), 'Preview GT/body identity differs')
        for regime in plan['regimes']:
            observed, scene, asset = (builder if regime == 'clean' else noisy).build(data, sample, identity, 'ego', (15,),
                noise_rng=None if regime == 'clean' else np.random.default_rng(entry['case_seed']))
            batch = collate_observations([(observed, scene, asset, 'ego')])
            known = model_inputs(cfg, batch)
            require(not any(key in known for key in ('motion', 'joints', 'contact_target', 'gt_position_weight', 'support_points')), 'Targets leaked into inference')
            fingerprint = hashlib.sha256()
            for key in ('trajectory', 'observation_meta', 'observation_uncertainty', 'reference_motion', 'reference_confidence'):
                fingerprint.update(known[key].detach().cpu().contiguous().numpy().tobytes())
            input_sha = fingerprint.hexdigest()
            for arm, settings in plan['arms'].items():
                key = entry['case'], regime, arm
                if key in done:
                    require(done[key]['input_sha256'] == input_sha, 'Resumed input differs')
                    continue
                active.clear()
                active.update(settings)
                model.observation_anchor_gain = settings['observation_anchor_gain']
                diagnostics.clear()
                write(out / 'STATUS.json', dict(status='evaluating', checked_at=now(), pid=os.getpid(),
                    case=entry['case'], sequence_id=entry['sequence_id'], regime=regime, arm=arm,
                    completed_predictions=len(rows), planned_predictions=plan['planned_predictions'], resources=helper.resources()))
                start = time.monotonic()
                with torch.no_grad():
                    result = sample_motion(model, known, cfg, steps=plan['ddim_steps'], seed=entry['case_seed'], use_scene=True)
                torch.cuda.synchronize()
                prediction = result[0].detach().cpu().numpy()
                del result
                require(np.isfinite(prediction).all(), 'Nonfinite prediction')
                elapsed = time.monotonic() - start
                artifact = out / 'motions' / f"case{entry['case']:02d}_{arm}_{regime}.npz"
                temporary = artifact.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, prediction=prediction, truth=sample['motion'], rest=sample['rest'],
                        observations=observed['trajectory'], observation_meta=observed['observation_meta'],
                        reference=known['reference_motion'][0].detach().cpu().numpy(),
                        identity_json=np.asarray(json.dumps(identity)), step=np.asarray(plan['checkpoint_step']),
                        arm=np.asarray(arm), input_sha256=np.asarray(input_sha))
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(artifact)
                metrics = support_metrics(prediction, sample['motion'], sample['rest'], observed['trajectory'], observed['observation_meta'], asset)
                metrics.update(surface_metrics(prediction, sample['motion'], asset, scene))
                # Reopen the persisted prediction for independent CPU kinematic verification.
                with np.load(artifact, allow_pickle=False) as saved:
                    verified = support_metrics(saved['prediction'], saved['truth'], saved['rest'], saved['observations'], saved['observation_meta'], asset)
                require(all(verified[k] == metrics[k] for k in verified), 'Persisted CPU metrics differ')
                row = dict(case=entry['case'], sequence_id=entry['sequence_id'], dataset=entry['dataset'], regime=regime,
                    arm=arm, checkpoint_step=plan['checkpoint_step'], input_sha256=input_sha,
                    artifact=str(artifact), artifact_sha256=sha(artifact), metrics=metrics,
                    diagnostics=dict(diagnostics), ddim_seconds=elapsed, checked_at=now())
                with (out / 'rows.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                rows.append(row)
                done[key] = row
                summarize(out, rows, plan)
                print(json.dumps(dict(case=entry['case'], regime=regime, arm=arm, rows=len(rows), metrics=metrics), ensure_ascii=False), flush=True)
            del known, batch, scene, asset, observed
            gc.collect()
            torch.cuda.empty_cache()
        data.arrays.clear()
    after = helper.model_tensor_digest(model.state_dict())
    require(before == after, 'Model tensors mutated')
    helper.verify_sources()
    require(len(rows) == plan['planned_predictions'], 'Incomplete planned coverage')
    require(sha(plan['checkpoint']) == plan['checkpoint_sha256'], 'Checkpoint changed')
    protocol.update(status='completed', finished_at=now(), model_tensor_digest_after=after, resources_after=helper.resources())
    write(out / 'PROTOCOL.json', protocol)
    write(out / 'VERIFY.json', dict(status='passed_complete', metric_rows_reopened_and_recomputed=len(rows),
        fixed_model_tensors=True, fixed_inputs_and_seeds=True, sealed_training_sources_unchanged=True,
        all_arms_keep_scene_and_original_physical_guidance=True, optimizer_updates=0))
    summarize(out, rows, plan, complete=True)
    write(out / 'STATUS.json', dict(status='completed', checked_at=now(), pid=os.getpid(),
        completed_predictions=len(rows), planned_predictions=plan['planned_predictions']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--memory-fraction', type=float, default=.22)
    args = parser.parse_args()
    require(0 < args.memory_fraction <= .3, 'Evaluation memory fraction must be in (0,.3]')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'job.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(out)
        if args.prepare_only:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        try:
            run(out, plan, args.memory_fraction)
        except Exception as error:
            write(out / 'STATUS.json', dict(status='failed', checked_at=now(), pid=os.getpid(), error=repr(error)))
            raise


if __name__ == '__main__':
    main()
