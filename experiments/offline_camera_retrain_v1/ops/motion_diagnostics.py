"""Read-only per-stage tracing of a frozen native30 DDIM sampler.

The observer preserves all original model calls. It records the final denoiser
pass, excluding the proposal and physical line-search candidate constraints.
Ground truth enters CPU evaluation only, never the inference payload.
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from .motion_ablation import BASE, FULL, load_module, now, read, require, sha, write

DEFAULT_OUT = BASE / 'runs/native30_live_monitor_oct16/foot_stage_diagnostics'
TRAIN = BASE / 'runs/native30_gait_v2_ego55k_oct15/training'
STAGES = ('raw_fk', 'learned_feedback', 'head_translation', 'head_ik', 'physical')
KEYS = ('mpjpe_mm', 'observed_head_position_error_mm',
        'foot_joint_horizontal_speed_gt_low_m_s', 'foot_joint_vertical_speed_gt_low_m_s',
        'foot_joint_y_abs_error_gt_low_mm')


def prepare(out):
    """Pin the model-only best atomically; retain no optimizer copy."""
    source_plan = BASE / 'runs/native30_live_monitor_oct16/support_ablation/PLAN.json'
    helper = load_module('stage_trace_source_helpers', FULL / 'runner.py')
    helper.verify_sources()
    if (out / 'PLAN.json').exists():
        plan = read(out / 'PLAN.json')
        require(sha(plan['checkpoint']) == plan['checkpoint_sha256'], 'Pinned checkpoint changed')
        return plan
    best = read(TRAIN / 'best.json')
    require(best['checkpoint_step'] == 19000, 'This comparison requires selected 19000-step best')
    checkpoint = out / 'checkpoint_eval.pt'
    if not checkpoint.exists():
        os.link(TRAIN / 'best.pt', checkpoint)
    require(sha(checkpoint) == best['checkpoint_sha256'], 'Best changed during atomic pin')
    write(out / 'BEST_SOURCE.json', best)
    dependencies = [Path(__file__), Path(__file__).with_name('motion_ablation.py'),
                    Path(__file__).with_name('support_metrics.py'), FULL / 'runner.py',
                    BASE / 'runs/native30_final_review_oct16/io_recovery.py']
    plan = dict(protocol='frozen-motion-stage-trace-v1', created_at=now(),
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), checkpoint_step=19000,
        model_tensor_sha256=best['compact_model_tensor_sha256'],
        source_best=str(TRAIN / 'best.pt'), source_best_metadata_sha256=sha(out / 'BEST_SOURCE.json'),
        storage='Atomic hard link of existing model-only best; no optimizer or duplicate data blocks.',
        source_case_plan=str(source_plan), source_case_plan_sha256=sha(source_plan),
        entries=read(source_plan)['entries'][:2], regimes=['clean', 'noisy'],
        stages=list(STAGES), ddim_steps=20, planned_predictions=4, observer_control_predictions=1,
        control=dict(case=1, regime='clean', require='byte-identical final sampler output'),
        dependency_sha256={str(p): sha(p) for p in dependencies},
        source_scope='365 sealed training/execution sources; no training modifications or optimizer updates.',
        targets='GT used only after generation for metrics; predicted contact used only for existing physical solver.',
        stage_scope='Final denoiser pass only; proposal constraints and physical line-search constraints excluded.',
        metric_scope='Horizontal foot speed on GT 3D-speed<0.2m/s frames; differs from training validation 3D speed.',
        interpretation='Paired local stage changes conditional on this actual DDIM trajectory, not end-to-end ablations.')
    write(out / 'PLAN.json', plan)
    return plan


class StageObserver:
    """Process-local wrappers call every original operation exactly once."""
    def __init__(self, model):
        from experiments.offline_camera_retrain_v1 import native30_refinement as ref
        from experiments.offline_camera_retrain_v1 import native30_physical_guidance as physical
        self.model, self.ref, self.physical = model, ref, physical
        self.original = (model.forward, model.constrain_motion, ref.confidence_anchor,
                         physical.guide_motion, physical.prepare_evidence)
        self.records = []
        self.active = False
        self.inside_physics = False

    @staticmethod
    def cpu(value):
        return value.detach().cpu().numpy().copy()

    def install(self):
        forward, constrain, anchor, guide, evidence = self.original

        def constrained(motion, timestep, batch):
            eligible = self.active and not self.inside_physics
            if eligible:
                self.constrain_calls += 1
                if self.constrain_calls == 2:
                    self.current['learned_feedback'] = self.cpu(motion)[0]
            result = constrain(motion, timestep, batch)
            if eligible and self.constrain_calls == 2:
                self.current['head_ik'] = self.cpu(result)[0]
            return result

        def anchored(motion, batch, signal, gain):
            result = anchor(motion, batch, signal, gain)
            if self.active and not self.inside_physics and self.constrain_calls == 2:
                self.current['head_translation'] = self.cpu(result)[0]
                delta = (result[..., :3] - motion[..., :3]) * 2
                self.current['diagnostics']['anchor_root_delta_mean_mm'] = float(delta.norm(dim=-1).mean()) * 1000
                self.current['diagnostics']['anchor_root_delta_y_mean_mm'] = float(delta[..., 1].mean()) * 1000
                self.current['diagnostics']['anchor_root_delta_max_mm'] = float(delta.norm(dim=-1).max()) * 1000
            return result

        def prepared(motion, batch, probability, config):
            result = evidence(motion, batch, probability, config)
            if self.active and self.inside_physics:
                labels = result['asset']['labels']
                import torch
                feet = torch.isin(labels, labels.new_tensor([7, 8, 10, 11]))
                cw, pw = result['contact_weight'], result['plane_weight']
                self.current['diagnostics'].update(
                    physical_evidence_prepared=True,
                    positive_contact_anchor_count=int((cw > 0).sum()),
                    positive_foot_contact_anchor_count=int((cw[:, feet] > 0).sum()),
                    positive_plane_anchor_count=int((pw > 0).sum()),
                    positive_foot_contact_weight_sum=float(cw[:, feet].sum()),
                    positive_plane_weight_sum=float(pw.sum()))
            return result

        def guided(motion, timestep, batch, details, config, constraint):
            if self.active:
                import numpy as np
                require(np.array_equal(self.current['head_ik'], self.cpu(motion)[0]), 'Physical input differs from IK output')
            self.inside_physics = True
            try:
                result, diagnostics = guide(motion, timestep, batch, details, config, constraint)
            finally:
                self.inside_physics = False
            if self.active:
                self.current['physical'] = self.cpu(result)[0]
                self.current['diagnostics'].update({k: float(v.detach().sum()) for k, v in diagnostics.items()
                    if k in ('physical_energy_before', 'physical_energy_after', 'physical_accepted_steps')})
            return result, diagnostics

        def monitored(*args, **kwargs):
            requested = kwargs.pop('return_details', False)
            self.current = {'diagnostics': dict(physical_evidence_prepared=False)}
            self.constrain_calls = 0
            self.active = True
            try:
                result, details = forward(*args, return_details=True, **kwargs)
            finally:
                self.active = False
            require(self.constrain_calls == 2, 'Expected proposal and final outer constraints')
            self.current['raw_unreconciled'] = self.cpu(details['raw_motion'])[0]
            probability = self.cpu(details['contact_logits'].sigmoid())[0]
            self.current['contact_probability'] = probability
            self.current['diagnostics'].update(
                foot_contact_above_threshold_fraction=float((probability[:, [7, 8, 10, 11]] > .5).mean()),
                foot_contact_probability_mean=float(probability[:, [7, 8, 10, 11]].mean()),
                outer_constrain_calls=self.constrain_calls)
            timestep = args[1]
            self.current['timestep'] = int(timestep[0])
            self.current['alpha'] = float(self.model.alphas[timestep][0])
            require('physical' in self.current, 'Original physical guidance did not run')
            import numpy as np
            require(np.array_equal(self.current['physical'], self.cpu(result)[0]), 'Forward return differs from physical output')
            self.records.append(self.current)
            return (result, details) if requested else result

        self.model.constrain_motion, self.model.forward = constrained, monitored
        self.ref.confidence_anchor, self.physical.guide_motion = anchored, guided
        self.physical.prepare_evidence = prepared

    def restore(self):
        self.model.forward, self.model.constrain_motion, self.ref.confidence_anchor, self.physical.guide_motion, self.physical.prepare_evidence = self.original


def report(out, rows, complete=False):
    lines = ['# 冻结 19000 步权重：脚部误差逐阶段诊断', '',
        '**' + ('已完成，包含无监控同 seed 控制。' if complete else '运行中，结果尚未完整。') + '**', '',
        '两段预选失败/走动窗口，clean/noisy 共4次真实DDIM20。这里比较同一次采样的局部阶段，不等价于删除对应模块的完整消融。', '',
        'raw_fk 是原始输出按FK一致化后的结果；随后是学习反馈、整身贴头平移、头IK、原物理修正。', '',
        '脚速是 GT 三维足速<0.2m/s 时预测的水平足速，不与训练验证中的三维足速混用。', '',
        '## 最后一步 t=0', '',
        '| 窗口/输入 | 阶段 | MPJPE mm | 头对输入 mm | 水平足速 m/s | 脚高绝对误差 mm |',
        '|---|---|---:|---:|---:|---:|']
    final = [r for r in rows if r['timestep'] == 0]
    for row in final:
        for stage in STAGES:
            m = row['metrics'][stage]
            lines.append(f"| {row['dataset']} / {row['regime']} | {stage} | {m['mpjpe_mm']:.3f} | {m['observed_head_position_error_mm']:.3f} | {m['foot_joint_horizontal_speed_gt_low_m_s']:.4f} | {m['foot_joint_y_abs_error_gt_low_mm']:.3f} |")
    summary = {'completed_windows': len(final), 'complete': complete, 'last_step': final, 'late_step_paired_deltas': {}}
    lines += ['', '## alpha>0.5 的逐步配对变化', '',
        '以下是局部阶段均值变化（后减前）；多个DDIM步彼此相关，不是独立样本。', '',
        '| 转换 | MPJPE mm | 头对输入 mm | 水平足速 m/s | 脚高绝对误差 mm |', '|---|---:|---:|---:|---:|']
    late = [r for r in rows if r['alpha'] > .5]
    for a, b in zip(STAGES[:-1], STAGES[1:]):
        delta = {k: statistics.mean(r['metrics'][b][k] - r['metrics'][a][k] for r in late) for k in KEYS} if late else {}
        summary['late_step_paired_deltas'][a + '→' + b] = delta
        if delta:
            lines.append(f"| {a} → {b} | {delta['mpjpe_mm']:+.3f} | {delta['observed_head_position_error_mm']:+.3f} | {delta['foot_joint_horizontal_speed_gt_low_m_s']:+.4f} | {delta['foot_joint_y_abs_error_gt_low_mm']:+.3f} |")
    summary['late_step_count'] = len(late)
    lines += ['', '物理门控记录直接来自该次原始 prepare_evidence，不新增场景查询；没有记录到的距离/法向门控不能推测为失败原因。',
        'GT仅在采样完成后的CPU评价使用；记录不代表新训练收益，也没有更改主训练。', '']
    write(out / 'SUMMARY.json', summary)
    (out / 'REPORT_zh.md').write_text('\n'.join(lines))


def run(out, plan, memory_fraction):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model, window_options
    from experiments.offline_camera_retrain_v1.native30_data import Native30Data
    from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
    from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, sample_motion
    from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    from experiments.offline_camera_retrain_v1 import native30_surface_cache as surfaces, native30_memory as memory
    from experiments.offline_camera_retrain_v1.native30_surface_cache import CanonicalSurfaceCache
    from .support_metrics import support_metrics

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    helper = load_module('stage_trace_eval_helpers', FULL / 'runner.py')
    effective = helper.verify_sources()
    for path, digest in plan['dependency_sha256'].items():
        require(sha(path) == digest, 'Diagnostic source changed: ' + path)
    state = torch.load(plan['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    require(state['step'] == 19000 and state['effective_config'] == effective, 'Frozen checkpoint identity mismatch')
    before = helper.model_tensor_digest(state['model'])
    require(before == plan['model_tensor_sha256'], 'Model tensor digest differs from best guardian')
    cfg = effective['config']
    model = make_model(cfg).cuda().eval()
    model.load_state_dict(state['model'], strict=True)
    del state
    observer = StageObserver(model)
    io = load_module('stage_trace_crc_reads', BASE / 'runs/native30_final_review_oct16/io_recovery.py')
    surfaces.np = io.NumpyProxy(surfaces.np, out / 'IO_READS.jsonl')
    memory.np = io.NumpyProxy(memory.np, out / 'IO_READS.jsonl')
    data = Native30Data(BASE / cfg['train_manifest'], 'validation', cfg['seed'] + 1, frames=cfg['frames'],
        memory_protocol=cfg['scene']['memory_protocol'], quality_config=cfg['gt_reliability'],
        exo_max_empty_frames=cfg['exo_max_empty_frames'], **window_options(cfg))
    data.runtime_cache_root = out / 'native30_runtime_cache_v2'
    old_path = data.scene_memory._cache_path
    data.scene_memory._cache_path = lambda *args: (p if (p := old_path(*args)) is not None and p.exists() else None)
    data.surface_cache = CanonicalSurfaceCache(data, limit=1)
    data.scene_memory.cached_streams = 1
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(), read(BASE / cfg['mount_defaults']), cfg['observation']['input_noise'])
    noisy = Native30ObservationBuilder(builder.skin_cache, builder.mounts, dict(builder.noise_config, clean_probability=0.))
    protocol = dict(status='running', pid=os.getpid(), started_at=now(), model_tensor_digest_before=before,
        plan_sha256=sha(out / 'PLAN.json'), optimizer_updates=0, resources_before=helper.resources())
    write(out / 'PROTOCOL.json', protocol)
    require(not (out / 'rows.jsonl').exists(), 'Trace is immutable; choose a new output for a repeat')
    (out / 'motions').mkdir(exist_ok=True)
    rows, checks = [], []
    for entry in plan['entries']:
        members = [i for i, (meta, _) in enumerate(data.members[entry['group']]) if meta['path'] == entry['path']]
        require(len(members) == 1, 'Nonunique source sample')
        sample, identity = data.sample(entry['group'], 'ego', member=members[0], start=entry['start'], yaw=0)
        require(identity['sequence_id'] == entry['sequence_id'] and identity['source_start_30fps'] == entry['source_first'], 'Wrong source identity')
        for regime in plan['regimes']:
            write(out / 'STATUS.json', dict(status='sampling', checked_at=now(), pid=os.getpid(),
                case=entry['case'], regime=regime, completed_predictions=len(checks), planned_predictions=4))
            observed, scene, asset = (builder if regime == 'clean' else noisy).build(data, sample, identity, 'ego', (15,),
                noise_rng=None if regime == 'clean' else np.random.default_rng(entry['case_seed']))
            batch = collate_observations([(observed, scene, asset, 'ego')])
            known = model_inputs(cfg, batch)
            require(not any(k in known for k in ('motion', 'joints', 'contact_target', 'gt_position_weight', 'support_points')), 'GT leaked into inference')
            fingerprint = hashlib.sha256()
            for key in ('trajectory', 'observation_meta', 'observation_uncertainty', 'reference_motion', 'reference_confidence'):
                fingerprint.update(known[key].detach().cpu().contiguous().numpy().tobytes())
            input_sha = fingerprint.hexdigest()
            start = time.monotonic()
            observer.records.clear()
            observer.install()
            try:
                with torch.no_grad():
                    prediction_tensor = sample_motion(model, known, cfg, steps=20, seed=entry['case_seed'], use_scene=True)
            finally:
                observer.restore()
            torch.cuda.synchronize()
            prediction = prediction_tensor[0].detach().cpu().numpy().copy()
            records = observer.records
            require(len(records) == 20 and records[-1]['timestep'] == 0, 'Incomplete DDIM trace')
            last = torch.from_numpy(records[-1]['physical'])[None].cuda()
            with torch.no_grad():
                # Native residual DDIM performs this exact subtract/add round
                # trip before its terminal FK reconciliation. Omitting it can
                # change low bits without any observer effect.
                terminal = known['reference_motion'] + (last - known['reference_motion'])
                recovered = reconcile_motion(terminal, known['rest'])[0].cpu().numpy()
                fk_error = float((forward_kinematics(last, known['rest']) - forward_kinematics(prediction_tensor, known['rest'])).abs().max())
            require(np.array_equal(recovered, prediction), 'Sampler differs from final physical result after reconciliation')
            control_identical = None
            if entry['case'] == 1 and regime == 'clean':
                with torch.no_grad():
                    control = sample_motion(model, known, cfg, steps=20, seed=entry['case_seed'], use_scene=True)[0].cpu().numpy()
                control_identical = bool(np.array_equal(control, prediction))
                require(control_identical, 'Observer changed actual DDIM output')
            arrays = dict(prediction=prediction, truth=sample['motion'], rest=sample['rest'], observations=observed['trajectory'],
                observation_meta=observed['observation_meta'], identity_json=np.asarray(json.dumps(identity)),
                input_sha256=np.asarray(input_sha), checkpoint_step=np.asarray(19000),
                timestep=np.asarray([r['timestep'] for r in records]), alpha=np.asarray([r['alpha'] for r in records]),
                contact_probability=np.stack([r['contact_probability'] for r in records]),
                raw_unreconciled=np.stack([r['raw_unreconciled'] for r in records]))
            rest_cpu = torch.as_tensor(sample['rest'])[None]
            for stage in STAGES:
                if stage == 'raw_fk':
                    with torch.no_grad():
                        arrays[stage] = np.stack([reconcile_motion(torch.from_numpy(r['raw_unreconciled'])[None], rest_cpu)[0].numpy() for r in records])
                else:
                    arrays[stage] = np.stack([r[stage] for r in records])
            artifact = out / 'motions' / f"case{entry['case']:02d}_{regime}.npz"
            with artifact.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            artifact.with_suffix('.tmp').replace(artifact)
            artifact_sha = sha(artifact)
            with np.load(artifact, allow_pickle=False) as saved:
                for index, record in enumerate(records):
                    metrics = {stage: support_metrics(saved[stage][index], saved['truth'], saved['rest'], saved['observations'], saved['observation_meta'], asset) for stage in STAGES}
                    row = dict(case=entry['case'], dataset=entry['dataset'], sequence_id=entry['sequence_id'], regime=regime,
                        timestep=record['timestep'], alpha=record['alpha'], metrics=metrics, diagnostics=record['diagnostics'],
                        input_sha256=input_sha, artifact=str(artifact), artifact_sha256=artifact_sha)
                    with (out / 'rows.jsonl').open('a') as stream:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    rows.append(row)
            check = dict(case=entry['case'], regime=regime, sampler_matches_final_stage=True,
                final_stage_fk_max_absolute_difference_m=fk_error, observer_control_byte_identical=control_identical,
                artifact=str(artifact), artifact_sha256=artifact_sha, total_seconds=time.monotonic() - start)
            checks.append(check)
            write(out / 'CHECKS.json', checks)
            report(out, rows)
            print(json.dumps(dict(event='case_complete', **check)), flush=True)
            del known, batch, scene, asset, observed, prediction_tensor, last, records, arrays
            gc.collect()
            torch.cuda.empty_cache()
        data.arrays.clear()
    after = helper.model_tensor_digest(model.state_dict())
    require(before == after, 'Model tensors mutated')
    helper.verify_sources()
    for path, digest in plan['dependency_sha256'].items():
        require(sha(path) == digest, 'Diagnostic source changed during execution: ' + path)
    require(sha(plan['checkpoint']) == plan['checkpoint_sha256'], 'Pinned checkpoint changed')
    require(len(rows) == 80, 'Incomplete coverage')
    protocol.update(status='completed', finished_at=now(), model_tensor_digest_after=after, resources_after=helper.resources())
    write(out / 'PROTOCOL.json', protocol)
    write(out / 'VERIFY.json', dict(status='passed_complete', optimizer_updates=0, traced_predictions=4,
        observer_control_predictions=1, observer_control_byte_identical=True, fixed_model_tensors=True,
        sealed_training_sources_unchanged=True, all_forward_outputs_equal_physical_stage=True,
        all_sampler_outputs_equal_reconciled_final_physical_stage=True, metrics_recomputed_from_saved_arrays=400,
        gt_excluded_from_inference_payload=True, checks=checks))
    report(out, rows, complete=True)
    write(out / 'STATUS.json', dict(status='completed', checked_at=now(), pid=os.getpid(), completed_predictions=4,
        observer_control_predictions=1, planned_predictions=4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--memory-fraction', type=float, default=.22)
    args = parser.parse_args()
    require(0 < args.memory_fraction <= .22, 'Memory fraction must be in (0,.22]')
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
