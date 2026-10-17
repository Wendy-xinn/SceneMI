"""Frozen, paired validation of original and yaw-only observation references.

Reuse the registered 14-window foot-trial cohort and its CPU metrics. A reused
control prediction is never regenerated: its original artifact remains intact,
while a provenance wrapper records the new, reference-excluding input digest.
Run with ``python -m experiments.offline_camera_retrain_v1.ops.reference_trial_evaluation``.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time

from . import foot_trial_evaluation as foot
from .motion_ablation import BASE, FULL, load_module, now, read, require, sha, surface_metrics, write

DEFAULT_OUT = BASE / 'runs/native30_live_monitor_oct16/paired_reference_trial/evaluation'
VARIANTS = {'control': 'original', 'yaw_only': 'yaw_only'}
REFERENCE_KEYS = ('reference_motion', 'reference_confidence', 'observation_anchor_reference')
INPUT_SCOPE = ('SHA256 of sorted inference tensor keys, dtypes, shapes and bytes, excluding '
    'reference_motion/reference_confidence/observation_anchor_reference. Non-tensor scene/skin '
    'objects are not part of this digest; source identities and common evaluation assets are checked separately.')
FORBIDDEN_INPUTS = ('motion', 'joints', 'contact_target', 'contact_valid', 'gt_position_weight',
                    'gt_rotation_weight', 'support_points', 'gt_contact')


def rows_at(path):
    if not Path(path).exists():
        return []
    raw = Path(path).read_bytes()
    require(not raw or raw.endswith(b'\n'), 'Partial row tail; preserve before explicit recovery')
    return [json.loads(line) for line in raw.splitlines()]


def digest_arrays(arrays):
    """Same tensor digest encoding as the sealed foot evaluator."""
    import numpy as np
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(json.dumps([key, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def input_fingerprints(known):
    import torch
    require(not any(key in known for key in FORBIDDEN_INPUTS), 'GT target leaked into inference')
    arrays = {key: value.detach().cpu().contiguous().numpy()
              for key, value in known.items() if isinstance(value, torch.Tensor)}
    require(all(key in arrays for key in REFERENCE_KEYS), 'Missing reference-residual inference tensors')
    common = {key: value for key, value in arrays.items() if key not in REFERENCE_KEYS}
    reference = {key: arrays[key] for key in REFERENCE_KEYS}
    return dict(input_sha256=digest_arrays(common), full_known_sha256=digest_arrays(arrays),
        reference_sha256={key: digest_arrays({key: value}) for key, value in reference.items()},
        common_arrays=common, reference_arrays=reference)


def declared_variant(item):
    return item.get('reference_variant', item.get('metadata', {}).get('reference_variant'))


def validate_state(state, declaration, label):
    """Always called on CPU before a model or observation builder touches CUDA."""
    require(isinstance(state.get('step'), int) and not isinstance(state['step'], bool)
            and state['step'] == declaration['expected_step'],
            f"{label}: unfinished/wrong checkpoint step; expected {declaration['expected_step']}, found {state.get('step')}")
    expected = declared_variant(declaration['metadata'])
    actual = state.get('reference_variant', 'original')
    require(actual == expected == VARIANTS[label], 'Checkpoint reference variant differs: ' + label)
    if label == 'yaw_only':
        require(state.get('reference_variant') == 'yaw_only', 'Yaw-only checkpoint needs explicit variant provenance')
        trial_sha = state.get('reference_trial_plan_sha256')
        require(isinstance(trial_sha, str) and re.fullmatch(r'[0-9a-f]{64}', trial_sha),
                'Yaw-only checkpoint lacks its declared training-plan SHA256')
        requested = declaration['metadata'].get('reference_trial_plan_sha256')
        require(requested is None or requested == trial_sha, 'Yaw-only training-plan SHA differs')
    require(state.get('effective_config') == read(foot.EFFECTIVE),
            'Candidate effective config differs from the unchanged sealed training config')


def check_checkpoint_cpu(path, declaration, label):
    import torch
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    validate_state(state, declaration, label)
    del state


def verify_dependencies(plan):
    for path, digest in plan['dependency_sha256'].items():
        require(sha(path) == digest, 'Reference-evaluation dependency changed: ' + path)


def freeze_models(out, manifest, selected, reuse_control=None):
    require(isinstance(manifest, dict) and set(manifest) == set(VARIANTS),
            'Manifest must declare exactly control and yaw_only')
    helper = load_module('reference_trial_freeze_helpers', FULL / 'runner.py')
    helper.verify_sources()
    (out / 'checkpoints').mkdir(exist_ok=True)
    frozen = {}
    for name, variant in VARIANTS.items():
        item = manifest[name]
        require(isinstance(item, dict) and declared_variant(item) == variant,
                f'{name}: explicitly declare reference_variant={variant}')
        step = item.get('expected_step')
        require(isinstance(step, int) and not isinstance(step, bool) and step > 0,
                'Each model requires a positive integer expected_step')
        source = Path(item['checkpoint']).resolve()
        frozen[name] = dict(source=str(source), checkpoint=str(out / 'checkpoints' / (name + '.pt')),
            expected_step=step, metadata=dict(item),
            evaluation_config='Original sealed architecture/config; declared reference policy installed only in this evaluation process; physical guidance crossed identically at inference.')
        if source.exists():
            check_checkpoint_cpu(source, frozen[name], name)
    require(frozen['control']['expected_step'] == frozen['yaw_only']['expected_step'],
            'Matched candidates must finish at the same absolute optimizer step')
    dependencies = [Path(__file__), Path(__file__).with_name('reference_prior.py'),
        Path(foot.__file__), Path(__file__).with_name('foot_trial_recheck.py'),
        Path(__file__).with_name('motion_ablation.py'), Path(__file__).with_name('support_metrics.py'),
        FULL / 'runner.py', foot.EFFECTIVE, BASE / 'runs/native30_final_review_oct16/io_recovery.py']
    plan = dict(protocol='frozen-yaw-reference-trial-evaluation-v1',
        cohort_sha256=sha(out / 'COHORT.json'), models=frozen,
        inference_config=str(foot.EFFECTIVE), inference_config_sha256=sha(foot.EFFECTIVE),
        primary='All 14 registered VAL windows × clean/noisy; independent physical guidance OFF for both models.',
        secondary='Same five registered core windows × clean/noisy; independent physical guidance ON for both models.',
        planned_predictions=2 * selected['planned_predictions_per_model'],
        reuse_control=None if reuse_control is None else str(Path(reuse_control).resolve()),
        dependency_sha256={str(path): sha(path) for path in dependencies},
        input_hash_scope=INPUT_SCOPE, reference_tensor_keys_excluded=list(REFERENCE_KEYS),
        reference_policy='Control uses original observation reference; candidate uses reference_prior.installed(). The measured trajectory/meta are unchanged. Derived motion/anchor references may differ; scalar reference confidence is held fixed.',
        targets='GT/contact/reliability targets are excluded from inference; GT is used only for CPU metrics and common evaluation masks.',
        metric_scope='Reuse unchanged foot_trial_evaluation/support_metrics/surface_metrics. Pose joint extent is not mesh stature; sparse GT-associated visible support proxies are not full-mesh collision.',
        scope='Paired short-training development comparison; no TEST or final long-training quality claim.')
    path = out / 'PLAN.json'
    if path.exists():
        require(read(path) == plan, 'Immutable reference-evaluation plan differs')
    else:
        write(path, plan)
    if reuse_control is not None:
        control_sources(out, selected, plan, Path(reuse_control).resolve())
    return plan


def pin_checkpoint(out, label, declaration, helper):
    source = Path(declaration['source'])
    require(source.exists(), 'Declared candidate checkpoint is not ready: ' + label)
    check_checkpoint_cpu(source, declaration, label)
    result = foot.pin_checkpoint(out, label, declaration, helper)
    check_checkpoint_cpu(result['checkpoint'], declaration, label)
    return result


def expected_keys(selected, labels):
    return {(label, entry['case'], regime, guidance)
            for label in labels for entry in selected['entries'] for regime in selected['regimes']
            for guidance in ('off', 'on')
            if guidance == 'off' or entry['case'] in selected['secondary']['cases']}


def control_sources(out, selected, plan, source_eval):
    """Read-only verification of the independently checked original 38 outputs."""
    require(source_eval != out, 'Control reuse source and destination must differ')
    source_plan = read(source_eval / 'PLAN.json')
    require(read(source_eval / 'COHORT.json') == selected
            and sha(source_eval / 'COHORT.json') == plan['cohort_sha256'], 'Control reuse cohort differs')
    report_path = source_eval / 'CPU_RECHECK_all.json'
    report = read(report_path)
    require(report['status'] == 'independent_cpu_passed' and 'control' in report['models']
            and report['plan_sha256'] == sha(source_eval / 'PLAN.json'),
            'Reuse requires a matching independent CPU check of the control outputs')
    for path, digest in source_plan['dependency_sha256'].items():
        require(sha(path) == digest, 'Control source dependency changed: ' + path)
    source_pin = read(source_eval / 'checkpoints/control.json')
    require(source_pin['declaration'] == source_plan['models']['control'], 'Source control pin declaration differs')
    require(source_pin['step'] == plan['models']['control']['expected_step'], 'Reused control stop step differs')
    require(sha(source_pin['checkpoint']) == source_pin['checkpoint_sha256'], 'Reused control checkpoint changed')
    require(sha(plan['models']['control']['source']) == source_pin['checkpoint_sha256'],
            'New control declaration is not the evaluated original control checkpoint')
    bank = {}
    evidence = {(item['case'], item['regime'], item['guidance']): item
                for item in report['checks'] if item['model'] == 'control'}
    entries = {entry['case']: entry for entry in selected['entries']}
    for row in rows_at(source_eval / 'rows.jsonl'):
        if row['model'] != 'control':
            continue
        key = row['case'], row['regime'], row['guidance']
        require(key not in bank, 'Repeated reused control prediction')
        entry = entries[row['case']]
        require(all(row[key] == entry[key] for key in ('dataset', 'sequence_id', 'start', 'source_first', 'case_seed')),
                'Reused control identity differs')
        require(row['checkpoint_step'] == source_pin['step']
                and row['checkpoint_sha256'] == source_pin['checkpoint_sha256'], 'Mixed control checkpoint')
        require(key in evidence and evidence[key]['artifact_sha256'] == row['artifact_sha256']
                and sha(row['artifact']) == row['artifact_sha256'], 'Reused control artifact lacks matching CPU evidence')
        bank[key] = row
    require({('control', *key) for key in bank} == expected_keys(selected, ['control'])
            and len(bank) == 38, 'Expected exactly the registered 38 control outputs')
    record = dict(status='verified_source_control', source_eval=str(source_eval),
        source_plan_sha256=sha(source_eval / 'PLAN.json'), source_cohort_sha256=sha(source_eval / 'COHORT.json'),
        independent_cpu_report=str(report_path), independent_cpu_report_sha256=sha(report_path),
        checkpoint_sha256=source_pin['checkpoint_sha256'], predictions=38,
        artifacts=[dict(case=key[0], regime=key[1], guidance=key[2], artifact=row['artifact'],
                        artifact_sha256=row['artifact_sha256'], legacy_full_known_sha256=row['input_sha256'])
                   for key, row in sorted(bank.items())],
        policy='Source NPZ files are never modified. New wrappers retain every source array byte-for-byte except input_sha256, which is replaced by the documented common non-reference tensor digest. Source SHA and legacy full-known digest remain explicit provenance.')
    path = out / 'CONTROL_REUSE.json'
    if path.exists():
        require(read(path) == record, 'Immutable source-control reuse evidence differs')
    else:
        write(path, record)
    return bank


def summarize(out, rows, plan, complete=False):
    foot.summarize(out, rows, plan, complete=complete)
    path = out / 'REPORT_zh.md'
    text = path.read_text().replace('# 脚步短训：统一推理口径配对验证', '# yaw-only 参考：配对短训验证', 1)
    text += ('\ncontrol 使用原 reference，yaw_only 使用相同规则训练及推理的 yaw-only fallback。'
             '公共输入指纹排除三个派生 reference tensor；派生 reference 单独记录，未要求其跨模型相同。'
             '这项比较同时涉及短训权重和声明的 reference 策略，不能当成单独推理开关效果或最终长训结论。\n')
    if plan['reuse_control'] is not None:
        text += ('\ncontrol 的 38 份预测复用已有独立 CPU 核验结果；源 NPZ/SHA/identity 保留。'
                 '新 wrapper 仅补公共指纹和 provenance，预测、GT、rest、观测逐数组字节不变。'
                 '旧采样耗时另保留，不与候选当前耗时作性能结论。\n')
    text += '\n完整新一轮独立 CPU 全量复核见 CPU_RECHECK_all.json（生成前不宣称已通过）。\n'
    path.write_text(text)
    summary = read(out / 'SUMMARY.json')
    summary.update(evaluation='yaw-only-reference-paired-short-train', input_hash_scope=INPUT_SCOPE,
                   reference_variants=VARIANTS, control_predictions_reused=plan['reuse_control'] is not None)
    write(out / 'SUMMARY.json', summary)


def append_row(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def write_artifact(path, arrays):
    import numpy as np
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def extra_input_arrays(fingerprints, variant):
    import numpy as np
    arrays = {'input_tensor__' + key: value for key, value in fingerprints['common_arrays'].items()}
    arrays.update({'reference_tensor__' + key: value for key, value in fingerprints['reference_arrays'].items()})
    arrays.update(input_sha256=np.asarray(fingerprints['input_sha256']), input_hash_scope=np.asarray(INPUT_SCOPE),
                  reference_variant=np.asarray(variant),
                  reference_tensor_sha256_json=np.asarray(json.dumps(fingerprints['reference_sha256'], sort_keys=True)))
    return arrays


def assert_same_array(first, second, description):
    import numpy as np
    first, second = np.asarray(first), np.asarray(second)
    require(first.dtype == second.dtype and first.shape == second.shape
            and first.tobytes() == second.tobytes(), 'Array bytes differ: ' + description)


def reference_audit(rows, complete=False):
    grouped = {}
    for row in rows:
        key = row['case'], row['regime']
        group = grouped.setdefault(key, {})
        value = dict(input_sha256=row['input_sha256'], reference_sha256=row['derived_reference_sha256'])
        require(row['model'] not in group or group[row['model']] == value,
                'Inputs/reference changed across physical guidance modes')
        group[row['model']] = value
    pairs = []
    for (case, regime), group in sorted(grouped.items()):
        if set(group) != set(VARIANTS):
            continue
        before, after = group['control'], group['yaw_only']
        require(before['input_sha256'] == after['input_sha256'], 'Common non-reference inference tensors differ')
        changed = {key: before['reference_sha256'][key] != after['reference_sha256'][key] for key in REFERENCE_KEYS}
        require(not changed['reference_confidence'], 'Scalar reference confidence changed in the yaw-only ablation')
        pairs.append(dict(case=case, regime=regime, common_input_sha256=before['input_sha256'],
                          reference_changed=changed, models=group))
    if complete:
        require(len(pairs) == 28, 'Expected 14×2 paired common-input/reference audits')
        require(any(row['reference_changed']['reference_motion'] for row in pairs),
                'No reference motion changed anywhere; yaw-only intervention was not demonstrated')
    return dict(status='passed_complete' if complete else 'partial', input_hash_scope=INPUT_SCOPE,
        reference_tensor_keys_excluded=list(REFERENCE_KEYS), pairs=pairs,
        derived_reference_expectation='Motion and observation-anchor references normally change when measured head pitch/roll is nonzero. Pure-yaw or missing-evidence windows may legitimately match. Scalar confidence is held fixed; derived hashes are not claimed equal across policies.',
        changed_reference_motion_pairs=sum(row['reference_changed']['reference_motion'] for row in pairs))


def run(out, selected, plan, memory_fraction, only=None):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model, window_options
    from experiments.offline_camera_retrain_v1.native30_data import Native30Data
    from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
    from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    from experiments.offline_camera_retrain_v1 import native30_refinement as refinement
    from experiments.offline_camera_retrain_v1 import native30_surface_cache as surfaces, native30_memory as memory
    from experiments.offline_camera_retrain_v1.native30_surface_cache import CanonicalSurfaceCache
    from . import reference_prior
    from .support_metrics import support_metrics

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    require(refinement.reference_from_observations is reference_prior._ORIGINAL_REFERENCE,
            'Unexpected preinstalled reference override in evaluation process')
    helper = load_module('reference_trial_eval_helpers', FULL / 'runner.py')
    effective = helper.verify_sources()
    verify_dependencies(plan)
    labels = list(VARIANTS) if only is None else [label for label in VARIANTS if label in only]
    require(labels and (only is None or set(labels) == set(only)), 'Unknown or empty --only labels')
    bank = {} if plan['reuse_control'] is None else control_sources(out, selected, plan, Path(plan['reuse_control']))
    # A --only yaw_only execution also imports the already-evaluated control when
    # needed; this performs no control model forward and no optimizer update.
    if bank and 'control' not in labels:
        labels.insert(0, 'control')
    pins = {label: pin_checkpoint(out, label, plan['models'][label], helper) for label in labels}
    # All requested checkpoint variant and stop-step checks precede CUDA setup.
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    cfg = effective['config']
    io = load_module('reference_trial_crc_reads', BASE / 'runs/native30_final_review_oct16/io_recovery.py')
    surfaces.np = io.NumpyProxy(surfaces.np, out / 'IO_READS.jsonl')
    memory.np = io.NumpyProxy(memory.np, out / 'IO_READS.jsonl')
    data = Native30Data(BASE / cfg['train_manifest'], 'validation', cfg['seed'] + 1, frames=cfg['frames'],
        memory_protocol=cfg['scene']['memory_protocol'], quality_config=cfg['gt_reliability'],
        exo_max_empty_frames=cfg['exo_max_empty_frames'], **window_options(cfg))
    data.runtime_cache_root = out / 'native30_runtime_cache_v2'
    cache_path = data.scene_memory._cache_path
    data.scene_memory._cache_path = lambda *args: (path if (path := cache_path(*args)) is not None and path.exists() else None)
    data.surface_cache = CanonicalSurfaceCache(data, limit=1)
    data.scene_memory.cached_streams = 1
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(), read(BASE / cfg['mount_defaults']), cfg['observation']['input_noise'])
    noisy = Native30ObservationBuilder(builder.skin_cache, builder.mounts, dict(builder.noise_config, clean_probability=0.))
    row_path = out / 'rows.jsonl'
    rows = rows_at(row_path)
    done = {(row['model'], row['case'], row['regime'], row['guidance']): row for row in rows}
    expected = expected_keys(selected, VARIANTS)
    require(len(done) == len(rows) and set(done).issubset(expected), 'Duplicate or unplanned saved predictions')
    fingerprints, denominators = {}, {}
    entries = {entry['case']: entry for entry in selected['entries']}
    for row in rows:
        require(row.get('input_hash_scope') == INPUT_SCOPE, 'Saved row uses a different input hash scope')
        require(row.get('reference_variant') == VARIANTS[row['model']], 'Saved reference variant differs')
        entry = entries[row['case']]
        require(all(row[key] == entry[key] for key in ('dataset', 'sequence_id', 'start', 'source_first', 'case_seed')),
                'Saved row cohort identity differs')
        require(sha(row['artifact']) == row['artifact_sha256'], 'Saved artifact changed')
        pin = read(out / 'checkpoints' / (row['model'] + '.json'))
        require(pin['declaration'] == plan['models'][row['model']]
                and sha(pin['checkpoint']) == pin['checkpoint_sha256'] == row['checkpoint_sha256'], 'Mixed saved checkpoint')
        key = row['case'], row['regime']
        require(key not in fingerprints or fingerprints[key] == row['input_sha256'], 'Saved common inputs differ')
        fingerprints[key] = row['input_sha256']
        value = row['metrics'].get('support_evaluation_anchor_count', 0), row['metrics']['gt_low_speed_foot_joint_samples']
        require(key not in denominators or denominators[key] == value, 'Saved GT evaluation denominators differ')
        denominators[key] = value
    reference_audit(rows)
    protocol = dict(status='running', pid=os.getpid(), started_at=now(), plan_sha256=sha(out / 'PLAN.json'),
        optimizer_updates=0, resumed_rows=len(rows), phase_models=labels, models={}, input_hash_scope=INPUT_SCOPE,
        resources_before=helper.resources(), control_forward_passes=0)
    if (out / 'PROTOCOL.json').exists():
        previous = read(out / 'PROTOCOL.json')
        require(previous['plan_sha256'] == protocol['plan_sha256'], 'Resumed plan differs')
        append_row(out / 'ATTEMPTS.jsonl', previous)
        protocol['models'].update(previous.get('models', {}))
    write(out / 'PROTOCOL.json', protocol)
    (out / 'motions').mkdir(exist_ok=True)
    for label in labels:
        pin = pins[label]
        pending_all = expected_keys(selected, [label]) - set(done)
        if not pending_all:
            continue
        model = None
        if not (label == 'control' and bank):
            state = torch.load(pin['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
            before = helper.model_tensor_digest(state['model'])
            require(before == pin['model_tensor_sha256'], 'Frozen model tensor identity differs')
            model = make_model(cfg).cuda().eval()
            model.load_state_dict(state['model'], strict=True)
            del state
        context = reference_prior.installed() if label == 'yaw_only' else nullcontext()
        with context:
            for entry in selected['entries']:
                pending = [(regime, guidance) for regime in selected['regimes'] for guidance in ('off', 'on')
                    if (label, entry['case'], regime, guidance) in pending_all]
                if not pending:
                    continue
                members = [i for i, (meta, _) in enumerate(data.members[entry['group']]) if meta['path'] == entry['path']]
                require(len(members) == 1, 'Nonunique source window')
                sample, identity = data.sample(entry['group'], 'ego', member=members[0], start=entry['start'], yaw=0)
                require(identity['sequence_id'] == entry['sequence_id']
                        and identity['source_start_30fps'] == entry['source_first'], 'Source window identity differs')
                for regime in selected['regimes']:
                    modes = [guidance for r, guidance in pending if r == regime]
                    if not modes:
                        continue
                    observed, scene, asset = (builder if regime == 'clean' else noisy).build(data, sample, identity, 'ego', (15,),
                        noise_rng=None if regime == 'clean' else np.random.default_rng(entry['case_seed']))
                    batch = collate_observations([(observed, scene, asset, 'ego')])
                    known = refinement.model_inputs(cfg, batch)
                    fingerprint = input_fingerprints(known)
                    input_sha = fingerprint['input_sha256']
                    key = entry['case'], regime
                    require(key not in fingerprints or fingerprints[key] == input_sha, 'Common observed tensors differ across models')
                    fingerprints[key] = input_sha
                    for guidance in modes:
                        write(out / 'STATUS.json', dict(status='sampling' if model is not None else 'importing_control',
                            checked_at=now(), pid=os.getpid(), model=label, case=entry['case'], regime=regime,
                            guidance=guidance, completed_predictions=len(rows), planned_predictions=plan['planned_predictions']))
                        started = time.monotonic()
                        source = bank.get((entry['case'], regime, guidance)) if label == 'control' else None
                        source_arrays = None
                        if source is not None:
                            require(source['checkpoint_sha256'] == pin['checkpoint_sha256'], 'Reused control checkpoint differs')
                            require(fingerprint['full_known_sha256'] == source['input_sha256'],
                                    'Regenerated original full-known input differs from the reused control inference')
                            with np.load(source['artifact'], allow_pickle=False) as saved:
                                source_arrays = {key: saved[key] for key in saved.files}
                            require(str(source_arrays['input_sha256']) == source['input_sha256'], 'Legacy artifact input SHA differs')
                            require(str(source_arrays['model']) == label and str(source_arrays['guidance']) == guidance
                                    and str(source_arrays['checkpoint_sha256']) == pin['checkpoint_sha256'], 'Legacy artifact arm differs')
                            saved_identity = json.loads(str(source_arrays['identity_json']))
                            require(saved_identity == identity, 'Reused control artifact identity differs')
                            for field, actual in [('truth', sample['motion']), ('rest', sample['rest']),
                                    ('observations', observed['trajectory']), ('observation_meta', observed['observation_meta'])]:
                                assert_same_array(source_arrays[field], actual, 'reused source ' + field)
                            arrays = dict(source_arrays)
                            arrays.update(legacy_full_known_sha256=np.asarray(source['input_sha256']),
                                source_artifact_sha256=np.asarray(source['artifact_sha256']), source_artifact=np.asarray(source['artifact']))
                        else:
                            model.guidance_enabled = guidance == 'on'
                            with torch.no_grad():
                                result = refinement.sample_motion(model, known, cfg, steps=selected['ddim_steps'], seed=entry['case_seed'], use_scene=True)
                            prediction = result[0].detach().cpu().numpy()
                            require(np.isfinite(prediction).all(), 'Nonfinite candidate prediction')
                            arrays = dict(prediction=prediction, truth=sample['motion'], rest=sample['rest'],
                                observations=observed['trajectory'], observation_meta=observed['observation_meta'],
                                identity_json=np.asarray(json.dumps(identity)), model=np.asarray(label), guidance=np.asarray(guidance),
                                checkpoint_sha256=np.asarray(pin['checkpoint_sha256']))
                            if label == 'control':
                                protocol['control_forward_passes'] += 1
                            del result
                        arrays.update(extra_input_arrays(fingerprint, VARIANTS[label]))
                        artifact = out / 'motions' / f"{label}_case{entry['case']:02d}_{regime}_physical_{guidance}.npz"
                        write_artifact(artifact, arrays)
                        with np.load(artifact, allow_pickle=False) as saved:
                            if source_arrays is not None:
                                for field in source_arrays:
                                    if field != 'input_sha256':
                                        assert_same_array(saved[field], source_arrays[field], 'source/wrapper ' + field)
                            metrics = support_metrics(saved['prediction'], saved['truth'], saved['rest'], saved['observations'], saved['observation_meta'], asset)
                            metrics.update(foot.extended_metrics(saved['prediction'], saved['truth'], saved['rest']))
                            metrics.update(surface_metrics(saved['prediction'], saved['truth'], asset, scene))
                        if source is not None:
                            require(set(metrics) == set(source['metrics']), 'Reused control metric schema differs')
                            for field, value in metrics.items():
                                old = source['metrics'][field]
                                require((value is None) == (old is None) and (value is None or abs(value - old) < 1e-3),
                                        'Recomputed control wrapper metric differs: ' + field)
                        denominator = metrics.get('support_evaluation_anchor_count', 0), metrics['gt_low_speed_foot_joint_samples']
                        require(key not in denominators or denominators[key] == denominator, 'GT evaluation masks differ')
                        denominators[key] = denominator
                        row = dict(model=label, checkpoint_sha256=pin['checkpoint_sha256'], checkpoint_step=pin['step'],
                            case=entry['case'], dataset=entry['dataset'], sequence_id=entry['sequence_id'], recording=entry['recording'],
                            start=entry['start'], source_first=entry['source_first'], case_seed=entry['case_seed'], regime=regime,
                            guidance=guidance, input_sha256=input_sha, input_hash_scope=INPUT_SCOPE,
                            non_reference_tensor_keys=sorted(fingerprint['common_arrays']), reference_variant=VARIANTS[label],
                            derived_reference_sha256=fingerprint['reference_sha256'], full_known_sha256=fingerprint['full_known_sha256'],
                            metrics=metrics, artifact=str(artifact), artifact_sha256=sha(artifact), seconds=time.monotonic() - started,
                            checked_at=now(), prediction_reused=source is not None)
                        if source is not None:
                            row.update(source_artifact=source['artifact'], source_artifact_sha256=source['artifact_sha256'],
                                legacy_full_known_sha256=source['input_sha256'], source_generation_seconds=source['seconds'],
                                reused_source_arrays_verified_byte_identical=True,
                                seconds_scope='Current provenance-wrapper/CPU metric preparation; source generation seconds separately retained')
                        else:
                            row['seconds_scope'] = 'Current DDIM generation, artifact write and CPU metrics; not comparable to reused-control preparation seconds'
                        append_row(row_path, row)
                        rows.append(row)
                        done[label, entry['case'], regime, guidance] = row
                        write(out / 'REFERENCE_INPUT_AUDIT.json', reference_audit(rows))
                        summarize(out, rows, plan)
                        print(json.dumps(dict(event='reference_prediction_complete', model=label, case=entry['case'],
                            regime=regime, guidance=guidance, reused=source is not None, completed=len(rows))), flush=True)
                    after_inputs = input_fingerprints(known)
                    require(after_inputs['input_sha256'] == input_sha
                            and after_inputs['reference_sha256'] == fingerprint['reference_sha256'], 'Inference mutated known inputs')
                    del known, batch, observed, scene, asset, fingerprint, after_inputs
                    gc.collect()
                    torch.cuda.empty_cache()
                data.arrays.clear()
        require(refinement.reference_from_observations is reference_prior._ORIGINAL_REFERENCE,
                'Yaw-only process-local patch was not restored')
        if model is not None:
            after = helper.model_tensor_digest(model.state_dict())
            require(before == after, 'Frozen candidate model tensors mutated')
            protocol['models'][label] = dict(model_tensor_digest_before=before, model_tensor_digest_after=after)
            del model
        else:
            protocol['models'][label] = dict(predictions_reused=True, checkpoint_sha256=pin['checkpoint_sha256'], model_forward_passes=0)
        write(out / 'PROTOCOL.json', protocol)
        gc.collect()
        torch.cuda.empty_cache()
    complete = set(done) == expected
    require(expected_keys(selected, labels).issubset(done), 'Incomplete requested evaluation phase')
    require(not complete or len(rows) == plan['planned_predictions'] == 76, 'Wrong complete prediction count')
    helper.verify_sources()
    verify_dependencies(plan)
    for pin in pins.values():
        require(sha(pin['checkpoint']) == pin['checkpoint_sha256'], 'Pinned checkpoint changed during evaluation')
    write(out / 'REFERENCE_INPUT_AUDIT.json', reference_audit(rows, complete))
    protocol.update(status='completed' if complete else 'phase_completed', finished_at=now(), resources_after=helper.resources())
    write(out / 'PROTOCOL.json', protocol)
    write(out / 'VERIFY.json', dict(status='passed_complete' if complete else 'passed_phase', completed_predictions=len(rows),
        optimizer_updates=0, fixed_model_tensors=True, common_non_reference_tensor_inputs_matched=True,
        derived_reference_tensors_recorded_separately=True, input_hash_scope=INPUT_SCOPE,
        metrics_from_committed_arrays=True, sealed_training_sources_unchanged=True,
        reference_patch_restored=True, independent_CPU_reverification='Not yet performed for this new evaluation; use --cpu-recheck'))
    summarize(out, rows, plan, complete)
    write(out / 'STATUS.json', dict(status='completed' if complete else 'phase_completed', checked_at=now(), pid=os.getpid(),
        completed_predictions=len(rows), planned_predictions=plan['planned_predictions'], phase_models=labels))


def cpu_recheck(out, labels=None):
    """Keep the sealed formula verifier; add exact saved common/reference audits."""
    import numpy as np
    import torch
    from . import foot_trial_recheck as recheck
    require(not torch.cuda.is_initialized(), 'Independent CPU recheck must not initialize CUDA')
    plan, selected = read(out / 'PLAN.json'), read(out / 'COHORT.json')
    verify_dependencies(plan)
    labels = list(VARIANTS) if labels is None else labels
    require(labels and set(labels).issubset(VARIANTS), 'Unknown CPU-recheck model labels')
    rows = [row for row in rows_at(out / 'rows.jsonl') if row['model'] in labels]
    found = {(row['model'], row['case'], row['regime'], row['guidance']) for row in rows}
    require(len(found) == len(rows) and found == expected_keys(selected, labels), 'CPU-recheck phase coverage differs')
    for row in rows:
        require(row.get('input_hash_scope') == INPUT_SCOPE, 'Wrong common-input scope')
        require(sha(row['artifact']) == row['artifact_sha256'], 'Artifact changed before CPU recheck')
        with np.load(row['artifact'], allow_pickle=False) as saved:
            common = {key.removeprefix('input_tensor__'): saved[key] for key in saved.files if key.startswith('input_tensor__')}
            references = {key.removeprefix('reference_tensor__'): saved[key] for key in saved.files if key.startswith('reference_tensor__')}
            require(sorted(common) == row['non_reference_tensor_keys']
                    and not set(common).intersection(REFERENCE_KEYS), 'Saved common tensor key scope differs')
            require(digest_arrays(common) == row['input_sha256'] == str(saved['input_sha256']), 'Saved common tensor digest differs')
            require(set(references) == set(REFERENCE_KEYS), 'Saved derived reference tensors missing')
            require({key: digest_arrays({key: value}) for key, value in references.items()} == row['derived_reference_sha256'],
                    'Saved derived-reference tensor digest differs')
            require(str(saved['reference_variant']) == row['reference_variant'] == VARIANTS[row['model']], 'Saved variant differs')
            if row.get('prediction_reused'):
                require(sha(row['source_artifact']) == row['source_artifact_sha256'], 'Preserved source-control artifact changed')
                with np.load(row['source_artifact'], allow_pickle=False) as source:
                    for key in source.files:
                        if key != 'input_sha256':
                            assert_same_array(saved[key], source[key], 'CPU source-control/wrapper ' + key)
                    require(str(source['input_sha256']) == row['legacy_full_known_sha256']
                            == str(saved['legacy_full_known_sha256']), 'Legacy full-known fingerprint differs')
    audit = reference_audit(rows, complete=set(labels) == set(VARIANTS))
    write(out / 'CPU_REFERENCE_INPUT_AUDIT.json', audit)
    # This independently regenerates source/observations/native CPU assets and
    # recomputes all 76 metric rows, including the separately coded formulas.
    recheck.run(out, labels)
    suffix = labels[0] if len(labels) == 1 else 'all'
    result_path = out / ('CPU_RECHECK_' + suffix + '.json')
    result = read(result_path)
    result['legacy_verifier_input_scope_text'] = result['input_scope']
    result['input_scope'] = (INPUT_SCOPE + ' The wrapper independently hashes every saved common tensor and derived reference tensor, '
        'checks source-control wrapper arrays byte-for-byte, and the unchanged CPU formula verifier rebuilds GT/rest/observed '
        'trajectory/meta from source. It does not independently regenerate every scene tensor or the GPU-derived reference from observations.')
    result.update(reference_wrapper_script_sha256=sha(Path(__file__)),
                  common_and_reference_saved_tensor_digests_independently_rechecked=True,
                  reference_input_audit_sha256=sha(out / 'CPU_REFERENCE_INPUT_AUDIT.json'))
    write(result_path, result)
    require(not torch.cuda.is_initialized(), 'CPU recheck unexpectedly initialized CUDA')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--models', type=Path, help='JSON: control/yaw_only checkpoint, expected_step, reference_variant')
    parser.add_argument('--reuse-control', type=Path, help='Original foot-evaluation directory with 38 independently checked control outputs')
    parser.add_argument('--prepare-only', action='store_true', help='Seal cohort/dependencies/declarations without initializing CUDA')
    parser.add_argument('--memory-fraction', type=float, default=.18)
    parser.add_argument('--only', nargs='+', help='Named models; --only yaw_only also imports missing registered control wrappers')
    parser.add_argument('--cpu-recheck', action='store_true', help='Independent CPU full-metric and saved-tensor recheck; no --models required')
    args = parser.parse_args()
    require(0 < args.memory_fraction <= .22, 'Memory fraction must be in (0,.22]')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'job.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.cpu_recheck:
            require(not args.prepare_only, '--cpu-recheck and --prepare-only are separate operations')
            cpu_recheck(out, args.only)
            return
        require(args.models is not None, 'Supply --models for reference evaluation preparation/execution')
        selected = foot.cohort(out)
        plan = freeze_models(out, read(args.models), selected, args.reuse_control)
        if args.prepare_only:
            print(json.dumps(dict(status='prepared', out=str(out), planned_predictions=plan['planned_predictions'],
                                 reference_variants=VARIANTS, input_hash_scope=INPUT_SCOPE), ensure_ascii=False))
            return
        try:
            run(out, selected, plan, args.memory_fraction, args.only)
        except Exception as error:
            write(out / 'STATUS.json', dict(status='failed', checked_at=now(), pid=os.getpid(), error=repr(error)))
            raise


if __name__ == '__main__':
    main()
