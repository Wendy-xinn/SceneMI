"""Pinned, resumable paired comparison of native EGO/EXO scene checkpoints.

Reuse the original model implementations, checked validation data, DDIM and
motion metrics. Balanced validation and previously inspected diagnostic cases
are reported separately. No training or independent-test arrays are opened.
"""
import argparse
from collections import defaultdict, OrderedDict
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tarfile
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.checked_scene_dataset import SceneSelection, CheckedLocalSceneDataset
from scenemi.data.exo_initialization import repair_payload
from scenemi.data.local_scene import inference_payload as local_payload
from scenemi.data.native_repair import NativeRepairDataset
from scenemi.exo_denoise import sample_denoising
from scenemi.metrics_batch import metric_rows, repair_rows
from scenemi.native_joint_data import SceneFreeEgoDataset, ego_payload
from scenemi.observations import INPUT_KEYS, public_observations
from scenemi.scene_joint_replay import SceneExoDataset, make_model, load_adapter_state
from scenemi.training import deterministic, ensure_bounded, now, tensor_digest, contract
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from .data import normalize
from .visualization import load_model as replay_model

CONFIG = ROOT/'configs/scene_joint128_comparison.yaml'
VARIANTS = ('no_scene', 'scene_1000')


def read_config(path):
    c = yaml.safe_load(path.read_text())
    if (c['version'] != 'scene-joint-paired-validation-v1' or c['test_used']
            or c['sources'] != ['egobody', 'trumans', 'grail']
            or c['exo_conditions'] != ['identity', 'tracking', 'occlusion', 'gap']
            or c['scene_step'] != 1000):
        raise ValueError('Unexpected paired validation contract')
    return c


def ranked_cases(cases, count, seed):
    """One hash-selected window from each hash-selected independent family."""
    families = defaultdict(list)
    for case in cases:
        families[case['family']].append(case)
    def rank(value):
        return hashlib.sha256((str(seed)+'|'+value).encode()).hexdigest()
    if len(families) < count:
        raise ValueError('Insufficient independent validation families')
    chosen = sorted(families, key=rank)[:count]
    return [min(families[f], key=lambda x:rank(x['path']+'|'+str(x['start']))) for f in chosen]


def pin_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Pinned record changed: '+str(path))
    atomic_json(path, value)


def prepare(c, out):
    manifest = ROOT/c['case_manifest']
    if sha256(manifest) != c['case_manifest_sha256']:
        raise ValueError('Original complete validation manifest changed')
    original = json.loads(manifest.read_text())
    if original['test_used'] or not original['full_validation']:
        raise ValueError('Require the original complete validation membership')
    scene = yaml.safe_load((ROOT/c['scene_config']).read_text())
    check = json.loads((ROOT/'runs/data_inventory/CURRENT_SELECTION_CHECK.json').read_text())
    if (check['state'] != 'passed' or check['selection_sha256'] != sha256(ROOT/scene['paths']['selection'])
            or original['selection_sha256'] != check['selection_sha256']):
        raise ValueError('Checked current validation selection changed')
    if c['scope'] != 'all_eligible_validation_windows' or not c['full_validation']:
        raise ValueError('This comparison requires complete validation, not a subset')
    jobs = []
    counts = {}
    for mode in ('ego', 'exo'):
        for source in c['sources']:
            selected = [x for x in original['cases'] if x['mode'] == mode and x['source'] == source]
            counts[mode+'_'+source] = len(selected)
            for condition in (('ego',) if mode == 'ego' else c['exo_conditions']):
                jobs.extend(dict(scope='validation', condition=condition, case=case) for case in selected)
    expected = dict(ego_egobody=401, ego_trumans=540, ego_grail=2530,
                    exo_egobody=470, exo_trumans=540, exo_grail=2530)
    if counts != expected:
        raise ValueError('Complete native validation coverage changed')
    diagnostic = ROOT/c['diagnostic_manifest']
    preview = json.loads(diagnostic.read_text())
    for index, chosen in enumerate(preview['cases']):
        jobs.append(dict(scope='diagnostic', condition=chosen['condition'], case=chosen['case'],
                         diagnostic_label=f'{index+1:02d}'))
    original_keys = {json.dumps(case,sort_keys=True) for case in original['cases']}
    metadata_checked = set()
    for index, job in enumerate(jobs):
        job['job_id'] = index
        case = job['case']
        if json.dumps(case,sort_keys=True) not in original_keys:
            raise ValueError('Comparison case outside original verified validation membership')
        if case['path'] not in metadata_checked:
            if sha256(Path(case['path'])/'metadata.json') != case['metadata_sha256']:
                raise ValueError('Comparison case metadata changed')
            metadata_checked.add(case['path'])
    value = dict(config=c, scene_config_sha256=sha256(ROOT/c['scene_config']),
        replay_config_sha256=sha256(ROOT/c['replay_config']),
        selection_sha256=check['selection_sha256'], diagnostic_manifest_sha256=sha256(diagnostic),
        jobs=jobs, counts=counts, selection='ALL eligible validation windows, stride 128 plus interval end-aligned tails',
        aggregation='within family, then equal datasets; EXO conditions reported separately',
        training_ablation=False, independent_test=False,
        limitation='Scene branch has extra adaptation and different EXO proposal basis; not a pure scene-training ablation.',
        exo_intrinsics='Primary pair uses identical /256 intrinsics. Also evaluate original no-scene checkpoint with historical pixel intrinsics as a separate control, retaining the training-distribution difference.',
        full_validation=True, test_used=False)
    out.mkdir(parents=True, exist_ok=True)
    pin_json(out/'PROTOCOL.json', value)
    return value


def datasets(scene, c, jobs):
    """Build checked selected records only, avoiding unrelated array scans."""
    config = yaml.safe_load((ROOT/c['scene_config']).read_text())
    bundle = SceneSelection(ROOT/config['paths']['selection'])
    result = {}
    for mode in ('ego', 'exo'):
        checked = {r['path']:r for r in bundle.records(mode, 'validation')}
        records = {}
        for job in jobs:
            case = job['case']
            if case['mode'] != mode:
                continue
            record = checked[case['path']]
            if (case['start'] not in record['starts'] or record['family'] != case['family']
                    or case['metadata_sha256'] != record.get('metadata_sha256', case['metadata_sha256'])):
                raise ValueError('Comparison window leaves checked accepted ranges')
            records[case['member']] = record
        cls = CheckedLocalSceneDataset if mode == 'ego' else SceneExoDataset
        data = cls.__new__(cls)
        data.split, data.config, data.records = 'validation', config, records
        data.common_pelvis = scene.base.shape.rest_template[0].detach().cpu().numpy()
        data.bundle = data.selection_bundle = bundle
        data.paths = bundle.paths
        data.metadata_cache = bundle.metadata_cache
        data.array_cache, data.prefix_cache = OrderedDict(), OrderedDict()
        if mode == 'exo':
            data.bank = copy.deepcopy(scene.base.native_body.bank).cpu().eval()
        reference = (SceneFreeEgoDataset if mode == 'ego' else NativeRepairDataset).__new__(
            SceneFreeEgoDataset if mode == 'ego' else NativeRepairDataset)
        reference.__dict__.update(data.__dict__)
        reference.array = bundle.array
        result[mode] = data, reference
    return result


def paired_sample(data, job, c):
    case = job['case']; mode = case['mode']
    config = data[0].config
    pairs = []
    for dataset in data:
        kw = dict(member=case['member'], start=case['start'], yaw=0.)
        rng = np.random.default_rng(c['observation_seed_base']+case['case_id'])
        pair = (dataset.sample(rng, **kw) if mode == 'ego' else
                dataset.sample(rng, config['frontend'], condition=job['condition'], **kw))
        pairs.append(normalize(*pair, case['source'], mode))
    scene, baseline = pairs[0][0], pairs[1][0]
    historical = pairs[1]
    equal = list(INPUT_KEYS)+['motion', 'target_rest', 'body_format', 'sensor_profile']
    if mode == 'exo':
        equal.remove('camera_intrinsics')
        equal += ['initial_motion', 'initial_rest', 'initial_reliability', 'initial_root_reliability',
                  'estimate_available', 'joint_visible']
        if not np.array_equal(scene['camera_intrinsics'], baseline['camera_intrinsics']/256.):
            raise ValueError('Unexpected EXO camera preprocessing difference')
        baseline = dict(baseline, camera_intrinsics=baseline['camera_intrinsics']/256.)
        pairs[1] = baseline, pairs[1][1]
        equal.append('camera_intrinsics')
    for key in equal:
        if not np.array_equal(scene[key], baseline[key]):
            raise ValueError('Paired public input/target mismatch: '+key)
    identities = [p[1] for p in pairs]
    for key in ('path', 'start', 'family', 'source_input_sha256', 'split'):
        if identities[0][key] != identities[1][key]:
            raise ValueError('Paired source identity mismatch: '+key)
    if mode == 'exo':
        visible = np.asarray(data[0].bundle.array(data[0].bundle.metadata(
            data[0].records[case['member']], 'validation'), 'exo_position_available'))
        if visible[case['start']:case['start']+128].any(-1).sum() < config['data']['minimum_tracked_frames_per_window']:
            raise ValueError('EXO initializer no longer satisfies declared availability')
    if mode == 'exo':
        pairs.append(historical)
    return pairs


def masked_mean(value, mask):
    return float(value[mask].mean()) if bool(mask.any()) else None


def extra_metrics(points, truth, available=None):
    """All quantities are computed on unaligned global joint coordinates."""
    delta = points[:, 0]-truth[:, 0]
    root = delta.norm(dim=-1)*1000
    available = torch.ones(len(points), device=points.device, dtype=torch.bool) if available is None else available.bool()
    pair = available[1:] & available[:-1]
    triple = pair[1:] & pair[:-1]
    fourth = triple[1:] & triple[:-1]
    accel = torch.diff(points, n=2, dim=0)*900
    gt_accel = torch.diff(truth, n=2, dim=0)*900
    jerk = torch.diff(points, n=3, dim=0)*27000
    gt_jerk = torch.diff(truth, n=3, dim=0)*27000
    velocity_error = (torch.diff(points, dim=0)-torch.diff(truth, dim=0)).norm(dim=-1).mean(-1)*30000
    missing = ~available
    return dict(root_first_mm=float(root[0]), root_last_mm=float(root[-1]), root_worst_frame_mm=float(root.max()),
        root_error_vector_change_mm=float((delta[-1]-delta[0]).norm()*1000),
        height_bias_mm=float(delta[:, 1].mean()*1000), height_max_abs_mm=float(delta[:, 1].abs().max()*1000),
        tracked_acceleration_m_s2=masked_mean(accel.norm(dim=-1).mean(-1), triple),
        tracked_acceleration_error_m_s2=masked_mean((accel-gt_accel).norm(dim=-1).mean(-1), triple),
        gt_tracked_acceleration_m_s2=masked_mean(gt_accel.norm(dim=-1).mean(-1), triple),
        tracked_jerk_m_s3=masked_mean(jerk.norm(dim=-1).mean(-1), fourth),
        gt_tracked_jerk_m_s3=masked_mean(gt_jerk.norm(dim=-1).mean(-1), fourth),
        tracked_velocity_error_mm_s=masked_mean(velocity_error, pair),
        missing_root_mean_mm=masked_mean(root, missing),
        missing_root_worst_frame_mm=float(root[missing].max()) if bool(missing.any()) else None,
        missing_mpjpe_mm=masked_mean((points-truth).norm(dim=-1).mean(-1)*1000, missing))


@torch.inference_mode()
def predict(model, batch, mode, variant, c):
    scene = variant == 'scene_1000'
    if mode == 'exo':
        payload = repair_payload(batch)
        if scene:
            payload.update(local_payload(batch))
        motion = sample_denoising(model.exo if scene else model, payload,
            steps=c['exo_ddim_steps'], start_timestep=c['exo_start_timestep'], seed=c['diffusion_seed'])
        return motion, forward_kinematics(motion, batch['initial_rest']), None
    base = model.base
    payload = local_payload(batch) if scene else ego_payload(batch)
    motion = sample_native30(model.ego if scene else base, payload,
        steps=c['ego_ddim_steps'], seed=c['diffusion_seed'], use_scene=scene)
    shape = base.predicted_body(public_observations(payload), payload['body_format'])
    shape['head_camera_translation'][payload['sensor_profile'] == 1] = base.pv_translation
    points = forward_kinematics(motion, shape['rest'])
    camera = points[:, :, 15]+(global_rotations(motion)[:, :, 15] @
        shape['head_camera_translation'][:, None, :, None])[..., 0]
    residual = (camera-batch['camera'][..., :3]*2).norm(dim=-1)*1000
    return motion, points, dict(camera_position_mean_mm=float(residual.mean()),
                               camera_position_max_mm=float(residual.max()))


def setup(c, protocol, device):
    for name in ('scene', 'replay'):
        if sha256(ROOT/c[name+'_checkpoint']) != c[name+'_sha256']:
            raise ValueError('Pinned checkpoint changed: '+name)
    scene_config = yaml.safe_load((ROOT/c['scene_config']).read_text())
    replay_config = yaml.safe_load((ROOT/c['replay_config']).read_text())
    scene = make_model(scene_config, 'cpu')
    state = torch.load(ROOT/c['scene_checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    if state['step'] != c['scene_step'] or state['parent_sha256'] != c['replay_sha256']:
        raise ValueError('Wrong scene checkpoint or parent')
    load_adapter_state(scene, state['adapter'])
    baseline = replay_model((ROOT/c['replay_checkpoint']).parent.parent, replay_config)
    scene.requires_grad_(False).eval(); baseline.requires_grad_(False).eval()
    data = datasets(scene, c, protocol['jobs'])
    paths = load_paths(ROOT/'configs/paths.local.yaml')
    seal = contract(c, paths, ROOT/c['case_manifest'])
    preserved = json.loads((ROOT/'runs/preserved/current_joint_conditioning_20261029_165943/PRESERVATION.json').read_text())
    old = {r['source']:r['sha256'] for r in preserved['sources']['files']}
    for name, digest in seal['sources'].items():
        if name in old and name != 'scenemi/__main__.py' and old[name] != digest:
            raise ValueError('Original preserved model dependency changed: '+name)
    return {'scene_1000': scene.to(device), 'no_scene': baseline.to(device)}, data, seal


def summary(rows):
    values = {}
    for key in rows[0]['metrics']:
        families = defaultdict(list)
        for row in rows:
            value = row['metrics'][key]
            if value is not None:
                families[row['family']].append(value)
        values[key] = float(np.mean([np.mean(x) for x in families.values()])) if families else None
    return dict(windows=len(rows), families=len({r['family'] for r in rows}), metrics=values)


def report(out, protocol, complete=False):
    jobs = protocol['jobs']; rows = []; done = []
    for job in jobs:
        file = out/'rows'/f'{job["job_id"]:04d}.json'
        if not file.exists():
            continue
        saved = json.loads(file.read_text())
        if saved['protocol_sha256'] != sha256(out/'PROTOCOL.json') or saved['job'] != job:
            raise ValueError('Cached pair belongs to another protocol')
        if sha256(out/saved['joints_file']) != saved['joints_sha256']:
            raise ValueError('Paired prediction evidence changed')
        rows.extend(saved['rows']); done.append(job['job_id'])
    groups = {}
    for row in rows:
        if row['scope'] not in ('validation', 'balanced'):
            continue
        key = row['mode']+'_'+row['condition']+'_'+row['source']
        groups.setdefault(key, {})
        if row['variant'] not in groups[key]:
            group = [r for r in rows if r['scope'] in ('validation', 'balanced') and r['mode'] == row['mode']
                     and r['condition'] == row['condition'] and r['source'] == row['source']
                     and r['variant'] == row['variant']]
            groups[key][row['variant']] = summary(group)
    for group in groups.values():
        a, b = group['no_scene']['metrics'], group['scene_1000']['metrics']
        group['scene_minus_no_scene'] = {k:b[k]-a[k] if a[k] is not None and b[k] is not None else None for k in a}
    result = dict(state='complete' if len(done) == len(jobs) else 'partial', completed_pairs=len(done),
        total_pairs=len(jobs), full_validation=len(done) == len(jobs), test_used=False, training_ablation=False,
        per_dataset=groups, diagnostics=[r for r in rows if r['scope'] == 'diagnostic'],
        protocol_sha256=sha256(out/'PROTOCOL.json'), updated_at_utc=now())
    atomic_json(out/('RESULT.json' if result['state'] == 'complete' else 'PARTIAL_RESULT.json'), result)
    lines = ['# 第 1000 步场景模型与无场景 replay 对照', '',
        f'状态：{result["state"]}；完成 {len(done)}/{len(jobs)} 对。负的差值代表场景模型误差更低。', '',
        '完整三数据集验证清单；同窗、同前端扰动、同设备同 seed、固定 batch=1。诊断九例不混入整体统计。',
        'EXO 主对照使用完全相同的 /256 内参，另保留历史像素内参的无场景输出。额外适配与结构差异使本结果不能当作纯场景训练消融。', '',
        '| 分组 | 窗口 | 无场景 MPJPE | 场景 MPJPE | Δ MPJPE | 无场景 root | 场景 root | 无场景 MPJVE | 场景 MPJVE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key, group in groups.items():
        a, b = group['no_scene']['metrics'], group['scene_1000']['metrics']
        values = [a['mpjpe_mm'], b['mpjpe_mm'], b['mpjpe_mm']-a['mpjpe_mm'],
                  a['root_error_mm'], b['root_error_mm'], a['mpjve_mm_s'], b['mpjve_mm_s']]
        lines.append(f'| {key} | {group["no_scene"]["windows"]} | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines += ['', 'MPJPE/root 单位 mm，MPJVE 单位 mm/s；全局指标不做根对齐或贴地。',
        'PA-MPJPE、缺失区域、最坏帧、相机一致性与正常跟踪段的加速度/jerk 见 JSON 逐例记录。',
        '这是已训练版本对照，尚非独立测试结果。只有所有清单窗口完成后才能称为完整验证结果。失败不静默跳过。']
    (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    return result


@torch.inference_mode()
def evaluate(c, out, protocol, device, preflight=False, limit=None):
    deterministic(c['diffusion_seed'], device)
    models, data, seal = setup(c, protocol, device)
    seal.update(comparison_device=device, logical_batch_size=1)
    seal['sha256'] = hashlib.sha256(json.dumps({k:v for k,v in seal.items() if k != 'sha256'},sort_keys=True).encode()).hexdigest()
    pin_json(out/'EVALUATOR_SEAL.json', seal)
    (out/'rows').mkdir(exist_ok=True); (out/'joints').mkdir(exist_ok=True)
    if not (out/'evaluator_sources.tar.gz').exists():
        with tarfile.open(out/'evaluator_sources.tar.gz', 'w:gz') as archive:
            for name in seal['sources']:
                path = Path(name) if Path(name).is_absolute() else ROOT/name
                archive.add(path, arcname=name if not Path(name).is_absolute() else 'external/'+path.name)
    jobs = protocol['jobs']
    if preflight:
        jobs = [next(j for j in jobs if j['scope'] == 'validation' and j['case']['mode'] == mode
            and j['case']['source'] == source and j['condition'] == ('ego' if mode == 'ego' else 'gap'))
            for mode in ('ego', 'exo') for source in c['sources']]
    if limit is not None:
        jobs = jobs[:limit]
    started = time.monotonic(); timings = []
    try:
        for job in jobs:
            identifier = job['job_id']; rowfile = out/'rows'/f'{identifier:04d}.json'
            if rowfile.exists():
                continue
            case = job['case']; mode = case['mode']; tick = time.monotonic()
            atomic_json(out/'STATUS.json', dict(state='evaluating', pid=os.getpid(), job=job,
                completed_pairs=len(list((out/'rows').glob('*.json'))), total_pairs=len(protocol['jobs']),
                device=device, updated_at_utc=now()))
            pairs = paired_sample(data[mode], job, c)
            batches = [collate([p[0]], device) for p in pairs]
            scene_batch, baseline_batch = batches[:2]
            truth = forward_kinematics(scene_batch['motion'], scene_batch['target_rest'])
            arrays = dict(truth=truth[0].cpu().numpy(), camera=scene_batch['camera'][0].cpu().numpy())
            initial = forward_kinematics(scene_batch['initial_motion'], scene_batch['initial_rest']) if mode == 'exo' else None
            available = scene_batch['estimate_available'][0] > 0 if mode == 'exo' else None
            visible = ((scene_batch['joint_visible'] > 0) & (scene_batch['estimate_available'][..., None] > 0)) if mode == 'exo' else None
            entries = []
            variants = [('no_scene', baseline_batch), ('scene_1000', scene_batch)]
            if mode == 'exo':
                variants.append(('historical_no_scene', batches[2]))
            for variant, batch in variants:
                motion, points, camera_metrics = predict(models['scene_1000' if variant == 'scene_1000' else 'no_scene'], batch, mode, variant, c)
                metrics = (repair_rows(points, truth, initial, visible, job['condition']) if mode == 'exo'
                           else metric_rows(points, truth))[0]
                metrics.update(extra_metrics(points[0], truth[0], available))
                if camera_metrics:
                    metrics.update(camera_metrics)
                arrays[variant] = points[0].cpu().numpy()
                entries.append(dict(scope=job['scope'], diagnostic_label=job.get('diagnostic_label'),
                    mode=mode, source=case['source'], family=case['family'], condition=job['condition'],
                    variant=variant, metrics=metrics))
            if mode == 'exo':
                arrays.update(initial=initial[0].cpu().numpy(), estimate_available=available.cpu().numpy(),
                              joint_visible=visible[0].cpu().numpy())
            file = out/'joints'/f'{identifier:04d}.npz'
            with file.with_suffix('.partial').open('wb') as stream:
                np.savez_compressed(stream, **arrays)
            file.with_suffix('.partial').replace(file)
            record = dict(job=job, rows=entries, identities=[p[1] for p in pairs],
                common_public_input_parity=True, input_sha256=[tensor_digest(ego_payload(b) if mode == 'ego' else repair_payload(b))
                    for b in batches],
                joints_file=str(file.relative_to(out)), joints_sha256=sha256(file),
                protocol_sha256=sha256(out/'PROTOCOL.json'), evaluator_sha256=seal['sha256'],
                elapsed_seconds=time.monotonic()-tick)
            atomic_json(rowfile, record); timings.append(record['elapsed_seconds'])
            print(json.dumps(dict(completed_pair=identifier, mode=mode, source=case['source'], condition=job['condition'],
                seconds=record['elapsed_seconds'], metrics={r['variant']:r['metrics']['mpjpe_mm'] for r in entries})),flush=True)
            if preflight or len(timings)%1000 == 0:
                report(out, protocol)
        result = report(out, protocol)
        state = 'complete' if result['state'] == 'complete' else ('preflight_passed' if preflight else 'partial')
        atomic_json(out/'STATUS.json', dict(state=state, pid=os.getpid(), completed_pairs=result['completed_pairs'],
            total_pairs=result['total_pairs'], device=device, elapsed_seconds=time.monotonic()-started, updated_at_utc=now()))
        if preflight:
            atomic_json(out/'PREFLIGHT.json', dict(state='passed', evaluated_pairs=len(jobs), device=device,
                new_pair_seconds=timings, paired_inputs_verified=True, evaluator_sha256=seal['sha256'],
                full_validation=False, test_used=False, updated_at_utc=now()))
        return result
    except BaseException as error:
        atomic_json(out/'STATUS.json', dict(state='failed', pid=os.getpid(), error=repr(error),
            job=job, device=device, updated_at_utc=now()))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'preflight', 'run', 'report'))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(argv); ensure_bounded()
    c = read_config(args.config); out = ROOT/c['output']
    if args.device == 'cuda':
        from experiments.offline_camera_retrain_v1.ops import resources
        admission = resources.admission('cuda', c['runtime']['memory_gib'])
        out.mkdir(parents=True,exist_ok=True); atomic_json(out/'ADMISSION.json',admission)
        if not admission['admitted']:
            raise RuntimeError('Standard GPU admission refused; use CPU or obtain a scoped sharing authorization')
        os.environ['SCENEMI_ALLOCATOR_GIB'] = str(c['runtime']['allocator_gib'])
        from scenemi.parallel_resources import configure_cuda
        configure_cuda()
    protocol = prepare(c, out)
    with (out/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'prepare':
            atomic_json(out/'STATUS.json',dict(state='prepared',total_pairs=len(protocol['jobs']),updated_at_utc=now()))
            print(json.dumps(dict(prepared_pairs=len(protocol['jobs']),output=str(out))),flush=True)
        elif args.action == 'report':
            print(json.dumps(report(out,protocol)),flush=True)
        else:
            evaluate(c, out, protocol, args.device, preflight=args.action == 'preflight',limit=args.limit)


if __name__ == '__main__':
    main()
