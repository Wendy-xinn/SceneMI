"""Evaluate released head-motion baselines on the retained four-corpus windows.

Preparation uses the existing SceneMI environment. Public-model inference uses
the isolated Python 3.12 EgoAllo environment. Neither path starts training.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ORDER = ('amass', 'egobody', 'trumans', 'grail')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def aggregate(rows):
    """Same equal-window-within-family, equal-family, equal-dataset reduction."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['identity']['dataset'], row['identity']['family'])].append(row['metrics'])
    per_source = defaultdict(list)
    for (source, _), members in grouped.items():
        per_source[source].append({k: float(np.mean([r[k] for r in members])) for k in members[0]})
    by_source = {s: {k: float(np.mean([r[k] for r in families])) for k in families[0]}
                 for s, families in per_source.items()}
    macro = {k: float(np.mean([by_source[s][k] for s in SOURCE_ORDER if s in by_source]))
             for k in next(iter(by_source.values()))}
    return dict(windows=len(rows), counts=dict(Counter(r['identity']['dataset'] for r in rows)),
                by_dataset=by_source, dataset_macro_mean=macro,
                aggregation='equal source dataset, then equal family')


def bounded(c, device):
    relative = Path('/proc/self/cgroup').read_text().strip().split(':', 2)[2]
    maximum = (Path('/sys/fs/cgroup') / relative.lstrip('/') / 'memory.max').read_text().strip()
    if maximum == 'max' or int(maximum) > c['runtime']['memory_gib'] * 2**30 or len(os.sched_getaffinity(0)) > c['runtime']['cpu_count']:
        raise RuntimeError('Use the baseline resource wrapper with its configured CPU/RAM bounds')
    torch.set_num_threads(c['runtime']['cpu_count'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(c['runtime']['torch_allocator_gib'] * 2**30 /
                                                  torch.cuda.get_device_properties(0).total_memory)


def manifest(c):
    cases = json.loads((ROOT / c['manifest']).read_text())['full']
    if dict(Counter(x['dataset'] for x in cases)) != c['expected_windows']:
        raise ValueError('Fixed validation counts changed')
    if c['fine_tuning'] or c['test_used']:
        raise ValueError('This protocol is pretrained validation only')
    return cases


def prepare(c):
    from scenemi.config import load_paths
    from scenemi.data.ego import FourEgoData
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    bounded(c, 'cpu')
    out = ROOT / c['output']; folder = out / 'data'; folder.mkdir(parents=True, exist_ok=True)
    cases = manifest(c); count = len(cases)
    if (folder / 'DATA_PROTOCOL.json').exists():
        old = json.loads((folder / 'DATA_PROTOCOL.json').read_text())
        if old['manifest_sha256'] != digest(ROOT / c['manifest']):
            raise ValueError('Prepared manifest differs')
        for name, expected in old['array_sha256'].items():
            if digest(folder / name) != expected:
                raise ValueError('Prepared baseline inputs changed')
        print('Verified existing baseline data export', flush=True); return
    paths = load_paths(ROOT / 'configs/paths.local.yaml')
    reference = yaml.safe_load((ROOT / c['reference_config']).read_text())
    data = FourEgoData(reference, paths, 'validation')
    arrays = {
        'camera.npy': np.lib.format.open_memmap(folder / 'camera.npy', mode='w+', dtype='float32', shape=(count, 128, 9)),
        'sensor_profile.npy': np.lib.format.open_memmap(folder / 'sensor_profile.npy', mode='w+', dtype='int64', shape=(count,)),
        'target.npy': np.lib.format.open_memmap(folder / 'target.npy', mode='w+', dtype='float32', shape=(count, 128, 22, 3)),
    }
    with (folder / 'cases.jsonl').open('w') as stream:
        for offset in range(0, count, 32):
            chunk = cases[offset:offset + 32]
            pairs = [data.sample_case(case) for case in chunk]
            values = [p[0] for p in pairs]
            if not all(np.all(v['camera_available'] == 1) for v in values):
                raise ValueError('Baseline input needs an explicit missing-observation policy')
            truth = forward_kinematics(torch.as_tensor(np.stack([v['motion'] for v in values])),
                                       torch.as_tensor(np.stack([v['target_rest'] for v in values]))).numpy()
            end = offset + len(chunk)
            arrays['camera.npy'][offset:end] = np.stack([v['camera'] for v in values])
            arrays['sensor_profile.npy'][offset:end] = [v['sensor_profile'] for v in values]
            arrays['target.npy'][offset:end] = truth
            for (sample, identity), case in zip(pairs, chunk):
                stream.write(json.dumps(dict(identity=identity, case=case), allow_nan=False) + '\n')
            print(f'Prepare validation inputs {end}/{count}', flush=True)
    for value in arrays.values():
        value.flush()
    scene_paths = yaml.safe_load((ROOT / reference['scene_paths']).read_text())
    mount_path = Path(scene_paths['pv_mount'])
    mount = json.loads(mount_path.read_text())['camera_wearer']
    if mount['fitting_split'] != 'train':
        raise ValueError('Camera calibration must come from train only')
    record = dict(state='complete', windows=count, counts=c['expected_windows'],
        manifest_sha256=digest(ROOT / c['manifest']), calibration=mount,
        calibration_sha256=digest(mount_path), personal_shape_input=False, test_used=False,
        array_sha256={name: digest(folder / name) for name in arrays},
        cases_sha256=digest(folder / 'cases.jsonl'))
    write_json(folder / 'DATA_PROTOCOL.json', record)
    print(json.dumps({k: v for k, v in record.items() if k != 'calibration'}, indent=2), flush=True)


def inputs(c):
    folder = ROOT / c['output'] / 'data'
    info = json.loads((folder / 'DATA_PROTOCOL.json').read_text())
    if info['manifest_sha256'] != digest(ROOT / c['manifest']):
        raise ValueError('Input manifest identity changed')
    rows = [json.loads(line) for line in (folder / 'cases.jsonl').read_text().splitlines()]
    arrays = {name: np.load(folder / (name + '.npy'), mmap_mode='r')
              for name in ('camera', 'sensor_profile', 'target')}
    return info, rows, arrays


def metric_rows(prediction, target):
    from scenemi.metrics_batch import metric_tensors
    from scenemi.metrics_temporal import temporal_metrics
    values = metric_tensors(prediction, target)
    values.update(temporal_metrics(prediction, target))
    names = list(values)
    matrix = torch.stack([values[k].double() for k in names], -1).cpu().tolist()
    return [dict(zip(names, row)) for row in matrix]


def seal(c, config):
    out = ROOT / c['output']
    sources = [Path(__file__), ROOT / 'scenemi/baseline_models.py',
               ROOT / 'scenemi/metrics_temporal.py', ROOT / 'scenemi/metrics_batch.py',
               ROOT / 'scenemi/metrics.py', ROOT / 'scenemi/baseline_runtime.py', config,
               ROOT / c['manifest'], out / 'data/DATA_PROTOCOL.json']
    allo_root = ROOT / c['egoallo_repository'] / 'src/egoallo'
    sources += [allo_root / name for name in ('network.py', 'inference_utils.py', 'sampling.py',
        'fncsmpl.py', 'fncsmpl_extensions.py', 'fncsmpl_jax.py', 'guidance_optimizer_jax.py',
        'hand_detection_structs.py', 'tensor_dataclass.py')]
    sources += list((allo_root / 'transforms').glob('*.py'))
    sources += list((ROOT / c['egogvae_repository'] / 'model').glob('*.py'))
    sources += [ROOT / c['egogvae_repository'] / 'utils/quat.py']
    sources += list((out / 'metric_sources').glob('*.py'))
    assets = [ROOT / c['ours_checkpoint'], ROOT / c['egogvae_checkpoint'],
              ROOT / c['egoallo_checkpoint'] / 'model.safetensors', Path(c['smplh_model'])]
    value = dict(config=c, source_sha256={str(p.resolve()): digest(p) for p in sources},
                 asset_sha256={str(p.resolve()): digest(p) for p in assets},
                 fine_tuning=False, test_used=False, sample_selection='one fixed-seed sample, no best-of-K',
                 calibration='fixed neutral virtual-camera-to-CPF and train-fit PV mount; no target shape',
                 initial_delta='first observed CPF repeated, without an extra context frame',
                 metrics=dict(jitter='mean 22-joint third finite difference * 30^3 / 100, units 100 m/s^3',
                    foot_sliding='AvatarJLM: GT displacement norm <=0.01m/frame; zero noncontact predicted XYZ deltas; abs().mean()*100, cm/frame',
                    horizontal='supplement: horizontal L2, contact-normalized, mm/contact-frame; zero with explicit has_contact=0 for empty support',
                    egoallo_skating='supplement: released height-weighted cumulative horizontal L1, meters/window',
                    pose='same SceneMI standard PA and unaligned 22-joint metrics as supplied table'),
                 metric_sources=['https://github.com/zxz267/AvatarJLM/blob/main/test.py',
                                 'https://github.com/zxz267/AvatarJLM/blob/main/utils/utils_metric.py',
                                 'https://github.com/brentyi/egoallo/blob/0c20ae8c8d48be4ff682803ea523bcb00a6a708d/src/egoallo/metrics_helpers.py'],
                 upstream_metric_limit='Pinned EgoGVAE metrics helper lacks compute_jitter and compute_foot_skate_gt_contact_batched imported by its evaluation entry.',
                 pretraining_overlap_note='Released EgoAllo config trained on AMASS train+val; AMASS validation is not an unseen pretraining split for that weight.',
                 preflight_sha256=digest(out / 'preflight/RESULT_cuda.json'))
    value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    path = out / 'EVALUATION_PROTOCOL.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Evaluation closure changed: preserve the existing evaluation and use a new output')
    write_json(path, value)
    return value


def summarize(c, model):
    folder = ROOT / c['output'] / model
    rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
    result = aggregate(rows)
    result.update(model=model, fine_tuned=False, test_used=False, state='complete')
    if result['counts'] != c['expected_windows']:
        raise ValueError('Full comparison cannot silently omit windows')
    if model == 'ours':
        reference = json.loads((ROOT / c['reference_metrics'] / 'RESULT.json').read_text())
        delta = max(abs(result['by_dataset'][s][k] - reference['by_dataset'][s][k])
                    for s in SOURCE_ORDER for k in ('mpjpe_mm', 'pa_mpjpe_mm', 'root_error_mm', 'head_error_mm', 'mpjve_mm_s'))
        result['reference_table_max_difference'] = delta
        if delta > .02:
            raise ValueError(f'Recomputed model does not reproduce supplied table: {delta}')
    write_json(folder / 'RESULT.json', result)
    return result


def evaluate(c, config, model_name, device):
    bounded(c, device)
    info, rows, arrays = inputs(c)
    protocol = seal(c, config)
    variants = ['egoallo', 'egoallo_raw'] if model_name == 'egoallo' else [model_name]
    count = len(rows); folders = {name: ROOT / c['output'] / name for name in variants}
    progress = {}
    for name, folder in folders.items():
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'PROGRESS.json'
        p = json.loads(path.read_text()) if path.exists() else dict(completed=0)
        if p.get('protocol_sha256', protocol['sha256']) != protocol['sha256']:
            raise ValueError('Resume protocol differs')
        progress[name] = p['completed']
    begin = min(progress.values())
    if begin == count:
        for name in variants:
            print(json.dumps(summarize(c, name)), flush=True)
        return
    prediction_files = {}
    for name, folder in folders.items():
        path = folder / 'predictions.npy'
        prediction_files[name] = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+',
                                    dtype='float32', shape=(count, 128, 22, 3))
        path = folder / 'metrics.jsonl'
        old = path.read_text().splitlines() if path.exists() else []
        path.write_text(''.join(line + '\n' for line in old[:begin]))
    adapter = ours = data = None
    if model_name == 'ours':
        from scenemi.checkpoint import load_trained_model
        from scenemi.config import load_paths
        from scenemi.data.ego import FourEgoData
        ours, reference, _ = load_trained_model(ROOT / c['ours_checkpoint'])
        ours = ours.to(device)
        data = FourEgoData(reference, load_paths(ROOT / 'configs/paths.local.yaml'), 'validation')
        batch_size = reference['validation']['eval_batch_size']
    else:
        from scenemi.baseline_models import PublicBaselines
        adapter = PublicBaselines(c, device, info['calibration'])
        adapter.load(model_name); batch_size = c['batch_size']
    started = time.monotonic()
    for offset in range(begin, count, batch_size):
        end = min(offset + batch_size, count)
        chunk = rows[offset:end]
        target = torch.as_tensor(np.array(arrays['target'][offset:end]), device=device)
        with torch.inference_mode():
            if model_name == 'ours':
                from scenemi.data.ego import collate, payload
                from experiments.offline_camera_retrain_v1.native30_model import sample_native30
                from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
                batch = collate([data.sample_case(row['case'])[0] for row in chunk], device)
                sample = sample_native30(ours, payload(batch), steps=reference['validation']['ddim_steps'],
                                         seed=reference['validation']['seed'], use_scene=False)
                _, details = ours(sample, torch.zeros(len(chunk), device=device, dtype=torch.long),
                                  payload(batch), use_scene=False, return_details=True)
                predicted = {'ours': forward_kinematics(sample, details['rest'])}
            else:
                cpf = adapter.cpf(np.array(arrays['camera'][offset:end]), np.array(arrays['sensor_profile'][offset:end]))
                ids = [r['case']['case_id'] for r in chunk]
                if model_name == 'egogvae':
                    predicted = {'egogvae': adapter.gvae_sample(cpf, ids)}
                else:
                    raw = adapter.allo_sample(cpf, ids)
                    raw_points = adapter.allo_joints(raw, cpf)
                    guided = adapter.guided(raw, cpf)
                    predicted = {'egoallo': adapter.allo_joints(guided, cpf), 'egoallo_raw': raw_points}
            for name, points in predicted.items():
                measures = metric_rows(points, target)
                prediction_files[name][offset:end] = points.cpu().numpy()
                prediction_files[name].flush()
                with (folders[name] / 'metrics.jsonl').open('a') as stream:
                    for row, measure in zip(chunk, measures):
                        stream.write(json.dumps(dict(**row, metrics=measure), allow_nan=False) + '\n')
                    stream.flush(); os.fsync(stream.fileno())
                elapsed = time.monotonic() - started
                state = dict(state='running', completed=end, total=count, elapsed_seconds=elapsed,
                    remaining_seconds=(count - end) * elapsed / (end - begin),
                    protocol_sha256=protocol['sha256'], test_used=False)
                write_json(folders[name] / 'PROGRESS.json', state)
        print(f'{model_name}: {end}/{count}; {(end - begin) / (time.monotonic() - started):.2f} windows/s', flush=True)
    for name in variants:
        result = summarize(c, name)
        write_json(folders[name] / 'PROGRESS.json', dict(state='complete', completed=count, total=count,
                    protocol_sha256=protocol['sha256']))
        print(json.dumps(result), flush=True)


def preflight(c, device):
    from scenemi.baseline_models import PublicBaselines
    from egoallo.sampling import run_sampling_with_stitching
    bounded(c, device)
    info, rows, arrays = inputs(c)
    selected = [next(i for i, row in enumerate(rows) if row['identity']['dataset'] == source) for source in SOURCE_ORDER]
    adapter = PublicBaselines(c, device, info['calibration'])
    cpf = adapter.cpf(np.array(arrays['camera'][selected]), np.array(arrays['sensor_profile'][selected]))
    adapter.load('egoallo')
    seed = c['seed'] + rows[selected[0]]['case']['case_id']
    torch.manual_seed(seed)
    reference = run_sampling_with_stitching(adapter.egoallo, body_model=adapter.body,
        guidance_mode='no_hands', guidance_inner=False, guidance_post=False, Ts_world_cpf=cpf[0],
        floor_z=0., hamer_detections=None, aria_detections=None, num_samples=1,
        device=adapter.device, guidance_verbose=False)
    custom = adapter.allo_sample(cpf[:1], [rows[selected[0]]['case']['case_id']])
    error = float((reference.pack() - custom.pack()).abs().max())
    if error > 2e-5:
        raise ValueError(f'Vectorized EgoAllo sampler changed released sampling: {error}')
    # The independent-conditions guidance must also match the released B=1 path.
    from egoallo.guidance_optimizer_jax import do_guidance_optimization
    single, _ = do_guidance_optimization(cpf[0, 1:], custom, adapter.body,
         'no_hands', 'post', None, None, False)
    batched = adapter.guided(custom, cpf[:1])
    guidance_error = float((single.pack() - batched.pack()).abs().max())
    if guidance_error > 2e-4:
        raise ValueError(f'Batched guidance changed released optimization: {guidance_error}')
    records = []
    for name in ('egogvae', 'egoallo'):
        with torch.inference_mode():
            ids = [rows[i]['case']['case_id'] for i in selected]
            if name == 'egoallo':
                raw = adapter.allo_sample(cpf, ids)
                guided = adapter.guided(raw, cpf)
                prediction = adapter.allo_joints(guided, cpf)
            else:
                prediction = adapter.gvae_sample(cpf, ids)
            target = torch.as_tensor(np.array(arrays['target'][selected]), device=device)
            values = metric_rows(prediction, target)
        records.extend(dict(model=name, dataset=source, metrics=value)
                       for source, value in zip(SOURCE_ORDER, values))
    result = dict(state='passed', sampler_max_difference=error,
        guidance_max_difference=guidance_error, rows=records,
        target_fields_enter_inference=False, device=device)
    write_json(ROOT / c['output'] / 'preflight' / ('RESULT_' + device + '.json'), result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'preflight', 'evaluate', 'summarize'))
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/pretrained_ego_baselines.yaml')
    parser.add_argument('--model', choices=('egoallo', 'egoallo_raw', 'egogvae', 'ours'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    config = args.config.resolve(); c = yaml.safe_load(config.read_text())
    if args.action == 'prepare':
        prepare(c)
    elif args.action == 'preflight':
        preflight(c, args.device)
    elif args.action == 'evaluate':
        if args.model not in ('egoallo', 'egogvae', 'ours'):
            parser.error('Choose egoallo, egogvae, or ours')
        evaluate(c, config, args.model, args.device)
    else:
        print(json.dumps(summarize(c, args.model), indent=2))


if __name__ == '__main__':
    main()
