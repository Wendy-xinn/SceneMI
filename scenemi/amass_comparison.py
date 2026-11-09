"""Matched held-out AMASS evaluation, using existing model and metric adapters.

This entry preserves sealed training sources. Preparation pins reflection and
window identity explicitly and exports one observation/target set for all models.
Use the current ``python -m scenemi resources run`` CPU/RAM wrapper.
"""
import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np
import torch
import yaml

from scenemi.baseline_evaluation import digest, metric_rows, write_json
from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import AmassDataset, collate, inference_payload
from scenemi.evaluation import summarize, test_windows, verify_training_seal
from scenemi.execution import configure_runtime
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

DEFAULT = ROOT / 'configs/amass_test_comparison.yaml'
MODELS = ('ours', 'ours_raw', 'egogvae', 'egoallo', 'egoallo_raw')


def check_identity(case, identity):
    expected = dict(source=case['source'], family=case['family'], corpus=case['corpus'],
                    reflected=case['reflected'], variant_sha256=case['variant_sha256'],
                    start=case['start_after_0b'], split='test', mode='ego', yaw=0.)
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Loaded test identity differs from manifest: {case['case_id']}")


def cases_for(c, paths):
    if digest(ROOT / c['manifest']) != c['manifest_sha256']:
        raise ValueError('Frozen test manifest changed')
    data = AmassDataset(ROOT / c['amass_index'], paths, 'test')
    cases = test_windows(dict(families=data.members))
    original = json.loads((ROOT / c['manifest']).read_text())['manifest']
    if (cases != original or len(cases) != c['expected_windows']
            or len({r['family'] for r in cases}) != c['expected_families']
            or dict(Counter(r['corpus'] for r in cases)) != c['expected_corpora']):
        raise ValueError('Test membership differs from the frozen protocol')
    return data, cases


def prepare(c):
    paths = load_paths(ROOT / c['paths'])
    data, cases = cases_for(c, paths)
    folder = ROOT / c['output'] / 'data'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'DATA_PROTOCOL.json').exists():
        load_inputs(c)
        print('Existing test arrays and actual source identities verified', flush=True)
        return
    arrays, rows = {}, []
    for i, case in enumerate(cases):
        sample, identity = data.sample(np.random.default_rng(c['seed']), 'ego',
            member=case['member'], start=case['start_after_0b'],
            reflected=case['reflected'], yaw=0., temporal_dropout=None)
        check_identity(case, identity)
        if not np.all(sample['camera_available'] == 1):
            raise ValueError('This protocol requires complete observed camera trajectories')
        if not arrays:
            for name, value in sample.items():
                arrays[name] = np.lib.format.open_memmap(folder / ('sample_' + name + '.npy'),
                    mode='w+', dtype=value.dtype, shape=(len(cases),) + value.shape)
            arrays['target'] = np.lib.format.open_memmap(folder / 'target.npy', mode='w+',
                dtype='float32', shape=(len(cases), 128, 22, 3))
        for name, value in sample.items():
            arrays[name][i] = value
        batch = collate([sample])
        arrays['target'][i] = forward_kinematics(batch['motion'], batch['target_rest'])[0].numpy()
        rows.append(dict(case=case, identity=identity))
        if (i + 1) % 32 == 0 or i + 1 == len(cases):
            print(f'Prepare exact test windows {i + 1}/{len(cases)}', flush=True)
    for array in arrays.values():
        array.flush()
    (folder / 'cases.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    record = dict(state='complete', windows=len(cases), source_families=c['expected_families'],
        manifest_sha256=c['manifest_sha256'], index_sha256=digest(ROOT / c['amass_index']),
        cases_sha256=digest(folder / 'cases.jsonl'), test_used=True,
        reflection_pinned_and_verified=True, source_windows_verified=True,
        array_sha256={p.name: digest(p) for p in sorted(folder.glob('*.npy'))},
        sample_keys=[k for k in arrays if k != 'target'], personal_shape_input=False,
        calibration=dict(rotation=np.eye(3).tolist(), translation_m=[0., 0., 0.]),
        calibration_note='PV mount unused: all cases are virtual camera, sensor_profile=0; fixed neutral virtual-camera-to-CPF in PublicBaselines',
        coordinate_system='Y up, meters, common 22 SMPL-H joints, native target shape',
        camera='same 128 observed eye-camera frames; fixed neutral conversion to CPF; first CPF repeated as initial context')
    write_json(folder / 'DATA_PROTOCOL.json', record)


def load_inputs(c):
    folder = ROOT / c['output'] / 'data'
    info = json.loads((folder / 'DATA_PROTOCOL.json').read_text())
    if (info['manifest_sha256'] != c['manifest_sha256']
            or digest(ROOT / c['manifest']) != c['manifest_sha256']
            or digest(ROOT / c['amass_index']) != info['index_sha256']
            or digest(folder / 'cases.jsonl') != info['cases_sha256']):
        raise ValueError('Test export identities changed')
    for name, expected in info['array_sha256'].items():
        if digest(folder / name) != expected:
            raise ValueError(f'Test export changed: {name}')
    rows = [json.loads(line) for line in (folder / 'cases.jsonl').read_text().splitlines()]
    original = json.loads((ROOT / c['manifest']).read_text())['manifest']
    if [r['case'] for r in rows] != original or len(rows) != c['expected_windows']:
        raise ValueError('Exported windows differ from test manifest')
    for row in rows:
        check_identity(row['case'], row['identity'])
    arrays = {k: np.load(folder / ('sample_' + k + '.npy'), mmap_mode='r') for k in info['sample_keys']}
    arrays['target'] = np.load(folder / 'target.npy', mmap_mode='r')
    return info, rows, arrays


def protocol(c, config):
    out = ROOT / c['output']
    sources = [Path(__file__), config, ROOT / c['manifest'], ROOT / c['paths'],
        ROOT / 'scenemi/baseline_models.py', ROOT / 'scenemi/baseline_evaluation.py',
        ROOT / 'scenemi/metrics_temporal.py', ROOT / 'scenemi/metrics_batch.py',
        ROOT / 'scenemi/metrics.py', ROOT / 'scenemi/evaluation.py',
        ROOT / 'scenemi/data/amass_dataset.py', ROOT / 'scenemi/execution.py',
        ROOT / 'scenemi/gpu_policy.py', ROOT / 'configs/runtime_policy.yaml',
        out / 'data/DATA_PROTOCOL.json']
    upstream = ROOT / c['egoallo_repository'] / 'src/egoallo'
    sources += [upstream / name for name in ('network.py', 'inference_utils.py', 'sampling.py',
        'fncsmpl.py', 'fncsmpl_extensions.py', 'fncsmpl_jax.py', 'guidance_optimizer_jax.py',
        'tensor_dataclass.py', 'hand_detection_structs.py')]
    sources += sorted((upstream / 'transforms').glob('*.py'))
    sources += sorted((ROOT / c['egogvae_repository'] / 'model').glob('*.py'))
    sources += [ROOT / c['egogvae_repository'] / 'utils/quat.py']
    allo = ROOT / c['egoallo_checkpoint']
    sources += [allo.parent / 'model_config.yaml', allo.parent / 'run_config.yaml']
    assets = {str((ROOT / c[name + '_checkpoint']).resolve()): c[name + '_sha256']
              for name in ('ours', 'egogvae')}
    assets[str((allo / 'model.safetensors').resolve())] = c['egoallo_sha256']
    for name, expected in assets.items():
        if digest(name) != expected:
            raise ValueError(f'Pinned checkpoint changed: {name}')
    assets[c['smplh_model']] = digest(c['smplh_model'])
    value = dict(config=c, source_sha256={str(p.resolve()): digest(p) for p in sources},
        asset_sha256=assets, samples_per_window=1, best_of_k=False, test_used=True,
        ours_selection='80k selected on validation before this test; no test selection',
        seed_policy='ours: seed 777 per B=1 window, matching retained evaluation; official models: 777 + case_id',
        primary='window_mean; family_macro_mean and per-corpus means also retained',
        metrics='shared metric_rows: standard PA; velocity mm/s; jerk m/s^3; jitter=jerk/100; GT-contact foot sliding cm/frame and horizontal mm/contact-frame',
        egoallo_guidance='released no_hands, post only; unguided retained separately',
        limitations=['Same new frozen test windows for all models, not exact paper random windows.',
            'Fixed neutral eye-camera-to-CPF conversion, not target personal calibration.',
            'EgoAllo official training uses train+val; ours and EgoGVAE train-only.',
            'Raw anchored/unanchored variants are diagnostics, not additional checkpoint selection.'])
    value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    path = out / 'EVALUATION_PROTOCOL.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Evaluation closure changed; use a new output directory')
    write_json(path, value)
    return value


def public_adapter(c, device, info):
    from scenemi.baseline_models import PublicBaselines
    return PublicBaselines(c, device, info['calibration'])


def preflight(c, config, device):
    from egoallo.sampling import run_sampling_with_stitching
    from egoallo.guidance_optimizer_jax import do_guidance_optimization
    info, rows, arrays = load_inputs(c)
    seal = protocol(c, config)
    adapter = public_adapter(c, device, info)
    cpf = adapter.cpf(np.array(arrays['camera'][:1]), np.zeros(1, dtype=np.int64))
    adapter.load('egoallo')
    torch.manual_seed(c['seed'] + rows[0]['case']['case_id'])
    reference = run_sampling_with_stitching(adapter.egoallo, body_model=adapter.body,
        guidance_mode='no_hands', guidance_inner=False, guidance_post=False, Ts_world_cpf=cpf[0],
        floor_z=0., hamer_detections=None, aria_detections=None, num_samples=1,
        device=adapter.device, guidance_verbose=False)
    custom = adapter.allo_sample(cpf, [rows[0]['case']['case_id']])
    sampler_error = float((reference.pack() - custom.pack()).abs().max())
    single, _ = do_guidance_optimization(cpf[0, 1:], custom, adapter.body,
        'no_hands', 'post', None, None, False)
    batched = adapter.guided(custom, cpf)
    guidance_error = float((single.pack() - batched.pack()).abs().max())
    if sampler_error > 2e-5 or guidance_error > 2e-4:
        raise ValueError(f'Public adapter differs from released inference: {sampler_error}, {guidance_error}')
    result = dict(state='passed', protocol_sha256=seal['sha256'],
        sampler_max_difference=sampler_error, guidance_max_difference=guidance_error,
        source_identity_verified=True, target_fields_enter_inference=False,
        environment={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'jax', 'jaxlib')})
    write_json(ROOT / c['output'] / 'preflight/RESULT.json', result)
    print(json.dumps(result), flush=True)


def load_ours(c, device):
    from scenemi.amass_continuation import load_config
    from scenemi.amass_training import AnchoredAmassPrior
    paths = load_paths(ROOT / c['paths'])
    state = torch.load(ROOT / c['ours_checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    config = load_config(ROOT / c['ours_config'])
    verify_training_seal(state['contract'], config, paths, ROOT / c['amass_index'])
    for path, expected in state['contract']['configuration_files'].items():
        if digest(path) != expected:
            raise ValueError(f'Training configuration changed: {path}')
    if state['step'] != c['ours_step'] or config['validation']['ddim_steps'] != c['ddim_steps']:
        raise ValueError('Expected the predeclared validation-selected step and sampler')
    model = AnchoredAmassPrior(config, paths)
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval(), state['contract']['sha256']


def evaluate(c, config, name, device):
    from experiments.offline_camera_retrain_v1.native30_model import sample_native30
    info, rows, arrays = load_inputs(c)
    seal = protocol(c, config)
    names = [name, name + '_raw'] if name in ('ours', 'egoallo') else [name]
    if name != 'ours':
        acceptance = json.loads((ROOT / c['output'] / 'preflight/RESULT.json').read_text())
        if acceptance['state'] != 'passed' or acceptance['protocol_sha256'] != seal['sha256']:
            raise ValueError('Released sampler equivalence preflight required')
    folders = {n: ROOT / c['output'] / n for n in names}
    progress, files = {}, {}
    for n, folder in folders.items():
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / 'PROGRESS.json'
        record = json.loads(p.read_text()) if p.exists() else dict(completed=0, protocol_sha256=seal['sha256'])
        if record['protocol_sha256'] != seal['sha256']:
            raise ValueError('Resume protocol differs')
        progress[n] = record['completed']
    begin = min(progress.values())
    if begin == len(rows):
        print(f'{name}: already complete and sealed', flush=True)
        return
    for n, folder in folders.items():
        path = folder / 'predictions.npy'
        if begin and not path.exists():
            raise ValueError('Committed predictions missing')
        files[n] = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+',
            dtype='float32', shape=arrays['target'].shape)
        path = folder / 'metrics.jsonl'
        old = path.read_text().splitlines() if path.exists() else []
        if len(old) < begin:
            raise ValueError('Committed metrics missing')
        path.write_text(''.join(line + '\n' for line in old[:begin]))
    model = adapter = None
    training_seal = None
    if name == 'ours':
        model, training_seal = load_ours(c, device)
        batch_size = c['ours_batch_size']
    else:
        adapter = public_adapter(c, device, info)
        adapter.load(name)
        batch_size = c['batch_size']
    started = time.monotonic()
    for offset in range(begin, len(rows), batch_size):
        end = min(offset + batch_size, len(rows))
        with torch.inference_mode():
            target = torch.as_tensor(np.array(arrays['target'][offset:end]), device=device)
            if name == 'ours':
                batch = {k: torch.as_tensor(np.array(arrays[k][offset:end]), device=device) for k in info['sample_keys']}
                public = inference_payload(batch)
                raw = sample_native30(model, public, steps=c['ddim_steps'], seed=c['seed'], use_scene=False)
                _, details = model(raw, torch.zeros(end - offset, device=device, dtype=torch.long),
                                   public, return_details=True)
                final = model.final_motion(raw, details)
                predicted = dict(ours=forward_kinematics(final, details['rest']),
                    ours_raw=forward_kinematics(raw, details['rest']))
            else:
                cpf = adapter.cpf(np.array(arrays['camera'][offset:end]), np.zeros(end - offset, dtype=np.int64))
                ids = [row['case']['case_id'] for row in rows[offset:end]]
                if name == 'egogvae':
                    predicted = dict(egogvae=adapter.gvae_sample(cpf, ids))
                else:
                    raw = adapter.allo_sample(cpf, ids)
                    predicted = dict(egoallo_raw=adapter.allo_joints(raw, cpf),
                        egoallo=adapter.allo_joints(adapter.guided(raw, cpf), cpf))
            for n, points in predicted.items():
                measures = metric_rows(points, target)
                files[n][offset:end] = points.cpu().numpy()
                files[n].flush()
                with (folders[n] / 'metrics.jsonl').open('a') as stream:
                    for row, metrics in zip(rows[offset:end], measures):
                        stream.write(json.dumps(dict(**row, metrics=metrics), allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                elapsed = time.monotonic() - started
                write_json(folders[n] / 'PROGRESS.json', dict(state='complete' if end == len(rows) else 'running',
                    completed=end, total=len(rows), protocol_sha256=seal['sha256'],
                    elapsed_seconds=elapsed, remaining_seconds=(len(rows) - end) * elapsed / (end - begin),
                    training_contract_sha256=training_seal))
        if name != 'ours' or end % 16 == 0 or end == len(rows):
            print(f'{name}: {end}/{len(rows)}; {(end - begin) / (time.monotonic() - started):.2f} windows/s', flush=True)
    for n, folder in folders.items():
        metrics = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
        result = summarize(metrics)
        result.update(state='complete', model=n, test_used=True, protocol_sha256=seal['sha256'])
        write_json(folder / 'RESULT.json', result)
        print(f"{n}: MPJPE {result['window_mean']['mpjpe_mm']:.3f}, jerk {result['window_mean']['jerk_m_s3']:.3f}", flush=True)


def independent_metrics(pred, gt, fps=30):
    """Independent NumPy reductions for the final array audit (no alignment)."""
    pred, gt = pred.astype(np.float64), gt.astype(np.float64)
    err = np.linalg.norm(pred - gt, axis=-1)
    mean = lambda x: x.reshape(len(x), -1).mean(1)
    delta = np.diff(pred[:, :, [7, 8, 10, 11]], axis=1)
    contact = np.linalg.norm(np.diff(gt[:, :, [7, 8, 10, 11]], axis=1), axis=-1) <= .01
    counts = contact.reshape(len(pred), -1).sum(1)
    return dict(mpjpe_mm=mean(err)*1000, root_error_mm=err[:, :, 0].mean(1)*1000,
        head_error_mm=err[:, :, 15].mean(1)*1000,
        mpjve_mm_s=mean(np.linalg.norm(np.diff(pred, axis=1) - np.diff(gt, axis=1), axis=-1))*fps*1000,
        jerk_m_s3=mean(np.linalg.norm(np.diff(pred, n=3, axis=1), axis=-1))*fps**3,
        gt_jerk_m_s3=mean(np.linalg.norm(np.diff(gt, n=3, axis=1), axis=-1))*fps**3,
        foot_sliding_avatarjlm_cm_frame=mean(np.abs(delta)*contact[..., None])*100,
        foot_sliding_horizontal_mm_contact_frame=(np.linalg.norm(delta[..., [0, 2]], axis=-1)*contact).reshape(len(pred), -1).sum(1)/counts.clip(min=1)*1000)


def report(c, config):
    info, identities, arrays = load_inputs(c)
    seal = protocol(c, config)
    results, checks, hashes = {}, {}, {}
    for name in MODELS:
        folder = ROOT / c['output'] / name
        progress = json.loads((folder / 'PROGRESS.json').read_text())
        if progress['completed'] != len(identities) or progress['protocol_sha256'] != seal['sha256']:
            raise ValueError('Cannot report a partial/different-protocol model')
        rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
        if [dict(case=r['case'], identity=r['identity']) for r in rows] != identities:
            raise ValueError(f'Per-model window identities differ: {name}')
        pred = np.load(folder / 'predictions.npy', mmap_mode='r')
        if pred.shape != arrays['target'].shape or not np.isfinite(pred).all():
            raise ValueError('Incomplete/nonfinite prediction arrays')
        independent_error = 0.
        for start in range(0, len(rows), 16):
            end = min(start + 16, len(rows))
            p, g = np.array(pred[start:end]), np.array(arrays['target'][start:end])
            reference = independent_metrics(p, g)
            for key, values in reference.items():
                error = np.max(np.abs(values - np.array([r['metrics'][key] for r in rows[start:end]])))
                independent_error = max(independent_error, float(error))
            # Recompute PA and the other shared metrics on CPU from stored arrays.
            recomputed = metric_rows(torch.from_numpy(p), torch.from_numpy(g))
            for a, b in zip(recomputed, rows[start:end]):
                if max(abs(v - b['metrics'][k]) for k, v in a.items()) > 1e-3:
                    raise ValueError(f'Stored metrics differ from CPU recomputation: {name}')
        if independent_error > 1e-4:
            raise ValueError(f'Independent NumPy audit failed: {name}, {independent_error}')
        result = summarize(rows)
        result['per_corpus'] = {corpus: summarize([r for r in rows if r['identity']['corpus'] == corpus])
                                for corpus in c['expected_corpora']}
        results[name] = result
        checks[name] = dict(windows=len(rows), finite=True, exact_case_identities=True,
                            independent_numpy_max_error=independent_error, cpu_recomputation_passed=True)
        hashes[name] = {p.name: digest(p) for p in (folder / 'predictions.npy', folder / 'metrics.jsonl')}
    comparison = dict(state='complete', test_used=True, protocol_sha256=seal['sha256'],
        windows=len(identities), families=c['expected_families'], primary_aggregation=c['primary_aggregation'],
        main_models=['ours', 'egogvae', 'egoallo'], diagnostic_models=['ours_raw', 'egoallo_raw'], results=results)
    out = ROOT / c['output']
    write_json(out / 'COMPARISON.json', comparison)
    keys = list(results['ours']['window_mean'])
    with (out / 'COMPARISON.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'aggregation'] + keys)
        writer.writeheader()
        for name, result in results.items():
            for aggregation in ('window_mean', 'family_macro_mean'):
                writer.writerow(dict(model=name, aggregation=aggregation, **result[aggregation]))
    audit = dict(state='passed', protocol_sha256=seal['sha256'], checks=checks,
        prediction_and_metrics_sha256=hashes, source_reflection_pinned=True,
        all_models_same_camera_target_and_windows=True, test_used_for_checkpoint_selection=False)
    write_json(out / 'FINAL_AUDIT.json', audit)
    print(json.dumps({name: result['window_mean'] for name, result in results.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'preflight', 'evaluate', 'report'))
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--model', choices=('ours', 'egogvae', 'egoallo'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    config = args.config.resolve()
    c = yaml.safe_load(config.read_text())
    if (c['version'] != 'amass-test-camera-comparison-v1' or c['frames'] != 128 or c['fps'] != 30
            or c['samples_per_window'] != 1 or c['ours_batch_size'] != 1):
        raise ValueError('Keep the declared matched-test inference protocol')
    configure_runtime(c, args.device)
    torch.set_num_threads(c['runtime']['cpu_count'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    out = ROOT / c['output']
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', str(out / 'jax-cache'))
    with (out / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'prepare':
            prepare(c)
        elif args.action == 'preflight':
            preflight(c, config, args.device)
        elif args.action == 'report':
            report(c, config)
        elif args.model:
            evaluate(c, config, args.model, args.device)
        else:
            parser.error('--model required for evaluate')


if __name__ == '__main__':
    main()
