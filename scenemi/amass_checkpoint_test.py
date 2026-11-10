"""Test a validation-selected AMASS branch on the retained matched test export.

Keep the historical evaluator and training sources unchanged. Reuse its exact
sampling loop and metrics, with a branch-aware model loader and a separate seal.
Public-baseline predictions are reused only after checking their recorded hashes.
"""
import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
import yaml

from scenemi import amass_comparison as matched
from scenemi.amass_finetuning import reference_config
from scenemi.amass_training import AnchoredAmassPrior
from scenemi.baseline_evaluation import digest, metric_rows, write_json
from scenemi.config import ROOT, load_paths
from scenemi.evaluation import summarize, verify_training_seal
from scenemi.execution import configure_runtime


def load_config(path):
    path = Path(path).resolve()
    spec = yaml.safe_load(path.read_text())
    if spec['version'] != 'amass-selected-checkpoint-test-v1':
        raise ValueError('Expected a selected-checkpoint test configuration')
    reference = ROOT / spec['reference_config']
    if digest(reference) != spec['reference_config_sha256']:
        raise ValueError('The retained comparison configuration changed')
    c = yaml.safe_load(reference.read_text())
    if (c['seed'], c['ddim_steps'], c['ours_batch_size'], c['samples_per_window'],
            c['frames'], c['fps']) != (777, 20, 1, 1, 128, 30):
        raise ValueError('Preserve the complete fixed test sampling protocol')
    c['reference_output'] = c['output']
    c.update({key: spec[key] for key in (
        'output', 'paths', 'ours_config', 'ours_checkpoint', 'ours_sha256', 'ours_step')})
    out, prior = (ROOT / c['output']).resolve(), (ROOT / c['reference_output']).resolve()
    if not out.is_relative_to(ROOT) or out.is_relative_to(prior) or prior.is_relative_to(out):
        raise ValueError('Use a separate evaluation output directory')
    c['checkpoint_test'] = dict(spec, config_path=str(path))
    return c


def check_selection(c, state, config):
    checkpoint = (ROOT / c['ours_checkpoint']).resolve()
    run = ROOT / config['output']
    if checkpoint != (run / 'selection/best.pt').resolve():
        raise ValueError('Test only the predeclared validation-selected checkpoint')
    best = json.loads((run / 'selection/BEST.json').read_text())
    metrics = run / 'validation' / f"step_{c['ours_step']:06d}_full/RESULT.json"
    if digest(metrics) != c['checkpoint_test']['validation_metrics_sha256']:
        raise ValueError('The predeclared selection validation changed')
    validation = json.loads(metrics.read_text())
    if (state['step'] != c['ours_step'] or best['step'] != c['ours_step']
            or state.get('validation_pending', True) or state['device'] != 'cuda'
            or state.get('best') != dict(step=best['step'], selection_score_mm=best['selection_score_mm'])
            or Path(best['checkpoint']).resolve() != checkpoint
            or validation['test_used'] is not False
            or config['validation']['test_used'] is not False
            or validation['selection_score_mm'] != best['selection_score_mm']):
        raise ValueError('Checkpoint must match completed validation selection, without test selection')


def load_verified_checkpoint(checkpoint, expected):
    """Deserialize the exact checked bytes; never reopen a checked path as mmap."""
    checkpoint = Path(checkpoint)
    before = checkpoint.stat()
    if not 0 < before.st_size <= 2 * 2**30:
        raise ValueError('Expected a checkpoint no larger than 2 GiB')
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    with io.BytesIO() as buffer:
        with checkpoint.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                if buffer.tell() + len(block) > 2 * 2**30:
                    raise ValueError('Checkpoint grew beyond its read budget')
                buffer.write(block)
        if identity(checkpoint.stat()) != identity(before):
            raise ValueError('Checkpoint identity changed during read')
        if hashlib.sha256(buffer.getbuffer()).hexdigest() != expected:
            raise ValueError('The predeclared test checkpoint changed')
        with zipfile.ZipFile(buffer) as archive:
            if archive.testzip() is not None:
                raise ValueError('Checkpoint archive CRC failed')
        buffer.seek(0)
        state = torch.load(buffer, map_location='cpu', weights_only=False)
        if hashlib.sha256(buffer.getbuffer()).hexdigest() != expected:
            raise ValueError('Checkpoint buffer changed during deserialization')
    return state


def load_ours(c, device):
    checkpoint = ROOT / c['ours_checkpoint']
    state = load_verified_checkpoint(checkpoint, c['ours_sha256'])
    if digest(checkpoint) != c['ours_sha256']:
        raise ValueError('The predeclared test checkpoint changed')
    config = reference_config(ROOT / c['ours_config'])
    paths = load_paths(ROOT / c['paths'])
    verify_training_seal(state['contract'], config, paths, ROOT / c['amass_index'])
    if state['contract'] != json.loads((ROOT / config['output'] / 'training/CONTRACT.json').read_text()):
        raise ValueError('Checkpoint and retained training contract differ')
    for path, expected in state['contract']['configuration_files'].items():
        if digest(path) != expected:
            raise ValueError(f'Training configuration changed: {path}')
    check_selection(c, state, config)
    if config['validation']['ddim_steps'] != c['ddim_steps']:
        raise ValueError('Preserve the validation sampler')
    if any(not torch.isfinite(value).all() for value in state['model'].values()):
        raise ValueError('Nonfinite checkpoint model tensor')
    model = AnchoredAmassPrior(config, paths)
    model.load_state_dict(state['model'], strict=True)
    if any(not torch.equal(value, state['model'][key]) for key, value in model.state_dict().items()):
        raise ValueError('Loaded CPU model differs from the verified checkpoint')
    write_json(ROOT / c['output'] / 'CHECKPOINT_READ.json', dict(state='passed',
        checkpoint=str(checkpoint.resolve()), sha256=c['ours_sha256'], step=state['step'],
        exact_verified_buffer_deserialized=True, archive_crc_valid=True,
        model_tensors_finite=True, strict_state_dict_copy_bitwise_equal=True,
        memory_mapped_checkpoint=False, training_contract_sha256=state['contract']['sha256']))
    return model.to(device).eval(), state['contract']['sha256']


def reference_records(c):
    root = ROOT / c['reference_output']
    records = {}
    for name, expected in c['checkpoint_test']['reference_records_sha256'].items():
        path = root / name
        if digest(path) != expected:
            raise ValueError(f'Retained baseline record changed: {path}')
        records[name] = json.loads(path.read_text())
    protocol = records['EVALUATION_PROTOCOL.json']
    payload = {key: value for key, value in protocol.items() if key != 'sha256'}
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != protocol['sha256']:
        raise ValueError('Invalid retained comparison protocol')
    audit = records['FINAL_AUDIT.json']
    if (audit['state'] != 'passed' or audit['protocol_sha256'] != protocol['sha256']
            or records['COMPARISON.json']['protocol_sha256'] != protocol['sha256']
            or protocol['config']['manifest_sha256'] != c['manifest_sha256']):
        raise ValueError('Retained baseline audit or test membership differs')
    return records


def protocol(c, config):
    reference_records(c)
    sources = [Path(__file__), Path(config), ROOT / c['checkpoint_test']['reference_config'],
        ROOT / c['manifest'], ROOT / c['paths'], ROOT / c['output'] / 'data/DATA_PROTOCOL.json']
    sources += [ROOT / name for name in (
        'scenemi/amass_comparison.py', 'scenemi/amass_finetuning.py',
        'scenemi/baseline_evaluation.py', 'scenemi/metrics_temporal.py',
        'scenemi/metrics_batch.py', 'scenemi/metrics.py', 'scenemi/evaluation.py',
        'scenemi/execution.py', 'scenemi/gpu_policy.py', 'configs/runtime_policy.yaml')]
    value = dict(version='amass-selected-checkpoint-test-v1', config=c,
        source_sha256={str(path.resolve()): digest(path) for path in sources},
        checkpoint_sha256=c['ours_sha256'], test_used=True,
        selection=f"Full-validation-selected best at step {c['ours_step']}; no test selection",
        manifest_sha256=c['manifest_sha256'], seed=777, batch_size=1,
        samples_per_window=1, best_of_k=False, ddim_steps=20,
        sampling_and_metrics='Unchanged scenemi.amass_comparison.evaluate and metric_rows',
        checkpoint_loading='SHA256 and archive CRC of one buffered snapshot; deserialize the same bytes without mmap',
        baseline_execution='Reuse previously audited official-checkpoint predictions on the identical export',
        limitations=reference_records(c)['EVALUATION_PROTOCOL.json']['limitations'])
    value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    path = ROOT / c['output'] / 'EVALUATION_PROTOCOL.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Evaluation closure changed; use a new output directory')
    write_json(path, value)
    return value


@contextmanager
def checkpoint_evaluator():
    """Bind only loader/provenance; keep the retained inference loop intact."""
    old_loader, old_protocol = matched.load_ours, matched.protocol
    matched.load_ours, matched.protocol = load_ours, protocol
    try:
        yield
    finally:
        matched.load_ours, matched.protocol = old_loader, old_protocol


def audit_predictions(folder, identities, target, *, expected=None):
    hashes = {name: digest(folder / name) for name in ('predictions.npy', 'metrics.jsonl')}
    if expected is not None and hashes != expected:
        raise ValueError(f'Retained predictions or metrics changed: {folder}')
    rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
    if [dict(case=row['case'], identity=row['identity']) for row in rows] != identities:
        raise ValueError(f'Exact test identities differ: {folder}')
    prediction = np.load(folder / 'predictions.npy', mmap_mode='r')
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError(f'Incomplete or nonfinite predictions: {folder}')
    maximum = 0.
    for start in range(0, len(rows), 16):
        end = min(start + 16, len(rows))
        pred, gt = np.array(prediction[start:end]), np.array(target[start:end])
        for key, values in matched.independent_metrics(pred, gt).items():
            maximum = max(maximum, float(np.max(np.abs(
                values - np.array([row['metrics'][key] for row in rows[start:end]])))))
        for computed, row in zip(metric_rows(torch.from_numpy(pred), torch.from_numpy(gt)), rows[start:end]):
            if max(abs(value - row['metrics'][key]) for key, value in computed.items()) > 1e-3:
                raise ValueError(f'CPU metric recomputation failed: {folder}')
    if maximum > 1e-4:
        raise ValueError(f'Independent NumPy metric recomputation failed: {folder}')
    result = summarize(rows)
    result['per_corpus'] = {name: summarize([row for row in rows if row['identity']['corpus'] == name])
                            for name in sorted({row['identity']['corpus'] for row in rows})}
    return result, dict(windows=len(rows), exact_case_identities=True, finite=True,
        independent_numpy_max_error=maximum, cpu_recomputation_passed=True,
        prediction_and_metrics_sha256=hashes)


def report(c, config):
    _, identities, arrays = matched.load_inputs(c)
    seal = protocol(c, config)
    records = reference_records(c)
    results, checks, provenance = {}, {}, {}
    for name in ('ours', 'ours_raw'):
        folder = ROOT / c['output'] / name
        progress = json.loads((folder / 'PROGRESS.json').read_text())
        if progress['completed'] != len(identities) or progress['protocol_sha256'] != seal['sha256']:
            raise ValueError('Cannot report incomplete or different-protocol predictions')
        label = name + '_' + str(c['ours_step'])
        results[label], checks[label] = audit_predictions(folder, identities, arrays['target'])
        provenance[label] = dict(execution='new', checkpoint=c['ours_checkpoint'],
            checkpoint_sha256=c['ours_sha256'], step=c['ours_step'],
            training_contract_sha256=progress['training_contract_sha256'])
    old_step = records['EVALUATION_PROTOCOL.json']['config']['ours_step']
    for name in matched.MODELS:
        label = name + '_' + str(old_step) if name.startswith('ours') else name
        expected = records['FINAL_AUDIT.json']['prediction_and_metrics_sha256'][name]
        results[label], checks[label] = audit_predictions(
            ROOT / c['reference_output'] / name, identities, arrays['target'], expected=expected)
        provenance[label] = dict(execution='reused; hashes and metrics rechecked',
            protocol_sha256=records['EVALUATION_PROTOCOL.json']['sha256'])
    if digest(ROOT / c['ours_checkpoint']) != c['ours_sha256']:
        raise ValueError('Checkpoint changed during test')
    out = ROOT / c['output']
    comparison = dict(state='complete', test_used=True, protocol_sha256=seal['sha256'],
        windows=len(identities), families=c['expected_families'],
        primary_aggregation=c['primary_aggregation'], results=results, model_provenance=provenance)
    write_json(out / 'COMPARISON.json', comparison)
    write_json(out / 'FINAL_AUDIT.json', dict(state='passed', protocol_sha256=seal['sha256'],
        checks=checks, all_models_same_camera_target_and_windows=True,
        test_used_for_checkpoint_selection=False))
    with (out / 'COMPARISON.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'aggregation'] + list(next(iter(results.values()))['window_mean']))
        writer.writeheader()
        for name, result in results.items():
            for aggregation in ('window_mean', 'family_macro_mean'):
                writer.writerow(dict(model=name, aggregation=aggregation, **result[aggregation]))
    print(json.dumps({name: result['window_mean'] for name, result in results.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    c = load_config(args.config)
    configure_runtime(c, args.device)
    torch.set_num_threads(c['runtime']['cpu_count'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    out = ROOT / c['output']
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reference_records(c)
        data = out / 'data'
        source = (ROOT / c['reference_output'] / 'data').resolve()
        if data.is_symlink() and data.resolve() != source:
            raise ValueError('Test export link changed')
        if not data.exists():
            data.symlink_to(source, target_is_directory=True)
        with checkpoint_evaluator():
            matched.evaluate(c, args.config.resolve(), 'ours', args.device)
        report(c, args.config.resolve())


if __name__ == '__main__':
    main()
