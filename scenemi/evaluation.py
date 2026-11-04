"""Reproducible AMASS EGO test evaluation of the sealed stage-A checkpoint.

Run via the resource wrapper: python -m scenemi.evaluation --paths ...
This separate entry preserves the immutable training CLI and its resume seal.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from scenemi.config import ROOT, DEFAULT_CONFIG, load_config, load_paths
from scenemi.data.amass_dataset import AmassDataset, collate, inference_payload
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.metrics import motion_metrics
from scenemi.models import build_model
from scenemi.training import deterministic, ensure_bounded, now
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

PROTOCOL = 'amass-ego128-released-deterministic-v1'


def test_windows(index):
    """Released EgoAllo 5_eval rule with a frozen lexicographic file list.

    Upstream 0b already drops frame zero. 5_eval asks for 129 poses and scores
    poses 1:129, so our 128-frame input starts one additional frame later.
    Upstream's file list depends on glob/thread completion order; that original
    list and paper's random windows are unavailable. This is a declared new list.
    """
    groups = []
    for member, row in enumerate(index['families']):
        if row['split'] == 'test':
            for variant in row['variants']:
                n = variant['frames_after_relative_pose_drop']
                if n >= 129:
                    groups.append(dict(member=member, source=row['source'], family=row['family'],
                        corpus=row['corpus'], reflected=variant['reflected'], variant=variant['path'],
                        variant_sha256=variant['sha256'], frames_after_0b=n))
    groups.sort(key=lambda row: row['variant'])
    if not groups:
        raise ValueError('No eligible AMASS test motions')
    count = sum(row['frames_after_0b'] for row in groups)//129
    result = []
    for i in range(count):
        row = dict(groups[i % len(groups)])
        valid_starts = row['frames_after_0b']-129
        start = ((i//len(groups))*129) % valid_starts if valid_starts else 0
        row.update(case_id=i, start_after_0b=start+1, upstream_129_start=start)
        result.append(row)
    return result


def verify_training_seal(seal, config, paths, index):
    payload = {key: value for key, value in seal.items() if key != 'sha256'}
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if actual != seal['sha256'] or seal['config'] != config:
        raise ValueError('Training contract or requested configuration changed')
    if sha256(index) != seal['index_sha256']:
        raise ValueError('Training data index changed')
    if sha256(paths['smplh_models']/'neutral/model.npz') != seal['body_model_sha256']:
        raise ValueError('Training body model changed')
    for name, digest in seal['sources'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'Sealed source changed: {name}')
    for name, version in seal['environment'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Training environment changed: {name}')


def summarize(rows):
    metrics = [row['metrics'] for row in rows]
    keys = metrics[0].keys()
    by_family = {}
    for row in rows:
        by_family.setdefault(row['identity']['family'], []).append(row['metrics'])
    return dict(
        windows=len(rows), independent_source_families=len(by_family),
        per_corpus_windows=dict(Counter(row['identity']['corpus'] for row in rows)),
        window_mean={key: float(np.mean([row[key] for row in metrics])) for key in keys},
        family_macro_mean={key: float(np.mean([np.mean([row[key] for row in family])
                            for family in by_family.values()])) for key in keys},
        # Reflections and repeated windows are not independent observations.
        family_standard_error={key: float(np.std([np.mean([row[key] for row in family])
            for family in by_family.values()], ddof=1)/np.sqrt(len(by_family)))
            if len(by_family)>1 else None for key in keys},
    )


@torch.inference_mode()
def run(args):
    ensure_bounded()
    paths, config = load_paths(args.paths), load_config(args.config)
    index_path = paths['output']/'index/complete.json'
    checkpoint = args.checkpoint or paths['output']/'training/last.pt'
    seal = json.loads((checkpoint.parent/'CONTRACT.json').read_text())
    verify_training_seal(seal, config, paths, index_path)
    deterministic(args.seed, args.device)
    # CPU loading avoids allocating the unused Adam states on the GPU.
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['contract'] != seal or state['step'] != config['training']['planned_updates']:
        raise ValueError('Expected the sealed final training checkpoint')
    model = build_model(config, paths['smplh_models']/'neutral/model.npz')
    model.load_state_dict(state['model'], strict=True)
    step = state['step']
    del state
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(6*2**30/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
    model = model.to(args.device).eval()
    data = AmassDataset(index_path, paths, split='test')
    # Manifest indices are relative to the test-only list used by this adapter.
    manifest_index = dict(families=data.members)
    cases = test_windows(manifest_index)
    if args.limit:
        cases = cases[:args.limit]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/'RESULT.json').exists() or (out/'metrics.jsonl').exists():
        raise ValueError('Choose an empty output directory; do not overwrite a recorded evaluation')
    checkpoint_hash = sha256(checkpoint)
    evaluator_sources = {str(Path(__file__).relative_to(ROOT)):sha256(Path(__file__)),
                         'scenemi/metrics.py':sha256(ROOT/'scenemi/metrics.py')}
    record = dict(protocol=PROTOCOL, checkpoint=str(checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
        step=step, training_contract_sha256=seal['sha256'], index_sha256=sha256(index_path),
        evaluator_sources=evaluator_sources, config_sha256=sha256(args.config), seed=args.seed,
        ddim_steps=args.steps, samples_per_window=1, best_of_k=False, batch_size=1,
        input='continuous virtual camera only; scene disabled; shape predicted',
        test_corpora=['Transitions_mocap', 'SSM_synced'], manifest=cases,
        limitations=['Fixed lexicographic list follows the released script rule, not exact paper windows.',
            'Virtual mount uses two eye vertices +2cm, not published four-vertex CPF.',
            'Unified EGO/EXO train-only training differs from EGO-only baselines.',
            'Baseline paper numbers are references; no baseline checkpoints are run here.',
            'Standard PA and released-code compatibility PA must not be conflated.'],
        diagnostic_only=bool(args.limit), started_at_utc=now(), pid=os.getpid(), device=args.device)
    atomic_json(out/'PROTOCOL.json', record)
    started = time.monotonic()
    rows = []
    try:
        with (out/'metrics.jsonl').open('x') as stream:
            for case in cases:
                sample, identity = data.sample(np.random.default_rng(args.seed), 'ego',
                    member=case['member'], start=case['start_after_0b'],
                    reflected=case['reflected'], yaw=0., temporal_dropout=None)
                if identity['variant_sha256'] != case['variant_sha256']:
                    raise ValueError('Test manifest changed at load')
                batch = collate([sample], args.device)
                payload = inference_payload(batch)
                pred = sample_native30(model, payload, steps=args.steps, seed=args.seed, use_scene=False)
                _, details = model(pred, torch.zeros(1, device=args.device, dtype=torch.long),
                                   payload, return_details=True)
                joints = forward_kinematics(pred, details['rest'])[0]
                truth = forward_kinematics(batch['motion'], batch['target_rest'])[0]
                row = dict(case_id=case['case_id'], identity=identity,
                    metrics=motion_metrics(joints, truth), elapsed_seconds=time.monotonic()-started)
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                status = dict(state='running', completed=len(rows), total=len(cases),
                    elapsed_seconds=time.monotonic()-started, updated_at_utc=now(), pid=os.getpid())
                atomic_json(out/'STATUS.json', status)
                if len(rows)%10 == 0 or len(rows)==1:
                    print(json.dumps(status), flush=True)
        if sha256(checkpoint) != checkpoint_hash:
            raise ValueError('Checkpoint changed during evaluation')
        result = dict(state='complete', protocol=PROTOCOL, diagnostic_only=bool(args.limit),
            checkpoint_sha256=checkpoint_hash, **summarize(rows),
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else 0,
            elapsed_seconds=time.monotonic()-started, completed_at_utc=now())
        atomic_json(out/'RESULT.json', result)
        atomic_json(out/'STATUS.json', dict(result, completed=len(rows), total=len(cases), pid=os.getpid()))
        return result
    except BaseException as exc:
        atomic_json(out/'STATUS.json', dict(state='failed', completed=len(rows), total=len(cases),
            error=repr(exc), updated_at_utc=now(), pid=os.getpid()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--limit', type=int, help='Diagnostic prefix only; never reported as full test')
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000 or (args.limit is not None and args.limit <= 0):
        parser.error('Positive diagnostic limit and 1..1000 DDIM steps required')
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == '__main__':
    main()
