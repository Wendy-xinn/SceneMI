"""Evaluate predeclared AMASS checkpoints on the frozen test manifest.

No model selection, body correction or camera redefinition is performed here.
Reuse the original evaluator's cases, seed, sampler, FK and metrics. Keep joint
arrays for independent metric checks and subsequent visual review.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import AmassDataset, collate, inference_payload
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.ego_training import load_ego_config
from scenemi.evaluation import summarize, test_windows, verify_training_seal
from scenemi.metrics import motion_metrics
from scenemi.models import build_model
from scenemi.parallel_resources import configure_cuda
from scenemi.training import deterministic, ensure_bounded, now
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def validate_cases(cases, original):
    if cases != original or not cases or any(c['corpus'] not in ('SSM_synced', 'Transitions_mocap') for c in cases):
        raise ValueError('Benchmark must exactly reuse the complete frozen test list')


@torch.inference_mode()
def evaluate(spec, policy, paths, data, cases, device):
    out = ROOT / policy['output'] / spec['name']
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / spec['checkpoint']
    digest = sha256(checkpoint)
    if digest != spec['checkpoint_sha256']:
        raise ValueError('Predeclared benchmark checkpoint changed')
    seal = json.loads((ROOT/spec['contract']).read_text())
    config = load_ego_config(ROOT/spec['config'])
    verify_training_seal(seal, config, paths, paths['output']/'index/complete.json')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if state['contract'] != seal or state['step'] != spec['step']:
        raise ValueError('Checkpoint is not the predeclared sealed candidate')
    deterministic(policy['seed'], device)
    model = build_model(config, paths['smplh_models']/'neutral/model.npz')
    model.load_state_dict(state['model'], strict=True)
    del state
    model = model.to(device).eval()
    protocol = dict(protocol=policy['protocol'], model=spec, manifest=cases,
        training_contract_sha256=seal['sha256'], seed=policy['seed'], ddim_steps=policy['ddim_steps'],
        batch_size=1, samples_per_window=1, best_of_k=False, split='test',
        input='continuous virtual camera only; two eye vertices +2cm; predicted shape; scene disabled',
        primary_reporting='window_mean; family_macro_mean also retained',
        no_test_selection=True, baseline_checkpoints_run=False,
        evaluator_sources={name:sha256(ROOT/name) for name in (
            'scenemi/benchmark.py', 'scenemi/evaluation.py', 'scenemi/metrics.py', 'scenemi/runtime.py')},
        limitations=['Frozen released-script cases are not the unpublished paper windows.',
            'Virtual camera and four-pupil CPF differ; baseline sensor adaptation remains separate.',
            'Standard PA is reported separately from released-code compatibility PA.',
            'Published baseline numbers are references, not local baseline executions.'])
    prior = out/'PROTOCOL.json'
    if prior.exists() and json.loads(prior.read_text()) != protocol:
        raise ValueError('An existing test protocol cannot be changed')
    atomic_json(prior, protocol)
    if (out/'RESULT.json').exists():
        return json.loads((out/'RESULT.json').read_text())
    # Interrupted evaluations may resume only the same sealed protocol.
    metrics_file = out/'metrics.jsonl'
    rows = [json.loads(s) for s in metrics_file.read_text().splitlines()] if metrics_file.exists() else []
    if [r['case_id'] for r in rows] != list(range(len(rows))):
        raise ValueError('Invalid partial evaluation order')
    arrays = out/'joints'
    arrays.mkdir(exist_ok=True)
    for row in rows:
        if sha256(arrays/f"{row['case_id']:04d}.npz") != row['joints_sha256']:
            raise ValueError('Saved evaluation joints changed')
    started = time.monotonic()
    try:
        with metrics_file.open('a') as stream:
            for case in cases[len(rows):]:
                sample, identity = data.sample(np.random.default_rng(policy['seed']), 'ego',
                    member=case['member'], start=case['start_after_0b'],
                    reflected=case['reflected'], yaw=0., temporal_dropout=None)
                if identity['variant_sha256'] != case['variant_sha256']:
                    raise ValueError('Test source changed')
                batch = collate([sample], device)
                payload = inference_payload(batch)
                pred = sample_native30(model, payload, steps=policy['ddim_steps'], seed=policy['seed'], use_scene=False)
                _, details = model(pred, torch.zeros(1, device=device, dtype=torch.long), payload, return_details=True)
                joints = forward_kinematics(pred, details['rest'])[0]
                target = forward_kinematics(batch['motion'], batch['target_rest'])[0]
                file = arrays/f"{case['case_id']:04d}.npz"
                np.savez_compressed(file, prediction=joints.cpu().numpy(), target=target.cpu().numpy(),
                    camera=payload['camera'][0].cpu().numpy(), predicted_rest=details['rest'][0].cpu().numpy(),
                    motion=pred[0].cpu().numpy())
                row = dict(case_id=case['case_id'], identity=identity, metrics=motion_metrics(joints, target),
                    joints_sha256=sha256(file))
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                status = dict(state='running', completed=len(rows), total=len(cases), pid=os.getpid(),
                    updated_at_utc=now(), elapsed_seconds=time.monotonic()-started)
                atomic_json(out/'STATUS.json', status)
                if len(rows)%20==0 or len(rows)==1:
                    print(json.dumps(dict(model=spec['name'], **status)), flush=True)
        if sha256(checkpoint) != digest:
            raise ValueError('Checkpoint changed during test')
        result = dict(state='complete', step=spec['step'], checkpoint_sha256=digest,
            no_test_selection=True, **summarize(rows), elapsed_seconds=time.monotonic()-started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0, completed_at_utc=now())
        atomic_json(out/'RESULT.json', result)
        atomic_json(out/'STATUS.json', dict(result, completed=len(rows), total=len(cases)))
        return result
    except BaseException as error:
        atomic_json(out/'STATUS.json', dict(state='failed', completed=len(rows), total=len(cases),
            error=repr(error), updated_at_utc=now()))
        raise
    finally:
        del model
        gc.collect()
        if device=='cuda':
            torch.cuda.empty_cache()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train-control':
        from scenemi.scene_ablation import main as control_main
        return control_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'scene':
        from scenemi.scene_evaluation import main as scene_main
        return scene_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, default=ROOT/'configs/amass_benchmark.json')
    parser.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    ensure_bounded()
    if args.device=='cuda':
        configure_cuda()
    paths = load_paths(args.paths)
    policy = json.loads(args.policy.read_text())
    if policy['batch_size']!=1 or policy['ddim_steps']!=20 or policy['seed']!=777:
        raise ValueError('Preserve the original fixed test sampling protocol')
    if any(s['checkpoint_sha256'] is None for s in policy['models']):
        raise ValueError('Freeze every checkpoint SHA256 before evaluating test')
    data = AmassDataset(paths['output']/'index/complete.json', paths, split='test')
    cases = test_windows(dict(families=data.members))
    original = json.loads((ROOT/'runs/amass128/evaluation/ego128/PROTOCOL.json').read_text())['manifest']
    validate_cases(cases, original)
    out = ROOT/policy['output']
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'FROZEN_POLICY.json', policy)
    results = {spec['name']:evaluate(spec, policy, paths, data, cases, args.device) for spec in policy['models']}
    atomic_json(out/'RESULTS.json', results)
    print(json.dumps({name:r['window_mean'] for name,r in results.items()}, indent=2), flush=True)


if __name__=='__main__':
    main()
