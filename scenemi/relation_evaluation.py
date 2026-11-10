"""Resumable fixed-checkpoint evaluation for the observed-relation adapter.

Keep the training dependency seal intact. The pilot repeats the original
predeclared identities, interventions and sampler; quick validation optionally
extends the correct-scene result to all original 176 windows, without selection.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.local_scene import LocalSceneDataset, inference_payload
from scenemi.metrics import motion_metrics
from scenemi.parallel_resources import configure_cuda
from scenemi.scene_followup import change_scene, read_ledger, verify_sources
from scenemi.scene_relations import make_model, scene_difference, settings
from scenemi.scene_selection import aggregate_scene_rows
from scenemi.training import deterministic, ensure_bounded, now
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def summarize_variants(ledger):
    return {variant: aggregate_scene_rows([r['variants'][variant] for r in ledger])
            for variant in ('correct', 'empty', 'mismatched')}


def sample(data, identity, c, device):
    value, actual = data.sample(np.random.default_rng(c['validation']['seed']),
        member=identity['member'], start=identity['start'], yaw=0.)
    actual['category'] = identity['category']
    if actual != identity:
        raise ValueError('Evaluation identity differs from the sealed reference')
    return collate([value], device)


@torch.inference_mode()
def predict(model, batch, payload, c, device):
    motion = sample_native30(model, payload, steps=c['validation']['ddim_steps'],
                            seed=c['validation']['seed'], use_scene=True)
    _, details = model(motion, torch.zeros(1, device=device, dtype=torch.long),
                       payload, return_details=True)
    joints = forward_kinematics(motion, details['rest'])[0]
    truth = forward_kinematics(batch['motion'], batch['target_rest'])[0]
    return motion_metrics(joints, truth), joints.cpu().numpy()


def evaluate(args):
    ensure_bounded()
    if args.device == 'cuda': configure_cuda()
    c = settings(args.config); deterministic(c['seed'], args.device)
    run = ROOT/c['output']; seal = json.loads((run/'training/CONTRACT.json').read_text())
    verify_sources(seal)
    if seal['config'] != c: raise ValueError('Evaluation configuration changed')
    checkpoint = run/'training/last.pt'; checkpoint_sha = sha256(checkpoint)
    model = make_model(c, load_paths(args.paths), args.scene_paths)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['contract'] != seal or state['step'] != c['training']['planned_updates']:
        raise ValueError('Only the fixed final adapter may be evaluated')
    model.load_adapter_state(state['adapter']); del state
    model = model.to(args.device).eval()
    data = LocalSceneDataset(args.scene_paths, model.shape.rest_template[:1].cpu().numpy(),
                             split='validation', config=c)
    reference_path = ROOT/'runs/scene_ego128_residual/selection/quick/step_003000.json'
    reference = json.loads(reference_path.read_text())
    intervention_path = ROOT/'runs/scene_ego128_residual_checks/PROTOCOL.json'
    intervention = json.loads(intervention_path.read_text())
    out = run/'evaluation/recovered'; out.mkdir(parents=True, exist_ok=True)
    protocol = dict(training_contract_sha256=seal['sha256'], checkpoint_sha256=checkpoint_sha,
        checkpoint_step=state_step(c), pilot_case_ids=seal['pilot_case_ids'],
        reference_sha256=sha256(reference_path), intervention_sha256=sha256(intervention_path),
        evaluator_source_sha256=sha256(__file__), seed=c['validation']['seed'],
        ddim_steps=c['validation']['ddim_steps'], device=args.device,
        quick_after_pilot=args.quick_after_pilot, test_used=False,
        reason='Separate evaluator repairs duplicate rows-key assembly; original training source unchanged')
    record = out/'PROTOCOL.json'
    if record.exists() and json.loads(record.read_text()) != protocol:
        raise ValueError('Started evaluation protocol changed')
    atomic_json(record, protocol)
    ids = seal['pilot_case_ids']; ledger = read_ledger(out/'metrics.jsonl')
    if [r['case_id'] for r in ledger] != ids[:len(ledger)]:
        raise ValueError('Invalid partial pilot order')
    arrays = out/'joints'; arrays.mkdir(exist_ok=True)
    for row in ledger:
        if sha256(arrays/f"{row['case_id']:04d}.npz") != row['joints_sha256']:
            raise ValueError('Saved paired predictions changed')
    started = time.monotonic()
    with (out/'metrics.jsonl').open('a') as stream:
        for i in ids[len(ledger):]:
            identity = reference['rows'][i]['identity']
            donor_identity = reference['rows'][intervention['donors'][i]]['identity']
            batch = sample(data, identity, c, args.device)
            donor = inference_payload(sample(data, donor_identity, c, args.device))
            payload = inference_payload(batch)
            teacher = dict(rest=batch['target_rest'], predicted_native_betas=batch['target_native_betas'],
                           predicted_template=batch['target_template'])
            with torch.no_grad():
                skin, _ = model.skin(batch['motion'][:, ::4], teacher, batch['body_format'])
            def compact(x):
                return dict(x, local_frame_points=x['local_frame_points'][:, ::4],
                            local_frame_owner=x['local_frame_owner'][:, ::4])
            geometry = scene_difference(compact(payload), compact(donor), skin, c['relation_pilot']['relation'])
            rows, predictions = {}, {}
            for variant in ('correct', 'empty', 'mismatched'):
                altered = payload if variant == 'correct' else change_scene(payload, variant, donor)
                metrics, predictions[variant] = predict(model, batch, altered, c, args.device)
                rows[variant] = dict(identity=identity, metrics=metrics)
            delta = np.linalg.norm(predictions['correct']-predictions['mismatched'], axis=-1)*1000
            file = arrays/f'{i:04d}.npz'; np.savez_compressed(file, **predictions)
            row = dict(case_id=i, variants=rows, joints_sha256=sha256(file),
                sensitivity=dict(case_id=i, identity=identity, donor_identity=donor_identity,
                    scene_difference=geometry, prediction_change_mean_mm=float(delta.mean()),
                    prediction_change_feet_mm=float(delta[:, [7,8,10,11]].mean()),
                    prediction_change_hips_mm=float(delta[:, [0,1,2]].mean())))
            ledger.append(row); stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
            atomic_json(out/'STATUS.json', dict(state='running', completed=len(ledger), total=len(ids),
                        pid=os.getpid(), updated_at_utc=now()))
    verify_sources(seal)
    if sha256(checkpoint) != checkpoint_sha: raise ValueError('Fixed checkpoint changed')
    result = dict(state='complete', contract_sha256=seal['sha256'], final_step=state_step(c),
        checkpoint_sha256=checkpoint_sha, variants=summarize_variants(ledger),
        parent_correct=aggregate_scene_rows([reference['rows'][i] for i in ids]),
        sensitivity=[r['sensitivity'] for r in ledger], pilot_only=True, test_used=False,
        elapsed_seconds=time.monotonic()-started, completed_at_utc=now(),
        interpretation='Predeclared 24-family validation pilot; fixed final 500 updates; no checkpoint selection')
    atomic_json(out/'RESULT.json', result)
    atomic_json(out/'STATUS.json', dict(state='complete', completed=len(ids), total=len(ids), updated_at_utc=now()))
    print(json.dumps({name: value['selection_score_mm'] for name, value in result['variants'].items()}), flush=True)
    if args.quick_after_pilot:
        from scenemi.local_training import evaluate as quick_evaluate
        quick_out = run/'evaluation/quick'; quick_out.mkdir(exist_ok=True)
        def progress(done, total, elapsed):
            atomic_json(quick_out/'STATUS.json', dict(state='running', completed=done, total=total,
                elapsed_seconds=elapsed, pid=os.getpid(), updated_at_utc=now()))
        quick = quick_evaluate(model, data, seal['validation_cases']['quick'], c, args.device, progress=progress)
        quick.update(state='complete', final_step=state_step(c), checkpoint_sha256=checkpoint_sha,
                     contract_sha256=seal['sha256'], test_used=False, selection_tier='fixed_final_quick')
        verify_sources(seal)
        if sha256(checkpoint) != checkpoint_sha: raise ValueError('Checkpoint changed during quick validation')
        atomic_json(quick_out/'RESULT.json', quick)
        atomic_json(quick_out/'STATUS.json', dict(state='complete', completed=len(quick['rows']),
                    total=len(quick['rows']), updated_at_utc=now()))


def state_step(c):
    return c['training']['planned_updates']


def main():
    if len(sys.argv)>1 and sys.argv[1]=='bounded':
        from scenemi import parallel_resources
        from scenemi.shared_runtime import install
        install(ROOT/'configs/gpu_shared_jobs.local.json')
        parallel_resources.MODULES |= {'scenemi.relation_evaluation'}
        sys.argv = [__file__]+sys.argv[2:]
        return parallel_resources.main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/scene_relations.yaml')
    parser.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--scene-paths', type=Path, default=ROOT/'configs/scene_paths.local.yaml')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--quick-after-pilot', action='store_true')
    evaluate(parser.parse_args())


if __name__=='__main__': main()
