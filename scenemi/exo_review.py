"""Audit completed EXO validation against its identical observed-joint inputs.

This reconstructs only declared validation observations and labels on CPU.
It never samples a model, fills hidden joints from labels, or treats the
partial input as a complete upstream body estimate.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import AmassDataset, collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.exo_training import INDEX, example, settings, verify_sources
from scenemi.training import ensure_bounded, now
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def family_mean(rows, field):
    families = defaultdict(list)
    for row in rows:
        if row[field] is not None:
            families[row['family']].append(row[field])
    return float(np.mean([np.mean(values) for values in families.values()])) if families else None


def aggregate(rows):
    return dict(windows=len(rows), families=len({row['family'] for row in rows}),
        family_macro_mean={key:family_mean(rows, key) for key in
            ['input_visible_mpjpe_mm', 'output_visible_mpjpe_mm',
             'output_hidden_mpjpe_mm', 'visible_worsening_mm']},
        windows_output_visible_worse_than_input=sum(row['visible_worsening_mm']>1e-3 for row in rows))


def review(run, output):
    ensure_bounded()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'RESULT.json').exists() or (output/'observed_inputs.jsonl').exists():
        raise ValueError('Choose an empty review output; preserve existing audit records')
    seal = json.loads((run/'PROTOCOL.json').read_text())
    verify_sources(seal)
    status = json.loads((run/'training/STATUS.json').read_text())
    if status['state'] != 'complete':
        raise ValueError('Review requires completed EXO training and final EGO validation')
    best = json.loads((run/'selection/BEST.json').read_text())
    source_files = [run/'PROTOCOL.json', run/'training/STATUS.json', run/'selection/BEST.json',
                    run/'VALIDATION_CASES.jsonl']
    sources = {str(p.relative_to(ROOT)):sha256(p) for p in source_files}
    sources['scenemi/exo_review.py'] = sha256(Path(__file__))
    final_folder = run/'selection'/f'step_{best["step"]:06d}'
    curves = []
    for path in sorted((run/'selection').glob('step_*/RESULT.json')):
        result = json.loads(path.read_text())
        sources[str(path.relative_to(ROOT))] = sha256(path)
        curves.append(dict(step=result['step'], score_mm=result['selection_score_mm'],
            conditions={key:value['family_macro_mean'] for key,value in result['conditions'].items()}))
    config, _, effective = settings(ROOT/'configs/amass_exo128.yaml')
    if config != seal['config']:
        raise ValueError('Requested observation recipe differs from sealed training')
    paths = load_paths(ROOT/'configs/paths.local.yaml')
    data = AmassDataset(INDEX, paths, 'val')
    cases = [json.loads(line) for line in (run/'VALIDATION_CASES.jsonl').read_text().splitlines()]
    if cases != seal['validation_cases']:
        raise ValueError('Review cases differ from full validation')
    started = time.monotonic()
    conditions = {}
    with (output/'observed_inputs.jsonl').open('x') as stream:
        for condition in config['validation']['conditions']:
            path = final_folder/f'{condition}.jsonl'
            sources[str(path.relative_to(ROOT))] = sha256(path)
            predictions = [json.loads(line) for line in path.read_text().splitlines()]
            if len(predictions) != len(cases):
                raise ValueError('Incomplete prediction metrics')
            rows = []
            for offset in range(0, len(cases), 32):
                chunk = cases[offset:offset+32]
                pairs = [example(data, np.random.default_rng(config['validation']['seed']+case['case_id']),
                    config, effective, case=case, condition=condition) for case in chunk]
                batch = collate([pair[0] for pair in pairs])
                truth = forward_kinematics(batch['motion'], batch['target_rest'])
                errors = (batch['trajectory'][..., :3]*2-truth).norm(dim=-1)*1000
                for i, case in enumerate(chunk):
                    prediction = predictions[offset+i]
                    _, identity = pairs[i]
                    if prediction['identity'] != identity or prediction['case_id'] != case['case_id']:
                        raise ValueError('Reconstructed input identity differs from recorded evaluation')
                    mask = batch['observation_meta'][i, ..., 0]>0
                    input_error = float(errors[i][mask].mean())
                    output_error = prediction['diagnostics']['visible_mpjpe_mm']
                    row = dict(case_id=case['case_id'], condition=condition, family=identity['family'],
                        corpus=identity['corpus'], pattern=identity['synthetic_exo_pattern'],
                        whole_window_fully_observed=bool(mask.all()), visible_joint_frames=int(mask.sum()),
                        input_visible_mpjpe_mm=input_error, output_visible_mpjpe_mm=output_error,
                        output_hidden_mpjpe_mm=prediction['diagnostics']['hidden_mpjpe_mm'],
                        visible_worsening_mm=output_error-input_error)
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                progress = dict(state='auditing_inputs', condition=condition,
                    completed=len(rows), total=len(cases), pid=os.getpid(),
                    elapsed_seconds=time.monotonic()-started, updated_at_utc=now())
                atomic_json(output/'STATUS.json', progress)
                if offset%256 == 0:
                    print(json.dumps(progress), flush=True)
            conditions[condition] = aggregate(rows)
            complete = [row for row in rows if row['whole_window_fully_observed']]
            conditions[condition]['fully_observed_windows'] = aggregate(complete) if complete else None
    before = json.loads((run/'ego_monitor/step_000000/RESULT.json').read_text())
    after_path = run/'ego_monitor'/f'step_{status["completed_updates"]:06d}'/'RESULT.json'
    after = json.loads(after_path.read_text())
    for path in [run/'ego_monitor/step_000000/RESULT.json', after_path]:
        sources[str(path.relative_to(ROOT))] = sha256(path)
    checkpoint = Path(best['checkpoint'])
    last = run/'training/last.pt'
    if sha256(checkpoint) != best['checkpoint_sha256']:
        raise ValueError('Selected checkpoint no longer matches validated metrics')
    for name, digest in sources.items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Review dependency changed: '+name)
    result = dict(state='complete', best=best, curve=curves, conditions=conditions,
        checkpoint_hash_verified=True, best_and_last_same_checkpoint=sha256(last)==sha256(checkpoint),
        ego_full_aggregates_exact=before['conditions']==after['conditions'],
        ego_family_macro_mpjpe_mm=after['selection_score_mm'],
        windows_per_condition=len(cases), validation_families=len(data.members),
        metrics_scope='Identical available joint-frame masks; family macro aggregation. Hidden input joints are never scored.',
        full_upstream_initialization_available=False, model_inference_performed=False,
        training_performed=False, test_used=False, sources=sources,
        elapsed_seconds=time.monotonic()-started, completed_at_utc=now())
    atomic_json(output/'RESULT.json', result)
    atomic_json(output/'STATUS.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'runs/amass_exo128')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    review(run, args.output or run/'review')


if __name__ == '__main__':
    main()
