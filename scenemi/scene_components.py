"""Fixed-checkpoint diagnosis of global scene, local context and feedback.

These are input interventions on validation windows, not training ablations.
All variants, references and identities are pinned before inference starts.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.local_scene import LocalSceneDataset
from scenemi import local_training as local
from scenemi.metrics import motion_metrics
from scenemi.parallel_resources import configure_cuda
from scenemi.scene_evaluation import check_parity, predict, verify_seal
from scenemi.scene_selection import aggregate_scene_rows
from scenemi.training import deterministic, ensure_bounded, now
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

VARIANTS = {'global_only': (False, False), 'local_without_feedback': (True, False)}


@contextmanager
def component_flags(model, variant):
    enabled = VARIANTS[variant]
    original = model.local_enabled, model.feedback_enabled
    model.local_enabled, model.feedback_enabled = enabled
    try:
        yield
    finally:
        model.local_enabled, model.feedback_enabled = original


def run(args):
    ensure_bounded()
    if args.device == 'cuda':
        configure_cuda()
    policy = json.loads(args.policy.read_text())
    if (policy['protocol'] != 'native-scene-components-v1' or
            policy['variants'] != list(VARIANTS) or policy['validation_tier'] != 'quick'):
        raise ValueError('Unexpected component diagnosis protocol')
    c = local.settings(ROOT/policy['config'])
    seal = json.loads((ROOT/policy['contract']).read_text())
    verify_seal(seal, c, args.scene_paths)
    checkpoint = ROOT/policy['checkpoint']
    for path_key, hash_key in [('checkpoint', 'checkpoint_sha256'),
            ('reference', 'reference_sha256'), ('control_reference', 'control_reference_sha256')]:
        if sha256(ROOT/policy[path_key]) != policy[hash_key]:
            raise ValueError('Pinned diagnostic input changed: '+path_key)
    reference = json.loads((ROOT/policy['reference']).read_text())
    control = json.loads((ROOT/policy['control_reference']).read_text())
    cases = seal['validation_cases']['quick']
    if (seal['sha256'] != policy['contract_sha256'] or
            reference['step'] != policy['step'] or reference['contract_sha256'] != seal['sha256'] or
            len(reference['rows']) != len(cases) or
            [r['identity'] for r in reference['rows']] != [r['identity'] for r in control['rows']]):
        raise ValueError('Selection references must share the fixed ordered quick manifest')
    out = ROOT/policy['output']; out.mkdir(parents=True, exist_ok=True)
    sources = {name:sha256(ROOT/name) for name in (
        'scenemi/scene_components.py', 'scenemi/scene_evaluation.py',
        'scenemi/benchmark.py', 'scenemi/runtime.py', 'scenemi/metrics.py')}
    protocol = dict(policy=policy, validation_cases=cases, sources=sources,
                    training_contract_sha256=seal['sha256'], test_used=False,
                    selection_reference='validation; diagnostic exploration, not independent test')
    record = out/'PROTOCOL.json'
    if record.exists() and json.loads(record.read_text()) != protocol:
        raise ValueError('A started component protocol cannot change')
    atomic_json(record, protocol)
    deterministic(c['seed'], args.device)
    model = local.make_model(c, load_paths(args.paths), args.scene_paths)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if state['contract'] != seal or state['step'] != policy['step']:
        raise ValueError('Checkpoint content differs from the pinned identity')
    model.load_state_dict(state['model'], strict=True); del state
    model = model.to(args.device).eval()
    data = LocalSceneDataset(args.scene_paths, model.shape.rest_template[:1].cpu().numpy(),
                             split='validation', config=c)
    rows_file = out/'metrics.jsonl'
    saved = [json.loads(line) for line in rows_file.read_text().splitlines()] if rows_file.exists() else []
    expected = [(i, v) for i in range(len(cases)) for v in VARIANTS]
    if [(r['case_id'], r['variant']) for r in saved] != expected[:len(saved)]:
        raise ValueError('Invalid partial component evaluation order')
    for row in saved:
        if row['identity'] != reference['rows'][row['case_id']]['identity']:
            raise ValueError('Partial diagnostic identity changed')
    rows = {v:[r for r in saved if r['variant']==v] for v in VARIANTS}
    parity, checked = [], set()
    started = time.monotonic()
    def status(state, **extra):
        value = dict(state=state, completed=sum(map(len, rows.values())),
            total=len(expected), pid=os.getpid(), updated_at_utc=now(),
            elapsed_seconds=time.monotonic()-started, **extra)
        atomic_json(out/'STATUS.json', value)
        return value
    try:
        with rows_file.open('a') as stream:
            for i, case in enumerate(cases):
                if all(len(rows[v])>i for v in VARIANTS) and case['dataset'] in checked:
                    continue
                sample, identity = data.sample(np.random.default_rng(c['validation']['seed']),
                    member=case['member'], start=case['start'], yaw=0.)
                identity['category'] = case['category']
                if identity != reference['rows'][i]['identity'] or identity['synthetic_chair']:
                    raise ValueError('Diagnostic window changed')
                batch = collate([sample], args.device)
                target = forward_kinematics(batch['motion'], batch['target_rest'])[0]
                if case['dataset'] not in checked:
                    pred = predict(model, batch, use_scene=True, local=True,
                        seed=c['validation']['seed'], steps=c['validation']['ddim_steps'], device=args.device)
                    metrics = motion_metrics(pred, target)
                    check_parity(metrics, reference['rows'][i]['metrics'])
                    checked.add(case['dataset']); parity.append(dict(case_id=i, metrics=metrics))
                for variant in VARIANTS:
                    if len(rows[variant])>i:
                        continue
                    with component_flags(model, variant):
                        pred = predict(model, batch, use_scene=True, local=True,
                            seed=c['validation']['seed'], steps=c['validation']['ddim_steps'], device=args.device)
                    row = dict(case_id=i, variant=variant, identity=identity, metrics=motion_metrics(pred, target))
                    rows[variant].append(row)
                    stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                if (i+1)%10==0 or i==0:
                    print(json.dumps(status('running')), flush=True)
        # No files in the explicit evaluator closure may change during execution.
        if any(sha256(ROOT/name)!=digest for name,digest in sources.items()):
            raise ValueError('Evaluator source changed during component diagnosis')
        result = dict(state='complete', protocol=protocol, parity_checks=parity,
            scene_full=aggregate_scene_rows(reference['rows']),
            independent_noscene=aggregate_scene_rows(control['rows']),
            variants={v:aggregate_scene_rows(r) for v,r in rows.items()}, completed_at_utc=now())
        atomic_json(out/'RESULT.json', result)
        status('complete', scores_mm={v:r['selection_score_mm'] for v,r in result['variants'].items()})
        return {v:r['selection_score_mm'] for v,r in result['variants'].items()}
    except BaseException as error:
        status('failed', error=repr(error)); raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, default=ROOT/'configs/scene_components_benchmark.json')
    p.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
    p.add_argument('--scene-paths', type=Path, default=ROOT/'configs/scene_paths.local.yaml')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    print(json.dumps(run(p.parse_args(argv)), indent=2), flush=True)


if __name__ == '__main__':
    main()
