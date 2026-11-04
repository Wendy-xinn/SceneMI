"""Full-validation checkpoint selection beside an immutable running trainer.

Every eligible validation recording is tiled at 128 frames, with an end-aligned
tail window. Only unreflected recordings are scored. Snapshots are atomic hard
links, so replacing last.pt cannot mix two checkpoints. Retain one best, one
last (owned by training), per-case metrics and identities; never consult test.
"""
import argparse
from collections import Counter
import fcntl
import gc
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import AmassDataset, collate, inference_payload
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.ego_training import CONFIG, INDEX, OUTPUT, load_ego_config
from scenemi.evaluation import summarize, verify_training_seal
from scenemi.metrics import motion_metrics
from scenemi.models import build_model
from scenemi.parallel_resources import configure_cuda
from scenemi.training import deterministic, ensure_bounded, now
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

POLICY = ROOT / 'configs/amass_ego_selection.json'


def window_starts(frames):
    if frames < 128:
        raise ValueError('An eligible validation recording must contain 128 frames')
    return sorted(set(range(0, frames-127, 128)) | {frames-128})


def validation_manifest(members):
    cases = []
    for member, row in enumerate(members):
        if row['split'] != 'val':
            raise ValueError('Checkpoint selection may read validation only')
        variant = next(v for v in row['variants'] if not v['reflected'])
        for start in window_starts(variant['frames_after_relative_pose_drop']):
            cases.append(dict(case_id=len(cases), member=member, family=row['family'],
                corpus=row['corpus'], variant_sha256=variant['sha256'], start=start))
    return cases


def pin_checkpoint(source, target):
    """Source may be atomically replaced concurrently; the pinned inode cannot."""
    target.unlink(missing_ok=True)
    os.link(source, target)


def save_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@torch.inference_mode()
def evaluate_candidate(model, data, cases, policy, device, out, status_base):
    rows = []
    started = time.monotonic()
    size = policy['batch_size']
    metrics_path = out / 'metrics.jsonl'
    # Resuming an interrupted candidate restarts that candidate, never mixes rows.
    with metrics_path.open('w') as stream:
        for offset in range(0, len(cases), size):
            chunk, samples, identities = cases[offset:offset+size], [], []
            for case in chunk:
                sample, identity = data.sample(np.random.default_rng(policy['seed']), 'ego',
                    member=case['member'], start=case['start'], reflected=False, yaw=0., temporal_dropout=None)
                if identity['variant_sha256'] != case['variant_sha256']:
                    raise ValueError('Validation source changed')
                samples.append(sample)
                identities.append(identity)
            batch = collate(samples, device)
            payload = inference_payload(batch)
            prediction = sample_native30(model, payload, steps=policy['ddim_steps'],
                                         seed=policy['seed'], use_scene=False)
            _, details = model(prediction, torch.zeros(len(chunk), device=device, dtype=torch.long),
                               payload, return_details=True)
            joints = forward_kinematics(prediction, details['rest'])
            truth = forward_kinematics(batch['motion'], batch['target_rest'])
            for i, case in enumerate(chunk):
                row = dict(case_id=case['case_id'], identity=identities[i],
                           metrics=motion_metrics(joints[i], truth[i]))
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            status = dict(status_base, state='evaluating', completed=len(rows), total=len(cases),
                elapsed_seconds=time.monotonic()-started, updated_at_utc=now(), pid=os.getpid())
            atomic_json(out.parent/'STATUS.json', status)
            if offset == 0 or len(rows)//100 != offset//100:
                print(json.dumps(status), flush=True)
    return dict(**summarize(rows), elapsed_seconds=time.monotonic()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0)


def run(args):
    ensure_bounded()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'watch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config, policy = load_ego_config(args.config), json.loads(args.policy.read_text())
    expected = json.loads(POLICY.read_text())
    if policy != expected or policy['split'] != 'val':
        raise ValueError('Unsupported validation protocol')
    paths = load_paths(args.paths)
    seal = json.loads((args.training/'CONTRACT.json').read_text())
    verify_training_seal(seal, config, paths, INDEX)
    deterministic(policy['seed'], args.device)
    if args.device == 'cuda':
        configure_cuda()
    data = AmassDataset(INDEX, paths, 'val')
    cases = validation_manifest(data.members)
    manifest_path = out/'CASES.jsonl'
    if manifest_path.exists():
        old = [json.loads(line) for line in manifest_path.read_text().splitlines()]
        if old != cases:
            raise ValueError('Frozen validation manifest changed')
    else:
        save_jsonl(manifest_path, cases)
    sources = {str(Path(p).relative_to(ROOT)): sha256(Path(p)) for p in
               (__file__, ROOT/'scenemi/parallel_resources.py', ROOT/'scenemi/metrics.py',
                ROOT/'scenemi/evaluation.py')}
    protocol = dict(policy=policy, training_contract_sha256=seal['sha256'],
        index_sha256=sha256(INDEX), manifest_sha256=sha256(manifest_path),
        independent_families=len(data.members), windows=len(cases),
        per_corpus_families=dict(Counter(r['corpus'] for r in data.members)),
        sources=sources, test_used=False, gt_shape_input=False,
        aggregation='Mean windows within source family, then equal weight per family; corpus means also reported.',
        sampling='Fixed ordered batches; same seed at each batch; one sample, no best-of-k.',
        coverage='All eligible unreflected validation frames; end-aligned tail overlaps when needed.',
        selection_scope='Best among fully evaluated captured checkpoints; overwritten earlier checkpoints unavailable.')
    protocol_path = out/'PROTOCOL.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Selection protocol or evaluator changed; use a new output directory')
    atomic_json(protocol_path, protocol)
    pinned = out/'.candidate.pt'
    best_path = out/'BEST.json'
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    history_path = out/'history.jsonl'
    history = [json.loads(s) for s in history_path.read_text().splitlines()] if history_path.exists() else []
    last_evaluated = max((r['step'] for r in history), default=-1)
    next_step = ((last_evaluated//policy['every_updates'])+1)*policy['every_updates'] if last_evaluated >= 0 else 0
    try:
        while True:
            status = json.loads((args.training/'STATUS.json').read_text())
            final = config['training']['planned_updates']
            if status['completed_updates'] >= next_step:
                pin_checkpoint(args.training/'last.pt', pinned)
                state = torch.load(pinned, map_location='cpu', weights_only=False, mmap=True)
                step = state['step']
                if state['contract'] != seal:
                    raise ValueError('Checkpoint contract changed')
                if step > last_evaluated and step >= next_step:
                    digest = sha256(pinned)
                    model = build_model(config, paths['smplh_models']/'neutral/model.npz')
                    model.load_state_dict(state['model'], strict=True)
                    del state
                    model.to(args.device).eval()
                    folder = out / f'step_{step:06d}'
                    folder.mkdir(exist_ok=True)
                    metadata = dict(step=step, checkpoint_sha256=digest,
                        protocol_sha256=sha256(protocol_path), started_at_utc=now())
                    atomic_json(folder/'CHECKPOINT.json', metadata)
                    result = evaluate_candidate(model, data, cases, policy, args.device, folder, metadata)
                    corpora = sorted({row['corpus'] for row in cases})
                    rows = [json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
                    result['per_corpus'] = {corpus: summarize([r for r in rows if r['identity']['corpus']==corpus])
                                            for corpus in corpora}
                    result.update(metadata, state='complete', completed_at_utc=now())
                    if any(sha256(ROOT/name) != digest for name, digest in sources.items()):
                        raise ValueError('Evaluator source changed during candidate evaluation')
                    score = result['family_macro_mean']['mpjpe_mm']
                    result['selection_score_mm'] = score
                    if best is None or score < best['selection_score_mm']:
                        temporary = out/'.best.pt'
                        pin_checkpoint(pinned, temporary)
                        temporary.replace(out/'best.pt')
                        best = dict(step=step, checkpoint_sha256=digest, selection_score_mm=score,
                            protocol_sha256=sha256(protocol_path), metrics=str(folder/'RESULT.json'),
                            checkpoint=str(out/'best.pt'), full_validation=True, updated_at_utc=now())
                        atomic_json(best_path, best)
                    result['selected_best_step'] = best['step']
                    atomic_json(folder/'RESULT.json', result)
                    with history_path.open('a') as stream:
                        stream.write(json.dumps(result, allow_nan=False)+'\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                    last_evaluated = step
                    next_step = (step//policy['every_updates']+1)*policy['every_updates']
                    del model, rows
                    gc.collect()
                    if args.device == 'cuda':
                        torch.cuda.empty_cache()
                    print(json.dumps(dict(evaluated_step=step, score_mm=score, best=best)), flush=True)
                else:
                    del state
                pinned.unlink(missing_ok=True)
            if last_evaluated == final or not args.watch:
                atomic_json(out/'STATUS.json', dict(state='complete' if last_evaluated==final else 'idle',
                    last_evaluated_step=last_evaluated, best=best, pid=os.getpid(), updated_at_utc=now()))
                return
            if status['state'] in ('failed', 'stopped', 'interrupted'):
                atomic_json(out/'STATUS.json', dict(state='training_'+status['state'],
                    last_evaluated_step=last_evaluated, best=best, updated_at_utc=now()))
                return
            # Detect a disappeared trainer; never wait forever on a stale status.
            trainer = Path('/proc') / str(status['pid'])
            if not trainer.exists() and status['state'] != 'complete':
                raise RuntimeError('Trainer process missing; retained completed selection results')
            atomic_json(out/'STATUS.json', dict(state='waiting_for_checkpoint', next_step=next_step,
                last_evaluated_step=last_evaluated, best=best, pid=os.getpid(), updated_at_utc=now()))
            time.sleep(20)
    except BaseException as error:
        atomic_json(out/'STATUS.json', dict(state='failed', error=repr(error),
            last_evaluated_step=last_evaluated, pid=os.getpid(), updated_at_utc=now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--policy', type=Path, default=POLICY)
    parser.add_argument('--training', type=Path, default=OUTPUT/'training')
    parser.add_argument('--output', type=Path, default=OUTPUT/'selection')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--watch', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
