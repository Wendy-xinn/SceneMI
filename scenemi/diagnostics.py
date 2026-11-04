"""Bounded validation diagnostics for the unchanged AMASS motion prior."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from scenemi.config import ROOT, load_config, load_paths
from scenemi.data.amass_dataset import AmassDataset, collate, inference_payload
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.evaluation import verify_training_seal, summarize
from scenemi.geometry.alignment import place_ego_motion
from scenemi.metrics import motion_metrics
from scenemi.models import build_model
from scenemi.training import deterministic, ensure_bounded, now
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d, IDENTITY_6D


@torch.inference_mode()
def ego_alignment(args):
    ensure_bounded()
    paths, config = load_paths(args.paths), load_config()
    checkpoint = paths['output']/'training/last.pt'
    index = paths['output']/'index/complete.json'
    seal = json.loads((checkpoint.parent/'CONTRACT.json').read_text())
    verify_training_seal(seal, config, paths, index)
    deterministic(args.seed, args.device)
    data = AmassDataset(index, paths, split='val')
    rng = np.random.default_rng(args.seed)
    selected = []
    for corpus in sorted({row['corpus'] for row in data.members}):
        members = [i for i, row in enumerate(data.members) if row['corpus']==corpus]
        selected.extend(int(i) for i in rng.choice(members, min(len(members), args.per_corpus), replace=False))
    cases = [data.sample(rng, 'ego', member=i, reflected=False, yaw=0., temporal_dropout=None) for i in selected]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use an empty diagnostic output directory')
    if args.device=='cuda':
        torch.cuda.set_per_process_memory_fraction(6*2**30/torch.cuda.get_device_properties(0).total_memory)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['contract'] != seal or state['step'] != 55000:
        raise ValueError('Expected the unchanged final 55k model')
    model = build_model(config, paths['smplh_models']/'neutral/model.npz')
    model.load_state_dict(state['model'], strict=True)
    del state
    model = model.to(args.device).eval()
    digest = sha256(checkpoint)
    atomic_json(output/'PROTOCOL.json', dict(
        protocol='ego-camera-placement-validation-v1', split='val', diagnostic_only=True,
        checkpoint_sha256=digest, training_contract_sha256=seal['sha256'],
        ddim_steps=20, seed=args.seed, cases=[identity for _, identity in cases],
        strategies=['raw', 'translation_only', 'camera_rigid'],
        calibration='predicted shape only; unchanged two-eye +2cm mount',
        train_updates=0, ground_truth_placement=False, local_articulation_changed=False,
        sources={name:sha256(ROOT/name) for name in ('scenemi/diagnostics.py',
                 'scenemi/geometry/alignment.py', 'scenemi/metrics.py', 'scenemi/evaluation.py')},
        started_at_utc=now()))
    started = time.monotonic()
    rows = []
    for sample, identity in cases:
        batch = collate([sample], args.device)
        if not torch.all(batch['camera_available']==1) or not torch.all(batch['camera_confidence']==1):
            raise ValueError('This diagnostic requires fully observed EGO cameras')
        payload = inference_payload(batch)
        motion = sample_native30(model, payload, steps=20, seed=args.seed, use_scene=False)
        _, details = model(motion, torch.zeros(1, device=args.device, dtype=torch.long), payload, return_details=True)
        strategies = dict(raw=motion,
            translation_only=place_ego_motion(motion, details['rest'], details['head_camera_translation'], batch['camera'], rotation=False),
            camera_rigid=place_ego_motion(motion, details['rest'], details['head_camera_translation'], batch['camera']))
        truth = forward_kinematics(batch['motion'], batch['target_rest'])[0]
        reference_rotation = rotation_from_6d(batch['camera'][..., 3:]-motion.new_tensor(IDENTITY_6D))
        row = dict(identity=identity, strategies={})
        for name, pred in strategies.items():
            joints = forward_kinematics(pred, details['rest'])
            rotation = global_rotations(pred)[:, :, 15]
            camera = joints[:, :, 15]+(rotation @ details['head_camera_translation'][:, None, :, None])[..., 0]
            angle = (((rotation.transpose(-1,-2) @ reference_rotation).diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1).acos()
            metrics = motion_metrics(joints[0], truth)
            metrics.update(camera_position_error_mm=float((camera-batch['camera'][..., :3]*2).norm(dim=-1).mean()*1000),
                           camera_rotation_error_deg=float(angle.mean()*180/torch.pi))
            row['strategies'][name] = metrics
            if name!='raw':
                if not torch.equal(pred[..., 9:135], motion[..., 9:135]):
                    raise AssertionError('Non-root local rotations changed')
        if abs(row['strategies']['raw']['pa_mpjpe_mm']-row['strategies']['camera_rigid']['pa_mpjpe_mm'])>.01:
            raise AssertionError('A root rigid transform changed standard PA-MPJPE')
        rows.append(row)
    if sha256(checkpoint)!=digest:
        raise ValueError('Checkpoint changed during diagnostic')
    atomic_json(output/'CASES.json', rows)
    result = dict(state='complete', diagnostic_only=True, split='val', windows=len(rows),
        independent_source_families=len({row['identity']['family'] for row in rows}),
        checkpoint_sha256=digest, elapsed_seconds=time.monotonic()-started,
        completed_at_utc=now(), strategies={name:summarize([dict(identity=row['identity'],
            metrics=row['strategies'][name]) for row in rows]) for name in strategies})
    atomic_json(output/'RESULT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostic', choices=('ego-alignment',))
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--per-corpus', type=int, default=4)
    args = parser.parse_args()
    if args.per_corpus<1:
        parser.error('Positive per-corpus count required')
    print(json.dumps(ego_alignment(args), indent=2))


if __name__=='__main__':
    main()
