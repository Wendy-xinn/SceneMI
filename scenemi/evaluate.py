"""Fixed-window EGO validation and an independent evaluation command."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from scenemi.config import ROOT, load_paths
from scenemi.checkpoint import load_trained_model
from scenemi.execution import configure_runtime, ensure_bounded
from scenemi.data.ego import FourEgoData, SOURCE_ORDER, collate, payload, source_cases
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.camera_noise import augment_camera_batch, case_seed
from scenemi.metrics import motion_metrics
from scenemi.training import now, deterministic
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['identity']['dataset'], row['identity'].get('family', 'unknown'))].append(row['metrics'])
    by_dataset = {}
    for (dataset, _), values in grouped.items():
        by_dataset.setdefault(dataset, []).append(values)
    dataset_metrics = {}
    for dataset, family_values in by_dataset.items():
        keys = family_values[0][0].keys()
        dataset_metrics[dataset] = {key: float(np.mean([np.mean([v[key] for v in fam]) for fam in family_values]))
                                    for key in keys}
    keys = next(iter(dataset_metrics.values())).keys()
    macro = {key: float(np.mean([dataset_metrics[d][key] for d in SOURCE_ORDER])) for key in keys}
    return dict(windows=len(rows), by_dataset=dataset_metrics,
                dataset_macro_mean=macro,
                selection_score_mm=macro['mpjpe_mm'],
                aggregation='equal source dataset, then equal family')


@torch.inference_mode()
def evaluate(model, data, cases, c, device, out=None, *, noise_policy=None, use_scene=None):
    scene_stage = c['version'] == 'ego-scene-adaptation-v1'
    use_scene = scene_stage if use_scene is None else use_scene
    pack, public_payload = collate, payload
    if scene_stage:
        if noise_policy is not None:
            raise ValueError('Scene evaluation requires a jointly transformed map for camera corruption')
        from scenemi.data.scene_ego import collate as pack, payload as public_payload
    rows = []
    model.eval()
    for offset in tqdm(range(0, len(cases), c['validation']['eval_batch_size']),
                       desc='Validation', unit='batch', dynamic_ncols=True, leave=False):
        chunk = cases[offset:offset+c['validation']['eval_batch_size']]
        pairs = [(data.sample_camera_case(case) if scene_stage and not use_scene else data.sample_case(case))
                 for case in chunk]
        batch = pack([x[0] for x in pairs], device)
        if noise_policy is not None:
            batch = augment_camera_batch(batch, noise_policy, probability=1.,
                seeds=[case_seed(case, noise_policy['seed']) for case in chunk])
        prediction = sample_native30(model, public_payload(batch), steps=c['validation']['ddim_steps'],
                                     seed=c['validation']['seed'], use_scene=use_scene)
        _, details = model(prediction, torch.zeros(len(chunk), device=device, dtype=torch.long),
                           public_payload(batch), use_scene=use_scene, return_details=True)
        predicted = forward_kinematics(prediction, details['rest'])
        truth = forward_kinematics(batch['motion'], batch['target_rest'])
        for i, case in enumerate(chunk):
            rows.append(dict(identity=pairs[i][1], case=case,
                             metrics=motion_metrics(predicted[i], truth[i])))
    result = aggregate(rows)
    result.update(state='complete', completed_at_utc=now())
    if scene_stage:
        result.update(scene_enabled=use_scene, scene_map_protocol='observed-recording-scene-v1',
                      causal=False, test_used=False)
    if noise_policy is not None:
        result.update(input_condition='synthetic_camera_error', camera_noise=noise_policy,
                      corruption_probability=1., noise_strength=1., frontend_calibrated=False)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'metrics.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
        atomic_json(out / 'RESULT.json', result)
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--camera-noise-config', type=Path,
                        help='Optional YAML camera_noise policy; evaluation corrupts every window')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Evaluation output already exists; choose a new directory')
    ensure_bounded()
    model, c, step = load_trained_model(args.checkpoint)
    configure_runtime(c, args.device)
    deterministic(c['seed'], args.device)
    paths = load_paths(ROOT / 'configs/paths.local.yaml')
    if c['version'] == 'ego-scene-adaptation-v1':
        from scenemi.data.scene_ego import SceneEgoData
        data = SceneEgoData(c, paths, 'validation')
    else:
        data = FourEgoData(c, paths, 'validation')
    policy = None
    if args.camera_noise_config is not None:
        import yaml
        from scenemi.data.camera_noise import validate_policy
        policy = validate_policy(yaml.safe_load(args.camera_noise_config.read_text())['camera_noise'])
    result = evaluate(model.to(args.device), data, source_cases(data), c, args.device, args.output,
                      noise_policy=policy)
    result.update(checkpoint_step=step, checkpoint_sha256=sha256(args.checkpoint), test_used=False)
    atomic_json(args.output / 'RESULT.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
