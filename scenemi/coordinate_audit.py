"""Paired frozen inference under changes of the raw world coordinate frame."""
import argparse
from collections import defaultdict
import fcntl
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import AmassDataset
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.scene_geometry_cache import PreparedAmass
from scenemi.data.scene_geometry_data import GeometryEgoData, collate, payload
from scenemi.data.rigid_world import world_view
from scenemi.heldout_evaluation import load_config as heldout_config, data_for, make_model, json_hash, same_or_write
from scenemi.training import ensure_bounded, deterministic, tensor_digest
from experiments.offline_camera_retrain_v1.data import anchor_rotation
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


class AmassAuditData(GeometryEgoData):
    """Validation-only adapter, with the retained sampling/canonicalization."""
    def __init__(self, c):
        self.c, self.split, self.native = c, 'validation', None
        original = AmassDataset(ROOT/c['amass_index'], load_paths(ROOT/'configs/paths.local.yaml'), 'val')
        self.amass = PreparedAmass(original, ROOT/c['prepared_data']['root']/'amass')
        self.datasets = {'amass': self.amass}

    def pairs(self, *args, **kwargs):
        raise RuntimeError('Coordinate audit cannot train')


def select_cases(c, rich):
    groups = defaultdict(list)
    for case in json.loads((ROOT/c['amass_manifest']).read_text())['full']:
        if case['dataset'] == 'amass':
            groups[case['family']].append(case)
    names = sorted(np.random.default_rng(c['selection_seed']).choice(
        sorted(groups), size=c['amass_families'], replace=False).tolist())
    result = [sorted(groups[name], key=lambda x:x['start'])[len(groups[name])//2] for name in names]
    streams = defaultdict(list)
    for case in rich.native.cases:
        streams[case['member']].append(case)
    result += [items[len(items)//2] for _, items in sorted(streams.items())]
    rng = np.random.default_rng(c['transform_seed'])
    selected = []
    for case in result:
        # Avoid a nearly identity rotation, while covering both yaw directions.
        yaw = float(rng.uniform(np.pi/6, 5*np.pi/6)*rng.choice([-1, 1]))
        translation = rng.uniform(-c['horizontal_translation_range_m'], c['horizontal_translation_range_m'], 3)
        translation[1] = 0
        selected.append(dict(case=case, yaw_rad=yaw, translation_m=translation.tolist()))
    return selected


def transform(spec, condition):
    yaw = spec['yaw_rad'] if 'yaw' in condition else 0.
    translation = spec['translation_m'] if 'translation' in condition else [0., 0., 0.]
    return Rotation.from_rotvec([0, yaw, 0]).as_matrix(), np.asarray(translation)


def frame(data, case, identity):
    if case['dataset'] == 'amass':
        return dict(origin=identity['origin'], anchor=identity['anchor'], ground=0.)
    row = data.native.records[case['member']]
    saved = data.native.cached(row)
    start = case['start']
    origin = np.asarray(saved['camera_position_scenemi_yup'][start], np.float64).copy()
    origin[1] = 0
    anchor = anchor_rotation(np.asarray(saved['camera_rotation_scenemi_yup'][start:start+128], np.float64))
    return dict(origin=origin.tolist(), anchor=anchor.tolist(), ground=float(data.ground(row)))


def restore_world(points, reference):
    """Undo the actual normalizer, including its float32 yaw matrix and floor."""
    points = np.asarray(points, np.float64).copy()
    points[..., 1] += reference['ground']
    return points @ np.linalg.inv(reference['anchor']).T + reference['origin']


def paired_error(base, changed, base_frame, changed_frame, rotation, translation):
    original = restore_world(base, base_frame)
    restored = (restore_world(changed, changed_frame)-translation) @ rotation
    distance = np.linalg.norm(restored-original, axis=-1)*1000
    return dict(mean_mm=float(distance.mean()), max_mm=float(distance.max()))


def input_difference(base, changed):
    # Compare precisely the public inference schema, without private labels.
    left, right = payload(collate([base], 'cpu')), payload(collate([changed], 'cpu'))
    differences = {}
    for key in left:
        a, b = left[key].numpy(), right[key].numpy()
        differences[key] = dict(max_abs=float(np.max(np.abs(a.astype(np.float64)-b))),
            changed_fraction=float(np.mean(a != b)))
    return differences


@torch.inference_mode()
def predict(model, samples, scene, c, device):
    batch = collate(samples, device)
    public = payload(batch)
    raw = sample_native30(model, public, steps=c['ddim_steps'], seed=c['sampling_seed'], use_scene=scene)
    _, details = model(raw, torch.zeros(len(raw), device=device, dtype=torch.long),
        public, use_scene=False, return_details=True)
    final = model.final_motion(raw, details)
    result = dict(raw=forward_kinematics(raw, details['rest']),
        final_b=forward_kinematics(final, details['rest']),
        target=forward_kinematics(batch['motion'], batch['target_rest']))
    return {key:value.cpu().numpy() for key, value in result.items()}


def summarize(rows, c):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['case']['dataset'], row['scene'], row['condition'])].append(row)
    result = []
    for (dataset, scene, condition), items in sorted(groups.items()):
        summary = dict(dataset=dataset, scene=scene, condition=condition, cases=len(items))
        for stage in ('raw', 'final_b', 'target'):
            errors = [item['errors'][stage] for item in items]
            summary[stage] = dict(mean_mm=float(np.mean([e['mean_mm'] for e in errors])),
                max_mm=max(e['max_mm'] for e in errors))
        summary['camera_max_abs'] = max(item['inputs']['camera']['max_abs'] for item in items)
        summary['static_points_max_abs_m'] = max(item['inputs']['static_points']['max_abs'] for item in items)
        summary['occupancy_changed_fraction'] = float(np.mean([item['inputs']['occupancy']['changed_fraction'] for item in items]))
        summary['ground_delta_max_abs_m'] = max(abs(item['ground_delta_m']) for item in items)
        summary['mpjpe_delta_mean_mm'] = float(np.mean([item['mpjpe_delta_mm'] for item in items]))
        summary['within_predeclared_tolerance'] = (summary['final_b']['mean_mm'] <= c['mean_joint_tolerance_mm']
            and summary['final_b']['max_mm'] <= c['max_joint_tolerance_mm'])
        result.append(summary)
    return result


def run(c, config, device, preflight=False):
    ensure_bounded(); deterministic(c['sampling_seed'], device)
    evaluation, _ = heldout_config(ROOT/c['reference_evaluation'])
    rich, mc = data_for(evaluation)
    amass = AmassAuditData(mc)
    selection = select_cases(c, rich)
    out = ROOT/c['output']; out.mkdir(parents=True, exist_ok=True)
    if preflight:
        selection = [next(s for s in selection if s['case']['dataset'] == name) for name in ('amass', 'rich')]
        c = dict(c, ddim_steps=2)
    mode = 'PREFLIGHT_'+device if preflight else 'AUDIT'
    # Freeze case/transform selection and dependency identities before inference.
    sources = [Path(__file__), ROOT/'scenemi/data/rigid_world.py', ROOT/c['reference_evaluation'], config,
        ROOT/'scenemi/heldout_evaluation.py', ROOT/'scenemi/data/heldout_geometry.py',
        ROOT/'scenemi/data/grounded_heldout.py', ROOT/'scenemi/geometry/ground_frame.py']
    trained = torch.load(ROOT/evaluation['candidate']['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    sources_sha256 = {str((ROOT/name).resolve()):digest for name, digest in trained['contract']['source_sha256'].items()}
    sources_sha256.update({str(p.resolve()):sha256(p) for p in sources})
    protocol = dict(config=c, selection=selection, source_sha256=sources_sha256,
        amass_manifest_sha256=sha256(ROOT/c['amass_manifest']),
        rich_data_protocol_sha256=sha256(ROOT/evaluation['output']/'DATA_PROTOCOL.json'),
        device=device, frozen=True, training_or_selection=False, preflight=preflight,
        pair_policy='same seed, steps, batch membership, order and size; no output alignment',
        transforms='raw world before original canonicalization; camera, root, joints, scene positions and normals together; yaw and horizontal translation only',
        selection_policy='seeded AMASS families, one middle existing window; every RICH person stream, one middle existing window; never selected by errors',
        limitation='Tests world-coordinate preprocessing and downstream invariance, not body-shape/mount/action generalization or intrinsic neural equivariance',
        map_control='cached surfaces rigidly transformed in main arms; separate static-map voxelization and normals rebuilt from transformed first hits')
    model, identity = make_model(evaluation, mc, 'candidate_scene_b', device)
    protocol['model'] = identity; protocol['sha256'] = json_hash(protocol)
    same_or_write(out/(mode+'_PROTOCOL.json'), protocol)
    before = tensor_digest(model.state_dict())
    records = []; started = time.time()
    rows_file = out/(mode+'_rows.jsonl')
    if rows_file.exists():
        raise FileExistsError(f'Refusing to overwrite existing audit rows: {rows_file}')
    with rows_file.open('x') as stream:
        for dataset, data in (('amass', amass), ('rich', rich)):
            selected = [s for s in selection if s['case']['dataset'] == dataset]
            for first in range(0, len(selected), c['batch_size']):
                specs = selected[first:first+c['batch_size']]
                pairs = [data.sample_case(s['case']) for s in specs]
                samples = [p[0] for p in pairs]
                frames = [frame(data, s['case'], p[1]) for s, p in zip(specs, pairs)]
                modes = (False,) if dataset == 'amass' else (False, True)
                bases = {scene:predict(model, samples, scene, c, device) for scene in modes}
                conditions = ['identity_repeat', 'yaw', 'translation', 'yaw_translation']
                if dataset == 'rich' and c['rebuild_static_map_control']:
                    conditions += ['static_rebuild_identity', 'static_rebuild_yaw_translation']
                for condition in conditions:
                    changed_samples, changed_frames, transforms = [], [], []
                    for spec in specs:
                        rotation, translation = transform(spec, condition)
                        view = data if condition == 'identity_repeat' else world_view(data, rotation, translation,
                            rebuild_static=condition.startswith('static_rebuild'))
                        sample, info = view.sample_case(spec['case'])
                        changed_samples.append(sample)
                        changed_frames.append(frame(view, spec['case'], info))
                        transforms.append((rotation, translation))
                    differences = [input_difference(a,b) for a,b in zip(samples, changed_samples)]
                    # Rebuilding maps is a scene-specific control. Main no-scene
                    # arms still receive the recomputed common floor datum.
                    for scene in ((True,) if condition.startswith('static_rebuild') else modes):
                        predictions = predict(model, changed_samples, scene, c, device)
                        for i, spec in enumerate(specs):
                            rotation, translation = transforms[i]
                            errors = {key:paired_error(bases[scene][key][i], predictions[key][i], frames[i],
                                changed_frames[i], rotation, translation) for key in ('raw', 'final_b', 'target')}
                            metrics = [float(np.linalg.norm(p['final_b'][i]-p['target'][i], axis=-1).mean()*1000)
                                for p in (bases[scene], predictions)]
                            row = dict(case=spec['case'], actual_source_identity=pairs[i][1], scene=scene,
                                condition=condition, errors=errors, inputs=differences[i],
                                ground_delta_m=changed_frames[i]['ground']-frames[i]['ground'],
                                base_mpjpe_mm=metrics[0], changed_mpjpe_mm=metrics[1], mpjpe_delta_mm=metrics[1]-metrics[0])
                            if condition == 'identity_repeat' and any(e['max_mm'] != 0 for e in errors.values()):
                                raise AssertionError('Identical input repeat was not deterministic')
                            records.append(row); stream.write(json.dumps(row, sort_keys=True)+'\n')
                        stream.flush()
                print(json.dumps(dict(dataset=dataset, done=min(first+c['batch_size'], len(selected)),
                    total=len(selected), rows=len(records), elapsed_s=round(time.time()-started,1))), flush=True)
    after = tensor_digest(model.state_dict())
    if after != before:
        raise AssertionError('Frozen model tensors changed')
    for path, digest in sources_sha256.items():
        if sha256(path) != digest:
            raise ValueError(f'Audit source changed while running: {path}')
    result = dict(protocol_sha256=protocol['sha256'], model_tensor_sha256=after,
        frozen_tensor_digest_unchanged=True, rows=len(records), elapsed_s=time.time()-started,
        summaries=summarize(records, c))
    atomic_json(out/(mode+'_RESULT.json'), result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    config = Path(args.config).resolve(); c = yaml.safe_load(config.read_text())
    if c['version'] != 'ego-coordinate-audit-v1' or not (ROOT/c['output']).resolve().is_relative_to(ROOT/'runs'):
        raise ValueError('Invalid audit configuration/output')
    out = ROOT/c['output']; out.mkdir(parents=True, exist_ok=True)
    with (out/'audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(c, config, args.device, args.preflight)


if __name__ == '__main__':
    main()
