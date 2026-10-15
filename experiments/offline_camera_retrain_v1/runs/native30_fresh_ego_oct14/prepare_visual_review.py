"""Archive actual VAL outputs; decode native meshes without pose corrections."""
import datetime
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.native30_memory import ObservedStaticMemory
from experiments.offline_camera_retrain_v1.native30_cache_io import verified_array

ROOT = Path(__file__).resolve().parent


def prepare():
    torch.set_num_threads(4)
    source = ROOT / 'training/validation_latest.npz'
    with np.load(source) as a:
        step = int(a['step'])
    out = ROOT / f'visual_review_step{step}'
    out.mkdir(exist_ok=True)
    archive = out / 'validation_outputs.npz'
    if not archive.exists():
        shutil.copyfile(source, archive)
    saved = np.load(archive, allow_pickle=False)
    assert int(saved['step']) == step
    validation = next(json.loads(x) for x in (ROOT / 'training/validation.jsonl').read_text().splitlines() if json.loads(x)['step'] == step)
    (out / 'VALIDATION.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    identities = json.loads(str(saved['identities_json']))
    entries = [json.loads(x) for x in (HERE / 'data/native30_official_memory_v3/manifest.jsonl').read_text().splitlines()]
    entries = [x for x in entries if x['split'] == 'validation']
    metadata = {}
    arrays = {}
    def load(folder, key):
        folder = str(folder)
        if folder not in metadata:
            metadata[folder] = json.loads((Path(folder) / 'metadata.json').read_text())
        if (folder, key) not in arrays:
            arrays[folder, key] = verified_array(Path(folder) / f'{key}.npy',
                metadata[folder]['output_sha256'][key + '.npy'], HERE / 'data/native30_verified_read_cache_v1')
        return arrays[folder, key]
    config = json.loads((HERE / 'native30_fresh_ego_config.json').read_text())
    cache_root = (HERE / config['train_manifest']).resolve().parent.parent / 'native30_runtime_cache_v2'
    memory = ObservedStaticMemory(load, entries, protocol=config['scene']['memory_protocol'], cache_root=cache_root / 'static_maps')
    records = []
    for row in validation['cases']:
        if row['mode'] != 'ego' or row['regime'] != 'clean' or row.get('scene_variant', 'scene_on') != 'scene_on':
            continue
        i = row['case']
        identity = identities[i]
        info = identity['native_body']
        model = load_model(info['model'], info['gender']).cpu()
        tracks, payload = {}, {}
        error = 0.
        for regime, key in (('truth', f'{i}_truth'), ('clean', f'{i}_clean_prediction'), ('noisy', f'{i}_noisy_prediction')):
            motion = saved[key].astype(np.float32)
            vertices, joints = [], []
            for first in range(0, len(motion), 24):
                v, faces, j = decode_native_mesh(motion[first:first+24], info, model)
                vertices.append(v.astype(np.float32))
                joints.append(j.astype(np.float32))
            vertices, joints = np.concatenate(vertices), np.concatenate(joints)
            rest = torch.from_numpy(saved[f'{i}_rest'])[None]
            expected = forward_kinematics(torch.from_numpy(motion)[None], rest)[0].numpy()
            delta = float(np.abs(joints - expected).max())
            error = max(error, delta)
            assert delta < 1e-4, (i, regime, delta)
            payload[regime + '_vertices'] = vertices
            payload[regime + '_joints'] = joints
            tracks[regime] = joints.tolist()
        static, _ = memory.window(identity, 'ego', len(motion))
        static = static[np.linspace(0, len(static)-1, min(4500, len(static)), dtype=int)] if len(static) else static
        folder, start = identity['path'], identity['start']
        frame_points = np.asarray(load(folder, 'visible_frame_points_scenemi_yup')[start:start+192])
        owners = np.asarray(load(folder, 'visible_frame_owner')[start:start+192])
        assert not np.any(owners == 100)
        origin, anchor = np.asarray(identity['origin']), np.asarray(identity['anchor'])
        dynamic = []
        for p, own in zip(frame_points, owners):
            p = ((p[own > 0] - origin) @ anchor.T).astype(np.float32)
            p = p[np.linspace(0, len(p)-1, min(150, len(p)), dtype=int)] if len(p) else p
            dynamic.append(p)
        payload.update(faces=np.asarray(faces, np.int32), static=static)
        for regime in ('clean', 'noisy'):
            payload[regime + '_observations'] = saved[f'{i}_{regime}_observations']
            payload[regime + '_observation_meta'] = saved[f'{i}_{regime}_observation_meta']
        for t, p in enumerate(dynamic):
            payload[f'dynamic_{t}'] = p
        mesh_path = out / f'case_{i}.npz'
        np.savez_compressed(mesh_path, **payload)
        metrics = {r['regime']: r['metrics'] for r in validation['cases'] if r['case'] == i and r.get('scene_variant', 'scene_on') == 'scene_on'}
        record = dict(case=i, dataset=row['dataset'], group=row['group'], panel=row['panel'],
            sequence_id=identity['sequence_id'], source_start_30fps=identity['source_start_30fps'],
            mesh_file=mesh_path.name, mesh_fk_max_error_m=error, metrics=metrics,
            tracks=tracks, static=static.tolist(), dynamic=[p.tolist() for p in dynamic],
            observations={k: (saved[f'{i}_{k}_observations'][..., :3] * 2).tolist() for k in ('clean', 'noisy')},
            available={k: (saved[f'{i}_{k}_observation_meta'][..., 0] > 0).tolist() for k in ('clean', 'noisy')})
        records.append(record)
        print('decoded', row['dataset'], row['panel'], 'mesh_fk_max_error_m', error, flush=True)
        del model, payload, vertices, joints, frame_points
        arrays.clear()
        gc.collect()
    def rounded(x):
        if isinstance(x, float):return round(x, 5)
        if isinstance(x, list):return [rounded(v) for v in x]
        if isinstance(x, dict):return {k: rounded(v) for k, v in x.items()}
        return x
    compact = rounded(dict(step=step, fps=30, frames=192, cases=records))
    (out / 'replay_data.json').write_text(json.dumps(compact, ensure_ascii=False, separators=(',', ':')))
    template = (ROOT / 'visual_replay_template.html').read_text()
    (out / 'REPLAY.html').write_text(template.replace('__DATA__', json.dumps(compact, ensure_ascii=False, separators=(',', ':'))))
    audit = dict(status='passed', created_at=datetime.datetime.now().astimezone().isoformat(), step=step,
        split='official validation', optimizer_steps=0, GT_alignment=False, floor_snap=False,
        hands_face='neutral display convention from 22-joint representation', source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        cases=[{k: v for k, v in c.items() if k not in ('tracks', 'static', 'dynamic', 'observations', 'available')} for c in records],
        scope='All six scheduled ego windows: three fixed and three rotating, clean/noisy, DDIM20 scene ON. No selection by outcome. Meshes are display-only; observed static/dynamic points only, no fabricated floor.')
    (out / 'VISUALIZATION_AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    return out


if __name__ == '__main__':
    print(prepare(), flush=True)
