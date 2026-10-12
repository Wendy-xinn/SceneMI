"""Raw native30 export, separate from every native20 cache and checkpoint.

First-hit observed surfaces only. Full bodies are used by the simulator for
occlusion; target skin is excluded from environmental conditions. A restricted
export is explicitly diagnostic, and cannot be mistaken for a full training set.
"""
import argparse
import csv
import hashlib
import json
import pickle
import time
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from smplx.lbs import blend_shapes, vertices2joints

from experiments.offline_camera_retrain_v1.data import HERE, PROJECTS, SOURCE, PARENTS
from experiments.offline_camera_retrain_v1.native30_splits import (
    SPLIT_PROTOCOL, LEGACY_SPLIT_PROTOCOL, normalize_split, official_sources)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from experiments.offline_camera_retrain_v1.export_native20_training_scenes import (
    read_fits, resample_fits, forward_chunk, raw_pv_camera, virtual_head_camera)
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from experiments.offline_camera_retrain_v1.prepare_rich_pseudo_ego import (
    RICH_GENDER, RICH_TO_SCENEMI, _read_transform, _load_smplx_model)
from experiments.offline_camera_retrain_v1.prepare_trumans_dynamic_v2 import load_objects
from experiments.offline_camera_retrain_v1.scene_visibility_v2 import (
    render_scene, static_ray_scene, wearer_faces)
from experiments.offline_camera_retrain_v1.export_egobody_exo import (
    camera_rays, cast_body, regional_faces)

PROTOCOL = 'native30-observer-target-v1-raw-firsthit'
DEFAULT_OUTPUT = HERE / 'data/native30_official_memory_v2'


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def inventory(split, *, split_protocol=SPLIT_PROTOCOL):
    split = normalize_split(split)
    if split_protocol not in (SPLIT_PROTOCOL, LEGACY_SPLIT_PROTOCOL):
        raise ValueError('Unknown split protocol')
    result = []
    exclusions = {r['sequence_id'] for r in json.loads((HERE / 'data/trumans_temporal_training_v3/exclusions.json').read_text())}
    # The old manifest provides identities/split boundaries, never 20Hz values.
    for group in ('trumans', 'egobody'):
        candidates = ([r for r in official_sources() if r['split'] == split]
                      if group == 'egobody' and split_protocol == SPLIT_PROTOCOL
                      else rows(SOURCE / f'{group}_sequences/{split}.jsonl'))
        for row in candidates:
            if row['sequence_id'] in exclusions:
                continue
            row = dict(row, group=row.get('role', 'trumans'), split_protocol=split_protocol)
            first = row.get('source_first_frame', 0)
            row['raw_first'] = int(first)
            row['raw_last_exclusive'] = int(row['source_last_frame_exclusive'] if 'source_last_frame_exclusive' in row
                else np.floor(first + (row['frames_20fps'] - 1) * 1.5) + 1)
            result.append(row)
    # RICH test was not downloaded; never silently substitute val for test.
    if split == 'test':
        return result
    root = HERE / f"data/scene_visibility_v2_oct05/rich_{'train' if split == 'train' else 'val'}_smpl_native20_faceout_oct07"
    for folder in sorted(root.iterdir()):
        if not (folder / 'metadata.json').exists():
            continue
        meta = json.loads((folder / 'metadata.json').read_text())
        source = PROJECTS / 'RICH/extracted' / ('train_body' if split == 'train' else 'val_body') / folder.name
        ids = sorted(int(p.name) for p in source.iterdir() if p.is_dir())
        result.append(dict(meta, group='rich', dataset='rich', split=split, split_protocol=split_protocol,
                           sequence_id=folder.name, raw_first=ids[0], raw_last_exclusive=ids[-1]+1))
    return result


def global_head(params, model, world_rotation):
    count = len(params['global_orient'])
    local = Rotation.from_rotvec(np.concatenate((params['global_orient'][:, None],
        params['body_pose'].reshape(count, -1, 3)), 1).reshape(-1, 3)).as_matrix().reshape(count, -1, 3, 3)
    result = [world_rotation[None] @ local[:, 0]]
    for j in range(1, 22):
        result.append(result[PARENTS[j]] @ local[:, j])
    return result[15]


def raw_scene(row, q, digest):
    """Return raw-native body parameters, world transform and observed-camera setup."""
    group, name, split = row['group'], row['sequence_id'], row['split']
    objects = []
    observation_valid = np.ones(len(q), bool)
    if group == 'trumans':
        root = PROJECTS / 'TRUMANS'
        clip = next(r for r in rows(root / 'processed/scene_expert_v1/clips.jsonl') if r['clip_name'] == name)
        path = root / clip['smplx_global']; digest.update(path.read_bytes())
        fit = pickle.loads(path.read_bytes())
        model = load_model('smplx', 'male')
        beta = np.asarray(fit.get('betas', []), np.float32).reshape(-1)
        if len(beta) > 10 and np.any(beta[10:] != 0):
            raise ValueError('Unsupported nonzero extra TRUMANS betas; do not truncate shape')
        beta = np.pad(beta[:10], (0, max(0, 10-len(beta[:10]))))
        params = {k: np.asarray(fit[k], np.float32)[q.astype(int)] for k in
                  ('global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'expression', 'transl')}
        params['betas'] = np.broadcast_to(beta, (len(q), 10)).copy()
        params['leye_pose'] = np.zeros((len(q),3),np.float32)
        params['reye_pose'] = np.zeros((len(q),3),np.float32)
        people = [(model, params)]
        transform = (1., np.eye(3), np.zeros(3))
        scan = root / clip['scene']['mesh']
        for name_obj, mesh_path, r, p, source, valid, _ in load_objects(root, clip, q):
            mesh = trimesh.load(mesh_path, force='mesh', process=False)
            digest.update(mesh_path.read_bytes())
            objects.append((static_ray_scene(mesh.vertices, mesh.faces), r, p, len(objects)+1, valid))
            observation_valid &= valid
        for source_file in [root / clip['object_tracks']['path'], root / 'object_flag.npy', root / 'object_mat.npy']:
            digest.update(str(source_file).encode())
            # Track matrices are large; record identity and file stats here.
            digest.update(str((source_file.stat().st_size, source_file.stat().st_mtime_ns)).encode())
        bad = np.load(root / 'bad_frames.npy')
        observation_valid &= ~np.isin(clip['global_start'] + q.astype(int), bad)
        kind, gender, camera, intr = 'smplx', 'male', None, None
    elif group in ('camera_wearer', 'interactee'):
        root = PROJECTS / 'egobody'; record = row['recording']
        sources = {r['sequence_id']: r for r in rows(SOURCE / 'sequences.jsonl')}
        info = next(r for r in csv.DictReader((root / 'data_info_release.csv').open()) if r['recording_name'] == record)
        people = []
        for role in (group, 'interactee' if group == 'camera_wearer' else 'camera_wearer'):
            fit_dir = Path(sources[f'{record}/{role}']['body_fits'])
            idx = int(fit_dir.name.rsplit('_', 1)[1]); gender_role = info[f'body_idx_{idx}'].split()[-1].lower()
            model = load_model('smpl', gender_role)
            fits = read_fits([fit_dir / f'results/frame_{int(f):05d}/000.pkl' for f in q], digest)
            params = resample_fits(fits, np.arange(len(q), dtype=float), 'smpl')
            # Per-person GT shape uses its original profile path; this is not
            # a TRAIN-fitted camera mount or population normalization statistic.
            profile_split=row.get('pose_profile_split',split)
            pose_meta = next(r for r in rows(HERE / f'data/poses/{profile_split}.jsonl')
                             if r['sequence_id'] == name) if role == group and profile_split is not None else None
            beta = np.asarray(pose_meta['betas']) if pose_meta else np.median(params['betas'], 0)
            params['betas'] = np.broadcast_to(beta, params['betas'].shape).copy().astype(np.float32)
            people.append((model, params))
        tf = root / 'calibrations' / record / 'cal_trans/kinect12_to_world' / f"{row['scene_id']}.json"
        matrix = np.asarray(json.loads(tf.read_text())['trans']); digest.update(tf.read_bytes())
        transform = (1., matrix[:3, :3], matrix[:3, 3])
        scan = Path(sources[f'{record}/{group}']['scene_mesh'])
        if row.get('allowed_modes')==['exo']:
            # Do not require a covered PV trajectory for a static master input.
            # Virtual head fields below are unused and never sampled as ego.
            camera,intr=None,None
        else:
            p, r, intr = raw_pv_camera(root, record, row['scene_id'], q, digest)
            camera = (p, r) if group == 'camera_wearer' else None
            if camera is None:
                intr = np.broadcast_to(np.median(intr[:20], 0), intr.shape).copy()
        kind = 'smpl'; gender = info[f"body_idx_{int(Path(sources[f'{record}/{group}']['body_fits']).name.rsplit('_',1)[1])}"].split()[-1].lower()
    elif group == 'rich':
        root = PROJECTS / 'RICH/extracted'; original = 'train' if split == 'train' else 'val'
        source = root / f'{original}_body' / name; subject = str(row['subject_id'])
        subjects = sorted(p.stem for p in (source / f'{int(q[0]):05d}').glob('*.pkl'))
        if not subjects:  # Some releases use unpadded integer directory names.
            folders = {int(p.name): p for p in source.iterdir() if p.is_dir()}
        else:
            folders = {int(p.name): p for p in source.iterdir() if p.is_dir()}
        subjects = sorted(p.stem for p in folders[int(q[0])].glob('*.pkl'))
        subjects.remove(subject); subjects.insert(0, subject); people = []
        for person in subjects:
            fits = read_fits([folders[int(f)] / f'{person}.pkl' for f in q], digest)
            model = _load_smplx_model(RICH_GENDER[int(person)])
            params = resample_fits(fits, np.arange(len(q), dtype=float), 'smplx')
            if not np.allclose(params['betas'], params['betas'][:1], atol=1e-4):
                raise ValueError('RICH shape changes')
            people.append((model, params))
        scene = row['scene']; variant = 'chair' if 'chair' in name else 'yoga'
        scan = root / 'scan_calibration' / scene / (f'scan_{variant}_scene_camcoord.ply' if scene == 'LectureHall' else 'scan_camcoord.ply')
        tf = root / 'multicam2world' / (f'LectureHall_{variant}_multicam2world.json' if scene == 'LectureHall' else f'{scene}_multicam2world.json')
        scale, r, t = _read_transform(tf); digest.update(tf.read_bytes())
        transform = (scale, RICH_TO_SCENEMI @ r.T, t @ RICH_TO_SCENEMI.T)
        camera, intr, kind, gender = None, None, 'smplx', RICH_GENDER[int(subject)]
    else:
        raise ValueError(group)
    mesh = trimesh.load(scan, force='mesh', process=False); digest.update(scan.read_bytes())
    # EgoBody scan is already in world; other scans need the same body transform.
    static_vertices = mesh.vertices if group in ('camera_wearer', 'interactee') else mesh.vertices * transform[0] @ transform[1].T + transform[2]
    static = static_ray_scene(static_vertices, mesh.faces)
    model, params = people[0]; head = global_head(params, model, transform[1])
    beta = params['betas'][0, :10].copy()
    with torch.no_grad():
        rest = vertices2joints(model.J_regressor, model.v_template[None] +
                               blend_shapes(torch.from_numpy(beta[None]), model.shapedirs[:, :, :10]))[0, :22].numpy() * transform[0]
    mount = None
    if camera is None:
        v, j = forward_chunk(model, params, 0, min(20, len(q)), transform)
        if kind == 'smplx':
            eye_ids = [9929, 9448]
        else:
            from smplx.vertex_ids import vertex_ids
            eye_ids = [vertex_ids['smplh']['leye'], vertex_ids['smplh']['reye']]
        offset = np.median(np.einsum('tji,tj->ti', head[:len(j)], v[:, eye_ids].mean(1)-j[:, 15]), 0)
        mount = dict(rotation=np.eye(3).tolist(), translation_m=(offset+[0, 0, .02]).tolist(),
                     position_p90_m=0., rotation_p90_rad=0., source='actual fixed raw30 virtual mount')
    if intr is None:
        intr = np.broadcast_to([64/np.tan(np.deg2rad(66.56/2)),
                                48/np.tan(np.deg2rad(40.49/2)), 64, 48], (len(q), 4)).copy()
    profile = dict(model=kind, gender=gender, betas=beta.tolist(), scale=float(transform[0]))
    return people, transform, static, camera, head, mount, intr, rest, profile, objects, observation_valid


def sampled_environment(points, owner, count=512):
    # Preserve partner surfaces but exclude the target (owner100).
    result = np.zeros((count, 3), np.float32); ids = np.full(count, -1, np.int32)
    eligible = np.flatnonzero((owner >= 0) & (owner != 100))
    if len(eligible) > count:
        # Reserve at most half for dynamics so small partner/object patches survive.
        dynamic = eligible[owner.ravel()[eligible] > 0]
        static = eligible[owner.ravel()[eligible] == 0]
        nd = min(len(dynamic), count//2); ns = min(len(static), count-nd)
        eligible = np.r_[static[np.linspace(0, len(static)-1, ns).astype(int)] if ns else [],
                         dynamic[np.linspace(0, len(dynamic)-1, nd).astype(int)] if nd else []].astype(int)
    result[:len(eligible)] = points.reshape(-1, 3)[eligible]
    ids[:len(eligible)] = owner.ravel()[eligible]
    return result, ids


def export(row, output=DEFAULT_OUTPUT, start=None, frames=None):
    torch.set_num_threads(1); started = time.monotonic()
    first = row['raw_first'] if start is None else int(start)
    n = row['raw_last_exclusive']-first if frames is None else int(frames)
    q = first + np.arange(n, dtype=np.float64)
    if first < row['raw_first'] or q[-1] >= row['raw_last_exclusive']:
        raise ValueError('Raw30 export crosses a split/segment boundary')
    diagnostic = first != row['raw_first'] or n != row['raw_last_exclusive']-row['raw_first']
    name = row['sequence_id'] + (f'__raw30_{first}_{n}' if diagnostic else '')
    folder = Path(output) / row['split'] / name; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'metadata.json').exists():
        meta = json.loads((folder / 'metadata.json').read_text())
        if meta['protocol'] != PROTOCOL or meta.get('export_code_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError('Stale native30 export; use a new directory')
        for key, digest in meta['output_sha256'].items():
            if hashlib.sha256((folder/key).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Corrupt native30 export: {key}')
        return dict(path=str(folder), **meta)
    digest = hashlib.sha256()
    people, transform, static, camera, head, mount, intr, rest, profile, objects, valid = raw_scene(row, q, digest)
    n = len(q); arrays = dict(source_frame_ids=q, observation_valid=valid, rest=rest)
    for key, shape, dtype in [('motion_world', (n, 201), 'float32'),
            ('camera_position_scenemi_yup', (n, 3), 'float32'), ('camera_rotation_scenemi_yup', (n, 3, 3), 'float32'),
            ('visible_frame_points_scenemi_yup', (n, 512, 3), 'float32'), ('visible_frame_owner', (n, 512), 'int32')]:
        arrays[key] = np.lib.format.open_memmap(folder / f'{key}.npy', mode='w+', dtype=dtype, shape=shape)
    arrays['camera_intrinsics_128x96'] = np.asarray(intr, np.float32)
    exo = row['group'] in ('camera_wearer', 'interactee')
    if exo:
        root = PROJECTS / 'egobody'; matrix = np.r_[np.c_[transform[1], transform[2]], [[0,0,0,1]]]
        calibration = json.loads((root / 'kinect_cam_params/kinect_master/Color.json').read_text())
        rays, _ = camera_rays(calibration, matrix, width=192, height=108)
        # static_ray_scene supports cast_rays; apply the same near clipping.
        import open3d as o3d
        clipped = rays.copy(); clipped[:, :3] += .1*clipped[:, 3:]
        static_depth = static.cast_rays(o3d.core.Tensor(clipped))['t_hit'].numpy()+.1
        arrays.update(exo_sensor_to_world=matrix.astype(np.float32),
                      exo_intrinsics=np.array([calibration['camera_mtx'][0][0]/1920,
                          calibration['camera_mtx'][1][1]/1080, calibration['camera_mtx'][0][2]/1920,
                          calibration['camera_mtx'][1][2]/1080], np.float32))
        digest.update(json.dumps(calibration, sort_keys=True).encode())
        for key, shape, dtype in [('exo_trajectory_world', (n, 22, 3), 'float32'),
            ('exo_position_available', (n, 22), 'bool'), ('exo_position_confidence', (n, 22), 'float32'),
            ('exo_frame_points_world', (n, 512, 3), 'float32'), ('exo_frame_owner', (n, 512), 'int32')]:
            arrays[key] = np.lib.format.open_memmap(folder / f'{key}.npy', mode='w+', dtype=dtype, shape=shape)
    selected = {}
    self_faces, _ = wearer_faces(people[0][0].faces, people[0][0].lbs_weights.numpy(), model_type=profile['model'])
    params = people[0][1]
    local = Rotation.from_rotvec(np.concatenate((params['global_orient'][:, None],
        params['body_pose'].reshape(n, -1, 3)[:, :21]), 1).reshape(-1,3)).as_matrix().reshape(n,22,3,3)
    local[:, 0] = transform[1][None] @ local[:, 0]
    rot6 = np.concatenate((local[..., :, 0], local[..., :, 1]), -1) - [1,0,0,0,1,0]
    for lo in range(0, n, 32):
        hi = min(lo+32, n); bodies = [forward_chunk(m, p, lo, hi, transform) for m, p in people]
        v, j = bodies[0]
        if camera is None:
            cam, r = virtual_head_camera(j[:,15], head[lo:hi],
                offset_head=np.asarray(mount['translation_m']) - [0,0,.02])
        else:
            cam, r = camera[0][lo:hi], camera[1][lo:hi]
        arrays['camera_position_scenemi_yup'][lo:hi] = cam; arrays['camera_rotation_scenemi_yup'][lo:hi] = r
        # Native rest-root compensation. TRUMANS body joints are native FK,
        # not a copied SMPL skeleton or 20Hz measured pelvis.
        arrays['motion_world'][lo:hi] = np.concatenate(((j[:,0]-rest[0])/2,
            rot6[lo:hi].reshape(hi-lo,132), j.reshape(hi-lo,66)/2), -1)
        for k, t in enumerate(range(lo,hi)):
            meshes = [(bodies[0][0][k], self_faces, 100)] + [(bodies[a][0][k], people[a][0].faces, 100+a) for a in range(1,len(people))]
            if valid[t]:
                hit = render_scene(meshes, cam[k], r[k], static_scene=static, intrinsics=intr[t],
                    rigid_objects=[(sc, rr[t], pp[t], owner) for sc,rr,pp,owner,ov in objects if ov[t]])
                p, ids = sampled_environment(hit['points'], hit['owner'])
            else:
                p, ids = np.zeros((512,3),np.float32), np.full(512,-1,np.int32)
            arrays['visible_frame_points_scenemi_yup'][t] = p; arrays['visible_frame_owner'][t] = ids
            if t in (0, n//2, n-1) and valid[t]:
                selected[f'ego_owner_{t}'] = hit['owner']; selected[f'ego_depth_{t}'] = hit['depth']
            if exo:
                depth = static_depth.copy(); owner = np.where(np.isfinite(depth)&(depth>=.1)&(depth<=12),0,-1)
                depth[owner<0] = np.inf; target_parts = None
                for a, (vertices, joints) in enumerate(bodies):
                    bd, primitive = cast_body(vertices[k], people[a][0].faces, rays)
                    bh = np.isfinite(bd)&(bd>=.1)&(bd<=12)
                    if a == 0:
                        target_parts = np.full(len(rays),-1,np.int32)
                        target_parts[bh] = regional_faces(people[0][0])[primitive[bh]]
                    take = bh & (bd<depth); depth[take]=bd[take]; owner[take]=100+a
                visible = np.bincount(target_parts[owner==100], minlength=22)
                reference = np.bincount(target_parts[target_parts>=0], minlength=22)
                fraction = visible/np.maximum(reference,1); available = (visible>=2)&(fraction>=.15)
                arrays['exo_position_available'][t] = available
                arrays['exo_position_confidence'][t] = np.where(available, fraction*np.minimum(visible/12,1),0)
                arrays['exo_trajectory_world'][t] = j[k]
                hp = np.zeros((len(rays),3),np.float32); mask = owner>=0
                hp[mask] = rays[mask,:3]+depth[mask,None]*rays[mask,3:]
                p, ids = sampled_environment(hp, owner)
                arrays['exo_frame_points_world'][t] = p; arrays['exo_frame_owner'][t] = ids
                if t in (0,n//2,n-1):
                    selected[f'exo_owner_{t}'] = owner.reshape(108,192)
                    selected[f'exo_parts_{t}'] = np.where(owner==100,target_parts,-1).reshape(108,192)
        print(f"raw30 {row['group']} {name}: {hi}/{n}", flush=True)
    # Rigid kinematics are source-known simulator conditions, not estimated
    # by matching unrelated points. Missing partner skin velocity stays invalid.
    if objects:
        arrays['object_rotations'] = np.stack([o[1] for o in objects], 1).astype(np.float32)
        arrays['object_positions'] = np.stack([o[2] for o in objects], 1).astype(np.float32)
        arrays['object_valid'] = np.stack([o[4] for o in objects], 1)
    if row['group'] == 'rich':
        from experiments.offline_camera_retrain_v1.export_rich_contact_native20 import read_smplx
        from experiments.offline_camera_retrain_v1.contact_supervision import vertex_regions
        regions = vertex_regions(people[0][0]); target = np.zeros((n,22),np.float32); mask = np.zeros((n,22),bool)
        for t,frame in enumerate(q.astype(int)):
            path = PROJECTS / 'RICH/extracted' / ('train_hsc' if row['split']=='train' else 'val_hsc') / row['sequence_id'] / f'{frame:05d}' / f"{row['subject_id']}.obj"
            if path.exists():
                contact = read_smplx(path); digest.update(path.read_bytes())
                if contact is not None:
                    for region,ids in enumerate(regions):
                        if len(ids):
                            target[t,region] = contact[ids].any(); mask[t,region] = True
        arrays['contact_target'], arrays['contact_valid'] = target, mask
    for key, value in arrays.items():
        if isinstance(value, np.memmap):
            value.flush()
        else:
            np.save(folder / f'{key}.npy', value)
    np.savez_compressed(folder / 'selected_raycast_audit.npz', **selected)
    meta = dict(protocol=PROTOCOL, export_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sequence_id=row['sequence_id'], bundle_id=name, split=row['split'], group=row['group'],
        dataset=row['dataset'], fps=30, frames=n, source_first=int(q[0]), source_last=int(q[-1]),
        native_body=profile, actual_virtual_mount=mount, diagnostic_only=diagnostic,
        observation_budget='consumer selects recording observed-static union or prefix; current-frame dynamics',
        scene_unknown_policy='observed surface occupancy only; absence of points means unknown',
        dynamic_velocity_protocol='source-known rigid object poses; partner velocities unavailable',
        input_sha256=digest.hexdigest(), elapsed_s=time.monotonic()-started)
    meta.update(split_protocol=row.get('split_protocol', LEGACY_SPLIT_PROTOCOL),
        recording=row.get('recording'), scene_id=row.get('scene_id',row.get('scene')),
        pose_profile_split=row.get('pose_profile_split', row['split']),
        allowed_modes=row.get('allowed_modes',['ego','exo'] if row['group'] in ('camera_wearer','interactee') else ['ego']))
    if row.get('official_split_csv_sha256'):
        meta['official_split_csv_sha256'] = row['official_split_csv_sha256']
    meta['world_frame_id'] = hashlib.sha256(np.r_[transform[0],
        np.asarray(transform[1]).ravel(), np.asarray(transform[2]).ravel()].tobytes()).hexdigest()
    meta['output_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('*.npy'))}
    temp = folder / 'metadata.tmp'; temp.write_text(json.dumps(meta,indent=2)); temp.replace(folder/'metadata.json')
    return dict(path=str(folder), **meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--split', choices=['train','val','validation','test'], default='train')
    parser.add_argument('--sequence'); parser.add_argument('--start', type=int)
    parser.add_argument('--frames', type=int); args = parser.parse_args()
    args.split = normalize_split(args.split)
    candidates = inventory(args.split)
    if args.sequence:
        candidates = [r for r in candidates if r['sequence_id']==args.sequence]
        if not candidates:
            raise ValueError('Unknown source sequence')
    args.output.mkdir(parents=True,exist_ok=True); manifest = args.output/'manifest.jsonl'
    existing = {(r['split'],r['bundle_id']):r for r in rows(manifest)} if manifest.exists() else {}
    for row in candidates:
        value = export(row,args.output,args.start,args.frames); existing[(value['split'],value['bundle_id'])] = value
        temp = manifest.with_suffix('.tmp'); temp.write_text(''.join(json.dumps(r)+'\n' for r in existing.values())); temp.replace(manifest)
    complete = all((args.split,r['sequence_id']) in existing and not existing[(args.split,r['sequence_id'])]['diagnostic_only'] for r in inventory(args.split))
    (args.output/f'{args.split}_coverage.json').write_text(json.dumps(dict(split=args.split,
        complete=complete, expected=len(inventory(args.split)), exported=sum(r['split']==args.split for r in existing.values())),indent=2))


if __name__ == '__main__':
    main()
