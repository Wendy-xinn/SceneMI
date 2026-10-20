"""CPU GRAIL first-hit exports, with an explicit unresolved-template gate.

Default ``auto`` refuses unresolved body provenance. Explicit ``standard`` or
``g1`` produces review-only complete recordings in review_manifest.jsonl;
these are never published as a training manifest. The alternative wide/down
camera is saved only on audit frames and does not alter primary inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from smplx.lbs import blend_shapes, vertices2joints

from . import grail_data, grail_source
from experiments.offline_camera_retrain_v1.native30_cache_io import atomic_text, verified_sha256
from experiments.offline_camera_retrain_v1.export_native20_training_scenes import forward_chunk
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from experiments.offline_camera_retrain_v1.export_egobody_exo import regional_faces
from experiments.offline_camera_retrain_v1.native30_export import global_head, sampled_environment
from experiments.offline_camera_retrain_v1.native30_exo import (
    camera_rays, first_hit, look_at_cv, serial_camera, EXO_PROTOCOL,
)
from experiments.offline_camera_retrain_v1.scene_visibility_v2 import (
    render_scene, static_ray_scene, wearer_faces,
)

EXPORT_PROTOCOL = 'grail-complete-recording-cpu-export-v1'
GROUND_HALF_EXTENT_M = 20.
PRIMARY_CAMERA = dict(hfov_deg=66.56, vfov_deg=40.49, pitch_down_deg=0., far_m=4.,
                      width=128, height=96, near_m=.05)
AUDIT_CAMERA = dict(hfov_deg=90., vfov_deg=60., pitch_down_deg=15., far_m=6.,
                    width=128, height=96, near_m=.05)


class UnverifiedBodyTemplate(ValueError):
    pass


def source_fingerprint():
    paths = [Path(__file__), Path(grail_source.__file__), Path(grail_data.__file__)]
    base = Path(__file__).parents[1]
    paths += [base / name for name in ('scene_visibility_v2.py', 'native30_export.py',
              'native30_exo.py', 'export_egobody_exo.py', 'export_native20_training_scenes.py',
              'export_rest_joints.py')]
    assets = Path(__file__).with_name('grail_assets.py')
    if assets.exists():
        paths.append(assets)
    return {str(path.resolve()): grail_source.file_sha256(path) for path in paths}


def normalized_family(row):
    """Unify both stair phases without merging unrelated category families."""
    family = str(row['object_family_id'])
    for prefix in ('stair_p1__', 'stair_p2__'):
        if family.startswith(prefix):
            return 'stair__' + family[len(prefix):]
    return family


def body_policy(record, body_template='auto'):
    if body_template not in ('auto', 'standard', 'g1'):
        raise ValueError('Unknown GRAIL body template mode')
    quality = record['quality']
    if body_template != 'auto':
        return dict(template=body_template, trainable=False,
                    reason='Explicit template hypothesis for review; no trainable claim')
    if not quality.get('body_template_verified', False):
        raise UnverifiedBodyTemplate('GRAIL body template is unresolved. Use an explicit --body-template standard/g1 only for review; no training export was created.')
    template = quality.get('body_template', record.get('body_template'))
    template = {'grail_g1_baked_v1': 'g1', 'standard_smplx_scaled': 'standard'}.get(template, template)
    if template not in ('standard', 'g1'):
        raise UnverifiedBodyTemplate('Verified source must declare quality.body_template=standard or g1')
    if template == 'g1' and not quality.get('native_body_assets_verified', False):
        raise UnverifiedBodyTemplate('G1 source is identified but native body skin/FK assets are not yet verified for training')
    return dict(template=template, trainable=True, reason='Source template and downstream native-body assets verified')


def resolve_body_model(record, policy):
    factory = getattr(grail_source, 'body_model_factory', None)
    if factory is not None:
        model = factory(policy['template'])
    elif policy['template'] == 'standard':
        model = load_model('smplx', 'neutral')
    else:
        raise UnverifiedBodyTemplate('G1 body-model factory is unavailable; refusing to substitute standard SMPL-X')
    model = model.cpu().eval()
    if next(model.parameters()).device.type != 'cpu':
        raise ValueError('GRAIL export is CPU-only')
    return model


def ground_mesh(half_extent=GROUND_HALF_EXTENT_M):
    if not np.isfinite(half_extent) or half_extent <= 0:
        raise ValueError('Ground extent must be positive')
    h = float(half_extent)
    return (np.array([[-h, 0, -h], [-h, 0, h], [h, 0, h], [h, 0, -h]], np.float32),
            np.array([[0, 1, 2], [0, 2, 3]], np.int64))


def camera_from_head(head_positions, head_rotations, eye_offset, options):
    """A fixed head-local mount; return metadata that exactly inverts it."""
    c = Rotation.from_euler('x', options['pitch_down_deg'], degrees=True).as_matrix()
    rotation = np.asarray(head_rotations) @ c
    translation = np.asarray(eye_offset, float) + .02 * c[:, 2]
    position = np.asarray(head_positions) + np.einsum('tij,j->ti', head_rotations, translation)
    mount = dict(rotation=c.tolist(), translation_m=translation.tolist(),
                 position_p90_m=0., rotation_p90_rad=0.,
                 source='Fixed shaped-rest eye midpoint with 2cm optical clearance; synthetic camera')
    w, h = options['width'], options['height']
    intrinsics = np.array([w / (2 * np.tan(np.deg2rad(options['hfov_deg'] / 2))),
                          h / (2 * np.tan(np.deg2rad(options['vfov_deg'] / 2))),
                          w / 2, h / 2], np.float32)
    return position.astype(np.float32), rotation.astype(np.float32), mount, intrinsics


def external_observer(initial_root):
    """Fixed world direction and initial-root-only target; no future fitting."""
    initial = np.asarray(initial_root, float)
    direction = np.array([1., 0., -1.]) / np.sqrt(2.)
    position = initial + 3. * direction
    position[1] = 1.6
    target = initial.copy()
    target[1] = 1.
    w, h = 192, 108
    f = w / (2 * np.tan(np.deg2rad(PRIMARY_CAMERA['hfov_deg'] / 2)))
    return dict(kind='synthetic_training_observer', view_id='initial_root_fixed_world_diagonal',
                position=position, rotation=look_at_cv(position, target),
                K=np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.]]),
                distortion=np.zeros(8), source_size=[w, h], size=[w, h], near=.1, far=12.,
                benchmark_camera_verified=False, calibration=None,
                selection='initial root plus 3m fixed world [1,0,-1] diagonal, camera y=1.6m; looks at initial root x/z and y=1m; no future trajectory or visibility selection')


def audit_indices(frames):
    return np.unique(np.linspace(0, frames - 1, min(5, frames)).round().astype(int))


def visibility_counts(hit):
    owner, points = hit['owner'], hit['points']
    static = owner == 0
    ground = static & (np.abs(points[..., 1]) <= .003)
    return dict(rays=int(owner.size), static=int(static.sum()), ground_y0=int(ground.sum()),
                object_above_ground=int((static & ~ground).sum()), target_body=int((owner == 100).sum()))


def _rest_geometry(model, params, scale):
    beta = np.asarray(params['betas'][0], np.float32)
    with torch.inference_mode():
        shaped = model.v_template[None] + blend_shapes(torch.from_numpy(beta[None]), model.shapedirs[:, :, :10])
        rest = vertices2joints(model.J_regressor, shaped)[0, :22].numpy() * scale
    zero = {key: np.zeros_like(value[:1]) for key, value in params.items()}
    zero['betas'][:] = beta
    vertices, joints = forward_chunk(model, zero, 0, 1, (scale, np.eye(3), np.zeros(3)))
    # SMPL-X stable eye vertex identities are used only for the fixed mount.
    eye_offset = vertices[0, [9929, 9448]].mean(0) - joints[0, 15]
    return rest.astype(np.float32), eye_offset.astype(np.float32)


def _verify_mount(cam, rotations, head_positions, head_rotations, mount):
    from experiments.offline_camera_retrain_v1.native30_observations import inverse_head
    from experiments.offline_camera_retrain_v1.data import rotation6d
    payload = np.c_[cam / 2, rotation6d(rotations)].astype(np.float32)
    observed, available, _ = inverse_head(payload, np.ones((len(cam), 2)), np.ones((len(cam), 2)), mount)
    error = float(np.max(np.abs(observed[:, :3] * 2 - head_positions)))
    rotation_error = float(np.max(np.abs(observed[:, 3:] - rotation6d(head_rotations))))
    if not available.all() or error > 2e-5 or rotation_error > 2e-5:
        raise ValueError('Virtual camera head mount failed forward/inverse roundtrip')
    return dict(max_position_error_m=error, max_rotation6d_error=rotation_error)


def _existing(folder, options, row, fingerprint):
    meta = json.loads((folder / 'metadata.json').read_text())
    if meta.get('export_options') != options or meta.get('export_fingerprint') != fingerprint:
        raise ValueError('Existing GRAIL export uses different options/code; use a fresh output directory')
    if meta['sequence_id'] != row['sequence_id']:
        raise ValueError('Existing GRAIL source identity changed')
    family = normalized_family(row)
    if meta['object_family_id'] != family or meta['split'] != grail_data.family_split(family):
        raise ValueError('Existing GRAIL object-family assignment changed')
    for key, source_field in (('recon', 'recon_path'), ('object_usd', 'usd_path'), ('meta', 'meta_path')):
        if row.get(source_field) and Path(meta['source_files'][key]).resolve() != Path(row[source_field]).resolve():
            raise ValueError('Existing GRAIL source path changed')
    for key, path in meta['source_files'].items():
        if grail_source.file_sha256(path) != meta['source_hashes'][key]:
            raise ValueError('Existing GRAIL source fingerprint changed')
    for name, expected in meta['output_sha256'].items():
        verified_sha256(folder / name, expected)
    verified_sha256(folder / 'selected_raycast_audit.npz', meta['audit_sha256'])
    return dict(meta, path=str(folder.resolve()))


def export_record(row, output, *, body_template='auto', source_fps=24., scope='pilot',
                  chunk_size=16, model=None, source_template=None):
    """Publish one complete recording atomically; model injection is for CPU tests."""
    if scope not in ('pilot', 'full') or chunk_size < 1:
        raise ValueError('Invalid GRAIL export scope or chunk size')
    torch.set_num_threads(2)
    started = time.monotonic()
    family = normalized_family(row)
    split = grail_data.family_split(family)
    root = Path(output).resolve()
    folder = root / split / row['sequence_id']
    effective_template = source_template or row.get('body_template')
    if body_template == 'g1':
        effective_template = 'grail_g1_baked_v1'
    elif body_template == 'standard':
        effective_template = 'standard_smplx_scaled'
    options = dict(protocol=EXPORT_PROTOCOL, body_template=body_template,
                   source_template=effective_template,
                   source_fps=float(source_fps), scope=scope, primary_camera=PRIMARY_CAMERA,
                   audit_camera=AUDIT_CAMERA, synthetic_ground_half_extent_m=GROUND_HALF_EXTENT_M)
    fingerprint = source_fingerprint()
    if folder.exists():
        return _existing(folder, options, row, fingerprint)
    record = grail_source.load_record(row, source_fps=source_fps, output_fps=30.,
                                      body_template=effective_template)
    profile = dict(model='smplx', gender='neutral', betas=record['params']['betas'][0].tolist(),
                   scale=record['human_scale'], template=(
                       'grail_g1_baked_v1' if record.get('body_template') == 'grail_g1_baked_v1'
                       else 'smplx_neutral_flat_v1'))
    template_info = record.get('source_provenance', {}).get('body_template', {})
    if template_info.get('sha256'):
        profile['template_sha256'] = template_info['sha256']
    assets_check = None
    if body_template == 'auto' and record['quality'].get('body_template_verified', False):
        from .grail_assets import validate_body_assets
        assets_check = validate_body_assets(profile)
        record['quality'] = dict(record['quality'], **assets_check)
    policy = body_policy(record, body_template)
    if not record['quality']['object_static']:
        raise ValueError('This exporter supports only genuinely static GRAIL objects')
    model = resolve_body_model(record, policy) if model is None else model.cpu().eval()
    params = record['params']
    scale, wr, wt = record['human_scale'], record['world_rotation'], record['world_offset']
    transform = (scale, wr, wt)
    n = len(record['output_timestamps_s'])
    rest, eye_offset = _rest_geometry(model, params, scale)
    head = global_head(params, model, wr)
    _, initial_joints = forward_chunk(model, params, 0, 1, transform)
    observer = external_observer(initial_joints[0, 0])
    object_vertices = (record['object_vertices'] @ record['object_rotations'][0].T + record['object_positions'][0]).astype(np.float32)
    object_faces = np.asarray(record['object_faces'], np.int64)
    object_min_y = float(object_vertices[:, 1].min())
    ground_quarantine = object_min_y < -.02
    if ground_quarantine:
        policy = dict(policy, trainable=False,
                      reason='Quarantined: object terrain extends below synthetic plane y=0; resolve ground support before training')
    ground_vertices, ground_faces = ground_mesh()
    vertices = np.concatenate((object_vertices, ground_vertices))
    faces = np.concatenate((object_faces, ground_faces + len(object_vertices)))
    static = static_ray_scene(vertices, faces)
    rays = camera_rays(observer)
    import open3d as o3d
    clipped = rays.copy()
    clipped[:, :3] += observer['near'] * clipped[:, 3:]
    static_depth = static.cast_rays(o3d.core.Tensor(clipped))['t_hit'].numpy() + observer['near']
    region_labels = regional_faces(model)
    self_faces, head_exclusion = wearer_faces(model.faces, model.lbs_weights.detach().cpu().numpy(), model_type='smplx')
    local = Rotation.from_rotvec(np.concatenate((params['global_orient'][:, None],
        params['body_pose'].reshape(n, 21, 3)), axis=1).reshape(-1, 3)).as_matrix().reshape(n, 22, 3, 3)
    local[:, 0] = wr[None] @ local[:, 0]
    six = np.concatenate((local[..., :, 0], local[..., :, 1]), axis=-1) - [1, 0, 0, 0, 1, 0]
    folder.parent.mkdir(parents=True, exist_ok=True)
    partial = folder.parent / ('.' + folder.name + '.partial.' + uuid.uuid4().hex)
    partial.mkdir()
    arrays, audit, stats = {}, {}, []
    selected = set(audit_indices(n).tolist())
    roundtrip = []
    try:
        for key, spec in grail_data.array_specs(n).items():
            arrays[key] = np.lib.format.open_memmap(partial / (key + '.npy'), mode='w+',
                dtype=spec['dtype'], shape=tuple(spec['shapes'][0]))
        arrays['source_frame_ids'][:] = np.arange(n)
        arrays['source_timestamps_s'][:] = record['output_timestamps_s']
        arrays['observation_valid'][:] = True
        arrays['rest'][:] = rest
        arrays['exo_sensor_to_world'][:] = np.eye(4)
        arrays['exo_sensor_to_world'][:3, :3] = observer['rotation']
        arrays['exo_sensor_to_world'][:3, 3] = observer['position']
        w, h = observer['source_size']
        k = observer['K']
        arrays['exo_intrinsics'][:] = [k[0, 0] / w, k[1, 1] / h, k[0, 2] / w, k[1, 2] / h]
        for key in ('reference', 'visible', 'person', 'static', 'rigid'):
            arrays['exo_' + key + '_counts'] = np.lib.format.open_memmap(
                partial / ('exo_' + key + '_counts.npy'), mode='w+', dtype='int32', shape=(n, 22))
        for lo in range(0, n, chunk_size):
            hi = min(lo + chunk_size, n)
            body_vertices, joints = forward_chunk(model, params, lo, hi, transform)
            cam, cr, mount, intrinsics = camera_from_head(joints[:, 15], head[lo:hi], eye_offset, PRIMARY_CAMERA)
            roundtrip.append(_verify_mount(cam, cr, joints[:, 15], head[lo:hi], mount))
            arrays['camera_position_scenemi_yup'][lo:hi] = cam
            arrays['camera_rotation_scenemi_yup'][lo:hi] = cr
            arrays['camera_intrinsics_128x96'][lo:hi] = intrinsics
            arrays['motion_world'][lo:hi] = np.concatenate(((joints[:, 0] - rest[0]) / 2,
                six[lo:hi].reshape(hi-lo, 132), joints.reshape(hi-lo, 66) / 2), axis=-1)
            for offset, t in enumerate(range(lo, hi)):
                meshes = [(body_vertices[offset], self_faces, 100)]
                hit = render_scene(meshes, cam[offset], cr[offset], static_scene=static,
                    intrinsics=intrinsics, near=PRIMARY_CAMERA['near_m'], far=PRIMARY_CAMERA['far_m'])
                points, owners = sampled_environment(hit['points'], hit['owner'])
                arrays['visible_frame_points_scenemi_yup'][t] = points
                arrays['visible_frame_owner'][t] = owners
                counts = visibility_counts(hit)
                counts['frame'] = t
                stats.append(counts)
                hp, owner, parts, depth, available, confidence, reference, visible, blocked = first_hit(
                    observer, rays, static_depth, [(body_vertices[offset], model.faces)], region_labels, [], t, True)
                arrays['exo_trajectory_world'][t] = np.where(available[:, None], joints[offset], 0)
                arrays['exo_position_available'][t] = available
                arrays['exo_position_confidence'][t] = confidence
                arrays['exo_frame_points_world'][t], arrays['exo_frame_owner'][t] = sampled_environment(hp, owner)
                arrays['exo_reference_counts'][t] = reference
                arrays['exo_visible_counts'][t] = visible
                for key, values in blocked.items():
                    arrays['exo_' + key + '_counts'][t] = values
                if t in selected:
                    audit['ego_owner_' + str(t)] = hit['owner']
                    audit['ego_depth_' + str(t)] = hit['depth']
                    audit['ego_camera_position_' + str(t)] = cam[offset]
                    audit['ego_camera_rotation_' + str(t)] = cr[offset]
                    audit['body_vertices_' + str(t)] = body_vertices[offset]
                    wide_p, wide_r, wide_mount, wide_k = camera_from_head(
                        joints[offset:offset+1, 15], head[t:t+1], eye_offset, AUDIT_CAMERA)
                    _verify_mount(wide_p, wide_r, joints[offset:offset+1, 15], head[t:t+1], wide_mount)
                    wide = render_scene(meshes, wide_p[0], wide_r[0], static_scene=static,
                        intrinsics=wide_k, near=AUDIT_CAMERA['near_m'], far=AUDIT_CAMERA['far_m'])
                    for key in ('owner', 'depth'):
                        audit['wide_' + key + '_' + str(t)] = wide[key]
                    audit['wide_camera_position_' + str(t)] = wide_p[0]
                    audit['wide_camera_rotation_' + str(t)] = wide_r[0]
                    audit['wide_visibility_' + str(t)] = np.array(list(visibility_counts(wide).values()), np.int64)
                    audit['exo_owner_' + str(t)] = owner.reshape(observer['size'][1], observer['size'][0])
                    audit['exo_depth_' + str(t)] = depth.reshape(observer['size'][1], observer['size'][0])
                    audit['exo_parts_' + str(t)] = parts.reshape(observer['size'][1], observer['size'][0])
            print(json.dumps(dict(sequence=row['sequence_id'], frames=hi, total=n, cpu_only=True)), flush=True)
        for value in arrays.values():
            value.flush()
        for key, value in params.items():
            np.save(partial / ('source_pose_' + key + '.npy'), np.asarray(value, np.float32))
        for key, value in dict(scene_object_vertices=object_vertices, scene_object_faces=object_faces,
                scene_ground_vertices=ground_vertices, scene_ground_faces=ground_faces,
                source_world_rotation=wr, source_world_offset=wt,
                source_original_timestamps_s=record['source_timestamps_s']).items():
            np.save(partial / (key + '.npy'), value)
        audit['body_faces'] = np.asarray(model.faces, np.int64)
        audit['selected_frames'] = np.array(sorted(selected), np.int64)
        np.savez_compressed(partial / 'selected_raycast_audit.npz', **audit)
        quality = dict(record['quality'], selected_template=policy['template'],
                       trainable=policy['trainable'], template_policy_reason=policy['reason'],
                       object_world_min_y_m=object_min_y, ground_plane_quarantined=ground_quarantine)
        meta = dict(protocol=grail_data.PROTOCOL, split_protocol=grail_data.SPLIT_PROTOCOL,
            resampling=grail_data.RESAMPLING, sequence_id=row['sequence_id'], bundle_id=row['sequence_id'],
            group='grail', dataset='grail', fps=30, frames=n, source_first=0, source_last=n-1,
            source_fps=record['source_fps'], source_frames=record['source_frames'],
            source_duration_s=float(record['source_timestamps_s'][-1]), category=row['category'],
            object_family_id=family, split=split, scope=scope, allowed_modes=['ego', 'exo'],
            diagnostic_only=False, trainable=policy['trainable'], quality=quality,
            native_body=profile, native_body_asset_audit=assets_check,
            actual_virtual_mount=mount, virtual_camera=PRIMARY_CAMERA,
            audit_virtual_camera=AUDIT_CAMERA, audit_frames=sorted(selected),
            mount_roundtrip=dict(max_position_error_m=max(v['max_position_error_m'] for v in roundtrip),
                                 max_rotation6d_error=max(v['max_rotation6d_error'] for v in roundtrip)),
            exo_protocol=EXO_PROTOCOL, exo_camera=serial_camera(observer),
            exo_input_kind='ideal_GT_position_tracker_simulation_with_firsthit_confidence',
            input_sha256=record['input_sha256'], source_files=record['source_files'],
            source_hashes=record['source_hashes'], source_provenance=record['source_provenance'],
            world_frame_id=hashlib.sha256(np.r_[scale, wr.ravel(), wt].tobytes()).hexdigest(),
            scene_unknown_policy='Only observed first-hit surfaces are inputs; full scene is simulator/audit geometry',
            scene_memory='Separate EGO/EXO owner=0 observed-static union; current body is excluded',
            synthetic_ground=dict(plane_world_y_m=0., half_extent_m=GROUND_HALF_EXTENT_M,
                source='Explicit GRAIL simulator support plane, following official replay renderer; not a released scene scan'),
            primary_visibility_by_frame=stats,
            ground_visibility_count_policy='owner0 and abs(worldY)<=3mm; object surfaces coplanar with ground may be counted as ground',
            head_exclusion=head_exclusion, contact_labels='Absent; not invented',
            full_pose_storage='source_pose_* remain pre-world-conversion model parameters; apply human scale, source_world_rotation, source_world_offset; no padded/repeated frames',
            export_options=options, export_fingerprint=fingerprint, elapsed_s=time.monotonic()-started,
            audit_sha256=grail_source.file_sha256(partial / 'selected_raycast_audit.npz'))
        meta['output_sha256'] = {p.name: grail_source.file_sha256(p) for p in sorted(partial.glob('*.npy'))}
        atomic_text(partial / 'metadata.json', json.dumps(meta, indent=2, allow_nan=False) + '\n')
        for mapped in arrays.values():
            mapped.flush()
            mapped._mmap.close()
        del mapped
        arrays.clear()
        os.replace(partial, folder)
        return dict(meta, path=str(folder))
    except BaseException:
        for mapped in arrays.values():
            if not mapped._mmap.closed:
                mapped._mmap.close()
        arrays.clear()
        shutil.rmtree(partial, ignore_errors=True)
        raise


def publish_manifest(root, row):
    root = Path(root)
    path = root / ('manifest.jsonl' if row['trainable'] else 'review_manifest.jsonl')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    by_identity = {value['sequence_id']: value for value in rows}
    by_identity[row['sequence_id']] = row
    atomic_text(path, ''.join(json.dumps(value, allow_nan=False) + '\n'
                            for _, value in sorted(by_identity.items())))
    return path


def export_inventory(inventory_path, output, *, limit=None, sequence=None, category=None,
                     body_template='auto', source_fps=24., scope='pilot', source_template=None):
    text = Path(inventory_path).read_text()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = payload['rows'] if isinstance(payload, dict) and 'rows' in payload else (
        [payload] if isinstance(payload, dict) and 'sequence_id' in payload else payload)
    if source_template is None and isinstance(payload, dict):
        source_template = payload.get('source_template')
    selected = [row for row in rows if (sequence is None or row['sequence_id'] == sequence)
                and (category is None or row['category'] == category)]
    if not selected:
        raise ValueError('No GRAIL inventory rows match this selection')
    if limit is not None:
        if limit < 1:
            raise ValueError('Export limit must be positive')
        selected = selected[:limit]
    Path(output).mkdir(parents=True, exist_ok=True)
    completed = []
    for row in selected:
        value = export_record(row, output, body_template=body_template, source_fps=source_fps,
                              scope=scope, source_template=source_template)
        publish_manifest(output, value)
        completed.append(value)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('export',))
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--sequence')
    parser.add_argument('--category', choices=grail_source.CATEGORIES)
    parser.add_argument('--body-template', choices=('auto', 'standard', 'g1'), default='auto')
    parser.add_argument('--source-template', choices=('grail_g1_baked_v1', 'standard_smplx_scaled'),
                        help='Explicit source-template contract; auto still verifies downstream body assets')
    parser.add_argument('--source-fps', type=float, default=24.)
    parser.add_argument('--scope', choices=('pilot', 'full'), default='pilot')
    args = parser.parse_args()
    rows = export_inventory(args.inventory, args.output, limit=args.limit, sequence=args.sequence,
                            category=args.category, body_template=args.body_template,
                            source_fps=args.source_fps, scope=args.scope, source_template=args.source_template)
    print(json.dumps(dict(status='complete', sources=len(rows),
                          trainable_sources=sum(row['trainable'] for row in rows), optimizer_steps=0)))


if __name__ == '__main__':
    main()
