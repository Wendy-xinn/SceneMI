"""Split-isolated recording maps and observed-only support for GIMO review.

The complete voxel map is a spatial index, not a learned global feature.
Support patches are shared with the scene refiner. Preview extrapolation is
explicitly uncertain and never replaces measured points or reliable support.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import yaml

from scenemi.data.gimo import ROOT, atomic_json, contiguous_ranges, digest
from scenemi.geometry.spatial_scene import ObservedSceneIndex
from scenemi.geometry.support_surfaces import (
    SupportSurfaceAtlas, extract_support_surfaces, validate_policy,
)

VERSION = 'gimo-recording-observed-support-v1'


def map_identity(row):
    """A recording may cross splits or metric world scales; neither is pooled."""
    return (row['split'], row['scene'], row['recording'],
            row['output_sha256']['scanner_to_world.npy'])


def read_bundle(folder):
    folder = Path(folder)
    meta = json.loads((folder/'CACHE.json').read_text())
    if not meta.get('complete'):
        raise ValueError('Incomplete GIMO scene bundle')
    arrays = {}
    for name, sha in meta['arrays'].items():
        if Path(name).name != name or not name.endswith('.npy') or digest(folder/name) != sha:
            raise ValueError('GIMO scene checksum differs: '+str(folder/name))
        arrays[Path(name).stem] = np.load(folder/name, mmap_mode='r', allow_pickle=False)
    return arrays, meta


def surface_preview(index, arrays, policy, spacing=.05):
    """Display bounded plane interpolation and a distinct uncertain extension.

    A cell is displayed only when ALL four corners are close to measured
    inliers. Each fitted height layer stays independent. The orange layer is
    a diagnostic illustration, never evidence for free space or contact.
    """
    meshes = {key: [[], []] for key in ('fitted', 'extended')}
    q = policy['query']
    for patch_id, coefficients in enumerate(arrays['coefficients']):
        lo, hi = arrays['sample_offsets'][patch_id:patch_id+2]
        points = index.points[arrays['sample_ids'][lo:hi]][:, [0, 2]]
        tree = cKDTree(points)
        radius = q['uncertain_distance_m']
        low = np.floor((points.min(0)-radius)/spacing).astype(int)
        high = np.ceil((points.max(0)+radius)/spacing).astype(int)
        xx, zz = np.meshgrid(np.arange(low[0], high[0]+1)*spacing,
                              np.arange(low[1], high[1]+1)*spacing, indexing='ij')
        xz = np.column_stack((xx.ravel(), zz.ravel()))
        distance = tree.query(xz, workers=1)[0]
        lo, hi = arrays['hull_offsets'][patch_id:patch_id+2]
        equations = arrays['hull_equations'][lo:hi]
        inside = (xz @ equations[:, :2].T+equations[:, 2]).max(-1) <= 1e-6
        vertices = np.column_stack((xz[:, 0], np.c_[xz, np.ones(len(xz))] @ coefficients, xz[:, 1]))
        ids = np.arange(len(xz)).reshape(xx.shape)
        corners = np.stack((ids[:-1, :-1], ids[1:, :-1], ids[1:, 1:], ids[:-1, 1:]), -1).reshape(-1, 4)
        nearby = (distance[corners] <= radius).all(-1)
        interior = inside[corners].all(-1)
        # Internal gaps are also uncertain: a convex hull alone cannot
        # certify unobserved space between distant samples.
        measured = interior & (distance[corners] <= q['reliable_distance_m']).all(-1)
        for name, keep in [('fitted', nearby & measured), ('extended', nearby & ~measured)]:
            faces = np.concatenate((corners[keep][:, [0, 1, 2]], corners[keep][:, [0, 2, 3]]))
            if not len(faces):
                continue
            used, inverse = np.unique(faces, return_inverse=True)
            offset = sum(len(v) for v in meshes[name][0])
            meshes[name][0].append(vertices[used].astype(np.float32))
            meshes[name][1].append(inverse.reshape(-1, 3).astype(np.int32)+offset)
    result = {}
    for name, (vertices, faces) in meshes.items():
        result[name+'_vertices'] = np.concatenate(vertices) if vertices else np.empty((0, 3), np.float32)
        result[name+'_faces'] = np.concatenate(faces) if faces else np.empty((0, 3), np.int32)
    return result


def prepare(config, output):
    """Build each observed recording once. No motion labels enter fitting."""
    from scenemi.data.scene_geometry_cache import fit_normals, _publish
    from scenemi.data.video_scene import voxel_observations

    output = Path(output)
    source_summary = json.loads((output/'summary.json').read_text())
    if not source_summary['complete']:
        raise ValueError('Complete source preprocessing before scene preparation')
    records = json.loads((output/'segments.json').read_text())['segments']
    groups = defaultdict(list)
    for row in records:
        if 'skipped' not in row:
            groups[map_identity(row)].append(row)
    support = validate_policy(yaml.safe_load((ROOT/config['scene']['support_policy']).read_text()))
    normal_policy = dict(normal_neighbours=config['scene']['normal_neighbours'],
        normal_radius_m=config['scene']['normal_radius_m'],
        minimum_neighbours=config['scene']['minimum_normal_neighbours'],
        minimum_spread_m=config['scene']['minimum_normal_spread_m'],
        maximum_planarity_ratio=config['scene']['maximum_planarity_ratio'])
    manifest = dict(version=VERSION, complete=False, maps=[], segment_maps={},
        full_scan_is_input=False, fitting_uses_body_labels=False, causal=False,
        scope='all available annotation observations of one split/recording/metric world',
        unannotated_video_frames_included=False, global_point_limit=None,
        support_policy=support, normal_policy=normal_policy,
        source_segments_sha256=digest(output/'segments.json'),
        implementation_sha256=digest(__file__), training_started=False)
    all_frames = defaultdict(set)
    for number, (key, members) in enumerate(sorted(groups.items()), 1):
        members.sort(key=lambda r: (r['first'], r['annotation']))
        identity = dict(version=VERSION, key=list(key), voxel_m=config['scene']['voxel_m'],
            normal_policy=normal_policy,
            sources={r['id']: digest(Path(r['folder'])/'metadata.json') for r in members})
        import hashlib
        identifier = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        folder = output/'maps'/identifier
        if not folder.exists():
            ps, fs, cs = [], [], []
            for row in members:
                path = Path(row['folder'])
                arrays = {}
                for name in ('visible_frame_points', 'visible_frame_owner', 'observation_valid',
                             'camera_world', 'source_frame_ids'):
                    if digest(path/(name+'.npy')) != row['output_sha256'][name+'.npy']:
                        raise ValueError('Changed source observations: '+row['id'])
                    arrays[name] = np.load(path/(name+'.npy'), allow_pickle=False)
                mask = (arrays['visible_frame_owner'] == 0) & arrays['observation_valid'][:, None]
                frame, _ = np.nonzero(mask)
                ps.append(arrays['visible_frame_points'][mask])
                fs.append(arrays['source_frame_ids'][frame])
                cs.append(arrays['camera_world'][frame, :3, 3])
            points, origins, counts = voxel_observations(np.concatenate(ps), np.concatenate(fs),
                np.concatenate(cs), config['scene']['voxel_m'])
            normals = np.zeros_like(points)
            valid = np.zeros(len(points), bool)
            # Bound PCA workspace even for the largest maps.
            for lo in range(0, len(points), 8192):
                hi = min(len(points), lo+8192)
                normals[lo:hi], valid[lo:hi] = fit_normals(points, points[lo:hi], origins[lo:hi], normal_policy)
            _publish(folder, identity, dict(points=points, normals=normals,
                normal_valid=valid, observations=counts))
        arrays, meta = read_bundle(folder)
        if meta['identity'] != identity:
            raise ValueError('Recording map identity differs')
        index = ObservedSceneIndex(**arrays)
        support_folder = folder/'support'
        support_identity = dict(map_sha256=digest(folder/'CACHE.json'), policy=support,
            implementation_sha256=digest(ROOT/'scenemi/geometry/support_surfaces.py'))
        if not support_folder.exists():
            patches = extract_support_surfaces(index, support, seed=int(identifier[:8], 16))
            _publish(support_folder, support_identity, patches)
        patches, support_meta = read_bundle(support_folder)
        if support_meta['identity'] != support_identity:
            raise ValueError('Support policy changed; use a new scene output')
        SupportSurfaceAtlas(patches, index.points, support)
        preview_folder = folder/'preview'
        preview_identity = dict(support_sha256=digest(support_folder/'CACHE.json'),
            policy=support, implementation_sha256=digest(__file__))
        if not preview_folder.exists():
            _publish(preview_folder, preview_identity, surface_preview(index, patches, support))
        _, preview_meta = read_bundle(preview_folder)
        if preview_meta['identity'] != preview_identity:
            raise ValueError('Scene preview implementation changed')
        frame_ids = set()
        for row in members:
            path = Path(row['folder'])
            valid = np.load(path/'observation_valid.npy')
            ids = np.load(path/'source_frame_ids.npy')[valid]
            frame_ids.update(map(int, ids))
            manifest['segment_maps'][row['id']] = identifier
        split, scene, recording, world = key
        all_frames[(scene, recording, split)].update(frame_ids)
        info = dict(id=identifier, split=split, scene=scene, recording=recording,
            folder=str(folder), map_sha256=digest(folder/'CACHE.json'),
            support_sha256=digest(support_folder/'CACHE.json'),
            preview_sha256=digest(preview_folder/'CACHE.json'),
            members=[r['id'] for r in members], observed_frame_ranges=contiguous_ranges(frame_ids),
            observed_frames=len(frame_ids), points=len(index.points),
            normal_valid_fraction=float(index.normal_valid.mean()) if len(index.points) else 0.,
            support_patches=len(patches['coefficients']),
            preview_only_extension=True)
        manifest['maps'].append(info)
        atomic_json(output/'scene_progress.json', dict(done=number, total=len(groups), last=identifier))
        print(json.dumps(dict(scene_maps_done=number, total=len(groups), points=info['points'],
            patches=info['support_patches'], recording=recording, split=split)), flush=True)
    for (scene, recording, split), frames in all_frames.items():
        for other in ('train', 'validation', 'test'):
            if other != split and frames & all_frames.get((scene, recording, other), set()):
                raise ValueError('Cross-split observations entered recording maps')
    manifest['complete'] = True
    atomic_json(output/'scene_manifest.json', manifest)
    return dict(complete=True, maps=len(groups), segments=len(manifest['segment_maps']),
                points=sum(r['points'] for r in manifest['maps']),
                support_patches=sum(r['support_patches'] for r in manifest['maps']),
                no_cross_split_observation_overlap=True)


def verify(output):
    manifest = json.loads((Path(output)/'scene_manifest.json').read_text())
    if manifest['version'] != VERSION or not manifest['complete']:
        raise ValueError('Incomplete GIMO recording maps')
    seen = defaultdict(set)
    for row in manifest['maps']:
        folder = Path(row['folder'])
        for relative, expected in [('', row['map_sha256']), ('support', row['support_sha256']),
                                   ('preview', row['preview_sha256'])]:
            if digest(folder/relative/'CACHE.json') != expected:
                raise ValueError('GIMO scene manifest changed')
            read_bundle(folder/relative)
        for first, end in row['observed_frame_ranges']:
            for frame in range(first, end):
                key = (row['scene'], row['recording'], frame)
                seen[key].add(row['split'])
                if len(seen[key]) > 1:
                    raise ValueError('Cross-split scene observation leakage')
    return dict(passed=True, maps=len(manifest['maps']), no_cross_split_observation_overlap=True)
