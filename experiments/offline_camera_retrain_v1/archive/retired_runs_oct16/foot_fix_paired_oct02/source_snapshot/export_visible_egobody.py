"""Extend EgoBody scan-derived visible maps over the entire known camera path."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parents[2]
MOTION_ROOT = PROJECTS / 'diffusion-motion-inbetweening'
if str(MOTION_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTION_ROOT))
from sample.prepare_trumans_static_scene import camera_cloud, masked_surface_sample

PREPARED = MOTION_ROOT / 'experiments/offline_sequence_v1/data/egobody_sequences'
EGOBODY = PROJECTS / 'egobody'
QUERY_RADIUS = 6.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit-per-split', type=int, default=0)
    parser.add_argument('--candidate-count', type=int, default=65536)
    parser.add_argument('--sample-stride', type=int, default=4)
    parser.add_argument('--output', type=Path, default=HERE / 'data/visible_egobody_5hz')
    args = parser.parse_args()
    surfaces = {}
    for split in ('train', 'validation', 'test'):
        rows = [json.loads(line) for line in (PREPARED / f'{split}.jsonl').read_text().splitlines() if line]
        if args.limit_per_split:
            rows = rows[:args.limit_per_split]
        manifest = []
        for row in rows:
            scene_id = row['scene_id']
            if scene_id not in surfaces:
                mesh_path = EGOBODY / 'scene_mesh' / scene_id / f'{scene_id}.obj'
                mesh = trimesh.load(mesh_path, force='mesh', process=False)
                seed = int.from_bytes(hashlib.sha256(scene_id.encode()).digest()[:4], 'big')
                points, normals = masked_surface_sample(mesh, args.candidate_count, seed)
                surfaces[scene_id] = (points, normals, cKDTree(points))
            points, normals, tree = surfaces[scene_id]
            folder = Path(row['folder'])
            output = args.output / split / row['sequence_id']
            output.mkdir(parents=True, exist_ok=True)
            saved = output / 'visible_static_points_world.npy'
            if saved.is_file() and (output / 'metadata.json').is_file():
                meta = json.loads((output / 'metadata.json').read_text())
                if (meta['sequence_id'] != row['sequence_id']
                        or meta['frames_20fps'] != row['frames_20fps']
                        or meta.get('sample_stride') != args.sample_stride
                        or meta.get('candidate_count') != args.candidate_count
                        or meta.get('query_radius_m') != QUERY_RADIUS):
                    raise ValueError(f'{row["sequence_id"]}: stale map export')
            else:
                camera = np.load(folder / 'camera_position_world.npy', mmap_mode='r')
                rotation = np.load(folder / 'camera_rotation_world.npy', mmap_mode='r')
                observed = [np.load(folder / 'visible_static_points_world.npy')]
                for frame in range(0, len(camera), args.sample_stride):
                    # A 4 m forward-depth frustum extends beyond a 4 m sphere
                    # at its 90x70-degree corners.
                    nearby = np.asarray(tree.query_ball_point(camera[frame], QUERY_RADIUS), dtype=np.int64)
                    if not len(nearby):
                        continue
                    cloud, mask = camera_cloud(points[nearby], normals[nearby],
                                               camera[frame], rotation[frame],
                                               512, 90, 70, .05, 4.)
                    if mask.any():
                        observed.append(cloud[mask, :3] @ rotation[frame].T + camera[frame])
                merged = np.concatenate(observed).astype(np.float32)
                _, indices = np.unique(np.floor(merged / .025).astype(np.int32), axis=0,
                                       return_index=True)
                merged = merged[np.sort(indices)]
                if len(merged) < 32:
                    raise ValueError(f'{row["sequence_id"]}: insufficient scene map')
                np.save(saved, merged)
                meta = dict(sequence_id=row['sequence_id'], split=split, scene_id=scene_id,
                            frames_20fps=len(camera), points=len(merged),
                            protocol=f'original cached rays plus static scan projection every {args.sample_stride}/20 seconds over full known camera path',
                            sample_stride=args.sample_stride,
                            candidate_count=args.candidate_count,
                            query_radius_m=QUERY_RADIUS,
                            caveat='Synthetic scan depth; moving partner is not an explicit scene condition.',
                            path=str(saved.resolve()))
                (output / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
            manifest.append(meta)
            print(f'{split}: {row["sequence_id"]}: {meta["points"]} points', flush=True)
        if not args.limit_per_split:
            (args.output / f'{split}.jsonl').write_text(
                ''.join(json.dumps(meta) + '\n' for meta in manifest))


if __name__ == '__main__':
    main()
