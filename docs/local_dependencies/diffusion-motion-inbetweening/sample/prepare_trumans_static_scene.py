"""Prepare static TRUMANS scene conditioning for the four head-only clips.

All geometry is kept in TRUMANS Y-up coordinates until it is transformed by
the same first-frame head-yaw anchor used by the head-control model.  Dynamic
object tracks are deliberately excluded from this first scene experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from sample.custom_clip_utils import head_local_yaw_features


BODY_QUERY_IDS = np.asarray([0, 4, 5, 7, 8, 10, 11, 15, 20, 21])
# Eye midpoint relative to SMPL-X head joint 15 in the supplied neutral model.
DEFAULT_CAMERA_OFFSET_HEAD = (-0.01031326, 0.04308906, 0.07903221)


def masked_surface_sample(mesh: trimesh.Trimesh, count: int, seed: int):
    points, face_ids = trimesh.sample.sample_surface(mesh, count, seed=seed)
    normals = mesh.face_normals[face_ids]
    return points.astype(np.float32), normals.astype(np.float32)


def select_global(points, normals, head_path, count, radius):
    distance = np.full(len(points), np.inf, dtype=np.float32)
    for start in range(0, len(points), 8192):
        chunk = points[start:start + 8192]
        distance[start:start + len(chunk)] = np.linalg.norm(
            chunk[:, None] - head_path[None], axis=-1
        ).min(axis=1)
    candidates = np.flatnonzero(distance <= radius)
    if len(candidates) < count:
        candidates = np.argsort(distance)[:count]
    else:
        candidates = candidates[np.linspace(0, len(candidates) - 1, count).astype(int)]
    return points[candidates], normals[candidates]


def camera_cloud(points, normals, camera_position, camera_rotation, count,
                 horizontal_fov, vertical_fov, near, far, image_width=64,
                 image_height=48, selection="nearest", occluder_depth=None):
    """Return a deterministic z-buffered cloud in +X right,+Y up,+Z forward."""
    camera_points = (points - camera_position) @ camera_rotation
    camera_normals = normals @ camera_rotation
    depth = camera_points[:, 2]
    tan_h = np.tan(np.deg2rad(horizontal_fov) / 2)
    tan_v = np.tan(np.deg2rad(vertical_fov) / 2)
    valid = (
        (depth >= near) & (depth <= far)
        & (np.abs(camera_points[:, 0] / np.maximum(depth, 1e-6)) <= tan_h)
        & (np.abs(camera_points[:, 1] / np.maximum(depth, 1e-6)) <= tan_v)
    )
    if occluder_depth is not None and valid.any():
        if np.asarray(occluder_depth).shape != (image_height, image_width):
            raise ValueError("occluder_depth must match camera_cloud image dimensions")
        valid_ids = np.flatnonzero(valid)
        ratio_x = camera_points[valid_ids, 0] / depth[valid_ids]
        ratio_y = camera_points[valid_ids, 1] / depth[valid_ids]
        pixel_x = np.clip(((ratio_x / tan_h + 1) * 0.5 * image_width).astype(int),
                          0, image_width - 1)
        # SceneMI camera coordinates are +Y-up, while image rows increase
        # downward.  Keep this identical to _body_depth()'s ray convention;
        # using the opposite sign makes the occluder lookup vertically flip
        # the body and lets background points leak through the actor.
        pixel_y = np.clip(((-ratio_y / tan_v + 1) * 0.5 * image_height).astype(int),
                          0, image_height - 1)
        body_depth = np.asarray(occluder_depth)[pixel_y, pixel_x]
        blocked = np.isfinite(body_depth) & (body_depth > 0) & (body_depth <= depth[valid_ids] + 1e-3)
        valid[valid_ids[blocked]] = False
    ids = np.flatnonzero(valid)
    if len(ids):
        ratio_x = camera_points[ids, 0] / depth[ids]
        ratio_y = camera_points[ids, 1] / depth[ids]
        pixel_x = np.clip(((ratio_x / tan_h + 1) * 0.5 * image_width).astype(int),
                          0, image_width - 1)
        pixel_y = np.clip(((-ratio_y / tan_v + 1) * 0.5 * image_height).astype(int),
                          0, image_height - 1)
        pixel = pixel_y * image_width + pixel_x
        order = np.argsort(depth[ids])
        _, first = np.unique(pixel[order], return_index=True)
        ids = ids[order[first]]
        if len(ids) > count:
            if selection == "pixel_uniform":
                # ``ids`` is now ordered by projected pixel index.  Uniformly
                # retain pixels across the whole image after z-buffering;
                # taking the nearest points globally collapses a ground plane
                # to a narrow strip when the view contains nearer geometry.
                ids = ids[np.linspace(0, len(ids) - 1, count).astype(np.int64)]
            elif selection == "nearest":
                ids = ids[np.argsort(depth[ids])[:count]]
            else:
                raise ValueError(f"Unknown camera-cloud selection: {selection}")
    features = np.zeros((count, 6), dtype=np.float32)
    mask = np.zeros(count, dtype=bool)
    used = min(len(ids), count)
    if used:
        features[:used, :3] = camera_points[ids[:used]]
        features[:used, 3:] = camera_normals[ids[:used]]
        mask[:used] = True
    return features, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motions", type=Path,
                        default=Path("data/custom_pelvis_head/four_clips_from_trumans.npz"))
    parser.add_argument("--world-head", type=Path,
                        default=Path("data/custom_world_head/four_clips_world_head_smplx.npz"))
    parser.add_argument("--trumans-root", type=Path, default=Path("../TRUMANS"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/custom_scene/four_clips_static_scene.npz"))
    parser.add_argument("--surface-candidates", type=int, default=65536)
    parser.add_argument("--global-points", type=int, default=2048)
    parser.add_argument("--local-points", type=int, default=128)
    parser.add_argument("--global-radius", type=float, default=4.0)
    parser.add_argument("--local-source-radius", type=float, default=4.5)
    parser.add_argument("--horizontal-fov", type=float, default=90.0)
    parser.add_argument("--vertical-fov", type=float, default=70.0)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--far", type=float, default=4.0)
    parser.add_argument("--camera-offset-head", nargs=3, type=float,
                        default=DEFAULT_CAMERA_OFFSET_HEAD)
    parser.add_argument("--contact-threshold", type=float, default=0.05)
    parser.add_argument("--distance-scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    motions = np.load(args.motions, allow_pickle=False)
    head = np.load(args.world_head, allow_pickle=False)
    if not np.array_equal(motions["names"], head["names"]):
        raise ValueError("Motion and head clip order differs")
    records = {
        item["clip_name"]: item
        for line in (args.trumans_root / "processed/scene_expert_v1/clips.jsonl").read_text().splitlines()
        if line.strip()
        for item in (json.loads(line),)
    }
    scene_paths = [args.trumans_root / records[str(name)]["scene"]["mesh"]
                   for name in motions["names"]]
    if len({path.resolve() for path in scene_paths}) != 1:
        raise ValueError("This initial static-scene pack expects one shared room mesh")
    mesh = trimesh.load(scene_paths[0], force="mesh", process=False)
    surface_world, normal_world = masked_surface_sample(
        mesh, args.surface_candidates, args.seed
    )

    head_features, anchor_position, anchor_rotation = head_local_yaw_features(
        head["world_head_position_m"], head["world_head_rotation_matrix"]
    )
    world_to_anchor = np.swapaxes(anchor_rotation, -1, -2)
    frames = head_features.shape[1]
    camera_offset = np.asarray(args.camera_offset_head, dtype=np.float32)
    head_rotation_anchor = np.einsum(
        "bij,btjk->btik", world_to_anchor, head["world_head_rotation_matrix"]
    )
    camera_position_anchor = (
        head_features[..., :3]
        + np.einsum("btij,j->bti", head_rotation_anchor, camera_offset)
    )
    camera_rotation_anchor = head_rotation_anchor

    global_features, global_masks = [], []
    for clip in range(len(motions["names"])):
        points_anchor = np.einsum(
            "ij,pj->pi", world_to_anchor[clip], surface_world - anchor_position[clip]
        )
        normals_anchor = np.einsum("ij,pj->pi", world_to_anchor[clip], normal_world)
        selected_points, selected_normals = select_global(
            points_anchor, normals_anchor, head_features[clip, :, :3],
            args.global_points, args.global_radius,
        )
        global_features.append(np.concatenate((selected_points, selected_normals), -1))
        global_masks.append(np.ones(args.global_points, dtype=bool))
    global_features = np.stack(global_features).astype(np.float32)
    global_masks = np.stack(global_masks)

    # Pair [head/body clip, counterfactual scene placement]. The diagonal is factual.
    local_features = np.zeros(
        (len(motions["names"]), len(motions["names"]), frames,
         args.local_points, 6), dtype=np.float32
    )
    local_masks = np.zeros(local_features.shape[:-1], dtype=bool)
    interaction = np.zeros(
        (len(motions["names"]), len(motions["names"]), frames,
         len(BODY_QUERY_IDS), 2), dtype=np.float32
    )
    body_world = motions["source_world_joints"][:, :, BODY_QUERY_IDS]
    body_anchor = np.einsum(
        "bij,btkj->btki", world_to_anchor,
        body_world - anchor_position[:, None, None]
    )
    for motion_id in range(len(motions["names"])):
        for scene_id in range(len(motions["names"])):
            scene_points = global_features[scene_id, :, :3]
            scene_normals = global_features[scene_id, :, 3:]
            for frame in range(frames):
                local_features[motion_id, scene_id, frame], local_masks[
                    motion_id, scene_id, frame
                ] = camera_cloud(
                    scene_points, scene_normals,
                    camera_position_anchor[motion_id, frame],
                    camera_rotation_anchor[motion_id, frame],
                    args.local_points, args.horizontal_fov, args.vertical_fov,
                    args.near, args.far,
                )
                distance = np.linalg.norm(
                    body_anchor[motion_id, frame, :, None] - scene_points[None],
                    axis=-1,
                ).min(axis=-1)
                interaction[motion_id, scene_id, frame, :, 0] = np.clip(
                    distance / args.distance_scale, 0.0, 1.0
                )
                interaction[motion_id, scene_id, frame, :, 1] = (
                    distance <= args.contact_threshold
                )

    camera_world_position = (
        head["world_head_position_m"]
        + np.einsum(
            "btij,j->bti", head["world_head_rotation_matrix"], camera_offset
        )
    )
    camera_world_from = np.zeros(
        (len(motions["names"]), frames, 4, 4), dtype=np.float32
    )
    camera_world_from[..., :3, :3] = head["world_head_rotation_matrix"]
    camera_world_from[..., :3, 3] = camera_world_position
    camera_world_from[..., 3, 3] = 1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        names=motions["names"],
        scene_name=np.asarray(records[str(motions["names"][0])]["scene"]["name"]),
        scene_mesh=np.asarray(str(scene_paths[0])),
        global_features=global_features,
        global_mask=global_masks,
        local_features_pair=local_features,
        local_mask_pair=local_masks,
        factual_local_features=local_features[np.arange(4), np.arange(4)],
        factual_local_mask=local_masks[np.arange(4), np.arange(4)],
        interaction_pair=interaction,
        factual_interactions=interaction[np.arange(4), np.arange(4)],
        body_query_ids=BODY_QUERY_IDS,
        body_query_anchor=body_anchor.astype(np.float32),
        head_anchor_position_world=anchor_position,
        head_anchor_rotation_world=anchor_rotation,
        camera_world_from=camera_world_from,
        head_from_camera_translation=camera_offset,
        head_from_camera_rotation=np.eye(3, dtype=np.float32),
        camera_axis_convention=np.asarray("right=+X, up=+Y, forward=+Z"),
        coordinate_system=np.asarray("TRUMANS Y-up; network geometry uses first-head-yaw anchor"),
        static_only=np.asarray(True),
        distance_scale_m=np.asarray(args.distance_scale),
        contact_threshold_m=np.asarray(args.contact_threshold),
        fps=head["fps"],
    )
    factual_valid = local_masks[np.arange(4), np.arange(4)].sum(axis=-1)
    print(f"Saved {args.output}")
    print("Global/local:", global_features.shape, local_features.shape)
    print("Factual visible points min/mean/max:",
          factual_valid.min(), factual_valid.mean(), factual_valid.max())
    print("Factual contact rates:", interaction[np.arange(4), np.arange(4), ..., 1].mean((1, 2)))
    print("Camera transform orthogonality max:", np.abs(
        np.einsum("...ji,...jk->...ik", camera_world_from[..., :3, :3],
                  camera_world_from[..., :3, :3]) - np.eye(3)
    ).max())


if __name__ == "__main__":
    main()
