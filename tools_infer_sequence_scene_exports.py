"""Run the current corrected scene EXO adapter on three local HSI exports.

This is an inference-only adapter.  The exports contain meshes/cameras/scene
points but no per-joint confidence, so a second pass derives a conservative
visibility proxy from the target mesh joints projected into the source PV
image.  The proxy is recorded as such and is never used as an evaluation
label.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from scenemi.config import ROOT
from scenemi.data.amass_dataset import six_columns
from scenemi.geometry.body import PARENTS, encode_body
from scenemi.scene_joint_training import settings
from scenemi import scene_joint_replay as historical
from experiments.offline_camera_retrain_v1.data import anchor_rotation
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.unified_geometry import occupancy_from_points


EXPORT_ROOT = Path('/home/wenxin/projects/sequence_scene_exports')
BUNDLE_ROOT = Path('/home/wenxin/projects/egobody_val3_share_20261026')
PREP_ROOT = ROOT / 'runs/exo_preparation/sequence_exports'
DEFAULT_OUT = ROOT / 'runs/scene_joint128_conditioning_corrected/sequence_scene_exports_step003000'
AXES_EXPORT_TO_YUP = np.diag([1., -1., 1.]).astype(np.float32)
SMPL_VISIBLE = np.array([0, 1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21])
BODY25_VISIBLE = np.array([8, 12, 9, 13, 10, 14, 11, 1, 0, 5, 2, 6, 3, 7, 4])


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy.core.numeric', 'numpy._core.numeric') and name == '_frombuffer':
            return np.core.numeric._frombuffer
        if module == 'numpy' and name == 'dtype':
            return np.dtype
        raise ValueError(f'unexpected pickle global: {module}.{name}')


def read_pickle(path):
    with Path(path).open('rb') as stream:
        return SafeUnpickler(stream).load()


def source_image(path):
    """Map the source machine path in the export to the local shared bundle."""
    path = Path(path)
    marker = Path('egobody_val3_share_20261026')
    parts = path.parts
    try:
        index = parts.index(str(marker))
        return BUNDLE_ROOT.joinpath(*parts[index + 1:])
    except ValueError:
        # Current exports use the absolute path above; this fallback keeps the
        # adapter useful when a manifest already contains a local path.
        return path


def camera_and_mesh(folder):
    value = read_pickle(folder / 'cameras.pkl')
    cam = value['cameras']['hsi']
    c2w_export = np.asarray(cam['camera_to_world_4x4'], np.float32)
    intrinsic = np.asarray(value['intrinsic_3x3'], np.float32)
    height, width = map(int, value['image_hw'])
    # The export is OpenCV (+Y down, +Z forward).  Reflect the world and the
    # local camera Y axis together, yielding SceneMI's right-handed Y-up frame.
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = AXES_EXPORT_TO_YUP @ c2w_export[:3, :3] @ AXES_EXPORT_TO_YUP
    c2w[:3, 3] = AXES_EXPORT_TO_YUP @ c2w_export[:3, 3]
    people = [p for p in read_pickle(folder / 'smpl.pkl')['people'] if p['kind'] == 'hsi']
    return c2w, intrinsic, (height, width), people


def project(points, folder):
    value = read_pickle(folder / 'cameras.pkl')
    w2c = np.asarray(value['cameras']['hsi']['world_to_camera_4x4'], np.float32)
    k = np.asarray(value['intrinsic_3x3'], np.float32)
    h, w = map(int, value['image_hw'])
    q = np.asarray(points) @ w2c[:3, :3].T + w2c[:3, 3]
    uv = q[:, :2] / np.maximum(q[:, 2:,], 1e-6) * k.diagonal()[:2] + k[[0, 1], [2, 2]]
    inside = (q[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv, inside


def load_sequence(directory, clip_id):
    frames = sorted(directory.glob('frame_*'), key=lambda p: int(p.name.split('_')[1]))
    if len(frames) != 299:
        raise ValueError(f'{directory.name}: expected 299 frames, got {len(frames)}')
    cameras, intrinsics, images, joints, valid, clouds = [], [], [], [], [], []
    for frame in frames:
        c2w, k, hw, people = camera_and_mesh(frame)
        cameras.append(c2w); intrinsics.append(k); images.append(source_image(json.loads((frame/'manifest.json').read_text())['source_image']))
        candidates = [p for p in people if str(p['track_id']) == '0']
        if len(candidates) > 1:
            raise ValueError(f'{frame}: duplicate selected track')
        if candidates:
            # The preparation audit used this fixed male SMPL regressor.  Read
            # the already-audited joint track rather than opening GT arrays.
            valid.append(True)
        else:
            valid.append(False)
        ply = json.loads((frame/'manifest.json').read_text())['files']['environment']['hsi']['file']
        cloud = np.asarray(trimesh.load(frame / ply, process=False).vertices, np.float32)
        if cloud.ndim != 2 or cloud.shape[1] != 3:
            raise ValueError(f'{frame}: invalid environment point cloud')
        clouds.append(cloud)
    audited = np.load(PREP_ROOT / f'{directory.name}.npz', allow_pickle=False)
    joints_export = np.asarray(audited['joints_world_unscaled'], np.float32)
    valid = np.asarray(audited['body_available'], bool)
    # These are the bundle's image-space OpenPose detections, not 3D body GT.
    # They are used only as a conservative image visibility proxy.
    annotation = np.load(BUNDLE_ROOT / 'clips' / f'clip{clip_id}' / 'gt/annotations_2d.npz', allow_pickle=False)
    openpose = np.asarray(annotation['openpose25'], np.float32)
    if openpose.shape[0] != len(frames):
        raise ValueError(f'{directory.name}: 2D annotation clock mismatch')
    cameras = np.asarray(cameras); intrinsics = np.asarray(intrinsics)
    return dict(frames=frames, cameras=cameras, intrinsics=intrinsics, images=images,
                joints_export=joints_export, valid=valid, clouds=clouds, openpose=openpose,
                image_hw=tuple(map(int, camera_and_mesh(frames[0])[2])))


def fill_track(joints, valid):
    """Interpolate only missing exported track frames for a finite initializer."""
    joints = joints.copy(); valid = valid.copy()
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        raise ValueError('fewer than two valid exported body frames')
    query = np.clip(np.arange(len(joints)), indices[0], indices[-1])
    for j in range(22):
        for k in range(3):
            joints[:, j, k] = np.interp(query, indices, joints[indices, j, k])
    return joints


def root_rotation(rest, joints):
    """Estimate a stable global root rotation from torso and hip directions."""
    def align(obs, ref):
        try:
            return Rotation.align_vectors(obs, ref)[0].as_matrix().astype(np.float32)
        except ValueError:
            return np.eye(3, dtype=np.float32)
    ref = np.stack((rest[3]-rest[0], rest[1]-rest[2], rest[12]-rest[0]))
    obs = np.stack((joints[3]-joints[0], joints[1]-joints[2], joints[12]-joints[0]))
    return align(obs, ref)


def estimate_local_rotations(rest, joints):
    """Fit a FK-consistent native pose from mesh-regressed joint positions.

    A joint's position is determined by its *parent* global rotation.  The
    former adapter aligned every child bone independently and then used those
    child rotations as parent rotations, which made the 201D rotation and
    redundant joint-position channels disagree.  Estimate each joint's global
    rotation from its outgoing bones instead, then encode only the FK-derived
    positions.  This keeps the initializer on the representation used by the
    trained prior even though the export has no native pose/betas.
    """
    t = len(joints); global_r = np.zeros((t, 22, 3, 3), np.float32)
    local = np.zeros_like(global_r)
    eye = np.eye(3, dtype=np.float32)
    children = [[index for index, parent in enumerate(PARENTS) if parent == joint]
                for joint in range(22)]
    for i in range(t):
        for j in range(22):
            parent = PARENTS[j]
            ref = [rest[k] - rest[j] for k in children[j]]
            obs = [joints[i, k] - joints[i, j] for k in children[j]]
            fallback = global_r[i, parent] if parent >= 0 else eye
            if not ref or any(np.linalg.norm(v) < 1e-5 for v in ref + obs):
                global_r[i, j] = fallback
                continue
            try:
                global_r[i, j] = Rotation.align_vectors(np.asarray(obs), np.asarray(ref))[0].as_matrix().astype(np.float32)
            except ValueError:
                global_r[i, j] = fallback
        local[i, 0] = global_r[i, 0]
        for j in range(1, 22):
            local[i, j] = global_r[i, PARENTS[j]].T @ global_r[i, j]
        # Numerical projection back to SO(3).
        for j in range(22):
            u, _, vh = np.linalg.svd(local[i, j]); local[i, j] = (u @ vh).astype(np.float32)
            if np.linalg.det(local[i, j]) < 0: local[i, j, :, -1] *= -1
    return local


def sample_points(cloud, count=512, seed=0):
    if len(cloud) == 0:
        return np.zeros((count, 3), np.float32), np.full(count, -1, np.int64)
    rng = np.random.default_rng(seed)
    take = np.linspace(0, len(cloud)-1, min(count, len(cloud)), dtype=np.int64)
    if len(cloud) > count:
        take = np.sort(rng.choice(len(cloud), count, replace=False))
    points = np.zeros((count, 3), np.float32); owners = np.full(count, -1, np.int64)
    points[:len(take)] = cloud[take]; owners[:len(take)] = 0
    return points, owners


def make_payload(seq, start, rest, visibility_mode):
    span = slice(start, start+128); cp = seq['cameras'][:, :3, 3]; cr = seq['cameras'][:, :3, :3]
    origin = cp[start].copy(); origin[1] = 0.
    anchor = anchor_rotation(cr[start:start+128])
    # Export points and camera are already in one world; transform both using
    # one reflection and one fixed yaw, preserving reprojection geometry.
    body_scene = np.einsum('ij,tkj->tki', AXES_EXPORT_TO_YUP, seq['joints_export'])
    body_local = (body_scene[span]-origin) @ anchor.T
    valid = seq['valid'][span].copy(); filled = fill_track(body_local, valid)
    local_rot = estimate_local_rotations(rest, filled)
    motion = encode_body(local_rot, filled[:, 0], rest)
    # encode_body writes redundant positions generated by the same rotations
    # and rest skeleton.  Do not overwrite them with mesh-regressed positions:
    # those positions are estimates, not an FK-consistent native SMPL track.
    visible = np.zeros((128, 22), np.float32)
    for i, frame in enumerate(seq['frames'][start:start+128]):
        # BODY25 confidence is the only image-derived visibility evidence
        # available in this bundle.  Do not use the HSI mesh projection here:
        # its recovered camera is not a calibrated PV camera.
        op = seq['openpose'][start+i]
        detected = np.zeros(22, bool)
        detected[SMPL_VISIBLE] = (op[BODY25_VISIBLE, 2] > .5)
        image_h, image_w = cv2.imread(str(seq['images'][start+i])).shape[:2]
        inside = np.zeros(22, bool)
        xy = op[BODY25_VISIBLE, :2]
        inside[SMPL_VISIBLE] = ((xy[:, 0] >= 0) & (xy[:, 0] < image_w) &
                                (xy[:, 1] >= 0) & (xy[:, 1] < image_h))
        visible[i] = (detected & inside).astype(np.float32) if valid[i] else 0.
    available = valid.astype(np.float32)
    if visibility_mode == 'all':
        visible[:] = 1.; reliability = np.ones_like(visible); root_reliability = np.ones(128, np.float32)
    else:
        reliability = visible * available[:, None]
        # The pelvis itself is usually outside the PV crop.  A visible torso,
        # shoulder or head still anchors the global translation of the tracked
        # HSI body, so use that group for root confidence.
        root_reliability = visible[:, [12, 13, 14, 15]].max(-1) * available
    camera_local = (cp[span]-origin) @ anchor.T / 2.
    camera_rot = np.einsum('ij,tjk->tik', anchor, cr[span])
    camera = np.concatenate((camera_local, six_columns(camera_rot)), -1).astype(np.float32)
    height, width = seq['image_hw']; k = seq['intrinsics'][span]
    intr = np.stack((k[:, 0, 0]/width, k[:, 1, 1]/height, k[:, 0, 2]/width, k[:, 1, 2]/height), -1).astype(np.float32)
    frame_pts, frame_owner, history = [], [], []
    for i in range(start, start+128):
        p, o = sample_points((AXES_EXPORT_TO_YUP @ seq['clouds'][i].T).T, 512, seed=9000+i)
        frame_pts.append((p-origin) @ anchor.T); frame_owner.append(o)
    for i in range(start):
        p, _ = sample_points((AXES_EXPORT_TO_YUP @ seq['clouds'][i].T).T, 512, seed=9000+i)
        history.append((p-origin) @ anchor.T)
    history = np.concatenate(history, 0) if history else np.zeros((0, 3), np.float32)
    if len(history) > 8192: history = history[np.linspace(0, len(history)-1, 8192, dtype=np.int64)]
    hp = np.zeros((8192, 3), np.float32); hv = np.zeros(8192, bool); hp[:len(history)] = history; hv[:len(history)] = True
    frame_pts, frame_owner = np.asarray(frame_pts, np.float32), np.asarray(frame_owner, np.int64)
    payload = dict(
        camera=camera[None], camera_available=np.ones((1,128,2), np.float32), camera_confidence=np.ones((1,128,2), np.float32),
        camera_context=np.asarray([[0.,1.,0.,1.,0.,-1.]], np.float32), camera_intrinsics=intr[None], camera_intrinsics_valid=np.ones((1,128), np.float32),
        trajectory=np.zeros((1,128,22,9), np.float32), observation_meta=np.zeros((1,128,22,5), np.float32), observation_uncertainty=np.zeros((1,128,22,2), np.float32),
        body_format=np.asarray([1], np.int64), sensor_profile=np.asarray([0], np.int64),
        occupancy=occupancy_from_points(history)[None], frame_scene_points=frame_pts[None], frame_scene_owner=frame_owner[None],
        local_history_points=hp[None], local_history_valid=hv[None], local_frame_points=frame_pts[None], local_frame_owner=frame_owner[None],
        initial_motion=motion[None], initial_rest=rest[None].astype(np.float32), initial_reliability=reliability[None].astype(np.float32),
        initial_root_reliability=root_reliability[None].astype(np.float32), estimate_available=available[None].astype(np.float32), joint_visible=visible[None].astype(np.float32))
    meta = dict(start=start, origin=origin.tolist(), anchor=anchor.tolist(), visible=visible, available=available,
                source_valid=valid, body_local=body_local, body_scene=body_scene[span])
    return payload, meta


@torch.no_grad()
def sample_scene_exo(model, payload, steps=20, start_timestep=200, seed=777):
    initial = payload['initial_motion']; model.exo._anchor_payload = None
    anchor = model.exo.anchor(payload)
    generator = torch.Generator(device=initial.device).manual_seed(seed)
    noise = torch.randn(initial.shape, device=initial.device, generator=generator)
    schedule = torch.linspace(start_timestep, 0, steps, device=initial.device).round().long()
    state = model.alphas[start_timestep].sqrt()*anchor + (1-model.alphas[start_timestep]).sqrt()*noise
    root = payload['initial_root_reliability'][..., None].expand(-1, -1, 3)
    rel = payload['initial_reliability'][..., None].expand(-1, -1, -1, 6).flatten(2)
    pos = torch.minimum(payload['initial_reliability'], payload['initial_root_reliability'][..., None])[..., None].expand(-1,-1,-1,3).flatten(2)
    confidence = torch.cat((root, rel, pos), -1).clamp(0,1)
    old = model.training; model.eval()
    try:
        for i, t in enumerate(schedule):
            prediction = model(state, t.expand(len(initial)), payload)
            if i == len(schedule)-1: return prediction
            following = schedule[i+1]
            eps = (state-model.alphas[t].sqrt()*prediction)/(1-model.alphas[t]).sqrt().clamp_min(1e-6)
            state = model.alphas[following].sqrt()*prediction + (1-model.alphas[following]).sqrt()*eps
            observed = model.alphas[following].sqrt()*anchor + (1-model.alphas[following]).sqrt()*noise
            state = torch.lerp(state, observed, confidence)
    finally:
        model.train(old)


def to_tensors(payload, device):
    out = {}
    for key, value in payload.items():
        out[key] = torch.as_tensor(value, device=device)
    return out


def draw_skeleton(image, joints, folder, color, thickness=2):
    uv, inside = project(joints, folder)
    out = image.copy()
    for j, parent in enumerate(PARENTS):
        if parent < 0: continue
        if inside[j] and inside[parent]:
            cv2.line(out, tuple(np.round(uv[j]).astype(int)), tuple(np.round(uv[parent]).astype(int)), color, thickness, cv2.LINE_AA)
    for j in range(22):
        if inside[j]: cv2.circle(out, tuple(np.round(uv[j]).astype(int)), 3, color, -1, cv2.LINE_AA)
    return out


def render_hsi_contact(seq, predictions, out):
    """Render the consistent HSI camera/scene frame, without pretending PV alignment."""
    rows = []
    for frame_i in [0, 100, 200, 298]:
        frame = seq['frames'][frame_i]
        c = read_pickle(frame / 'cameras.pkl'); k = np.asarray(c['intrinsic_3x3'], np.float32)
        h, w = map(int, c['image_hw']); r = np.asarray(c['cameras']['hsi']['world_to_camera_4x4'], np.float32)
        canvas = np.full((h, w, 3), 248, np.uint8)
        points = seq['clouds'][frame_i]
        q = points @ r[:3, :3].T + r[:3, 3]
        uv = q[:, :2] / np.maximum(q[:, 2:,], 1e-6) * k.diagonal()[:2] + k[[0, 1], [2, 2]]
        keep = (q[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        take = np.flatnonzero(keep)
        if len(take) > 9000: take = take[np.linspace(0, len(take)-1, 9000).astype(int)]
        canvas[np.round(uv[take, 1]).astype(int), np.round(uv[take, 0]).astype(int)] = (205, 205, 205)
        panels = []
        for name, color in [('initial', (150, 80, 30)), ('pv_visibility', (30, 30, 220))]:
            panel = draw_skeleton(canvas, predictions[name][frame_i], frame, color)
            cv2.putText(panel, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
            panels.append(cv2.resize(panel, (344, 192)))
        rows.append(np.concatenate(panels, 1))
    if rows: cv2.imwrite(str(out), np.concatenate(rows, 0))


def render_contact(seq, folder, predictions, out):
    rows = []
    for frame_i in [0, 100, 200, 298]:
        frame = seq['frames'][frame_i]; image = cv2.imread(str(seq['images'][frame_i]))
        if image is None: continue
        panel = image.copy(); op = seq['openpose'][frame_i]
        for j, body25 in enumerate(BODY25_VISIBLE):
            x, y, conf = op[body25]
            if not (0 <= x < panel.shape[1] and 0 <= y < panel.shape[0]): continue
            color = (40, 170, 40) if conf > .5 else (80, 80, 220)
            cv2.circle(panel, (int(round(x)), int(round(y))), 6, color, -1, cv2.LINE_AA)
        cv2.putText(panel, 'PV OpenPose visibility audit (green=conf>0.5)', (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .75, (30, 30, 220), 2, cv2.LINE_AA)
        rows.append(cv2.resize(panel, (688, 384)))
    if rows: cv2.imwrite(str(out), np.concatenate(rows, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = settings(ROOT/'configs/scene_joint128_conditioning_corrected.yaml')
    device = torch.device(args.device)
    model = historical.make_model(config, str(device))
    state = torch.load(ROOT/'runs/scene_joint128_conditioning_corrected/training/last.pt', map_location='cpu', weights_only=False)
    historical.load_adapter_state(model, state['adapter']); model.eval()
    rest = model.base.shape.rest_template.detach().cpu().numpy().astype(np.float32)
    report = dict(state='running', checkpoint_step=int(state['step']), checkpoint=str(ROOT/'runs/scene_joint128_conditioning_corrected/training/last.pt'),
                  visibility_proxy='bundle OpenPose BODY25 confidence > 0.5 and image bounds; mapped to 15 SMPL joints; outside/missing gets zero joint reliability',
                  coordinate_transform='export OpenCV world -> SceneMI Y-up via diag(1,-1,1), then fixed first-window camera yaw; no GT alignment',
                  windows=[], outputs=str(args.out))
    (args.out/'PROTOCOL.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    for clip_id, directory in enumerate(sorted(EXPORT_ROOT.glob('sequence_*')), 1):
        seq = load_sequence(directory, clip_id); clip_out = args.out/directory.name; clip_out.mkdir(exist_ok=True)
        all_pred = {'initial': np.full((299,22,3), np.nan, np.float32), 'pv_visibility': np.full((299,22,3), np.nan, np.float32)}
        all_vis = np.zeros((299,22), np.float32); all_valid = seq['valid'].copy()
        starts = [0, 128, 171]
        for start in starts:
            payload_np, meta = make_payload(seq, start, rest, 'pv')
            payload = to_tensors(payload_np, device)
            # Initial FK in the local window, and the same motion after scene
            # repair using the PV-derived confidence proxy.
            init = payload['initial_motion']; init_j = forward_kinematics(init, payload['initial_rest'])[0].cpu().numpy()
            out_motion = sample_scene_exo(model, payload, steps=args.steps, seed=777+start)
            out_j = forward_kinematics(out_motion, payload['initial_rest'])[0].cpu().numpy()
            anchor = np.asarray(meta['anchor'], np.float32); origin = np.asarray(meta['origin'], np.float32)
            init_world = init_j @ anchor + origin; pred_world = out_j @ anchor + origin
            init_export = (init_world @ AXES_EXPORT_TO_YUP.T).astype(np.float32)
            pred_export = (pred_world @ AXES_EXPORT_TO_YUP.T).astype(np.float32)
            all_pred['initial'][start:start+128] = init_export; all_pred['pv_visibility'][start:start+128] = pred_export
            all_vis[start:start+128] = meta['visible'];
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            report['windows'].append(dict(sequence=directory.name, start=start, frames=128,
                source_valid_frames=int(meta['source_valid'].sum()), pv_visible_joint_fraction=float(meta['visible'].mean()),
                lower_body_visible_fraction=float(meta['visible'][:, [4,5,7,8,10,11]].mean()),
                missing_export_frames=int((~meta['source_valid']).sum())))
            np.savez_compressed(clip_out/f'window_{start:04d}.npz', initial_joints_export=init_export,
                pv_visibility_joints_export=pred_export, projected_visibility=meta['visible'], source_available=meta['source_valid'])
        np.savez_compressed(clip_out/'predictions.npz', initial_joints_export=all_pred['initial'],
            pv_visibility_joints_export=all_pred['pv_visibility'], projected_visibility=all_vis, source_available=all_valid)
        # The PV panel is kept as an image audit; model joints are rendered in
        # a separate HSI camera projection because HSI is not PV-calibrated.
        render_contact(seq, seq['frames'], all_pred, clip_out/'contact_rgb_visibility_audit.jpg')
        render_hsi_contact(seq, all_pred, clip_out/'contact_hsi_projection_initial_vs_pv_visibility.jpg')
    report['state']='complete'; report['sequences']=sorted(p.name for p in args.out.iterdir() if p.is_dir())
    (args.out/'RESULT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    (args.out/'STATUS.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
