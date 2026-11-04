"""Inspect native source motion and exported EXO inputs without model inference.

The shared MotionViewer supplies playback and layer controls. Moving previews
use separately recast observations with declared synthetic optics.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np

from experiments.offline_camera_retrain_v1.ops.motion_viewer import MotionViewer, atomic_json
from experiments.offline_camera_retrain_v1.ops.viewer_geometry import (
    camera_layers, instant_scene_layers, readonly_array,
)
from experiments.offline_camera_retrain_v1.ops.viewer_observation import exo_first_hit_evidence

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'native-exo-source-preview-v1'


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def camera_motion(matrix):
    """Measure trajectory variation without assuming a calibrated optical axis."""
    matrix = np.asarray(matrix)
    if matrix.ndim == 2:
        matrix = matrix[None]
    if matrix.shape[1:] != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Invalid camera transforms')
    positions, rotations = matrix[:, :3, 3], matrix[:, :3, :3]
    if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=2e-4):
        raise ValueError('Camera rotations are not orthogonal')
    relative = rotations @ rotations[0].T
    angles = np.rad2deg(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1)))
    return dict(position_path_m=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
                position_max_from_first_m=float(np.linalg.norm(positions-positions[0], axis=1).max()),
                rotation_max_from_first_deg=float(angles.max()))


def observations(identity, length):
    start = identity['start']
    def read(name):
        return np.asarray(readonly_array(identity, name))[start:start+length]
    available, confidence = read('exo_position_available'), read('exo_position_confidence')
    evidence = exo_first_hit_evidence(read('exo_reference_counts'), read('exo_visible_counts'),
        {k: read('exo_'+k+'_counts') for k in ('person', 'static', 'rigid')},
        valid=read('observation_valid'))
    if not np.array_equal(available > 0, evidence['exported_position_available']):
        raise ValueError('EXO availability disagrees with saved first-hit evidence')
    if not np.allclose(confidence, evidence['exported_position_confidence'], atol=1e-6):
        raise ValueError('EXO confidence disagrees with saved first-hit evidence')
    position = read('exo_trajectory_world')
    observation = np.zeros((length, 22, 9), np.float32)
    observation[..., :3] = np.where(available[..., None] > 0, position/2, 0)
    meta = np.zeros((length, 22, 5), np.float32)
    meta[..., 0], meta[..., 2], meta[..., 4] = available, confidence, 1
    return dict(observations=observation, observation_meta=meta,
                source_occluded=evidence['has_occluding_surface'])


def prepare(selection, paths, output):
    import torch
    import trimesh
    import yaml
    from experiments.offline_camera_retrain_v1.data import PROJECTS
    from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
    from experiments.offline_camera_retrain_v1.ops.motion_viewer import reference_scene
    from experiments.offline_camera_retrain_v1.prepare_trumans_dynamic_v2 import load_objects
    from scenemi.training import ensure_bounded

    ensure_bounded()
    torch.set_num_threads(1)
    policy = json.loads(selection.read_text())
    native_path = Path(yaml.safe_load(paths.read_text())['native_manifest'])
    rows = {r['sequence_id']: r for r in map(json.loads, native_path.read_text().splitlines())
            if r.get('dataset') == 'trumans' and r['split'] == 'validation'}
    root = PROJECTS/'TRUMANS'
    clips = {r['clip_name']: r for r in map(json.loads,
        (root/'processed/scene_expert_v1/clips.jsonl').read_text().splitlines())}
    model = load_model('smplx', 'male')
    output.mkdir(parents=True, exist_ok=True)
    cases, audit = [], []
    for chosen in policy['cases']:
        row, start, length = dict(rows[chosen['sequence_id']]), chosen['start'], chosen['frames']
        if not 0 <= start <= row['frames']-length or length < 2:
            raise ValueError('Preview window is outside source sequence')
        metadata_path = Path(row['path'])/'metadata.json'
        if {k: v for k, v in row.items() if k != 'path'} != json.loads(metadata_path.read_text()):
            raise ValueError('Source manifest and metadata disagree')
        identity = dict(row, mode='exo', start=start, origin=[0., 0., 0.], anchor=np.eye(3).tolist())
        q = np.asarray(readonly_array(row, 'source_frame_ids'))[start:start+length].astype(int)
        clip = clips[row['sequence_id']]
        source_fit = root/clip['smplx_global']
        with source_fit.open('rb') as stream:
            fit = pickle.load(stream)
        beta = np.asarray(fit['betas'], np.float32).reshape(-1)
        if np.any(beta[10:] != 0):
            raise ValueError('Unsupported extra nonzero native shape coefficients')
        beta = np.pad(beta[:10], (0, max(0, 10-len(beta[:10]))))
        if not np.allclose(beta, row['native_body']['betas']):
            raise ValueError('Native personal shape differs from exported source')
        params = {k: np.asarray(fit[k], np.float32)[q] for k in
            ('global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'expression', 'transl')}
        params.update(betas=np.broadcast_to(beta[:10], (length, 10)).copy(),
                      leye_pose=np.zeros((length, 3), np.float32), reye_pose=np.zeros((length, 3), np.float32))
        vertices, joints = [], []
        for lo in range(0, length, 16):
            with torch.inference_mode():
                body = model(**{k: torch.from_numpy(v[lo:lo+16].copy()) for k, v in params.items()})
            vertices.append(body.vertices.numpy()); joints.append(body.joints[:, :22].numpy())
        vertices, joints = np.concatenate(vertices), np.concatenate(joints)
        target = np.asarray(readonly_array(row, 'motion_world'))[start:start+length, -66:].reshape(length, 22, 3)*2
        error = float(np.abs(target-joints).max())
        if error > 1e-5:
            raise ValueError(f'Native mesh/exported joint alignment error: {error}')
        center = (joints.reshape(-1, 3).min(0)+joints.reshape(-1, 3).max(0))/2
        radius = max(1., float(np.linalg.norm(joints-center, axis=-1).max()))
        arrays = dict(vertices=vertices.astype(np.float32), joints=joints.astype(np.float32),
            faces=np.asarray(model.faces, np.int32), rest=np.asarray(readonly_array(row, 'rest')),
            center=center.astype(np.float32), radius=np.asarray(radius, np.float32))
        scene, scene_note = reference_scene(identity, center, radius)
        camera, camera_note = camera_layers(identity, length, frustum_depth_m=.8)
        current, current_note = instant_scene_layers(identity, length)
        arrays.update(scene, **camera, **current, **observations(identity, length))
        # Display-only union over this selected window; explicitly not a causal input map.
        union = np.concatenate([current[f'current_static_{i}'] for i in range(length)])
        if len(union):
            _, keep = np.unique(np.floor(union/.025).astype(np.int64), axis=0, return_index=True)
            union = union[np.sort(keep)]
        arrays['visible_static'] = union.astype(np.float32)
        released = np.asarray(readonly_array(row, 'trumans_released_camera_to_world_reference'))
        arrays['release_positions'] = released[start:start+length, :3, 3].astype(np.float32)
        object_notes = []
        saved_positions = np.asarray(readonly_array(row, 'object_positions'))[start:start+length]
        saved_rotations = np.asarray(readonly_array(row, 'object_rotations'))[start:start+length]
        saved_valid = np.asarray(readonly_array(row, 'object_valid'))[start:start+length]
        objects = load_objects(root, clip, q)
        if len(objects) != saved_positions.shape[1]:
            raise ValueError('Object ordering/count differs from source export')
        for i, (name, mesh_path, rotation, position, protocol, valid, _) in enumerate(objects):
            if (not np.allclose(position, saved_positions[:, i], atol=1e-5)
                    or not np.allclose(rotation, saved_rotations[:, i], atol=1e-5)
                    or not np.array_equal(valid, saved_valid[:, i])):
                raise ValueError('Dynamic object transform differs from export')
            mesh = trimesh.load(mesh_path, force='mesh', process=False)
            arrays[f'object_{i}_vertices'] = np.asarray(mesh.vertices, np.float32)
            arrays[f'object_{i}_faces'] = np.asarray(mesh.faces, np.int32)
            arrays[f'object_{i}_rotations'] = saved_rotations[:, i]
            arrays[f'object_{i}_positions'] = saved_positions[:, i]
            arrays[f'object_{i}_valid'] = saved_valid[:, i]
            object_notes.append(dict(name=name, mesh=str(mesh_path), sha256=sha256(mesh_path), protocol=protocol))
        exo = np.asarray(readonly_array(row, 'exo_sensor_to_world'))
        provided = arrays['observation_meta'][..., 0] > 0
        observation_summary = dict(mean_available_joints=float(provided.sum(-1).mean()),
            frames_with_no_joint=int((provided.sum(-1) == 0).sum()),
            valid_frames=int(np.asarray(readonly_array(row, 'observation_valid'))[start:start+length].sum()))
        for kind in ('reference', 'visible', 'person', 'static', 'rigid'):
            counts = np.asarray(readonly_array(row, 'exo_'+kind+'_counts'))[start:start+length]
            observation_summary[kind+'_pixel_count'] = int(counts.sum())
        audit_row = dict(sequence_id=row['sequence_id'], start=start, frames=length,
            source_frames=row['frames'], reason=chosen['reason'], scene_id=row['scene_id'],
            source_metadata_sha256=sha256(metadata_path), source_fit_sha256=sha256(source_fit),
            mesh_joint_max_error_m=error, fixed_camera_motion=camera_motion(exo),
            released_camera_full_motion=camera_motion(released),
            released_camera_window_motion=camera_motion(released[start:start+length]),
            observation_summary=observation_summary,
            release_reference_only=True, release_K_verified=False, release_optical_axes_verified=False,
            source_exo_input_kind=row['exo_input_kind'], dynamic_objects=object_notes,
            native_body_asset_sha256=sha256(ROOT/'body_models/smplx/SMPLX_MALE.npz'))
        metadata = dict(data_only=True, model_inference=False, full_sequence=False, identity=identity,
            source_frames=row['frames'], data_frames=length, fps=30, audit=audit_row,
            display_geometry=dict(scene=scene_note, camera=camera_note, current_scene=current_note),
            static_union_causal=False, static_union_scope='selected preview window only, display-only')
        arrays['metadata_json'] = np.asarray(json.dumps(metadata, ensure_ascii=False))
        cache = output/(row['sequence_id']+f'_{start:06d}.npz')
        with cache.with_suffix('.npz.tmp').open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        cache.with_suffix('.npz.tmp').replace(cache)
        cases.append(dict(label=row['sequence_id'], display_label=chosen['label'], dataset='trumans',
            sequence_id=row['sequence_id'], start=start, data_frames=length, regimes=['clean'],
            models=[dict(key='native_source', label='原始 SMPL-X 重建（未生成）', mode='exo',
                         case=dict(cache=str(cache.resolve()), cache_sha256=sha256(cache))) ]))
        audit.append(audit_row)
        print(json.dumps(audit_row, ensure_ascii=False), flush=True)
    atomic_json(output/'PREPARED_CASES.json', dict(schema=SCHEMA, data_only=True,
        model_inference=False, selection=str(selection.resolve()), selection_sha256=sha256(selection),
        cases=cases, implementation_sha256=sha256(__file__)))
    atomic_json(output/'AUDIT.json', dict(schema=SCHEMA, cases=audit, test_used=False, training_started=False))


@contextmanager
def source_labels():
    import viser
    api = viser.GuiApi
    original = {k: getattr(api, k) for k in ('add_markdown', 'add_checkbox', 'add_slider')}
    def markdown(self, content, *args, **kwargs):
        if content.startswith('## 已保存动作'):
            content = ('## TRUMANS · EXO 输入检查\n\n紫色：原始 SMPL-X；绿/橙点：实际导出关节。'
                       '橙色视锥：本案例实际输入相机；青色轨迹：源拟合恢复的相机位置。'
                       '案例名称区分移动相机与旧固定对照。未运行生成模型。')
        elif content.startswith('输入参考人体是'):
            content = ('关节位置是按可见性遮罩提供的真值模拟观测，尚非真实 3D tracker。'
                       '完整静态场景和动态物体仅供检查；可见点并集包含窗口后续帧，不作为因果输入。'
                       '移动案例使用源外参、明确声明的合成内参与固定光学轴转换，逐帧重算遮挡与场景点；'
                       '尚非原 RGB 标定或真实 tracker。旧固定案例保留原输入。')
        return original['add_markdown'](self, content, *args, **kwargs)
    def checkbox(self, label, initial_value, *args, **kwargs):
        initial_value = {'GT 人体（蓝）': False, '实际输入关节点 + 置信颜色': True,
            '完整静态场景（GT参考）': True, '本录制可见静态并集（实际输入）': False,
            '当前帧其他人 / 物体': True, '输入相机轨迹': True}.get(label, initial_value)
        label = {'完整静态场景（GT参考）': '完整源场景（仅展示）',
            '本录制可见静态并集（实际输入）': '窗口可见点并集（含后续帧，仅展示）',
            '当前帧其他人 / 物体': '动态物体网格与可见点',
            '跟随 GT 人体': '跟随原始重建人体'}.get(label, label)
        return original['add_checkbox'](self, label, initial_value, *args, **kwargs)
    def slider(self, label, *args, **kwargs):
        label = {'帧 / 30 Hz': '窗口内帧 / 30 Hz', '预测人体不透明度': '原始重建人体不透明度'}.get(label, label)
        return original['add_slider'](self, label, *args, **kwargs)
    api.add_markdown, api.add_checkbox, api.add_slider = markdown, checkbox, slider
    try:
        yield
    finally:
        for name, method in original.items():
            setattr(api, name, method)


class NativeSourceViewer(MotionViewer):
    COLORS = dict(MotionViewer.COLORS, native_source=(185, 40, 160))

    def __init__(self, cases, port):
        self.show_release = None
        self.follow_observer = None
        with source_labels():
            super().__init__(cases, port)
        self.show_release = self.server.gui.add_checkbox('显示恢复的移动相机位置与轨迹（参考）', initial_value=True)
        self.show_release.on_update(lambda _: self.update())
        button = self.server.gui.add_button('从当前 EXO 相机看')
        button.on_click(lambda _: self.observer_view())
        self.follow_observer = self.server.gui.add_checkbox('播放时跟随 EXO 相机视角', initial_value=False)
        self.follow_observer.on_update(lambda _: self.update())

    def read_case(self, index):
        if index not in self.cache:
            entry = self.cases[index]['models'][0]
            with np.load(entry['case']['cache'], allow_pickle=False) as saved:
                data = {k: np.asarray(saved[k]) for k in saved.files}
            meta = json.loads(str(data['metadata_json']))
            for regime in ('clean',):
                data[regime+'_vertices'] = data[regime+'_reference_vertices'] = data['vertices']
                data[regime+'_joints'] = data[regime+'_reference_joints'] = data['joints']
                data[regime+'_observations'], data[regime+'_observation_meta'] = data['observations'], data['observation_meta']
            data['truth_vertices'], data['truth_joints'] = data['vertices'], data['joints']
            self.cache[index] = [dict(entry=entry, data=data, meta=meta)]
        self.cache.move_to_end(index)
        while len(self.cache) > 1:
            self.cache.popitem(last=False)
        return self.cache[index]

    def load_case(self):
        with self.lock, self.server.atomic():
            super().load_case()
            self.regime.visible = False
            for mode, panel in self.panels.items():
                for k in ('gt', 'reference', 'uncertainty', 'reference_source'):
                    panel[k].visible = False
                    if k != 'reference_source':
                        panel[k].value = False
                for button in self.preset_buttons[mode].values():
                    button.visible = False
            self.panels['ego']['status'].content = '此页仅检查 EXO 数据输入。'
            rt = self.runtime['exo'];data = rt['data']
            rt['released_position'] = self.points('/exo/released/position', (15, 180, 180), .07)
            rt['released_path'] = self.lines('/exo/released/path', self.segments(data['release_positions']), (15, 180, 180), 4.)
            rt['objects'] = [self.mesh('/exo/objects/'+str(i), data[f'object_{i}_vertices'],
                data[f'object_{i}_faces'], (190, 155, 90), .8) for i in range(len(rt['meta']['audit']['dynamic_objects']))]
            rt['scan'].opacity = .16
            self.update()

    def observer_view(self):
        rt = self.runtime['exo'];data = rt['data'];t = int(self.frame.value)
        for client in self.server.get_clients().values():
            client.camera.position = data['camera_positions'][t]
            corners = data['camera_frustum_segments'][t, :4, 1]
            client.camera.look_at = corners.mean(0)
            # EXO convention is +Y down. Source K controls the browser vertical FOV.
            identity = rt['meta']['identity'];intr = data.get('camera_intrinsics')
            if intr is None:
                intr = readonly_array(identity, 'exo_intrinsics')
            k = intr if intr.ndim == 1 else intr[identity['start']+t]
            if 'camera_rotations' in data:
                rotation = data['camera_rotations'][t]
            else:
                matrix = readonly_array(identity, 'exo_sensor_to_world')
                rotation = matrix[:3,:3] if matrix.ndim == 2 else matrix[identity['start']+t,:3,:3]
            client.camera.up_direction = tuple(-rotation[:, 1])
            client.camera.fov = float(2*np.arctan(.5/k[1]))

    def update(self):
        if self.loading or 'exo' not in self.runtime:
            return
        with self.lock, self.server.atomic():
            rt = self.runtime['exo'];data = rt['data'];panel = self.panels['exo']
            t = int(self.frame.value);enabled = bool(panel['enabled'].value)
            for k in ('gt', 'reference', 'uncertain_joints', 'uncertain_bones', 'gt_path', 'occluded_gt', 'occluded_reference'):
                rt[k].visible = False
            for handle in rt['gt_feet']:
                handle.visible = False
            meta = data['observation_meta'][t];active = meta[:, 0] > 0
            rt['input'].points = data['observations'][t, active, :3]*2
            rt['input'].colors = np.asarray([[40, 190, 90] if c >= .99 else [240, 160, 25] for c in meta[active, 2]], np.uint8).reshape(-1, 3)
            rt['input'].visible = enabled and bool(panel['observations'].value)
            for k in ('scan', 'static', 'camera_path'):
                rt[k].visible = enabled and bool(panel[k].value)
            for name, field in (('current', 'current_static_'), ('dynamic', 'current_dynamic_')):
                rt[name].points = data[field+str(t)];rt[name].visible = enabled and bool(panel[name].value)
            for name in ('camera', 'camera_axes', 'camera_position'):
                rt[name].visible = enabled and bool(panel['camera'].value)
            rt['camera'].points, rt['camera_axes'].points = data['camera_frustum_segments'][t], data['camera_axis_segments'][t]
            rt['camera_position'].points = data['camera_positions'][t:t+1]
            for item in rt['models']:
                visible = enabled and bool(panel['models'][item['entry']['key']].value)
                item['mesh'].vertices = data['vertices'][t];item['mesh'].opacity = float(self.opacity.value);item['mesh'].visible = visible
                item['path'].visible = visible and bool(panel['paths'].value)
                item['occluded'].points = data['joints'][t, data['source_occluded'][t]]
                item['occluded'].visible = visible and bool(panel['occluded'].value)
                for foot, handle in zip((10, 11), item['feet']):
                    handle.points = self.segments(data['joints'][max(0, t-29):t+1, foot]);handle.visible = visible and bool(panel['feet'].value)
            if 'released_position' in rt:
                show = enabled and (self.show_release is None or bool(self.show_release.value))
                rt['released_position'].points = data['release_positions'][t:t+1]
                rt['released_position'].visible = rt['released_path'].visible = show
            for i, handle in enumerate(rt.get('objects', [])):
                handle.visible = enabled and bool(panel['dynamic'].value) and bool(data[f'object_{i}_valid'][t])
                if handle.visible:
                    handle.vertices = data[f'object_{i}_vertices'] @ data[f'object_{i}_rotations'][t].T + data[f'object_{i}_positions'][t]
            a = rt['meta']['audit'];absolute = a['start']+t
            self.info.content = (f"**原始数据预览，未运行模型** · 原序列第 {absolute} 帧 / {a['source_frames']} 帧\n\n"
                f"当前窗口 {a['start']}–{a['start']+a['frames']-1} · 30 Hz。{a['reason']}")
            moving = rt['meta'].get('camera_mode') == 'moving'
            panel['status'].content = ('**移动 EXO · 源外参＋合成光学设置**' if moving else
                                       '**旧固定合成 EXO 观测**') + ' · 原生 SMPL-X 人体及源动态物体。'
            motion = a['moving_camera_motion'] if moving else a['fixed_camera_motion']
            panel['info'].content = (f"本帧提供 {int(active.sum())}/22 个关节。位置来自真值模拟 tracker，缺失位置为零。\n\n"
                f"本案例输入相机窗口最大位移 {motion['position_max_from_first_m']:.4f} m；"
                f"恢复的移动相机全序列最大位移 {a['released_camera_full_motion']['position_max_from_first_m']:.2f} m。\n\n"
                + ('本案例逐帧重算静态、动态物体及人体首命中。未核验原 RGB 内参，光学轴转换是明确的模拟假设。'
                   if moving else '青色轨迹仅作参考，未用于生成此旧固定案例的输入。')
                + '完整网格仅展示，不是模型输入。')
            if moving:
                panel['info'].content += (f"\n\n全窗口平均提供 {a['observation_summary']['mean_available_joints']:.2f}/22 个关节；"
                    f"旧固定对照为 {a['fixed_observation_summary']['mean_available_joints']:.2f}/22。")
            if a.get('observation_summary', {}).get('frames_with_no_joint') == a['frames']:
                panel['info'].content += (f"\n\n**本窗口 {a['frames']} 帧均没有关节输入。** "
                    '检查出画及首命中遮挡记录；该例保留作数据问题对照。')
            if self.follow.value:
                self.camera()
            if self.follow_observer is not None and self.follow_observer.value:
                self.observer_view()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    build = commands.add_parser('prepare')
    build.add_argument('--selection', type=Path, required=True)
    build.add_argument('--paths', type=Path, default=ROOT/'configs/scene_paths.local.yaml')
    build.add_argument('--output', type=Path, required=True)
    moving = commands.add_parser('move-camera', help='Recast moving EXO observations for verified source previews')
    moving.add_argument('--source-manifest', type=Path, required=True)
    moving.add_argument('--camera-config', type=Path, default=ROOT/'configs/trumans_exo_moving.yaml')
    moving.add_argument('--output', type=Path, required=True)
    serve = commands.add_parser('serve')
    serve.add_argument('--manifest', type=Path, required=True)
    serve.add_argument('--port', type=int, default=8844)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        return prepare(args.selection, args.paths, args.output)
    if args.action == 'move-camera':
        from .data.exo_camera import prepare as prepare_moving
        return prepare_moving(args.source_manifest, args.camera_config, args.output)
    manifest = json.loads(args.manifest.read_text())
    if manifest['schema'] != SCHEMA or manifest.get('model_inference') is not False:
        raise ValueError('Expected a native source-only preview manifest')
    for case in manifest['cases']:
        entry = case['models'][0]['case']
        if sha256(entry['cache']) != entry['cache_sha256']:
            raise ValueError('Preview cache content changed')
    viewer = NativeSourceViewer(manifest['cases'], args.port)
    atomic_json(args.manifest.parent/'SERVER.json', dict(pid=os.getpid(), url=f'http://localhost:{args.port}/',
        manifest=str(args.manifest.resolve()), bind='127.0.0.1', data_only=True, model_inference=False,
        case_count=len(manifest['cases']), camera_policy=manifest.get('camera_policy')))
    print(f'READY http://localhost:{args.port}/', flush=True)
    viewer.run()


if __name__ == '__main__':
    main()
