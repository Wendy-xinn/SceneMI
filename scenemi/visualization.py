"""Source-data playback adapters for the retained shared MotionViewer."""
import argparse
import json
from contextlib import contextmanager
import os
from pathlib import Path
import time

from experiments.offline_camera_retrain_v1.ops.grail_preview import (
    GrailSourceViewer, WARNING, _source_ui_labels, validate_prepared,
)
from experiments.offline_camera_retrain_v1.ops.motion_viewer import (
    MotionViewer, atomic_json, validate_prepared_cases,
)


@contextmanager
def source_labels():
    """Match the current shared UI while preserving source-only semantics."""
    import viser
    with _source_ui_labels():
        markdown = viser.GuiApi.add_markdown
        checkbox = viser.GuiApi.add_checkbox

        def add_markdown(api, content, *args, **kwargs):
            if content.startswith('## 已保存动作 · 模型对照'):
                content = ('## GRAIL · 原始动作与场景\n\n'
                           '紫色人体：数据集原始重建。EGO / EXO 显示同一完整动作的两种合成观测。'
                           '在“案例”中选择动作，勾选“播放”或拖动帧滑块查看。')
            return markdown(api, content, *args, **kwargs)

        def add_checkbox(api, label, initial_value, *args, **kwargs):
            if label == '完整静态场景（GT参考）':
                initial_value = True
            elif label == '本录制可见静态并集（实际输入）':
                initial_value = False
            return checkbox(api, label, initial_value, *args, **kwargs)

        viser.GuiApi.add_markdown = add_markdown
        viser.GuiApi.add_checkbox = add_checkbox
        try:
            yield
        finally:
            viser.GuiApi.add_markdown = markdown
            viser.GuiApi.add_checkbox = checkbox


class GrailPlayback(GrailSourceViewer):
    """Reuse source geometry/playback with the current per-panel button API."""
    def __init__(self, cases, port):
        self.full_recording = None
        with source_labels():
            MotionViewer.__init__(self, cases, port)
        if any('quality_review' in case for case in cases):
            self.server.gui.add_markdown('**EGO 筛选复核**：随机窗口及合成补椅子来源按案例标签区分；'
                '如有旧问题对照会单独标注。默认循环选定的 128 帧，帧号从 0 开始。')
            self.full_recording = self.server.gui.add_checkbox('播放完整原序列（含筛除段）', initial_value=False)
            self.full_recording.on_update(lambda _: self.apply_playback_limits())
            self.apply_playback_limits()

    def apply_playback_limits(self):
        with self.lock, self.server.atomic():
            case = self.cases[self.index]
            review = case.get('quality_review')
            if not review:
                return
            full = review['kind'] == 'excluded_comparison' or bool(
                self.full_recording is not None and self.full_recording.value)
            start = 0 if full else review['start']
            end = case['data_frames']-1 if full else start+127
            self.frame.min = 0
            self.frame.max = case['data_frames']-1
            if not start <= self.frame.value <= end:
                self.frame.value = start
            self.frame.min, self.frame.max = start, end
            if self.full_recording is not None:
                self.full_recording.disabled = review['kind'] == 'excluded_comparison'
            self.update()

    def load_case(self):
        with self.lock, self.server.atomic():
            self.frame.min = 0
            MotionViewer.load_case(self)
            self.regime.visible = False
            self.regime.disabled = True
            for panel in self.panels.values():
                for key in ('gt', 'reference', 'uncertainty', 'reference_source'):
                    panel[key].visible = False
                    panel[key].disabled = True
                panel['gt'].value = panel['reference'].value = panel['uncertainty'].value = False
            for group in self.preset_buttons.values():
                for button in group.values():
                    button.visible = False
                    button.disabled = True
            if 'quality_review' in self.cases[self.index]:
                self.panels['exo']['enabled'].value = False
                for runtime in self.runtime.values():
                    runtime['scan'].opacity = .65
                self.apply_playback_limits()
            self.update()

    def update(self):
        super().update()
        if self.loading:
            return
        review = self.cases[self.index].get('quality_review')
        if not review:
            return
        t = int(self.frame.value)
        quality = review['quality']
        covered = any(a <= t <= b+127 for a, b in quality['eligible_start_ranges'])
        flagged = any(a <= t <= b for a, b in quality['bad_frame_ranges'])
        status = '保留窗口覆盖' if covered else '不进入本次 EGO 训练窗口'
        if flagged:
            status += ' · 接触异常或事件缓冲区'
        note = review.get('note', '')
        self.info.content = (f'**EGO 筛选复核 · {status}**\n\n'
            f'原序列第 **{t}** 帧（从 0 开始） · 当前播放范围 **{int(self.frame.min)}–{int(self.frame.max)}**\n\n'
            + (f'随机保留窗口：**{review["start"]}–{review["start"]+127}**。\n\n'
               if review['kind'] == 'random_retained' else '')
            + f'{note}\n\n'
            '紫色为原始重建；场景使用原始坐标，未做贴地修正。完整场景仅供检查。'
            '通过筛查仍可能有规则未覆盖的小幅接触误差。')

    def run(self):
        while True:
            time.sleep(1/(30*float(self.speed.value)))
            if self.play.value and not self.loading:
                with self.lock:
                    lo, hi = int(self.frame.min), int(self.frame.max)
                    self.frame.value = lo+(int(self.frame.value)-lo+1) % (hi-lo+1)


def prepared_playback(manifest):
    """Select an existing playback adapter and verify its saved geometry."""
    from .data.amass_prepare import sha256
    from experiments.offline_camera_retrain_v1.ops.motion_viewer import SCHEMA
    value = json.loads(manifest.read_text())
    schema, protocol = value.get('schema'), value.get('protocol')
    if schema == SCHEMA:
        return validate_prepared_cases(value, manifest.parent), MotionViewer
    if schema == 'grail-source-preview-v1':
        return validate_prepared(manifest), GrailPlayback
    if schema == 'native-exo-source-preview-v1':
        from .native_visualization import NativeSourceViewer
        viewer = NativeSourceViewer
    elif protocol == 'amass-native-source-preview-v1':
        from .amass_visualization import AmassPlayback
        viewer = AmassPlayback
    elif protocol == 'replay-prediction-preview-v1':
        from .replay.visualization import ReplayViewer
        viewer = ReplayViewer
    elif protocol == 'ego-prediction-preview-v1':
        viewer = ego_prediction_viewer()
    else:
        raise ValueError(f'Unsupported prepared playback schema/protocol: {schema!r}/{protocol!r}')
    if not value.get('cases'):
        raise ValueError('Prepared playback requires at least one case')
    for case in value['cases']:
        if not case.get('models'):
            raise ValueError('Prepared case has no model entries')
        for entry in case['models']:
            source = entry['case']
            path = Path(source['cache'])
            path = path if path.is_absolute() else manifest.parent / path
            if sha256(path) != source['cache_sha256']:
                raise ValueError('Prepared playback changed')
            source['cache'] = str(path.resolve())
    return value['cases'], viewer


def ego_prediction_viewer():
    """Reuse prediction playback without historical experiment labels."""
    from .replay.visualization import ReplayViewer

    class EgoPredictionViewer(ReplayViewer):
        def __init__(self, cases, port):
            import viser
            markdown = viser.GuiApi.add_markdown

            def add_markdown(api, content, *args, **kwargs):
                if content.startswith('输入参考人体是'):
                    inputs = ('相机轨迹与整段已观测场景输入模型（AMASS 无场景），离线非因果。'
                              if any(case.get('scene_fed_to_model') for case in cases)
                              else '仅相机轨迹输入模型，场景仅供显示。')
                    content = ('EGO：蓝色为 GT，紫色为预测。' + inputs +
                               '体型由模型预测。固定 CPU seed 预览与 CUDA 验证噪声不同。'
                               '坐标保持窗口原始归一化，无逐帧对齐或贴地修正。')
                return markdown(api, content, *args, **kwargs)

            viser.GuiApi.add_markdown = add_markdown
            try:
                MotionViewer.__init__(self, cases, port)
            finally:
                viser.GuiApi.add_markdown = markdown

    return EgoPredictionViewer


def prepare_ego_preview(checkpoint, output):
    """Export one fixed validation example per dataset from a saved EGO model."""
    import numpy as np
    import torch
    from .config import ROOT, load_paths
    from .data.amass_prepare import sha256
    from .data.ego import FourEgoData, collate, payload, SOURCE_ORDER
    from .checkpoint import load_trained_model
    from .geometry.native_shape import GENDERS
    from .training import deterministic, ensure_bounded
    from .replay.visualization import mesh, display_geometry
    from experiments.offline_camera_retrain_v1.native30_model import sample_native30
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d
    ensure_bounded()
    if output.exists():
        raise ValueError('Preview output already exists; choose a new directory')
    model, c, step = load_trained_model(checkpoint)
    use_scene = c['version'] == 'ego-scene-adaptation-v1'
    if use_scene:
        from .data.scene_ego import SceneEgoData as FourEgoData, collate, payload
    # Select before inference from the completed fixed validation, rather than
    # searching generated samples for visually pleasing cases.
    metrics = checkpoint.resolve().parent
    if metrics.name == 'candidates':
        metrics = metrics.parent
    if metrics.name == 'training':
        metrics = metrics.parent / 'selection'
    metrics = metrics / f'full_step_{step:06d}' / 'metrics.jsonl'
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line]
    selected = []
    for source in SOURCE_ORDER:
        group = sorted((row for row in rows if row['identity']['dataset'] == source),
                       key=lambda row: row['metrics']['mpjpe_mm'])
        if not group:
            raise ValueError(f'No saved validation windows for {source}')
        selected.append(group[(len(group) - 1) // 2])
    digest = sha256(checkpoint)
    output.mkdir(parents=True)
    atomic_json(output / 'CASES.json', dict(checkpoint_sha256=digest, checkpoint_step=step,
        metrics_sha256=sha256(metrics), cases=selected, test_used=False,
        scene_fed_to_model=use_scene,
        selection='Per dataset, median saved validation-window MPJPE; targeted review, not aggregate performance'))
    deterministic(c['validation']['seed'], 'cpu')
    paths = load_paths(ROOT / 'configs/paths.local.yaml')
    data = FourEgoData(c, paths, 'validation')
    deterministic(c['validation']['seed'], 'cpu')
    prepared, meshes = [], {}

    def amass_mesh(motion, rest, betas):
        from smplx.lbs import lbs
        if 'smplh' not in meshes:
            with np.load(paths['smplh_models'] / 'neutral/model.npz', allow_pickle=False) as z:
                parents = z['kintree_table'][0].astype(np.int64); parents[0] = -1
                meshes['smplh'] = dict(v_template=torch.tensor(z['v_template'], dtype=torch.float32),
                    shapedirs=torch.tensor(z['shapedirs'], dtype=torch.float32),
                    posedirs=torch.tensor(z['posedirs'].reshape(-1, z['posedirs'].shape[-1]).T.copy(), dtype=torch.float32),
                    J_regressor=torch.tensor(z['J_regressor'], dtype=torch.float32),
                    parents=torch.tensor(parents), lbs_weights=torch.tensor(z['weights'], dtype=torch.float32))
                meshes['smplh_faces'] = z['f'].astype(np.int32)
        native = meshes['smplh']
        shaped = native['v_template'] + torch.einsum('vci,i->vc', native['shapedirs'], torch.as_tensor(betas))
        native_pelvis = (native['J_regressor'] @ shaped)[0]
        pelvis_offset = torch.as_tensor(rest[0]) - native_pelvis
        local = rotation_from_6d(torch.as_tensor(motion[:, 3:135].reshape(128, 22, 6)))
        full_pose = torch.eye(3).expand(128, len(native['parents']), 3, 3).clone()
        full_pose[:, :22] = local
        vertices, joints = [], []
        for start in range(0, 128, 16):
            v, j = lbs(torch.as_tensor(betas)[None].expand(16, -1), full_pose[start:start+16],
                       pose2rot=False, **native)
            shift = torch.as_tensor(motion[start:start+16, :3]) * 2 + pelvis_offset
            vertices.append(v.numpy() + shift[:, None].numpy())
            joints.append(j[:, :22].numpy() + shift[:, None].numpy())
        vertices, joints = np.concatenate(vertices), np.concatenate(joints)
        error = float(np.abs(joints - forward_kinematics(torch.as_tensor(motion)[None],
                                                       torch.as_tensor(rest)[None])[0].numpy()).max())
        if error > 2e-4:
            raise ValueError(f'SMPL-H display mesh/FK mismatch: {error}')
        return vertices, meshes['smplh_faces'], joints, error

    with torch.inference_mode():
        for index, row in enumerate(selected):
            case = row['case']; source = case['dataset']
            sample, identity = data.sample_case(case)
            if identity != row['identity']:
                raise ValueError('Preview input differs from saved validation identity')
            batch = collate([sample], 'cpu')
            prediction = sample_native30(model, payload(batch), steps=c['validation']['ddim_steps'],
                                         seed=c['validation']['seed'], use_scene=use_scene)
            _, details = model(prediction, torch.zeros(1, dtype=torch.long), payload(batch),
                               use_scene=use_scene, return_details=True)
            rest = details['rest'][0].numpy(); motion = prediction[0].numpy()
            if source == 'amass':
                pv, faces, pj, pe = amass_mesh(motion, rest, details['predicted_betas'][0].numpy())
                tv, _, tj, te = amass_mesh(sample['motion'], sample['target_rest'], sample['target_betas'])
                geometry = dict(scene_vertices=np.empty((0, 3), np.float32),
                                scene_faces=np.empty((0, 3), np.int32), visible_static=np.empty((0, 3), np.float32))
            else:
                meta = data.native.selection_bundle.metadata(data.native.records[case['member']], 'validation')
                info = dict(meta['native_body'], gender=GENDERS[int(details['predicted_template'][0])],
                            betas=details['predicted_native_betas'][0].numpy().tolist())
                pv, faces, pj, pe = mesh(motion, rest, info, meshes)
                tv, truth_faces, tj, te = mesh(sample['motion'], sample['target_rest'], meta['native_body'], meshes)
                if not np.array_equal(faces, truth_faces):
                    raise ValueError('Preview body topologies disagree')
                geometry = display_geometry(data.native.selection_bundle, meta, dict(start=case['start']), tj, 'ego')
                if use_scene:
                    scene_sample, _ = data.native.sample(np.random.default_rng(c['validation']['seed']),
                        member=case['member'], start=case['start'], yaw=0.)
                    geometry['visible_static'] = scene_sample['local_history_points'][scene_sample['local_history_valid']]
            error = pj[:, 0] - tj[:, 0]
            measured = dict(world_mpjpe_mm=float(np.linalg.norm(pj - tj, axis=-1).mean() * 1000),
                root_mean_error_mm=float(np.linalg.norm(error, axis=-1).mean() * 1000),
                root_start_error_mm=float(np.linalg.norm(error[0]) * 1000),
                root_end_error_mm=float(np.linalg.norm(error[-1]) * 1000),
                root_error_vector_change_mm=float(np.linalg.norm(error[-1] - error[0]) * 1000),
                mesh_fk_max_error_m=max(pe, te))
            cp = sample['camera'][:, :3] * 2
            six = sample['camera'][:, 3:]
            rotation = np.stack((six[:, :3], six[:, 3:], np.cross(six[:, :3], six[:, 3:])), -1)
            ends = cp[:, None] + rotation.swapaxes(-1, -2) * .3
            artifact = output / f'case_{index + 1:02d}.npz'
            np.savez_compressed(artifact, truth_vertices=tv, truth_joints=tj, faces=faces,
                preview_vertices=pv, preview_joints=pj, preview_reference_vertices=pv, preview_reference_joints=pj,
                preview_observations=sample['trajectory'], preview_observation_meta=sample['observation_meta'],
                source_occluded=np.zeros((128, 22), bool), center=tj[:, 0].mean(0) + np.array([0, .4, 0]),
                radius=np.float32(max(2.6, np.linalg.norm(tj[:, 0] - tj[:, 0].mean(0), axis=-1).max() + 1)),
                camera_positions=cp, camera_axis_segments=np.stack((np.broadcast_to(cp[:, None], ends.shape), ends), 2),
                camera_frustum_segments=np.empty((128, 0, 2, 3), np.float32), camera_path=np.stack((cp[:-1], cp[1:]), 1),
                metadata_json=np.array(json.dumps(dict(identity=identity, metrics=measured, checkpoint_step=step,
                    checkpoint_sha256=digest, scene_fed_to_model=use_scene and source != 'amass', target_shape_used_for_prediction=False,
                    sampler='CPU DDIM with saved validation seed; differs from CUDA RNG', scene_errors=[]))), **geometry)
            label = f'{index + 1:02d} · EGO · {source} · step {step}'
            prepared.append(dict(label=label, display_label=label, regimes=['preview'], metrics=measured,
                scene_fed_to_model=use_scene and source != 'amass',
                selection_reason='验证窗口 MPJPE 中位数；固定 CPU seed 预览，不代表整体指标。',
                models=[dict(key='prediction', label=f"{'EGO + scene' if use_scene else 'EGO-only'} {step}", mode='ego',
                             case=dict(cache=str(artifact.resolve()), cache_sha256=sha256(artifact)))]))
            print(f'Prepared {index + 1}/4 · {source} · MPJPE {measured["world_mpjpe_mm"]:.2f} mm', flush=True)
    atomic_json(output / 'PREPARED_CASES.json', dict(protocol='ego-prediction-preview-v1', cases=prepared,
        checkpoint_step=step, checkpoint_sha256=digest, test_used=False, scene_fed_to_model=use_scene))
    return output / 'PREPARED_CASES.json'


def serve_prepared(argv):
    """Serve any prepared shared-viewer manifest without model inference."""
    parser = argparse.ArgumentParser(description='Shared Viser playback of prepared motion exports')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8840)
    parser.add_argument('--check-only', action='store_true', help='Validate assets without opening a server')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    manifest = args.manifest.resolve()
    cases, viewer_class = prepared_playback(manifest)
    if args.check_only:
        print(json.dumps(dict(state='passed', manifest=str(manifest), cases=len(cases), model_inference=False)))
        return
    import torch
    from .training import ensure_bounded
    ensure_bounded()
    torch.set_num_threads(1)
    viewer = viewer_class(cases, args.port)
    atomic_json(manifest.parent / 'SERVER.json', dict(pid=os.getpid(),
        url=f'http://localhost:{args.port}/', bind='127.0.0.1', manifest=str(manifest),
        cases=len(cases), cpu_only=True, model_inference=False))
    print(f'READY http://127.0.0.1:{args.port}/ · {len(cases)} cases', flush=True)
    viewer.run()


def grail_review_specs(annotations, include_excluded=False, blind=False):
    """Read review-only annotations; never change the selected dataset."""
    from .data.amass_prepare import sha256
    rows = [json.loads(line) for line in Path(annotations).read_text().splitlines() if line.strip()]
    specs = []
    for row in rows:
        if (not blind and not row.get('flags') and not row.get('review_control')) or (not include_excluded and not row['in_current_selection']):
            continue
        path = Path(row['path']) / 'metadata.json'
        if sha256(path) != row['source_metadata_sha256']:
            raise ValueError('Review source metadata changed: ' + row['sequence_id'])
        meta = json.loads(path.read_text())
        if meta['split'] not in ('train', 'validation') or not 0 <= row['start'] <= meta['frames'] - 128:
            raise ValueError('Review source split/window is invalid')
        use = {'train': '训练', 'validation': '验证'}.get(row['current_use'], '已排除对照')
        kind = '座面重叠线索' if row['category'] == 'sitting' else '脚底间隙线索'
        if row.get('review_control'):
            kind = '未标记对照'
        elif row.get('manual_only'):
            kind = '人工指出的问题'
        if row.get('scene_stage_withheld'):
            kind += ' · 场景暂不使用'
        label = f"源例{row['viewer_case_index']:03d} · {use} · {kind} · {row['sequence_id'][:65]}"
        if blind:
            label = f"盲审{len(specs) + 1:03d} · {row['sequence_id'][:75]}"
            row = dict(row, blind_review=True)
        specs.append(dict(row=dict(path=row['path'], metadata_sha256=row['source_metadata_sha256'],
            sequence_id=row['sequence_id'], split=row['split'], category=row['category']),
            status='blind_review' if blind else row['current_use'], start=row['start'], label=label, annotation=row))
    if not specs:
        raise ValueError('No annotated review cases in the requested selection')
    if blind:
        return specs
    # Exact terrain frame ranges first; seat summaries still need localization.
    return sorted(specs, key=lambda s: (
        s['status'] == 'not_selected_at_this_window', s['row']['category'] == 'sitting',
        not any(f['type'] == 'both_soles_clearance_over_5cm' for f in s['annotation']['flags']),
        s['annotation']['viewer_case_index']))


def grail_review_viewer():
    """Add annotation navigation to the existing source-surface viewer."""
    import numpy as np
    from .data.grail_alignment import AlignmentViewer
    from .data.grail_contact import ContactBody

    class GrailReviewViewer(AlignmentViewer):
        def __init__(self, specs, config, port):
            from .config import ROOT
            self.review_body = ContactBody(json.loads((ROOT / 'configs/grail_ego_contact.json').read_text()))
            self.review_ui_ready = False
            import viser
            markdown = viser.GuiApi.add_markdown
            def review_markdown(api, content, *args, **kwargs):
                if content.startswith('## GRAIL · 源人体与实际场景输入'):
                    content = '## GRAIL · 待复核案例\n\n紫色：原始人体皮肤；灰色：原始完整场景参考。红点是检查线索，尚未确认错误。'
                    if specs[0]['annotation'].get('blind_review'):
                        content = '## GRAIL · 原始动作盲审\n\n紫色：原始人体皮肤；灰色：原始场景。隐藏旧筛选结论和自动标记，请直接判断动作是否可用。'
                elif content.startswith('青色为整条原序列实际观测到的静态点云'):
                    content = '本页只检查源 GT，不展示模型预测。源坐标和动作未修正；案例标记不会改变数据选择或训练。'
                return markdown(api, content, *args, **kwargs)
            viser.GuiApi.add_markdown = review_markdown
            try:
                super().__init__(specs, config, port)
            finally:
                viser.GuiApi.add_markdown = markdown
            self.synthetic_control.value = False
            self.synthetic_control.label = '补椅点（训练增强，非原始椅子）'
            self.opacity.label = '人体皮肤不透明度'
            self.review_marks = self.server.gui.add_checkbox('显示红色待复核皮肤点', initial_value=True)
            self.review_next = self.server.gui.add_button('跳到下一段标记')
            self.review_zoom = self.server.gui.add_button('近看脚底 / 座面')
            self.review_next.on_click(lambda _: self.next_mark())
            self.review_zoom.on_click(lambda _: self.zoom_review())
            self.review_marks.on_update(lambda _: self.update())
            self.review_ui_ready = True
            if specs[0]['annotation'].get('blind_review'):
                self.review_marks.value = False
                self.review_marks.visible = self.review_next.visible = self.synthetic_control.visible = False
            self.update()

        def read_case(self, index):
            items = super().read_case(index)
            data = items[0]['data']
            if 'review_mask' in data:
                return items
            spec = self.specs[index]
            mask = np.zeros(128, bool); preferred = []
            if (spec['annotation'].get('blind_review') or spec['annotation'].get('review_control')
                    or spec['annotation'].get('manual_only')):
                data.update(review_mask=mask, review_focus=0)
                return items
            explicit_frames = False
            for flag in spec['annotation']['flags']:
                for lo, hi in flag.get('source_frame_ranges_inclusive') or []:
                    a, b = max(0, lo - spec['start']), min(127, hi - spec['start'])
                    if a <= b:
                        explicit_frames = True
                        mask[a:b + 1] = True
                        if flag['type'] == 'both_soles_clearance_over_5cm':
                            preferred.append((a + b) // 2)
            if spec['row']['category'] == 'sitting':
                import open3d as o3d
                from .data.grail_contact import sustained
                from experiments.offline_camera_retrain_v1.scene_visibility_v2 import static_ray_scene
                vertices, faces = data['scene_vertices'], data['scene_faces']
                triangles = vertices[faces]
                normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                horizontal = np.abs(normal[:, 1]) >= .7 * np.maximum(np.linalg.norm(normal, axis=-1), 1e-12)
                support = horizontal & (triangles[..., 1].mean(-1) > .15)
                gap = np.full(128, np.nan)
                if support.any():
                    scene = static_ray_scene(vertices, faces[support])
                    skin = data['truth_vertices'][:, self.review_body.seat_ids]
                    closest = scene.compute_closest_points(o3d.core.Tensor(skin.copy()), nthreads=1)['points'].numpy()
                    valid = np.linalg.norm((skin - closest)[..., [0, 2]], axis=-1) <= .04
                    joints = data['truth_joints']
                    thigh = joints[:, [4, 5]] - joints[:, [1, 2]]
                    seated = ((np.abs(thigh[..., 1]) / np.maximum(np.linalg.norm(thigh, axis=-1), 1e-8) < .6).all(-1)
                              & (np.abs(np.gradient(joints[:, 0], axis=0)[:, 1] * 30) < .25))
                    stable = sustained(seated, 8)
                    for t in np.flatnonzero((valid.mean(-1) >= .1) & stable):
                        gap[t] = np.quantile((skin - closest)[t, valid[t], 1], .1)
                data['review_seat_gap_m'] = gap
                if not explicit_frames:
                    mask = np.isfinite(gap) & (gap < -.02)
                    preferred = [int(np.nanargmin(gap))] if np.isfinite(gap).any() else []
            indices = np.flatnonzero(mask)
            data['review_mask'] = mask
            data['review_focus'] = preferred[0] if preferred else (int(indices[0]) if len(indices) else 0)
            return items

        def load_case(self):
            super().load_case()
            panel = self.panels['ego']; panel['scan'].value = True
            panel['static'].value = panel['current'].value = False
            panel['camera'].value = panel['paths'].value = False
            self.panels['exo']['enabled'].value = False
            self.extras['review_skin'] = self.points('/ego/diagnostic/review_skin', (245, 40, 35), .009)
            self.frame.value = int(self.runtime['ego']['data']['review_focus'])
            self.update()

        def next_mark(self):
            with self.lock:
                indices = np.flatnonzero(self.runtime['ego']['data']['review_mask'])
                if len(indices):
                    starts = indices[np.r_[True, np.diff(indices) > 1]]
                    later = starts[starts > int(self.frame.value)]
                    self.frame.value = int(later[0] if len(later) else starts[0])

        def zoom_review(self):
            with self.lock:
                data = self.runtime['ego']['data']; t = int(self.frame.value)
                sitting = self.specs[self.index]['row']['category'] == 'sitting'
                ids = self.review_body.seat_ids if sitting else np.concatenate(self.review_body.foot_ids)
                center = data['truth_vertices'][t, ids].mean(0)
                position = center + np.asarray([.65, .35, .80])
                self.server.initial_camera.position = tuple(position)
                self.server.initial_camera.look_at = tuple(center)
                for client in self.server.get_clients().values():
                    client.camera.position, client.camera.look_at = tuple(position), tuple(center)

        def update(self):
            super().update()
            if self.loading or not self.runtime:
                return
            spec = self.specs[self.index]; annotation = spec['annotation']
            data = self.runtime['ego']['data']; t = int(self.frame.value)
            sitting = spec['row']['category'] == 'sitting'
            ids = self.review_body.seat_ids if sitting else np.concatenate(self.review_body.foot_ids)
            mark = self.extras.get('review_skin')
            flagged = bool(data['review_mask'][t])
            if mark is not None:
                mark.points = data['truth_vertices'][t, ids]
                mark.visible = flagged and (not self.review_ui_ready or self.review_marks.value)
            if annotation.get('blind_review'):
                self.panels['ego']['status'].content = '源 GT 盲审 · 隐藏原筛选结论及自动几何标记'
                self.panels['ego']['info'].content = f"录制：{annotation['sequence_id']}\n\n请直接观察原始动作与场景是否可用，不依据自动阈值判断。"
                self.info.content = (f"**{spec['label']}**\n\n窗口帧 **{t}** / 源帧 **{spec['start'] + t}**。\n\n"
                    '这是随机混合的原保留/排除案例。紫色为原始人体，灰色为原始场景；未修正坐标，也不影响训练。')
                return
            if annotation.get('review_control'):
                self.panels['ego']['status'].content = '未标记对照 · 未触发本次严重规则，仍可人工复核'
                self.panels['ego']['info'].content = f"录制：{annotation['sequence_id']}"
                self.info.content = (f"**源例 {annotation['viewer_case_index']:03d} · 未标记对照**\n\n"
                    f"窗口帧 **{t}** / 源帧 **{spec['start'] + t}**。\n\n"
                    '紫色为原始人体，灰色为原始场景；未修改数据，也没有模型预测。')
                return
            if annotation.get('manual_only'):
                self.panels['ego']['status'].content = '用户已指出问题 · 场景阶段暂时隔离 · 具体异常帧尚待定位'
                self.panels['ego']['info'].content = f"录制：{annotation['sequence_id']}\n\n此例按用户反馈保留，不冒充自动规则检出的结果。"
                self.info.content = (f"**源例 {annotation['viewer_case_index']:03d} · 人工标记**\n\n"
                    f"窗口帧 **{t}** / 源帧 **{spec['start'] + t}**。\n\n"
                    '紫色为原始人体，灰色为原始场景；可播放前后动作检查，原始数据没有修改。')
                return
            if sitting:
                value = data['review_seat_gap_m'][t]
                evidence = (f'当前臀部皮肤相对最近水平支撑面高度：**{value * 100:.2f} cm**。'
                            if np.isfinite(value) else '当前帧未找到足够的邻近水平支撑证据。')
                evidence += ' 原标记来自整录制统计；这里定位当前128帧的候选，未做完整身体体积穿透判定。'
            else:
                ranges = [f['source_frame_ranges_inclusive'] for f in annotation['flags']
                          if f.get('source_frame_ranges_inclusive')]
                evidence = f'原始标记帧区间（源帧，从0开始）：**{ranges}**。支撑/摆动尚待人工判断。'
            self.panels['ego']['status'].content = '源 GT 审阅 · 待复核线索 · 不代表已确认错误'
            self.panels['ego']['info'].content = f"录制：{annotation['sequence_id']}\n\n{evidence}"
            self.info.content = (f"**源例 {annotation['viewer_case_index']:03d} · {annotation['current_use']}**\n\n"
                f"窗口帧 **{t}** / 源帧 **{spec['start'] + t}**；当前帧{'有' if flagged else '无'}红色复核标记。\n\n"
                '紫色是原始人体皮肤，灰色是原始完整场景参考。红点仅用于定位待复核部位。'
                '没有模型预测、贴地或对齐修正；不改变数据选择或训练。')

    return GrailReviewViewer


def serve_grail_review(argv):
    from .config import ROOT
    parser = argparse.ArgumentParser(description='Review marked GRAIL source geometry without changing training')
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8095)
    parser.add_argument('--include-excluded', action='store_true')
    parser.add_argument('--blind', action='store_true', help='Hide old selection decisions and geometric flags')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    specs = grail_review_specs(args.annotations, args.include_excluded, args.blind)
    if args.check_only:
        print(json.dumps(dict(state='passed', cases=len(specs), data_only=True)))
        return
    import torch
    import yaml
    import numpy as np
    from .training import ensure_bounded
    ensure_bounded(); torch.set_num_threads(2)
    config = yaml.safe_load((ROOT / 'configs/ego_scene_adaptation.yaml').read_text())
    viewer = grail_review_viewer()(specs, config, args.port)
    checks = []
    for index, spec in enumerate(specs):
        data = viewer.read_case(index)[0]['data']
        gap = data.get('review_seat_gap_m')
        checks.append(dict(sequence_id=spec['row']['sequence_id'],
            source_case_index=spec['annotation']['viewer_case_index'], current_use=spec['status'],
            source_frame_range_inclusive=[spec['start'], spec['start'] + 127],
            focus_source_frame=spec['start'] + int(data['review_focus']),
            marked_frames=int(data['review_mask'].sum()), source_body_fk_verified=True,
            seat_gap_min_m=float(gap[np.isfinite(gap)].min()) if gap is not None and np.isfinite(gap).any() else None))
        print(f'Checked source geometry {index + 1}/{len(specs)}', flush=True)
    atomic_json(args.annotations.parent / 'VIEWER_CHECK.json', dict(state='passed',
        cases=checks, source_arrays_modified=False, model_inference=False, dataset_selection_changed=False))
    record = dict(pid=os.getpid(), url=f'http://localhost:{args.port}/', cases=len(specs),
                  annotations=str(args.annotations.resolve()), data_only=True, model_inference=False,
                  dataset_changed=False, training_changed=False, blind_review=args.blind)
    atomic_json(args.annotations.parent / 'VIEWER_SERVER.json', record)
    print(f'READY http://localhost:{args.port}/ · {len(specs)} review cases', flush=True)
    viewer.run()


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'prepare':
        parser = argparse.ArgumentParser(description='Export four EGO checkpoint validation examples on CPU')
        parser.add_argument('--checkpoint', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(argv[1:])
        print(prepare_ego_preview(args.checkpoint.resolve(), args.output.resolve()))
        return
    if argv and argv[0] == 'serve':
        return serve_prepared(argv[1:])
    if argv and argv[0] == 'grail-review':
        return serve_grail_review(argv[1:])
    if argv and argv[0] == 'replay':
        from .replay.visualization import main as replay_main
        return replay_main(argv[1:])
    if argv and argv[0] == 'trumans':
        from .native_visualization import main as native_main
        return native_main(argv[1:])
    if argv and argv[0] == 'amass':
        from .amass_visualization import main as amass_main
        return amass_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'EGO prediction export: view prepare --checkpoint MODEL.pt --output NEW_DIRECTORY. '
        'Shared playback: view serve --manifest PREPARED_CASES.json --port 8840. '
        'Other adapters: view {replay,trumans,amass} --help.'))
    parser.add_argument('dataset', choices=('grail',))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8840)
    args = parser.parse_args(argv)
    manifest = args.manifest.resolve()
    cases = validate_prepared(manifest)
    for case in cases:
        review = case.get('quality_review')
        if review and review['kind'] != 'excluded_comparison':
            start, quality = review['start'], review['quality']
            if (not 0 <= start <= case['data_frames']-128
                    or not any(a <= start <= b for a, b in quality['eligible_start_ranges'])
                    or any(a <= start+127 and b >= start for a, b in quality['bad_frame_ranges'])):
                raise ValueError('Preview window does not pass the recorded EGO selection')
    viewer = GrailPlayback(cases, args.port)
    atomic_json(manifest.parent/'SERVER.json', dict(pid=os.getpid(), url=f'http://localhost:{args.port}/',
        bind='127.0.0.1', manifest=str(manifest), data_only=True, warning=WARNING,
        model_inference=False, case_count=len(viewer.cases), default_scene='complete_source_scene'))
    viewer.run()


if __name__ == '__main__':
    main()
