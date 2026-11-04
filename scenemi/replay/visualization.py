"""Export fixed replay validation windows and serve the shared Viser viewer.

CPU preview noise is explicitly separate from the original CUDA validation
noise. No training, target-shape inference input, or display alignment is used.
"""
import argparse
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.checked_scene_dataset import SceneSelection
from scenemi.data.exo_initialization import repair_payload
from scenemi.data.native_repair import NativeRepairDataset
from scenemi.geometry.native_shape import GENDERS
from scenemi.geometry.temporal import NativeJointMotionPrior
from scenemi.native_joint_data import SceneFreeEgoDataset, ego_payload
from scenemi.native_joint_training import NativeMotionPrior
from scenemi.training import deterministic, ensure_bounded, now
from experiments.offline_camera_retrain_v1.data import anchor_rotation
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.ops.motion_viewer import MotionViewer, reference_scene
from scenemi.exo_denoise import sample_denoising
from .data import normalize

CONFIG = ROOT/'configs/scene_joint128_replay.yaml'


def select_cases(run, output):
    """Choose cases from preexisting metrics, before generating any previews."""
    manifest = json.loads((run/'VALIDATION_CASES.json').read_text())
    cases = []
    for source in ('egobody', 'trumans', 'grail'):
        for mode, condition, quantiles in [('ego', 'ego', [.5, .9]), ('exo', 'gap', [.9])]:
            file = run/'selection/step_002000'/f'{mode}_{condition}_{source}.json'
            shard = json.loads(file.read_text())
            if shard['identity']['checkpoint_sha256'] != json.loads(
                    (run/'selection/step_002000/RESULT.json').read_text())['checkpoint_sha256']:
                raise ValueError('Selection metrics belong to different weights')
            rows = sorted((r for r in shard['rows'] if r['variant'] == 'repair'),
                          key=lambda r:r['metrics']['root_error_mm'])
            for q in quantiles:
                row = rows[round((len(rows)-1)*q)]
                identity = row['identity']
                matching = [c for c in manifest['cases'] if c['mode'] == mode
                    and c['source'] == source and c.get('path') == identity['path']
                    and c['start'] == identity['start']]
                if len(matching) != 1:
                    raise ValueError('Expected one pinned validation case')
                cases.append(dict(case=matching[0], condition=condition, root_error_quantile=q,
                    original_metrics=row['metrics'], original_identity=identity,
                    metrics_sha256=sha256(file)))
    result = dict(protocol='replay-prediction-preview-v1', checkpoint_step=2000,
        checkpoint_sha256=shard['identity']['checkpoint_sha256'],
        validation_cases_sha256=sha256(run/'VALIDATION_CASES.json'), test_used=False,
        selection='Per dataset: EGO root-error window quantiles 0.5/0.9; EXO gap quantile 0.9. '
                  'Targeted review; neither random nor representative of family-macro averages.',
        sampler='CPU DDIM seed 777; CPU RNG differs from the original CUDA validation noise',
        cases=cases)
    target = output/'CASES.json'
    if target.exists() and json.loads(target.read_text()) != result:
        raise ValueError('Existing preview selection differs')
    atomic_json(target, result)
    return result


def load_model(run, config):
    path = run/'candidates/step_002000.pt'
    expected = json.loads((run/'selection/step_002000/RESULT.json').read_text())['checkpoint_sha256']
    if sha256(path) != expected:
        raise ValueError('Replay checkpoint changed')
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if state['step'] != 2000 or state['contract']['config'] != config:
        raise ValueError('Unexpected replay checkpoint/configuration')
    weights = state['model']
    bases = {kind:{key:weights[f'base.native_body.bank.{kind}_{key}']
        for key in ('rest_template', 'rest_directions', 'eye_template', 'eye_directions')}
        for kind in ('smpl', 'smplx', 'grail_g1')}
    paths = load_paths(ROOT/'configs/paths.local.yaml')
    sp = yaml.safe_load((ROOT/'configs/scene_paths.local.yaml').read_text())
    mount = json.loads(Path(sp['pv_mount']).read_text())['camera_wearer']
    base = NativeMotionPrior(paths['smplh_models']/'neutral/model.npz', mount, bases=bases)
    model = NativeJointMotionPrior(base, **config['model'])
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(False).eval()
    return model


def sample_case(model, bundle, chosen, config):
    case = chosen['case']; mode = case['mode']
    cls = SceneFreeEgoDataset if mode == 'ego' else NativeRepairDataset
    dataset = cls.__new__(cls)
    dataset.split = 'validation'
    dataset.common_pelvis = model.base.shape.rest_template[0].numpy()
    dataset.bundle = dataset.selection_bundle = bundle
    dataset.array = bundle.array
    record = dict(path=case['path'], dataset=case['source'], family=case['family'],
                  starts=np.array([case['start']]), metadata_sha256=case['metadata_sha256'])
    dataset.records = {case['member']:record}
    if mode == 'exo':
        dataset.bank = copy.deepcopy(model.base.native_body.bank).cpu().eval()
    rng = np.random.default_rng(777+case['case_id'])
    kw = dict(member=case['member'], start=case['start'], yaw=0.)
    pair = (dataset.sample(rng, **kw) if mode == 'ego' else
            dataset.sample(rng, config['frontend'], condition=chosen['condition'], **kw))
    sample, identity = normalize(*pair, case['source'], mode)
    expected = chosen['original_identity']
    # Scene integration added display transforms after the replay validation.
    # Every originally recorded field must still match exactly.
    if (any(identity.get(k) != v for k, v in expected.items())
            or set(identity)-set(expected)-{'origin', 'anchor'}):
        differences = {k:dict(expected=expected.get(k), actual=identity.get(k))
                       for k in set(expected)|set(identity) if expected.get(k) != identity.get(k)}
        raise ValueError('Preview observation identity differs: '+json.dumps(differences))
    return sample, identity, bundle.metadata(record, 'validation')


def body_model(info, cache):
    key = ('grail_g1', 'neutral') if info.get('template') == 'grail_g1_baked_v1' else (info['model'], info['gender'])
    if key not in cache:
        if key[0] == 'grail_g1':
            from experiments.offline_camera_retrain_v1.ops.grail_source import body_model_factory
            cache[key] = body_model_factory('g1')
        else:
            from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
            cache[key] = load_model(*key).cpu().eval()
    return cache[key]


def mesh(motion, rest, info, cache):
    """Preserve the common pelvis and the actual shape used by FK."""
    model = body_model(info, cache)
    beta = torch.tensor(info['betas'], dtype=torch.float32)
    native = model.v_template + torch.einsum('vci,i->vc', model.shapedirs[..., :10], beta)
    pelvis = (model.J_regressor @ native)[0].numpy()*info['scale']
    offset = np.asarray(rest)[0]-pelvis
    vertices, joints = [], []
    for start in range(0, 128, 16):
        v, faces, j = decode_native_mesh(motion[start:start+16], info, model)
        vertices.append(v+offset); joints.append(j+offset)
    vertices, joints = np.concatenate(vertices), np.concatenate(joints)
    expected = forward_kinematics(torch.as_tensor(motion)[None], torch.as_tensor(rest)[None])[0].numpy()
    error = float(np.abs(joints-expected).max())
    if error > 2e-4:
        raise ValueError(f'Display mesh/FK mismatch: {error}')
    return vertices, faces.astype(np.int32), joints, error


def display_geometry(bundle, meta, sample, truth, mode):
    span = slice(sample['start'], sample['start']+128)
    if mode == 'ego':
        rotation = anchor_rotation(np.asarray(bundle.array(meta, 'camera_rotation_scenemi_yup')[span]))
        point_name, owner_name = 'visible_frame_points_scenemi_yup', 'visible_frame_owner'
    else:
        sensor = np.asarray(bundle.array(meta, 'exo_sensor_to_world'))
        rotation = anchor_rotation(sensor[None, :3, :3] if sensor.ndim == 2 else sensor[span, :3, :3])
        point_name, owner_name = 'exo_frame_points_world', 'exo_frame_owner'
    world = np.asarray(bundle.array(meta, 'motion_world')[span, 135:]).reshape(128, 22, 3)*2
    shift = truth[0, 0]-world[0, 0]@rotation.T
    if np.abs(world@rotation.T+shift-truth).max() > 2e-4:
        raise ValueError('Source scene and displayed motion coordinates disagree')
    transform = lambda p:(np.asarray(p)@rotation.T+shift).astype(np.float32)
    points = np.asarray(bundle.array(meta, point_name)[span])
    owners = np.asarray(bundle.array(meta, owner_name)[span])
    visible = transform(points[(owners >= 0)&(owners < 100)&np.isfinite(points).all(-1)])
    visible = visible[np.linspace(0, len(visible)-1, min(len(visible), 40000)).astype(int)] if len(visible) else visible
    geometry = dict(visible_static=visible, scene_vertices=np.empty((0, 3), np.float32),
                    scene_faces=np.empty((0, 3), np.int32))
    if meta.get('dataset', meta.get('group')) == 'grail':
        vertices = [np.asarray(bundle.array(meta, 'scene_'+k+'_vertices')) for k in ('ground', 'object')]
        faces = [np.asarray(bundle.array(meta, 'scene_'+k+'_faces')) for k in ('ground', 'object')]
        geometry.update(scene_vertices=transform(np.concatenate(vertices)),
                        scene_faces=np.concatenate([faces[0], faces[1]+len(vertices[0])]).astype(np.int32))
    else:
        identity = dict(meta, origin=(-shift@rotation).tolist(), anchor=rotation.tolist())
        full, _ = reference_scene(identity, truth[:, 0].mean(0), 4.)
        geometry.update(full)
    return geometry


@torch.inference_mode()
def prepare(run, output):
    ensure_bounded(); deterministic(777, 'cpu'); torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    c = yaml.safe_load(CONFIG.read_text())
    selection = select_cases(run, output)
    check = json.loads((ROOT/'runs/data_inventory/CURRENT_SELECTION_CHECK.json').read_text())
    if check['state'] != 'passed' or check['selection_sha256'] != sha256(ROOT/c['selection']):
        raise ValueError('Current selection check required')
    model = load_model(run, c)
    bundle = SceneSelection(ROOT/c['selection'])
    prepared, models = [], {}
    for index, chosen in enumerate(selection['cases']):
        case = chosen['case']; mode = case['mode']; artifact = output/f'case_{index+1:02d}.npz'
        sample, identity, meta = sample_case(model, bundle, chosen, c)
        batch = collate([sample], 'cpu')
        truth = forward_kinematics(batch['motion'], batch['target_rest'])[0].numpy()
        if mode == 'ego':
            payload = ego_payload(batch)
            motion = sample_native30(model.base, payload, steps=c['validation']['ego_ddim_steps'], seed=777, use_scene=False)
            _, details = model.base(motion, torch.zeros(1, dtype=torch.long), payload, return_details=True)
            rest = details['rest'][0].numpy()
            info = dict(meta['native_body'], gender=GENDERS[int(details['predicted_template'][0])],
                        betas=details['predicted_native_betas'][0].numpy().tolist())
            reference, reference_rest, reference_info = motion[0].numpy(), rest, info
        else:
            motion = sample_denoising(model, repair_payload(batch), steps=c['diffusion']['ddim_steps'],
                                      start_timestep=c['diffusion']['start_timestep'], seed=777)
            rest = sample['initial_rest']
            # Reconstruct the exact noisy frontend beta draw, including sigma=0.
            rng = np.random.default_rng(777+case['case_id'])
            beta = np.asarray(meta['native_body']['betas']) + rng.normal(0, identity['synthetic_beta_sigma'], 10)
            info = dict(meta['native_body'], betas=beta.tolist())
            reference, reference_rest, reference_info = sample['initial_motion'], rest, info
        prediction = motion[0].numpy()
        pv, faces, pj, pe = mesh(prediction, rest, info, models)
        tv, tf, tj, te = mesh(sample['motion'], sample['target_rest'], meta['native_body'], models)
        rv, rf, rj, re = mesh(reference, reference_rest, reference_info, models)
        if not np.array_equal(faces, tf) or not np.array_equal(faces, rf):
            raise ValueError('Preview body topologies disagree')
        error = pj[:, 0]-tj[:, 0]
        metrics = dict(world_mpjpe_mm=float(np.linalg.norm(pj-tj, axis=-1).mean()*1000),
            root_mean_error_mm=float(np.linalg.norm(error, axis=-1).mean()*1000),
            root_start_error_mm=float(np.linalg.norm(error[0])*1000),
            root_end_error_mm=float(np.linalg.norm(error[-1])*1000),
            root_error_vector_change_mm=float(np.linalg.norm(error[-1]-error[0])*1000),
            mesh_fk_max_error_m=max(pe, te, re))
        geometry = display_geometry(bundle, meta, dict(start=case['start']), truth, mode)
        cp = sample['camera'][:, :3]*2
        six = sample['camera'][:, 3:]; rotation = np.stack((six[:, :3], six[:, 3:], np.cross(six[:, :3], six[:, 3:])), -1)
        ends = cp[:, None]+rotation.swapaxes(-1, -2)*.3
        arrays = dict(truth_vertices=tv, truth_joints=tj, faces=faces,
            preview_vertices=pv, preview_joints=pj, preview_reference_vertices=rv,
            preview_reference_joints=rj, preview_observations=sample['trajectory'],
            preview_observation_meta=sample['observation_meta'],
            source_occluded=(sample.get('joint_visible', np.ones((128, 22))) <= 0),
            center=(truth[:, 0].min(0)+truth[:, 0].max(0))/2+np.array([0, .4, 0]),
            radius=np.float32(max(2.6, np.linalg.norm(truth[:, 0]-truth[:, 0].mean(0), axis=-1).max()+1)),
            camera_positions=cp, camera_axis_segments=np.stack((np.broadcast_to(cp[:, None], ends.shape), ends), 2),
            camera_frustum_segments=np.empty((128, 0, 2, 3), np.float32),
            camera_path=np.stack((cp[:-1], cp[1:]), 1), **geometry)
        metadata = dict(identity=identity, metrics=metrics, checkpoint_sha256=selection['checkpoint_sha256'],
            original_metrics=chosen['original_metrics'], sampler=selection['sampler'], scene_fed_to_model=False,
            target_shape_used_for_prediction=False, scene_errors=[])
        arrays['metadata_json'] = np.array(json.dumps(metadata))
        with artifact.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        label = f'{index+1:02d} · {mode.upper()} · {case["source"]} · root P{int(chosen["root_error_quantile"]*100)} · {chosen["condition"]}'
        prepared.append(dict(label=label, display_label=label, regimes=['preview'],
            selection_reason=f'{Path(case["path"]).name} · frames {case["start"]}–{case["start"]+127}. '
                '按既有 CUDA validation 根误差选例；当前 CPU 固定 seed 重新采样，无逐帧对齐或贴地修正。',
            models=[dict(key='prediction', label='无场景 replay 2000', mode=mode,
                         case=dict(cache=str(artifact.resolve()), cache_sha256=sha256(artifact)))],
            metrics=metrics))
        atomic_json(output/'STATUS.json', dict(state='exporting', completed=index+1, total=len(selection['cases']),
                                               latest=metrics, updated_at_utc=now()))
        print(json.dumps(dict(prepared=index+1, label=label, metrics=metrics)), flush=True)
    atomic_json(output/'PREPARED_CASES.json', dict(protocol=selection['protocol'], cases=prepared,
        checkpoint_sha256=selection['checkpoint_sha256'], selection_sha256=sha256(output/'CASES.json'),
        source_sha256=sha256(Path(__file__)), test_used=False, scene_fed_to_model=False))
    atomic_json(output/'STATUS.json', dict(state='complete', completed=len(prepared), updated_at_utc=now()))


@contextmanager
def labels():
    import viser
    original = viser.GuiApi.add_markdown
    def markdown(api, content, *args, **kwargs):
        if content.startswith('输入参考人体是'):
            content = ('无场景 replay 第 2000 步。蓝色 GT；紫色预测；浅绿色 EXO 合成估计输入。'
                'EGO 只有相机输入。所有场景仅供参考，均未输入模型。'
                '显示保持同一全局坐标，无逐帧对齐、根居中、缩放或贴地修正。'
                '128 帧 / 30 Hz，不能代表长序列累计漂移。CPU 预览与原 CUDA validation 噪声不同。')
        return original(api, content, *args, **kwargs)
    viser.GuiApi.add_markdown = markdown
    try:
        yield
    finally:
        viser.GuiApi.add_markdown = original


class ReplayViewer(MotionViewer):
    def __init__(self, cases, port):
        with labels():
            super().__init__(cases, port)

    def load_case(self):
        super().load_case()
        for mode, rt in self.runtime.items():
            rt['root_error_line'] = self.lines('/'+mode+'/diagnosis/root_error',
                np.zeros((1, 2, 3), np.float32), (245, 145, 30), 3.)
        for mode, panel in self.panels.items():
            panel['scan'].value = True
            panel['static'].value = True
            for key in ('current', 'dynamic', 'occluded'):
                panel[key].value = False
                panel[key].visible = False
            panel['reference'].value = mode == 'exo'
            panel['reference'].disabled = mode == 'ego' or mode not in self.runtime
            panel['reference'].visible = mode == 'exo' and mode in self.runtime
            panel['static'].label = '窗口观测场景点（仅供参考）'
            for button in self.preset_buttons[mode].values():
                button.visible = False
        self.regime.visible = False
        self.update()

    def update(self):
        super().update()
        if self.loading:
            return
        for mode, rt in self.runtime.items():
            m = rt['meta']['metrics']
            t = int(self.frame.value)
            data = rt['data']
            delta = data['preview_joints'][t, 0]-data['truth_joints'][t, 0]
            available = bool((data['preview_observation_meta'][t, :, 0] > 0).any())
            if 'root_error_line' in rt:
                rt['root_error_line'].points = np.stack((data['truth_joints'][t, 0],
                    data['preview_joints'][t, 0]))[None]
                rt['root_error_line'].visible = bool(self.panels[mode]['enabled'].value
                    and self.panels[mode]['gt'].value
                    and self.panels[mode]['models']['prediction'].value)
            self.panels[mode]['info'].content = (
                f'当前第 **{t}** 帧：预测已保存；根误差 **{np.linalg.norm(delta)*1000:.1f} mm**，'
                f'高度偏差 **{delta[1]*1000:+.1f} mm**。橙线连接 GT 与预测根位置。\n\n'
                + ('当前 EXO 输入估计缺失；紫色仍是模型补全输出，浅绿是填充初始化。\n\n'
                   if mode == 'exo' and not available else '')
                +
                f'World MPJPE **{m["world_mpjpe_mm"]:.1f} mm**；根平均误差 **{m["root_mean_error_mm"]:.1f} mm**。\n\n'
                f'根首帧 / 末帧误差：**{m["root_start_error_mm"]:.1f} / {m["root_end_error_mm"]:.1f} mm**；'
                f'根误差向量首尾变化：**{m["root_error_vector_change_mm"]:.1f} mm**。\n\n'
                + ('EGO 无人体初始化；仅相机输入。' if mode == 'ego' else
                   '浅绿为合成估计输入；缺失段由已有估计构造，GT 未作为修复输入。')
                + '\n\n轨迹和误差保留全局位置；场景只用于显示。')


def audit(output):
    """Measure the unchanged playback, including tracked-only temporal errors."""
    ensure_bounded()
    manifest = json.loads((output/'PREPARED_CASES.json').read_text())
    rows = []
    for case in manifest['cases']:
        entry = case['models'][0]
        path = Path(entry['case']['cache'])
        digest = sha256(path)
        if digest != entry['case']['cache_sha256']:
            raise ValueError('Prepared playback changed')
        with np.load(path, allow_pickle=False) as data:
            truth, pred = data['truth_joints'], data['preview_joints']
            delta = pred[:, 0]-truth[:, 0]
            error = np.linalg.norm(delta, axis=-1)*1000
            vertices = data['preview_vertices']
            row = dict(label=case['label'], cache_sha256=digest, frames=len(pred),
                finite_prediction_frames=np.isfinite(vertices).all(axis=(1, 2)).sum().item(),
                root_first_mm=float(error[0]), root_last_mm=float(error[-1]),
                root_mean_mm=float(error.mean()), root_max_mm=float(error.max()),
                root_height_min_mm=float(delta[:, 1].min()*1000),
                root_height_max_mm=float(delta[:, 1].max()*1000),
                root_height_max_frame=int(delta[:, 1].argmax()),
                mesh_fk_max_error_m=json.loads(str(data['metadata_json']))['metrics']['mesh_fk_max_error_m'])
            if entry['mode'] == 'ego':
                camera = data['camera_positions']
                gt_mount = np.linalg.norm(truth[:, 15]-camera, axis=-1)
                pred_mount = np.linalg.norm(pred[:, 15]-camera, axis=-1)
                row.update(gt_head_camera_distance_range_m=[float(gt_mount.min()), float(gt_mount.max())],
                    prediction_head_camera_distance_range_m=[float(pred_mount.min()), float(pred_mount.max())])
            if entry['mode'] == 'exo':
                available = (data['preview_observation_meta'][..., 0] > 0).any(-1)
                edges = np.diff(np.pad((~available).astype(np.int8), (1, 1)))
                initial = data['preview_reference_joints']
                pair = available[1:] & available[:-1]
                triple = pair[1:] & pair[:-1]
                temporal = {}
                for name, points in [('input', initial), ('prediction', pred), ('gt', truth)]:
                    acceleration = np.linalg.norm(np.diff(points, n=2, axis=0)*900, axis=-1).mean(-1)
                    velocity_error = np.linalg.norm((np.diff(points, axis=0)-np.diff(truth, axis=0))*30000, axis=-1).mean(-1)
                    temporal[name] = dict(
                        tracked_mean_acceleration_m_s2=float(acceleration[triple].mean()) if triple.any() else None,
                        tracked_mean_velocity_error_mm_s=float(velocity_error[pair].mean()) if pair.any() else None)
                row.update(missing_intervals_half_open=list(map(list, zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))), tracked_frames=int(available.sum()),
                    missing_root_mean_mm=float(error[~available].mean()) if (~available).any() else None,
                    first_input_to_gt_root_mm=float(np.linalg.norm(initial[0, 0]-truth[0, 0])*1000),
                    first_prediction_to_input_root_mm=float(np.linalg.norm(pred[0, 0]-initial[0, 0])*1000),
                    temporal=temporal)
            rows.append(row)
    report = dict(protocol='replay-playback-audit-v1', checkpoint_sha256=manifest['checkpoint_sha256'],
        prepared_manifest_sha256=sha256(output/'PREPARED_CASES.json'),
        fps=30, coordinates='unchanged global metres, +Y up', new_inference=False,
        temporal_mask='All frames of each finite difference must have an actual frontend estimate; excludes outage boundaries.',
        cases=rows, updated_at_utc=now())
    atomic_json(output/'PLAYBACK_AUDIT.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'serve', 'audit'))
    p.add_argument('--run', type=Path, default=ROOT/'runs/scene_joint128_replay')
    p.add_argument('--output', type=Path)
    p.add_argument('--port', type=int, default=8842)
    p.add_argument('--initial-case', type=int, help='One-based playback case to open')
    a = p.parse_args(argv); output = (a.output or a.run/'visualization').resolve()
    if a.action == 'prepare':
        return prepare(a.run.resolve(), output)
    if a.action == 'audit':
        return audit(output)
    ensure_bounded(); torch.set_num_threads(1)
    manifest = json.loads((output/'PREPARED_CASES.json').read_text())
    for case in manifest['cases']:
        for entry in case['models']:
            if sha256(Path(entry['case']['cache'])) != entry['case']['cache_sha256']:
                raise ValueError('Prepared playback changed')
    viewer = ReplayViewer(manifest['cases'], a.port)
    if a.initial_case is not None:
        if not 1 <= a.initial_case <= len(manifest['cases']):
            p.error('--initial-case must identify an existing case')
        viewer.choice.value = manifest['cases'][a.initial_case-1]['label']
        viewer.load_case()
    atomic_json(output/'SERVER.json', dict(pid=os.getpid(), url=f'http://localhost:{a.port}/',
        bind='127.0.0.1', cases=len(manifest['cases']), model_inference=False,
        checkpoint_sha256=manifest['checkpoint_sha256'], updated_at_utc=now()))
    viewer.run()


if __name__ == '__main__':
    main()
