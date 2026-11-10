"""Random native AMASS windows and their exact stage-A observation streams.

Uses the shared MotionViewer. Ground and native meshes are review references;
the stage-A model receives neither. No training or inference is launched here.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path

import numpy as np
import torch

from .config import load_paths
from .data.amass_dataset import AmassDataset, INDEX_PROTOCOL
from .data.amass_prepare import atomic_json, sha256, source_indices
from .geometry.body import AMASS_ZUP_TO_YUP
from .observations import INPUT_KEYS
from experiments.offline_camera_retrain_v1.ops.motion_viewer import MotionViewer


def native_mesh(model, path, start, hands):
    """Exact native SMPL-H LBS with hand poses on the BODY source clock."""
    from smplx.lbs import lbs
    with np.load(path, allow_pickle=False) as saved:
        sl = slice(start+1, start+129)  # Same single 0b frame drop as training.
        pose = np.concatenate([saved['root_orient'][sl], saved['pose_body'][sl], hands], -1)
        beta, trans = saved['betas'][sl].copy(), saved['trans'][sl].copy()
    verts, joints = [], []
    with torch.inference_mode():
        for lo in range(0, 128, 16):
            v, j = lbs(torch.from_numpy(beta[lo:lo+16]), torch.from_numpy(pose[lo:lo+16]),
                model['template'], model['shape'], model['pose'], model['regressor'],
                model['parents'], model['weights'])
            verts.append(v.numpy()+trans[lo:lo+16, None])
            joints.append(j[:, :22].numpy()+trans[lo:lo+16, None])
    return np.concatenate(verts) @ AMASS_ZUP_TO_YUP.T, np.concatenate(joints) @ AMASS_ZUP_TO_YUP.T


def unanchor(points, identity):
    return np.asarray(points) @ np.asarray(identity['anchor']) + np.asarray(identity['origin'])


def prepare(paths, output, count=8, seed=2026102201):
    from .training import ensure_bounded
    ensure_bounded()
    torch.set_num_threads(1)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = paths['prepared_amass']/'manifest.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    families = [row for row in rows if row['state'] == 'prepared' and all(v['eligible_128'] for v in row['variants'])]
    model_path = paths['smplh_models']/'neutral/model.npz'
    # A review-only index cannot pass the formal training index requirement.
    index = dict(protocol=INDEX_PROTOCOL, complete=False, diagnostic_only=True,
        purpose='source_visualization_only', frames=128, fps=30,
        body_model_sha256=sha256(model_path), families=families)
    atomic_json(output/'review_index.json', index)
    dataset = AmassDataset(output/'review_index.json', paths, diagnostic=True)
    with np.load(model_path, allow_pickle=False) as z:
        parents = z['kintree_table'][0].astype(np.int64)
        parents[0] = -1
        model = dict(template=torch.from_numpy(z['v_template'].astype(np.float32)),
            shape=torch.from_numpy(z['shapedirs'].astype(np.float32)),
            pose=torch.from_numpy(z['posedirs'].reshape(-1, z['posedirs'].shape[-1]).T.copy().astype(np.float32)),
            regressor=torch.from_numpy(z['J_regressor'].astype(np.float32)),
            weights=torch.from_numpy(z['weights'].astype(np.float32)), parents=torch.from_numpy(parents))
        faces = z['f'].astype(np.int32)
    rng = np.random.default_rng(seed)
    corpora = sorted({row['corpus'] for row in dataset.members})
    if not 1 <= count <= len(corpora):
        raise ValueError('Choose one window from each of 1..13 distinct training corpora')
    chosen = rng.choice(corpora, size=count, replace=False).tolist()
    cases = []
    for ordinal, corpus in enumerate(chosen, 1):
        members = [i for i, row in enumerate(dataset.members) if row['corpus'] == corpus]
        mi = int(rng.choice(members))
        row = dataset.members[mi]
        reflected = bool(rng.integers(2))
        variant = next(v for v in row['variants'] if v['reflected'] == reflected)
        start = int(rng.integers(variant['frames_after_relative_pose_drop']-127))
        observation_seed = int(rng.integers(2**32))
        samples, identities = {}, {}
        for mode in ('ego', 'exo'):
            samples[mode], identities[mode] = dataset.sample(np.random.default_rng(observation_seed), mode,
                member=mi, start=start, reflected=reflected, yaw=0.)
        raw = paths['amass_raw']/row['source']
        if sha256(raw) != row['raw_sha256']:
            raise ValueError('Selected original AMASS source changed')
        body_clock = source_indices(row['raw_frames'], row['source_fps'])[1:][start:start+128]
        with np.load(raw, allow_pickle=False) as z:
            hands = z['poses'][body_clock, 66:156].astype(np.float32)
        # Upstream reflects body/root, but preserves raw hand poses. Match it.
        vertices, joints = native_mesh(model, paths['prepared_amass']/variant['path'], start, hands)
        for mode in ('ego', 'exo'):
            decoded = samples[mode]['motion'][:, 135:].reshape(128, 22, 3)*2
            error = float(np.linalg.norm(unanchor(decoded, identities[mode])-joints, axis=-1).max())
            if error > 2e-4:
                raise ValueError(f'Native mesh joints differ from actual {mode} training labels: {error}')
            identities[mode]['native_fk_error_m'] = error
        origin = np.array(identities['ego']['origin'])
        anchor = np.array(identities['ego']['anchor'])
        display = lambda points: ((points-origin) @ anchor.T).astype(np.float32)
        vertices, joints = display(vertices), display(joints)
        center = joints[:, 0].mean(0)
        half = max(3., float(np.linalg.norm(joints[:,0]-center, axis=-1).max())+1.)
        ground = np.array([[-half,0,-half],[-half,0,half],[half,0,half],[half,0,-half]],np.float32)
        ground[:,[0,2]] += center[[0,2]]
        arrays = dict(vertices=vertices, joints=joints, faces=faces,
            center=(center+[0., .4, 0.]).astype(np.float32), radius=np.float32(half),
            scene_vertices=ground, scene_faces=np.array([[0,1,2],[0,2,3]],np.int32),
            visible_static=np.empty((0,3),np.float32))
        for mode in ('ego', 'exo'):
            sample, identity = samples[mode], identities[mode]
            for key in INPUT_KEYS:
                arrays[f'{mode}_input__{key}'] = sample[key]
            obs = sample['trajectory'].copy()
            valid = sample['observation_meta'][..., 0] > 0
            obs[..., :3] = np.where(valid[...,None], display(unanchor(obs[...,:3]*2, identity))/2, 0.)
            arrays[f'{mode}__observations'] = obs
            arrays[f'{mode}__observation_meta'] = sample['observation_meta']
            if mode == 'ego':
                position = display(unanchor(sample['camera'][:, :3]*2, identity))
                # No intrinsics exist in AMASS: show measured axes, no invented FOV.
                six = sample['camera'][:,3:]
                right, up = six[:,:3], six[:,3:]
                rotation = np.stack((right, up, np.cross(right,up)), -1)
                ends = position[:,None] + rotation.swapaxes(-1,-2)*.3
                arrays.update(ego__camera_positions=position,
                    ego__camera_axis_segments=np.stack((np.broadcast_to(position[:,None],ends.shape),ends),2),
                    ego__camera_frustum_segments=np.empty((128,0,2,3),np.float32),
                    ego__camera_path=np.stack((position[:-1],position[1:]),1))
        cache = output/f'case_{ordinal:02d}.npz'
        with cache.open('wb') as stream:
            np.savez(stream, **arrays)
        case = dict(label=row['family'], dataset='amass', data_frames=128, start=start,
            display_label=f'{ordinal:02d} · {corpus} · {Path(row["source"]).stem} · {start}–{start+127}'
                + (' · 镜像' if reflected else ''),
            identity=identities, observation_seed=observation_seed, data_only=True,
            native_hand_pose='raw hands on verified body source clock; upstream body-only reflection; display only',
            models=[dict(key='amass_source',label='原始动作（未生成）',mode=mode,color=[190,55,195],
                case=dict(cache=str(cache),cache_sha256=sha256(cache))) for mode in ('ego','exo')])
        cases.append(case)
        print(json.dumps(dict(prepared=ordinal,total=count,source=row['source'],start=start)),flush=True)
    result = dict(protocol='amass-native-source-preview-v1', data_only=True, model_inference=False,
        seed=seed, method='Uniform distinct TRAIN corpora, then uniform family, original/reflected variant, and valid window. First draws retained.',
        source_manifest_sha256=sha256(manifest), formal_data_index_verified=(paths['output']/'index/complete.json').exists(),
        fps=30, frames=128, cases=cases)
    atomic_json(output/'PREPARED_CASES.json', result)
    return result


@contextmanager
def review_labels():
    import viser
    original = {key:getattr(viser.GuiApi,key) for key in ('add_markdown','add_checkbox','add_slider')}
    def markdown(api, content, *args, **kwargs):
        if content.startswith('## 已保存动作'):
            content = ('## AMASS · 训练数据检查\n\n紫色是原始 SMPL-H 动作，尚未训练生成模型。'
                '每例 128 帧、30 Hz；从 8 个不同训练子库随机抽取，未按视觉效果重选。'
                'EGO 默认显示相机位置和方向；展开 EXO 可查看同一窗口保留的 3D 关节。')
        elif content.startswith('输入参考人体是'):
            content = ('灰色地面仅供检查，不是场景输入。坐下或登高动作可能依赖 AMASS 未提供的物体。'
                'EGO 无人体关节输入；EXO 是真值关节的合成缺失，未模拟估计误差或真实遮挡。'
                '体形是训练标签，推理时由模型预测。')
        return original['add_markdown'](api,content,*args,**kwargs)
    def checkbox(api,label,initial_value,*args,**kwargs):
        label={'完整静态场景（GT参考）':'参考地面（不输入模型）',
            '实际输入关节点 + 置信颜色':'保留的 3D 关节',
            '输入相机位置 + 视锥':'输入相机位置和方向',
            '跟随 GT 人体':'跟随人体'}.get(label,label)
        return original['add_checkbox'](api,label,initial_value,*args,**kwargs)
    def slider(api,label,*args,**kwargs):
        label={'预测人体不透明度':'原始人体不透明度','帧 / 30 Hz':'窗口帧 / 30 Hz'}.get(label,label)
        return original['add_slider'](api,label,*args,**kwargs)
    viser.GuiApi.add_markdown,viser.GuiApi.add_checkbox,viser.GuiApi.add_slider=markdown,checkbox,slider
    try:
        yield
    finally:
        for key,value in original.items():
            setattr(viser.GuiApi,key,value)


class AmassPlayback(MotionViewer):
    """Display a shared source mesh with each observer's actual input stream."""

    def __init__(self, cases, port):
        with review_labels():
            super().__init__(cases, port)

    def read_case(self, index):
        if index not in self.cache:
            case = self.cases[index]
            with np.load(case['models'][0]['case']['cache']) as z:
                arrays = {key: z[key] for key in z.files}
            common = {key: value for key, value in arrays.items() if '__' not in key}
            common.update(truth_vertices=common['vertices'], truth_joints=common['joints'])
            for regime in ('clean', 'noisy'):
                for suffix in ('', 'reference_'):
                    common[f'{regime}_{suffix}vertices'] = common['vertices']
                    common[f'{regime}_{suffix}joints'] = common['joints']
            records = []
            for entry in case['models']:
                mode = entry['mode']
                observer = {key[len(mode)+2:]: value for key, value in arrays.items()
                            if key.startswith(mode+'__')}
                data = dict(common, **observer)
                for regime in ('clean', 'noisy'):
                    data[f'{regime}_observations'] = data['observations']
                    data[f'{regime}_observation_meta'] = data['observation_meta']
                records.append(dict(entry=entry, data=data,
                                    meta=dict(identity=case['identity'][mode])))
            self.cache.clear()
            self.cache[index] = records
        return self.cache[index]

    def load_case(self):
        super().load_case()
        self.regime.visible = False
        for mode, panel in self.panels.items():
            for key in ('gt', 'reference', 'uncertainty', 'reference_source',
                        'static', 'current', 'dynamic', 'occluded'):
                panel[key].visible = False
                if key != 'reference_source':
                    panel[key].value = False
            panel['enabled'].value = mode == 'ego'
            panel['scan'].value = True
            panel['observations'].value = mode == 'exo'
            for button in self.preset_buttons[mode].values():
                button.visible = False
            self.runtime[mode]['scan'].opacity = .5
        self.update()

    def update(self):
        if self.loading:
            return
        with self.lock, self.server.atomic():
            t = int(self.frame.value)
            for mode, rt in self.runtime.items():
                panel, data = self.panels[mode], rt['data']
                enabled = panel['enabled'].value
                for key in ('gt', 'reference', 'uncertain_joints', 'uncertain_bones',
                            'occluded_gt', 'occluded_reference', 'static', 'current',
                            'dynamic', 'gt_path'):
                    rt[key].visible = False
                for handle in rt['gt_feet']:
                    handle.visible = False
                rt['scan'].visible = bool(enabled and panel['scan'].value)
                valid = data['observation_meta'][t, :, 0] > 0
                rt['input'].points = data['observations'][t, valid, :3]*2
                rt['input'].visible = bool(enabled and panel['observations'].value)
                for name in ('camera', 'camera_axes', 'camera_position'):
                    rt[name].visible = bool(enabled and mode == 'ego' and panel['camera'].value)
                rt['camera_path'].visible = bool(enabled and mode == 'ego' and panel['camera_path'].value)
                if mode == 'ego':
                    rt['camera'].points = data['camera_frustum_segments'][t]
                    rt['camera_axes'].points = data['camera_axis_segments'][t]
                    colors = np.array([[220, 50, 50], [50, 190, 80], [50, 100, 230]], np.uint8)
                    rt['camera_axes'].colors = np.repeat(colors[:, None], 2, 1)
                    rt['camera_position'].points = data['camera_positions'][t:t+1]
                for item in rt['models']:
                    show = bool(enabled and panel['models'][item['entry']['key']].value)
                    item['mesh'].visible = show
                    item['mesh'].vertices = data['vertices'][t]
                    item['mesh'].opacity = float(self.opacity.value)
                    item['path'].visible = bool(show and panel['paths'].value)
                    for handle, foot in zip(item['feet'], (10, 11)):
                        handle.points = self.segments(data['joints'][max(0, t-30):t+1, foot])
                        handle.visible = bool(show and panel['feet'].value)
                    item['occluded'].visible = False
                panel['status'].content = f'**{mode.upper()} 输入检查** · 同一原始动作，未运行生成模型。'
                if mode == 'ego':
                    panel['info'].content = ('相机 6DoF 轨迹；没有关节输入。'
                        'AMASS 无图像内参，只显示方向，不绘制假设视锥。')
                else:
                    panel['info'].content = (f'本帧保留 {valid.sum()}/22 个 3D 关节；'
                        '位置与可用性掩码，无真值旋转输入。合成缺失，不等于真实估计器。')
            identity = self.cases[self.index]['identity']['ego']
            self.info.content = (f'**窗口第 {t} 帧 / 127 · {t/30:.2f} 秒**\n\n'
                f'源文件：`{identity["source"]}`\n\n'
                f'训练划分：**train**；30 Hz 起始 {identity["start"]}；'
                f'镜像：{identity["reflected"]}。源动作和体形均保留，未追加贴地或接触修正。手指只作显示，不属于模型目标。')
            if self.follow.value:
                self.camera()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    prep = actions.add_parser('prepare')
    prep.add_argument('--paths', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--count', type=int, default=8)
    prep.add_argument('--seed', type=int, default=2026102201)
    serve = actions.add_parser('serve')
    serve.add_argument('--manifest', type=Path, required=True)
    serve.add_argument('--port', type=int, default=8842)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        return prepare(load_paths(args.paths), args.output, args.count, args.seed)
    payload = json.loads(args.manifest.read_text())
    if payload['protocol'] != 'amass-native-source-preview-v1' or not payload['data_only']:
        raise ValueError('Expected a native source-only AMASS preview')
    for case in payload['cases']:
        for entry in case['models'][:1]:
            if sha256(entry['case']['cache']) != entry['case']['cache_sha256']:
                raise ValueError('Source preview cache changed')
    viewer = AmassPlayback(payload['cases'], args.port)
    atomic_json(args.manifest.parent/'SERVER.json', dict(pid=os.getpid(),
        url=f'http://localhost:{args.port}/', data_only=True, model_inference=False,
        formal_training_started=False, case_count=len(payload['cases'])))
    viewer.run()
