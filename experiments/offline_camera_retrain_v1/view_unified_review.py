"""Playback of all registered native mesh comparisons in their input frame."""
import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import viser

ROOT = Path(__file__).parent / 'runs/ego_exo_surface_continue_oct11/visual_review'
COLORS = {'gt': (55, 145, 245), 'original55k': (175, 175, 185),
          'unified': (45, 215, 100), 'no_scene': (245, 90, 65)}
LABELS = {'gt': 'GT 蓝色', 'original55k': '原 55k 灰色',
          'unified': '续训 3k 绿色', 'no_scene': '续训模型关闭全部场景 红色'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--port', type=int, default=8781)
    args = parser.parse_args()
    manifest = json.loads((args.root / 'manifest.json').read_text())
    assert manifest['status'] in ('exporting', 'completed')
    cases = {f"{c['number']:02d} · {c['group']} · {c['mode']} · {c['identity']['sequence_id']}": c
             for c in manifest['cases']}
    server = viser.ViserServer(host='0.0.0.0', port=args.port, label='SceneMI · 55k / 续训 3k 可视化验收')
    server.scene.set_up_direction('+y')
    server.gui.add_markdown('### 55k 与 ego/exo 续训 3k\n\n蓝：GT；灰：原 55k；绿：续训 3k；红：续训模型关闭全部场景。\n\n'
                            '18 个已登记开发案例全部保留；同输入、同噪声、20 步 DDIM。原 55k 的 exo 为零样本。'
                            '仅显示各模式实际输入的点云与轨迹，GT 只作显示参考。')
    select = server.gui.add_dropdown('案例', options=list(cases))
    frame = server.gui.add_slider('帧（20 Hz）', min=0, max=127, step=1, initial_value=0)
    play = server.gui.add_checkbox('播放', initial_value=False)
    speed = server.gui.add_dropdown('倍速', options=['0.25', '0.5', '1'], initial_value='0.5')
    bodies = {k: server.gui.add_checkbox(v, initial_value=k != 'no_scene') for k, v in LABELS.items()}
    layout = server.gui.add_dropdown('显示方式', options=['同坐标叠加', '并排比较'], initial_value='并排比较')
    scene_on = server.gui.add_checkbox('实际输入静态点云', initial_value=True)
    dynamic_on = server.gui.add_checkbox('当前帧可见动态点', initial_value=True)
    inputs_on = server.gui.add_checkbox('实际输入位置与轨迹 黄色', initial_value=True)
    camera_on = server.gui.add_checkbox('观测相机 紫色', initial_value=True)
    anchors_on = server.gui.add_checkbox('绿色身体的 67 个锚点', initial_value=False)
    opacity = server.gui.add_slider('人体透明度', min=.15, max=1., step=.05, initial_value=.9)
    reset = server.gui.add_button('重置观察视角')
    info = server.gui.add_markdown('')
    lock = threading.RLock()
    active = None
    geometry = None
    handles = {}

    def offset(label):
        return np.array([list(COLORS).index(label) * 2.5, 0., 0.], np.float32) if layout.value == '并排比较' else np.zeros(3, np.float32)

    def reset_camera(client):
        if geometry is None:
            return
        center = geometry['gt_vertices'][int(frame.value)].mean(0)
        if layout.value == '并排比较':
            enabled = [offset(label) for label in COLORS if bodies[label].value]
            center = center + (np.mean(enabled, axis=0) if enabled else 0.) + [.6, 0., 0.]
        client.camera.fov = np.deg2rad(50.)
        client.camera.position = tuple(center + ([0., 1., 7.] if layout.value == '并排比较' else [2.8, 1.2, 3.2]))
        client.camera.up_direction = (0., 1., 0.)
        client.camera.look_at = tuple(center)

    def update():
        nonlocal active, geometry, handles
        with lock, server.atomic():
            case = cases[select.value]
            t = int(frame.value)
            current = (select.value, layout.value)
            if active != current:
                if active is None or active[0] != select.value:
                    with np.load(args.root / case['file']) as saved:
                        geometry = {k: saved[k] for k in saved.files}
                active = current
                server.scene.reset()
                handles = {}
                for label, color in COLORS.items():
                    shift = offset(label)
                    handles[label] = server.scene.add_mesh_simple('/body/' + label, geometry[label + '_vertices'][t] + shift,
                                                                  geometry['faces'], color=color, opacity=opacity.value)
                    handles[label + '_scene'] = server.scene.add_point_cloud('/scene/' + label, geometry['static_points'] + shift,
                                                                            colors=(205, 210, 215), point_size=.004)
                    handles[label + '_input'] = server.scene.add_point_cloud('/inputs/' + label, np.zeros((0, 3), np.float32),
                                                                            colors=(255, 215, 30), point_size=.045)
                    handles[label + '_dynamic'] = server.scene.add_point_cloud('/dynamic/' + label, np.zeros((0, 3), np.float32),
                                                                              colors=(255, 150, 40), point_size=.02)
                    segments = []
                    meta = geometry['observation_meta'][..., 0] > 0
                    for joint in (15, 20, 21):
                        valid = meta[1:, joint] & meta[:-1, joint]
                        p = geometry['input_positions'][:, joint]
                        segments.append(np.stack((p[:-1][valid], p[1:][valid]), axis=1))
                    handles[label + '_path'] = server.scene.add_line_segments('/input_path/' + label, np.concatenate(segments) + shift,
                                                                             colors=(255, 215, 30), thickness=2, thickness_units='screen')
                handles['anchors'] = server.scene.add_point_cloud('/body_anchors', np.zeros((67, 3), np.float32),
                                                                  colors=(220, 50, 220), point_size=.027)
                handles['camera'] = server.scene.add_line_segments('/camera', np.zeros((8, 2, 3), np.float32),
                                                                    colors=(160, 70, 230), thickness=2, thickness_units='screen')
                center = geometry['gt_vertices'][t].mean(0)
                server.initial_camera.look_at = tuple(center)
                server.initial_camera.position = tuple(center + [2.8, 1.2, 3.2])
                for client in server.get_clients().values():
                    reset_camera(client)
            lo, hi = geometry['dynamic_offsets'][t:t + 2]
            observed = geometry['observation_meta'][t, :, 0] > 0
            for label in COLORS:
                shift = offset(label)
                handles[label].vertices = geometry[label + '_vertices'][t] + shift
                handles[label].visible = bodies[label].value
                handles[label].opacity = opacity.value
                context = bodies[label].value if layout.value == '并排比较' else label == 'gt'
                handles[label + '_scene'].visible = scene_on.value and context
                handles[label + '_input'].points = geometry['input_positions'][t, observed] + shift
                handles[label + '_input'].visible = inputs_on.value and context
                handles[label + '_path'].visible = inputs_on.value and context
                handles[label + '_dynamic'].points = geometry['dynamic_points'][lo:hi] + shift
                handles[label + '_dynamic'].visible = dynamic_on.value and context
            handles['anchors'].points = geometry['unified_vertices'][t, geometry['anchor_ids']] + offset('unified')
            handles['anchors'].visible = anchors_on.value and bodies['unified'].value
            camera = geometry['camera'][t]
            position = camera[:3] * 2
            x, y = camera[3:6], camera[6:9]
            rotation = np.stack((x, y, np.cross(x, y)), axis=1)
            corners = np.array([[-.3, -.2, .5], [.3, -.2, .5], [.3, .2, .5], [-.3, .2, .5]]) @ rotation.T + position
            handles['camera'].points = np.concatenate((np.stack((np.tile(position, (4, 1)), corners), 1),
                                                       np.stack((corners, np.roll(corners, -1, axis=0)), 1))).astype(np.float32)
            handles['camera'].visible = camera_on.value
            metrics = case['metrics']
            table = '| 模型 | 全局误差 mm | PA mm | 脚 marker 支撑期速度 mm/s |\n|---|---:|---:|---:|\n'
            for label in ('original55k', 'unified', 'no_scene'):
                m = metrics[label]
                table += f"| {LABELS[label]} | {m['world_unaligned_mpjpe_mm']:.1f} | {m['pa_mpjpe_mm']:.1f} | {m['native_foot_marker_gt_stance_speed_mm_s']:.1f} |\n"
            info.content = (f"帧 {t}/127 · {t / 20:.2f} 秒 · {case['mode']} · 位置输入 {int(observed.sum())} 个\n\n" + table +
                            '\n\n当前模型场景消融（同噪声）：\n\n' +
                            '\n\n'.join(f"{label}：全局误差 {metrics[label]['world_unaligned_mpjpe_mm']:.1f} mm，输出变化 {metrics[label]['output_change_from_full_mm']:.1f} mm"
                                          for label in ('global_off', 'attention_off')) +
                            '\n\n脚 marker 速度只是代理指标；尚未验收真实表面穿模。并排模式只为显示施加偏移，指标与输入不变。')

    @select.on_update
    def change_case(event):
        play.value = False
        frame.value = 0
        update()

    for control in [frame, layout, opacity, scene_on, dynamic_on, inputs_on, camera_on, anchors_on, *bodies.values()]:
        control.on_update(lambda event: update())

    @reset.on_click
    def reset_view(event):
        for client in server.get_clients().values():
            reset_camera(client)

    @server.on_client_connect
    def connect(client):
        reset_camera(client)

    update()
    print(f'READY http://127.0.0.1:{args.port}', flush=True)
    refreshed_at = time.monotonic()
    while True:
        time.sleep(.05 / float(speed.value))
        if play.value:
            frame.value = (int(frame.value) + 1) % 128
        if time.monotonic() - refreshed_at > 5:
            refreshed_at = time.monotonic()
            try:
                latest = json.loads((args.root / 'manifest.json').read_text())
            except json.JSONDecodeError:
                continue
            if len(latest['cases']) > len(cases):
                cases.update({f"{c['number']:02d} · {c['group']} · {c['mode']} · {c['identity']['sequence_id']}": c
                              for c in latest['cases']})
                select.options = tuple(cases)


if __name__ == '__main__':
    main()
