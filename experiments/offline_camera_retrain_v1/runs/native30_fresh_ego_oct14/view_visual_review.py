"""Loopback-only native mesh playback of archived VAL predictions."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import time
import numpy as np
import viser


def main(folder, port):
    audit = json.loads((folder / 'VISUALIZATION_AUDIT.json').read_text())
    cases = audit['cases']
    cache = OrderedDict()
    def load(case):
        path = folder / case['mesh_file']
        if path not in cache:
            with np.load(path) as saved:
                cache[path] = dict(saved)
        cache.move_to_end(path)
        while len(cache) > 2:
            cache.popitem(last=False)
        return cache[path]
    initial = next(i for i, c in enumerate(cases) if c['dataset'] == 'egobody' and c['panel'] == 'fixed')
    data = load(cases[initial])
    server = viser.ViserServer(host='127.0.0.1', port=port)
    server.scene.set_up_direction('+y')
    gt = server.scene.add_mesh_simple('/GT', data['truth_vertices'][96], data['faces'], color=(45,115,225), opacity=.45, side='double')
    pred = server.scene.add_mesh_simple('/prediction', data['clean_vertices'][96], data['faces'], color=(190,25,130), opacity=.65, side='double')
    static = server.scene.add_point_cloud('/observed_static', data['static'], colors=(155,164,175), point_size=.013)
    dynamic = server.scene.add_point_cloud('/observed_dynamic', np.zeros((1,3), np.float32), colors=(15,175,180), point_size=.02)
    observed = server.scene.add_point_cloud('/trusted_input', np.zeros((1,3), np.float32), colors=(245,155,20), point_size=.04)
    errors = server.scene.add_line_segments('/foot_errors', np.zeros((4,2,3), np.float32), colors=(235,80,50), thickness=2, thickness_units='screen')
    server.gui.add_markdown(f'## 第 {audit["step"]} 步真实 VAL\n\n蓝色＝GT，紫红＝预测，橙色＝可信输入。原生 SMPL/SMPL-X mesh；没有额外对齐、贴地或修正。\n\n目前训练尚未完成。本播放器不是最终 1 万步结果。')
    labels = [f'{c["dataset"]} / {c["panel"]} / {c["sequence_id"]}' for c in cases]
    choice = server.gui.add_dropdown('片段', options=labels, initial_value=labels[initial])
    noise = server.gui.add_dropdown('输入', options=('clean','noisy'), initial_value='clean')
    frame = server.gui.add_slider('帧 / 30Hz', min=0, max=191, step=1, initial_value=96)
    play = server.gui.add_checkbox('播放', initial_value=False)
    speed = server.gui.add_slider('倍速', min=.25, max=1., step=.25, initial_value=.5)
    show_gt = server.gui.add_checkbox('显示 GT', initial_value=True)
    show_pred = server.gui.add_checkbox('显示预测', initial_value=True)
    show_scene = server.gui.add_checkbox('显示已见场景', initial_value=True)
    show_error = server.gui.add_checkbox('显示足部误差连线', initial_value=True)
    wire = server.gui.add_checkbox('GT 线框', initial_value=False)
    opacity = server.gui.add_slider('预测透明度', min=.1, max=1., step=.05, initial_value=.65)
    reset = server.gui.add_button('重置视角')
    info = server.gui.add_markdown('')
    server.gui.add_markdown('包含全部 3 个 fixed 和 3 个 rotating ego 验证窗，可切换干净／带噪输入。192帧／6.4秒，DDIM20，scene ON。灰色为空间中已见的表面，空白区域仍未知；没有虚构地板。手／脸按22关节模型约定中性显示。')
    def camera():
        c = load(cases[labels.index(choice.value)])
        center = c['truth_joints'][:, 0].mean(0)
        position = center + np.array([2.5, 1.2, 2.7])
        server.initial_camera.position = tuple(position)
        server.initial_camera.look_at = tuple(center - [0,.12,0])
        for client in server.get_clients().values():
            client.camera.position = tuple(position)
            client.camera.look_at = tuple(center - [0,.12,0])
    def update(_=None):
        case = cases[labels.index(choice.value)]
        c, t, n = load(case), int(frame.value), noise.value
        truth, prediction = c['truth_joints'][t], c[n + '_joints'][t]
        with server.atomic():
            gt.vertices, gt.faces, gt.visible, gt.wireframe = c['truth_vertices'][t], c['faces'], show_gt.value, wire.value
            pred.vertices, pred.faces, pred.visible, pred.opacity = c[n + '_vertices'][t], c['faces'], show_pred.value, float(opacity.value)
            static.points, static.visible = c['static'], show_scene.value
            points = c[f'dynamic_{t}']
            dynamic.points, dynamic.visible = points if len(points) else np.zeros((1,3),np.float32), bool(show_scene.value and len(points))
            mask = c[n + '_observation_meta'][t,:,0] > 0
            points = c[n + '_observations'][t,mask,:3] * 2
            observed.points, observed.visible = points if len(points) else np.zeros((1,3),np.float32), bool(len(points))
            errors.points = np.stack((truth[[7,8,10,11]], prediction[[7,8,10,11]]), axis=1)
            errors.visible = show_error.value
            mm = float(np.linalg.norm(prediction-truth, axis=-1).mean()*1000)
            feet = float(np.linalg.norm(prediction[[7,8,10,11]]-truth[[7,8,10,11]], axis=-1).mean()*1000)
            metrics = case['metrics'][n]
            info.content = f'{case["dataset"]} / {case["panel"]} · source起点 {case["source_start_30fps"]}\n\n第{t}帧 / {t/30:.2f}秒 · 本帧全身 {mm:.1f}mm，踝脚 {feet:.1f}mm\n\n整窗全身 {metrics["mpjpe_mm"]["value"]:.1f}mm；GT支撑期脚速 {metrics["foot_speed_during_GT_stance_m_s"]["value"]:.3f}m/s。脚速低不等于落点正确。'
    for handle in (noise, frame, show_gt, show_pred, show_scene, show_error, wire, opacity):
        handle.on_update(update)
    choice.on_update(lambda _: (update(), camera()))
    reset.on_click(lambda _: camera())
    server.on_client_connect(lambda _: camera())
    update()
    camera()
    print(f'READY http://127.0.0.1:{port} / step {audit["step"]}', flush=True)
    while True:
        time.sleep(1 / (30 * float(speed.value)))
        if play.value:
            frame.value = (int(frame.value) + 1) % 192
            update()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--port', type=int, default=8788)
    args = p.parse_args()
    main(args.folder, args.port)
