"""Overlay native GT, same-model forecast and state-relative refined forecast."""
import argparse,json,threading,time
from pathlib import Path
import numpy as np
import viser
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
ROOT=Path(__file__).parent/'runs/state_relative_spline_oct10/demo'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8780);parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--scene-refinement',type=Path);args=parser.parse_args()
    frame_count=48 if args.scene_refinement else 128
    mesh_keys=['gt','original','refined','scene_contact'] if args.scene_refinement else ['gt','official55k','official55k_joint','original','refined']
    colors={'gt':(40,135,245),'official55k':(245,145,30),'official55k_joint':(175,80,230),'original':(240,65,60),'refined':(155,80,215) if args.scene_refinement else (45,205,90),'scene_contact':(45,205,90)}
    labels={'gt':'GT蓝色','official55k':'55k原相机输入橙色','official55k_joint':'55k统一头部输入紫色','original':'微调固定历史红色','refined':'旧样条修正紫色' if args.scene_refinement else '样条修正绿色','scene_contact':'场景接触修正绿色'}
    scene_rows={r['index']:r for r in json.loads((args.scene_refinement/'rows.json').read_text())} if args.scene_refinement else {}
    manifest=json.loads((args.root/'manifest.json').read_text());cases={};models={}
    for case in manifest['cases']:
        arrays=dict(np.load(args.root/f"{case['index']}.npz"));body=case['identity']['native_body'];key=(body['model'],body['gender'])
        if key not in models:models[key]=load_model(*key)
        if args.scene_refinement:
            arrays['scene_contact_motion']=np.load(args.scene_refinement/f"{case['index']}_motions.npz")['scene_contact'];case['metrics']['scene_contact']={'short32':scene_rows[case['index']]['metrics']['scene_contact']}
        for label in mesh_keys:
            vertices,faces,joints=decode_native_mesh(arrays[label+'_motion'][:frame_count],body,models[key]);arrays[label+'_vertices']=(vertices@arrays['anchor_rotation']+arrays['camera'][0]).astype(np.float32);arrays['faces']=faces
        name=f"{case['index']} · {case['identity']['sequence_id']}";cases[name]=(case,arrays)
    server=viser.ViserServer(port=args.port,label='SceneMI · 状态相对生成');server.scene.set_up_direction('+y')
    if not args.scene_refinement:server.gui.add_markdown('### 状态衔接与软头部约束\n蓝色GT · 橙色55k原权重/原相机输入 · 紫色55k原权重/统一头部输入 · 红色微调模型/固定历史 · 绿色红色基础上样条修正。同一场景与种子；橙色使用原相机输入，紫红绿使用统一头部输入。橙紫没有身体历史输入，红绿前16帧固定实际历史代理。未来头部来自GT模拟，非真实视频效果；默认播放完整未来5.6秒，蓝GT与红参考显示，其他对照可手动开启。修正不读取未来身体或接触GT；动态物体只显示当前姿态。')
    if args.scene_refinement:server.gui.add_markdown('**场景接触开发诊断：仅未来1.6秒。** 蓝GT · 红微调固定历史 · 紫旧样条 · 绿场景+软接触。局部观察表面约束不是完整碰撞证书；GT本身与场景有厘米级几何冲突信号，当前不宣称通过物理验收。')
    select=server.gui.add_dropdown('例子',options=list(cases),initial_value=next(iter(cases)))
    play=server.gui.add_checkbox('播放',initial_value=True);speed=server.gui.add_slider('速度',min=.25,max=2,step=.25,initial_value=.75)
    range_select=server.gui.add_dropdown('播放范围',options=['短期预测16–47','含历史0–47'] if args.scene_refinement else ['短期预测16–47','完整预测16–127','含历史0–127'],initial_value='短期预测16–47' if args.scene_refinement else '完整预测16–127')
    frame=server.gui.add_slider('帧',min=0,max=frame_count-1,step=1,initial_value=16);opacity=server.gui.add_slider('人物透明度',min=.1,max=1,step=.05,initial_value=.55)
    checks={label:server.gui.add_checkbox(label,initial_value=(label not in ['55k原相机输入橙色','55k统一头部输入紫色','旧样条修正紫色','样条修正绿色','场景接触修正绿色'])) for label in [*[labels[key] for key in mesh_keys],'静态记忆','当前可见点','动态物体','相机与轨迹']}
    follow=server.gui.add_checkbox('观察相机视角',initial_value=False);reset=server.gui.add_button('外部视角');info=server.gui.add_markdown('')
    handles={};active=None;lock=threading.RLock()
    def external(client,g):
        center=g['gt_vertices'][int(frame.value)].mean(0);client.camera.position=tuple(center+[2,.8,2]);client.camera.look_at=tuple(center);client.camera.up_direction=(0,1,0)
    def update():
        nonlocal handles,active
        with lock,server.atomic():
            case,g=cases[select.value];t=int(frame.value)
            if active!=select.value:
                active=select.value;server.scene.reset();handles={}
                for key,color in [(key,colors[key]) for key in mesh_keys]:handles[key]=server.scene.add_mesh_simple('/'+key,g[key+'_vertices'][t],g['faces'],color=color,opacity=opacity.value)
                handles['static']=server.scene.add_point_cloud('/static',g['static_points'],colors=(160,165,160),point_size=.006)
                handles['visible']=server.scene.add_point_cloud('/visible',g['visible_points'][t],colors=(240,210,40),point_size=.012)
                handles['objects']=[server.scene.add_mesh_simple(f'/object/{i}',g[f'obj_{i}_v'],g[f'obj_{i}_f'],color=(240,165,45)) for i in range(len(case['objects']))]
                handles['frustum']=server.scene.add_line_segments('/camera',np.zeros((8,2,3),np.float32),colors=(180,45,215),thickness=2,thickness_units='screen')
                handles['path']=server.scene.add_line_segments('/head_camera_path',np.stack((g['camera'][:frame_count-1],g['camera'][1:frame_count]),1),colors=(180,45,215),thickness=2,thickness_units='screen')
                center=g['gt_vertices'][16].mean(0);server.initial_camera.look_at=tuple(center);server.initial_camera.position=tuple(center+[2,.8,2])
                for client in server.get_clients().values():external(client,g)
            for key,label in [(key,labels[key]) for key in mesh_keys]:handles[key].vertices=g[key+'_vertices'][t];handles[key].visible=checks[label].value;handles[key].opacity=opacity.value
            known=g['static_times']<=g['source_frames'][t];handles['static'].points=g['static_points'][known];handles['static'].visible=checks['静态记忆'].value
            valid=(g['visible_owners'][t]>=0)&(g['visible_owners'][t]<100);handles['visible'].points=g['visible_points'][t][valid];handles['visible'].visible=checks['当前可见点'].value
            for i,node in enumerate(handles['objects']):node.position=g[f'obj_{i}_t'][t];node.wxyz=np.roll(Rotation.from_matrix(g[f'obj_{i}_r'][t]).as_quat(),1);node.visible=checks['动态物体'].value
            pos=g['camera'][t];r=g['rotation'][t];corners=np.array([[-.3,-.2,.5],[.3,-.2,.5],[.3,.2,.5],[-.3,.2,.5]])@r.T+pos
            handles['frustum'].points=np.concatenate((np.stack((np.tile(pos,(4,1)),corners),1),np.stack((corners,np.roll(corners,-1,axis=0)),1))).astype(np.float32)
            handles['frustum'].visible=handles['path'].visible=checks['相机与轨迹'].value
            if follow.value:
                for client in server.get_clients().values():client.camera.position=tuple(pos);client.camera.look_at=tuple(pos+r[:,2]);client.camera.up_direction=tuple(r[:,1])
            horizon='short32' if args.scene_refinement or range_select.value=='短期预测16–47' else 'future112';old=case['metrics']['original'][horizon];new=case['metrics']['refined'][horizon];official=case['metrics']['official55k'][horizon]
            info.content=f"帧{t} · {(t-16)/20:.2f}s（预测起点=16）\n\n世界MPJPE：55k相机输入 {official['mpjpe_cm']:.2f} / 微调固定历史 {old['mpjpe_cm']:.2f} / 修正 {new['mpjpe_cm']:.2f} cm\n\nPA-MPJPE：微调固定历史{old['pa_mpjpe_mm']:.1f} / 新{new['pa_mpjpe_mm']:.1f} mm\n\n骨盆朝向误差：微调固定历史{old['pelvis_orientation_mean_deg']:.1f} / 新{new['pelvis_orientation_mean_deg']:.1f}°\n\nGT支撑期脚滑：微调固定历史{old['gt_stance_slide_cm_frame']:.3f} / 新{new['gt_stance_slide_cm_frame']:.3f} cm/帧\n\n固定失败例子850/906，用于观察；平均效果见完整开发/确认集报告。未证明mesh无碰撞或实时仿真跟踪。"
            if args.scene_refinement:
                current=case['metrics']['scene_contact']['short32'];info.content=f"未来1.6秒开发诊断，帧{t}。\n\n世界MPJPE：旧样条 {new['mpjpe_cm']:.2f} / 场景接触 {current['mpjpe_cm']:.2f} cm\n\n脚滑：旧 {new['gt_stance_slide_cm_frame']:.3f} / 场景接触 {current['gt_stance_slide_cm_frame']:.3f} cm/帧\n\n观察半空间P95深度：{current['observed_halfspace_p95_depth_cm']:.2f} cm（局部代理，非完整mesh验收）。\n\n头部误差：{current['head_cm']:.2f} cm。完整场景独立诊断及GT冲突见 observed_surface_refine_oct10 报告。"
    @select.on_update
    def _(_):play.value=False;frame.value=16;update()
    for control in [frame,range_select,opacity,*checks.values(),follow]:control.on_update(lambda _:update())
    @reset.on_click
    def _(_):
        follow.value=False
        for client in server.get_clients().values():external(client,cases[select.value][1])
    @server.on_client_connect
    def _(client):external(client,cases[select.value][1])
    update();print(f'READY http://127.0.0.1:{args.port}',flush=True)
    while True:
        time.sleep(.05/float(speed.value))
        if play.value:
            low,high=(16,47) if range_select.value=='短期预测16–47' else ((16,127) if range_select.value=='完整预测16–127' else (0,frame_count-1))
            frame.value=low if int(frame.value)>=high or int(frame.value)<low else int(frame.value)+1

if __name__=='__main__':main()
