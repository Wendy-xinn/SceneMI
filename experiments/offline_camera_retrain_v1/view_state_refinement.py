"""Overlay native GT, same-model forecast and state-relative refined forecast."""
import argparse,json,threading,time
from pathlib import Path
import numpy as np
import viser
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from experiments.offline_camera_retrain_v1.generated_head_camera import head_world_track,apply_camera_mount,diagnostic_fixed_mount,camera_errors,frustum_lines
ROOT=Path(__file__).parent/'runs/state_relative_spline_oct10/demo'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8780);parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--scene-refinement',type=Path);parser.add_argument('--legacy-history-diagnostic',action='store_true');parser.add_argument('--turn-coordination',type=Path);parser.add_argument('--turn-training',type=Path);parser.add_argument('--support-refinement',type=Path);parser.add_argument('--fresh55k-replay',type=Path);parser.add_argument('--fresh55k-winding',type=Path);parser.add_argument('--fresh55k-physics',type=Path);parser.add_argument('--fresh55k-coverage',type=Path);parser.add_argument('--scale-body-scene',type=Path);args=parser.parse_args()
    scale_mode=bool(args.scale_body_scene)
    if scale_mode:
        if args.fresh55k_coverage:parser.error('Choose one coverage trial')
        args.fresh55k_coverage=args.scale_body_scene
    coverage_mode=bool(args.fresh55k_coverage)
    if coverage_mode:
        if args.fresh55k_physics:parser.error('Choose one physical trial')
        args.fresh55k_physics=args.fresh55k_coverage
    physics_mode=bool(args.fresh55k_physics)
    if physics_mode:
        if args.fresh55k_replay or args.fresh55k_winding:parser.error('Choose one fresh55k diagnostic')
        args.fresh55k_replay=args.fresh55k_physics
    winding_mode=bool(args.fresh55k_winding)
    if winding_mode:
        if args.fresh55k_replay:parser.error('Choose replay or winding')
        args.fresh55k_replay=args.fresh55k_winding
    gt_free=not (args.legacy_history_diagnostic or args.scene_refinement)
    frame_count=48 if args.scene_refinement else 128
    mesh_keys=['gt','original','refined','scene_contact'] if args.scene_refinement else ['gt','official55k','official55k_joint','original','refined']
    colors={'gt':(40,135,245),'official55k':(245,145,30),'official55k_joint':(175,80,230),'original':(240,65,60),'refined':(155,80,215) if args.scene_refinement else (45,205,90),'scene_contact':(45,205,90)}
    labels={'gt':'GT蓝色','official55k':'55k原相机输入橙色','official55k_joint':'55k统一头部输入紫色','original':'微调固定历史红色','refined':'旧样条修正紫色' if args.scene_refinement else '样条修正绿色','scene_contact':'场景接触修正绿色'}
    if gt_free:
        mesh_keys=['gt','official55k'];colors['official55k']=(240,65,60);labels['official55k']='55k无身体GT初始化红色'
    if sum(bool(x) for x in [args.turn_training,args.turn_coordination,args.support_refinement,args.fresh55k_replay])>1:parser.error('Choose one turn diagnostic')
    turn_root=args.fresh55k_replay or args.support_refinement or args.turn_training or args.turn_coordination
    turn_variant=('coverage_support' if coverage_mode else ('winding_scene' if physics_mode else ('winding_guard' if winding_mode else 'rollout_replay'))) if args.fresh55k_replay else ('sole_support' if args.support_refinement else ('turn_path' if args.turn_training else 'camera_soft'))
    if turn_root:
        if not gt_free:parser.error('Turn diagnostics require GT-free default mode')
        mesh_keys.append('turn_coordination');colors['turn_coordination']=(45,205,90);labels['turn_coordination']='足底支撑400步绿色（未通过）' if args.support_refinement else ('转向路径600步绿色（未通过）' if args.turn_training else '转向保脚开发候选绿色')
        if args.support_refinement:
            mesh_keys.extend(['prior_turn600','turn_control']);colors.update(prior_turn600=(245,145,30),turn_control=(155,80,215));labels.update(prior_turn600='上轮转向600步橙色',turn_control='继续转向400步紫色')
        if args.fresh55k_replay:
            labels['turn_coordination']='55k回放微调绿色（开发候选）';mesh_keys.append('continued_control');colors['continued_control']=(155,80,215);labels['continued_control']='55k标准加噪对照紫色'
        if physics_mode:
            labels['turn_coordination']='原55k独立场景接触组绿色（未通过）';labels['continued_control']='原55k独立累计转向参考橙色';colors['continued_control']=(245,145,30)
        if coverage_mode:
            labels['turn_coordination']='原55k独立支撑覆盖组绿色（开发候选）';labels['continued_control']='原55k独立上轮场景接触参考橙色'
        if winding_mode:
            labels['turn_coordination']='55k累计转向微调绿色（开发候选）';labels['continued_control']='55k回放标准DDIM紫色'
            mesh_keys.extend(['guard_terminal','replay_terminal']);colors.update(guard_terminal=(245,145,30),replay_terminal=(80,205,210));labels.update(guard_terminal='累计转向末端一次橙色',replay_terminal='回放末端一次青色')
        if args.turn_training:
            mesh_keys.append('continued_control');colors['continued_control']=(155,80,215);labels['continued_control']='继续原目标600步紫色'
    if scale_mode:
        turn_variant='body_local';labels['turn_coordination']='身体局部场景绿色（开发候选）';labels['continued_control']='尺度校准紫色（开发候选）';colors['continued_control']=(155,80,215)
        mesh_keys.append('support_reference');colors['support_reference']=(245,145,30);labels['support_reference']='上轮支撑覆盖橙色（参考）'
    turn_rows={r['index']:r for r in json.loads((turn_root/'rows.json').read_text()) if r['scope']=='demo'} if turn_root else {}
    init_audit={r['index']:r for r in json.loads((ROOT.parent.parent/'no_body_initialization_oct10/audit.json').read_text())['rows']} if gt_free else {}
    scene_rows={r['index']:r for r in json.loads((args.scene_refinement/'rows.json').read_text())} if args.scene_refinement else {}
    manifest=json.loads((args.root/'manifest.json').read_text());cases={};models={}
    for case in manifest['cases']:
        arrays=dict(np.load(args.root/f"{case['index']}.npz"));body=case['identity']['native_body'];key=(body['model'],body['gender'])
        if key not in models:models[key]=load_model(*key)
        if args.scene_refinement:
            arrays['scene_contact_motion']=np.load(args.scene_refinement/f"{case['index']}_motions.npz")['scene_contact'];case['metrics']['scene_contact']={'short32':scene_rows[case['index']]['metrics']['scene_contact']}
        if turn_root:
            with np.load(turn_root/f"{case['index']}.npz") as turn_arrays:
                arrays['turn_coordination_motion']=turn_arrays[turn_variant]
                if args.turn_training or args.fresh55k_replay:arrays['continued_control_motion']=turn_arrays['scale_calibrated' if scale_mode else ('scene_reference' if coverage_mode else ('guard_reference' if physics_mode else ('replay_reference' if winding_mode else ('denoise_control' if args.fresh55k_replay else 'continued_control'))))]
                if scale_mode:
                    arrays['support_reference_motion']=turn_arrays['support_reference']
                    for qkey in ('body_query_joints','body_query_surface_points','body_query_valid','body_query_dynamic'):arrays[qkey]=turn_arrays[qkey]
                    for qkey in ('body_query_joints','body_query_surface_points'):arrays[qkey]=arrays[qkey]@arrays['anchor_rotation']+arrays['camera'][0]
                if winding_mode:
                    arrays['guard_terminal_motion']=turn_arrays['guard_terminal'];arrays['replay_terminal_motion']=turn_arrays['replay_terminal']
                if args.support_refinement:arrays['turn_control_motion']=turn_arrays['turn_control']
            if args.support_refinement:
                with np.load(ROOT.parent.parent/f"turn_path_training_oct10/{case['index']}.npz") as prior:arrays['prior_turn600_motion']=prior['turn_path']
        camera_heads={}
        for label in mesh_keys:
            vertices,faces,joints=decode_native_mesh(arrays[label+'_motion'][:frame_count],body,models[key]);arrays[label+'_vertices']=(vertices@arrays['anchor_rotation']+arrays['camera'][0]).astype(np.float32);arrays['faces']=faces
            camera_heads[label]=head_world_track(arrays[label+'_motion'][:frame_count],joints,arrays['anchor_rotation'],arrays['camera'][0])
        mount_r,mount_t=diagnostic_fixed_mount(*camera_heads['gt'],arrays['camera'][:frame_count],arrays['rotation'][:frame_count])
        case['generated_camera_diagnostic']={'calibration':'GT first-frame fixed mount, display/scoring only; no motion modification','mount_translation_m':mount_t.tolist(),'mount_rotation':mount_r.tolist(),'variants':{}}
        for label,(hp,hr) in camera_heads.items():
            cp,cr=apply_camera_mount(hp,hr,mount_r,mount_t);arrays[label+'_camera_position']=cp;arrays[label+'_camera_rotation']=cr
            arrays[label+'_camera_position_error_cm']=np.linalg.norm(cp-arrays['camera'][:frame_count],axis=-1)*100
            arrays[label+'_camera_rotation_error_deg']=np.rad2deg(Rotation.from_matrix((arrays['rotation'][:frame_count].transpose(0,2,1)@cr).copy()).magnitude())
            case['generated_camera_diagnostic']['variants'][label]=camera_errors(cp,cr,arrays['camera'][:frame_count],arrays['rotation'][:frame_count])
        name=f"{case['index']} · {case['identity']['sequence_id']}";cases[name]=(case,arrays)
    server=viser.ViserServer(port=args.port,label='SceneMI · 状态相对生成');server.scene.set_up_direction('+y')
    if gt_free:server.gui.add_markdown('**850/906：严格无身体GT初始化基线。** 蓝GT仅对照 · 红真正原55k，从随机扩散噪声生成全部128帧。输入仅已知相机/场景与配置身体模板，没有GT身体前缀或初始姿态对齐。TRUMANS相机仍是模拟已知条件，非真实视频估计。红色已切回原55k，与此前微调+GT历史红色不同；可选绿色为转向开发诊断，尚未通过验收。')
    if scale_mode:server.gui.add_markdown('**两项独立600步对照，均从原55k开始。** 红原55k · 橙上轮支撑覆盖 · 紫约束尺度校准 · 绿同一校准加身体局部场景查询。全128帧从噪声生成，无GT身体初始化或推理投影。局部查询来自每步预生成FK关节，静态记忆按观测时间截断，动态几何只取当前帧。不是完整碰撞约束，联合结果见control_scale_body_scene_oct11报告。')
    if coverage_mode and not scale_mode:server.gui.add_markdown('**支撑覆盖配对实验，全部从原55k独立初始化。** 红原55k · 橙上轮场景接触参考 · 绿支撑patch覆盖修复。全128帧纯噪声，无GT身体初态，无推理投影；头部约束与上轮相同。本轮是否通过必须查看fresh55k_support_coverage_oct11评估，默认不启用开发候选。')
    if physics_mode and not coverage_mode:server.gui.add_markdown('**本轮全部从原55k独立初始化。** 红原55k · 橙累计转向参考 · 绿累计转向＋已观测地面接触组。全128帧从噪声生成，无GT身体初始化、无推理投影。850改善，906转向改善但转身后仍离地滑动，未通过。参考组只用于对照，没有作为绿色训练起点。')
    if winding_mode:server.gui.add_markdown('**均从原55k独立开始，未接着旧候选微调。** 红原55k · 紫回放600步参考 · 绿新增累计转向/速率尾部600步标准DDIM · 橙新组末端一次修正 · 青回放参考末端一次。绿色与紫色均全128帧由噪声生成；橙/青先冻结原55k生成再修正一次。无GT身体初始化或GT投影，联合验收及脚步幅度见fresh55k_winding_oct11报告。')
    if args.fresh55k_replay and not winding_mode and not physics_mode:server.gui.add_markdown('**直接从原55k开始的对照。** 红原55k · 紫同目标标准加噪微调 · 绿55k生成轨迹回放微调。两组都未加载此前候选；全128帧无GT身体初始化，没有GT投影。联合验收见fresh55k_replay_oct11报告，开发结果未证明物理可跟踪。')
    if args.support_refinement:server.gui.add_markdown('**足底支撑400步诊断：** 红原55k · 橙上轮转向600步 · 紫继续转向400步 · 绿新增足底支撑400步。全部无GT身体初始化；native足底监督用训练GT包络，未做推理GT投影。906漂浮减少但支撑未恢复；850脚滑仍退化，未通过联合验收。')
    if args.turn_training:server.gui.add_markdown('**600步转向诊断，未通过足部验收。** 绿色使用同一原55k架构和相机语义，无GT身体初始化。906净转向改为负向、峰值降低，但悬浮代理3.63cm；850脚滑增加约38%。紫色是相同采样/种子的继续训练对照。')
    if not gt_free and not args.scene_refinement:server.gui.add_markdown('**历史实验诊断，含GT前16帧身体代理，不作为无GT初始化结果。** 蓝GT · 橙55k相机 · 紫55k头部 · 红微调固定历史 · 绿旧样条。')
    if args.scene_refinement:server.gui.add_markdown('**场景接触开发诊断：仅未来1.6秒。** 蓝GT · 红微调固定历史 · 紫旧样条 · 绿场景+软接触。局部观察表面约束不是完整碰撞证书；GT本身与场景有厘米级几何冲突信号，当前不宣称通过物理验收。')
    select=server.gui.add_dropdown('例子',options=list(cases),initial_value=next(iter(cases)))
    play=server.gui.add_checkbox('播放',initial_value=True);speed=server.gui.add_slider('速度',min=.25,max=2,step=.25,initial_value=.75)
    options=['完整生成0–127','短期预测16–47'] if gt_free else (['短期预测16–47','含历史0–47'] if args.scene_refinement else ['短期预测16–47','完整预测16–127','含历史0–127'])
    range_select=server.gui.add_dropdown('播放范围',options=options,initial_value=options[0] if gt_free or args.scene_refinement else '完整预测16–127')
    frame=server.gui.add_slider('帧',min=0,max=frame_count-1,step=1,initial_value=0 if gt_free else 16);opacity=server.gui.add_slider('人物透明度',min=.1,max=1,step=.05,initial_value=.55)
    checks={label:server.gui.add_checkbox(label,initial_value=(label not in ['身体局部场景绿色（开发候选）','尺度校准紫色（开发候选）','上轮支撑覆盖橙色（参考）','原55k独立支撑覆盖组绿色（开发候选）','原55k独立上轮场景接触参考橙色','原55k独立场景接触组绿色（未通过）','原55k独立累计转向参考橙色','55k原相机输入橙色','55k统一头部输入紫色','旧样条修正紫色','样条修正绿色','场景接触修正绿色','转向保脚开发候选绿色','转向路径600步绿色（未通过）','继续原目标600步紫色','足底支撑400步绿色（未通过）','上轮转向600步橙色','继续转向400步紫色','55k回放微调绿色（开发候选）','55k标准加噪对照紫色','55k累计转向微调绿色（开发候选）','55k回放标准DDIM紫色','累计转向末端一次橙色','回放末端一次青色'])) for label in [*[labels[key] for key in mesh_keys],'静态记忆','当前可见点','动态物体','输入相机与轨迹','生成相机与轨迹']}
    server.gui.add_markdown('**相机诊断：** 紫色是已知输入；红/绿/紫生成相机随对应mesh开关显示。相机由生成身体FK头部重建，未吸附输入。固定安装偏置由GT首帧仅作绘图标定，非生成输入。位置和朝向误差均在完整128帧记录。')
    body_queries=server.gui.add_checkbox('身体局部查询（最后扩散步预生成关节）',initial_value=False) if scale_mode else None
    camera_size=server.gui.add_slider('相机线框深度（米）',min=.04,max=.8,step=.02,initial_value=.5)
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
                for key in mesh_keys:
                    if key=='gt':continue
                    handles[key+'_camera']=server.scene.add_line_segments('/generated_camera/'+key,frustum_lines(g[key+'_camera_position'][t],g[key+'_camera_rotation'][t]),colors=colors[key],thickness=2,thickness_units='screen')
                    pos_track=g[key+'_camera_position'];handles[key+'_camera_path']=server.scene.add_line_segments('/generated_camera_path/'+key,np.stack((pos_track[:-1],pos_track[1:]),1).astype(np.float32),colors=colors[key],thickness=2,thickness_units='screen')
                if scale_mode:handles['body_queries']=server.scene.add_line_segments('/body_local_queries',np.zeros((22,2,3),np.float32),colors=(40,205,205),thickness=2,thickness_units='screen')
                center=g['gt_vertices'][16].mean(0);server.initial_camera.look_at=tuple(center);server.initial_camera.position=tuple(center+[2,.8,2])
                for client in server.get_clients().values():external(client,g)
            for key,label in [(key,labels[key]) for key in mesh_keys]:handles[key].vertices=g[key+'_vertices'][t];handles[key].visible=checks[label].value;handles[key].opacity=opacity.value
            known=g['static_times']<=g['source_frames'][t];handles['static'].points=g['static_points'][known];handles['static'].visible=checks['静态记忆'].value
            valid=(g['visible_owners'][t]>=0)&(g['visible_owners'][t]<100);handles['visible'].points=g['visible_points'][t][valid];handles['visible'].visible=checks['当前可见点'].value
            for i,node in enumerate(handles['objects']):node.position=g[f'obj_{i}_t'][t];node.wxyz=np.roll(Rotation.from_matrix(g[f'obj_{i}_r'][t]).as_quat(),1);node.visible=checks['动态物体'].value
            if scale_mode:
                validq=g['body_query_valid'][t];segments=np.stack((g['body_query_joints'][t],g['body_query_surface_points'][t]),axis=1)
                segments[~validq]=g['body_query_joints'][t][~validq,None,:];handles['body_queries'].points=segments.astype(np.float32);handles['body_queries'].visible=body_queries.value
            pos=g['camera'][t];r=g['rotation'][t]
            handles['frustum'].points=frustum_lines(pos,r,camera_size.value)
            handles['frustum'].visible=handles['path'].visible=checks['输入相机与轨迹'].value
            for key in mesh_keys:
                if key=='gt':continue
                handles[key+'_camera'].points=frustum_lines(g[key+'_camera_position'][t],g[key+'_camera_rotation'][t],camera_size.value)
                handles[key+'_camera'].visible=handles[key+'_camera_path'].visible=checks['生成相机与轨迹'].value and checks[labels[key]].value
            if follow.value:
                for client in server.get_clients().values():client.camera.position=tuple(pos);client.camera.look_at=tuple(pos+r[:,2]);client.camera.up_direction=tuple(r[:,1])
            if not gt_free:
                horizon='short32' if args.scene_refinement or range_select.value=='短期预测16–47' else 'future112';old=case['metrics']['original'][horizon];new=case['metrics']['refined'][horizon];official=case['metrics']['official55k'][horizon]
                info.content=f"帧{t} · {(t-16)/20:.2f}s（预测起点=16）\n\n世界MPJPE：55k相机输入 {official['mpjpe_cm']:.2f} / 微调固定历史 {old['mpjpe_cm']:.2f} / 修正 {new['mpjpe_cm']:.2f} cm\n\nPA-MPJPE：微调固定历史{old['pa_mpjpe_mm']:.1f} / 新{new['pa_mpjpe_mm']:.1f} mm\n\n骨盆朝向误差：微调固定历史{old['pelvis_orientation_mean_deg']:.1f} / 新{new['pelvis_orientation_mean_deg']:.1f}°\n\nGT支撑期脚滑：微调固定历史{old['gt_stance_slide_cm_frame']:.3f} / 新{new['gt_stance_slide_cm_frame']:.3f} cm/帧\n\n固定失败例子850/906，用于观察；平均效果见完整开发/确认集报告。未证明mesh无碰撞或实时仿真跟踪。"
            if gt_free:
                m=init_audit[case['index']]['full128_metrics'];info.content=f"帧{t} · {t/20:.2f}s。全部128帧由噪声生成，无GT初始身体。\n\n完整6.4秒世界MPJPE {m['mpjpe_cm']:.2f} cm / PA {m['pa_mpjpe_mm']:.1f} mm\n\n骨盆朝向误差 {m['pelvis_orientation_mean_deg']:.1f}° / 脚滑 {m['gt_stance_slide_cm_frame']:.3f} cm/帧\n\n身体/接触/历史GT破坏测试输出差0；当前是原55k问题诊断，非已修好版本。"
            if turn_root:
                candidate=turn_rows[case['index']]['metrics'][turn_variant]
                info.content+=f"\n\n绿色开发候选：朝向 {candidate['pelvis_orientation_mean_deg']:.1f}° / 脚滑 {candidate['gt_stance_slide_cm_frame']:.3f} cm/帧。开发候选，联合验收结果见报告。"
                if args.turn_training or args.support_refinement or args.fresh55k_replay:info.content+=f"\n\n绿色净转向 {candidate['pelvis_turn_deg']:.1f}° / 根部最大步长 {candidate['root_step_max_deg']:.2f}° / 悬浮代理 {candidate['support_floating_m']*100:.2f}cm（相对GT足部支撑高度，非场景地板真值）。"
            if args.support_refinement or args.fresh55k_replay:
                info.content+=f"\n\n最后48帧：脚滑 {candidate['post80_stance_slide_cm_frame']:.3f}cm/帧 / 悬浮代理 {candidate['post80_floating_proxy_cm']:.2f}cm。native足底包络绝对误差 {candidate['native_sole_envelope_abs_cm']:.2f}cm。"
            if args.scene_refinement:
                current=case['metrics']['scene_contact']['short32'];info.content=f"未来1.6秒开发诊断，帧{t}。\n\n世界MPJPE：旧样条 {new['mpjpe_cm']:.2f} / 场景接触 {current['mpjpe_cm']:.2f} cm\n\n脚滑：旧 {new['gt_stance_slide_cm_frame']:.3f} / 场景接触 {current['gt_stance_slide_cm_frame']:.3f} cm/帧\n\n观察半空间P95深度：{current['observed_halfspace_p95_depth_cm']:.2f} cm（局部代理，非完整mesh验收）。\n\n头部误差：{current['head_cm']:.2f} cm。完整场景独立诊断及GT冲突见 observed_surface_refine_oct10 报告。"
            info.content+='\n\n**生成相机 vs 已知输入（固定安装标定）**'
            for key in mesh_keys:
                if key=='gt' or not checks[labels[key]].value:continue
                cm=case['generated_camera_diagnostic']['variants'][key]
                info.content+=f"\n\n{labels[key]}：本帧 {g[key+'_camera_position_error_cm'][t]:.2f}cm / {g[key+'_camera_rotation_error_deg'][t]:.1f}°；整段均值 {cm['position_mean_cm']:.2f}cm / {cm['orientation_mean_deg']:.1f}°，P95 {cm['position_p95_cm']:.2f}cm / {cm['orientation_p95_deg']:.1f}°。"
    @select.on_update
    def _(_):play.value=False;frame.value=0 if gt_free else 16;update()
    for control in [frame,range_select,opacity,camera_size,*checks.values(),follow]:control.on_update(lambda _:update())
    if body_queries is not None:body_queries.on_update(lambda _:update())
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
