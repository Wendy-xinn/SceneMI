"""GIMO source playback using the existing shared Viser MotionViewer."""
from collections import OrderedDict
from contextlib import contextmanager
import json
import os
from pathlib import Path

import numpy as np

from scenemi.data.gimo import atomic_json, body_assets, read_scan_obj, transform_points
from experiments.offline_camera_retrain_v1.ops.motion_viewer import MotionViewer


def review(output):
    source=json.loads((output/'segments.json').read_text())['segments']
    available=[r for r in source if 'skipped' not in r and r['windows_128']]
    selected=[]
    # Every prepared annotation is navigable; each gets its first complete
    # 128-frame window. Additional worst-residual windows expose quality.
    for row in available:
        folder=Path(row['folder'])
        starts=row['window_starts']
        residual=np.load(folder/'camera_head_position_residual_m.npy')
        worst=max(starts,key=lambda t:float(residual[t:t+128].mean()))
        for start,kind in [(starts[0],'原始动作')]+([(worst,'标定残差较大窗口')] if worst!=starts[0] else []):
            selected.append(dict(label=f"{row['id']} · {start} · {kind}",
                display_label=f"{row['split']} · {row['scene']} · a{row['annotation']:03d} · 帧 {start} · {kind}",
                dataset='gimo',sequence_id=row['id'],start=start,folder=row['folder'],
                data_frames=128,regimes=['clean'],source=row,
                visualization_group='GIMO 原始数据 · 128帧',
                selection_reason='每个已准备标注区间首个完整窗口；额外列出区间内标定残差最大窗口。'))
    atomic_json(output/'review.json',dict(protocol='gimo-source-shared-viewer-v1',cases=selected,
        source_only=True,model_inference=False))
    return dict(manifest=str(output/'review.json'),cases=len(selected))


@contextmanager
def source_labels():
    import viser
    markdown=viser.GuiApi.add_markdown
    checkbox=viser.GuiApi.add_checkbox
    slider=viser.GuiApi.add_slider
    def md(api,content,*args,**kwargs):
        if content.startswith('## 已保存动作'):
            content=('## GIMO · 原始人体 / 实测相机 / 可见场景\n\n'
                '紫：数据集原始网格；橙：实测 PV 相机。切换案例、拖动帧滑块或勾选播放。'
                '完整扫描用于检查，可见静态并集是该标注区间的离线输入，含后续帧。')
        return markdown(api,content,*args,**kwargs)
    def cb(api,label,initial_value,*args,**kwargs):
        if label=='完整静态场景（GT参考）':initial_value=True
        if label=='本录制可见静态并集（实际输入）':initial_value=False
        if label=='输入相机轨迹':initial_value=True
        label={'本录制可见静态并集（实际输入）':'本标注区间可见静态并集（含后续帧）',
               '实际输入关节点 + 置信颜色':'实测 PV 相机位置'}.get(label,label)
        return checkbox(api,label,initial_value,*args,**kwargs)
    def sl(api,label,*args,**kwargs):
        return slider(api,{'预测人体不透明度':'原始人体不透明度','帧 / 30 Hz':'窗口帧 / nominal 30 Hz'}.get(label,label),*args,**kwargs)
    viser.GuiApi.add_markdown,viser.GuiApi.add_checkbox,viser.GuiApi.add_slider=md,cb,sl
    try:yield
    finally:viser.GuiApi.add_markdown,viser.GuiApi.add_checkbox,viser.GuiApi.add_slider=markdown,checkbox,slider


class GimoViewer(MotionViewer):
    COLORS=dict(MotionViewer.COLORS,gimo_source=(185,40,160))

    def __init__(self,cases,port):
        for case in cases:
            case['models']=[dict(key='gimo_source',label='GIMO 原始人体',mode='ego')]
        self.scan_cache=OrderedDict()
        self.head_handle=None
        self.link_handle=None
        self.rgb_toggle=None
        with source_labels():super().__init__(cases,port)
        self.server.scene.set_up_direction('+y')
        self.help.content=('原始人体未做贴地或对齐修正；头部顶点框架来自标准 SMPL-X 顶点编号的刚体拟合。'
            '相机到头部的固定安装残差仅用于质量诊断；测试段 GT 标定不作为模型输入。'
            '场景点是扫描网格渲染的合成观测；身体可遮挡，头部表面不参与遮挡。'
            '源数据的 BVH 对应索引存在非单调帧，页面保留原始顺序并显示真实 PV 时间。')
        self.head_handle=self.server.scene.add_point_cloud('/calibration/head_vertices',np.zeros((1,3),np.float32),
            colors=(250,210,20),point_size=.012)
        self.link_handle=self.server.scene.add_line_segments('/calibration/camera_to_head',np.zeros((1,2,3),np.float32),
            colors=(250,100,20),thickness=3)
        self.ego_button=self.server.gui.add_button('进入当前 PV 相机视角')
        self.ego_button.on_click(lambda _:self.ego_view())
        self.rgb_toggle=self.server.gui.add_checkbox('显示原始 RGB（播放时读取源图片）',initial_value=False)
        self.rgb_image=self.server.gui.add_image(np.zeros((72,128,3),np.uint8),label='原始 PV RGB',visible=False)
        self.depth_image=self.server.gui.add_image(np.zeros((72,128,3),np.uint8),label='合成场景深度 · 近亮远暗')
        self.rgb_toggle.on_update(lambda _:self.update())
        self.update()

    def read_case(self,index):
        if index in self.cache:return self.cache[index]
        case=self.cases[index];row=case['source'];folder=Path(case['folder']);start=case['start'];n=128
        def a(name):return np.load(folder/(name+'.npy'),mmap_mode='r')[start:start+n]
        v=a('vertices_world');jname='joints_world' if (folder/'joints_world.npy').exists() else 'joints_mesh_regressed_world'
        j=a(jname);c=a('camera_world');intr=a('intrinsics')
        transform=np.load(folder/'scanner_to_world.npy')
        key=(row['scene_mesh'],tuple(transform.ravel()))
        if key not in self.scan_cache:
            sv,sf=read_scan_obj(row['scene_mesh'])
            if len(sf)>100000:sf=sf[np.linspace(0,len(sf)-1,100000).astype(int)]
            ids,rev=np.unique(sf,return_inverse=True)
            self.scan_cache[key]=(transform_points(sv[ids],transform).astype(np.float32),rev.reshape(-1,3).astype(np.int32))
            while len(self.scan_cache)>2:self.scan_cache.popitem(last=False)
        sv,sf=self.scan_cache[key]
        p=c[:,:3,3];r=c[:,:3,:3]
        pixels=np.array([[0,0],[128,0],[128,72],[0,72]])
        xy=(pixels-intr[:,None,2:])/intr[:,None,:2]
        xy[:,:,1]*=-1
        local=np.concatenate([xy,-np.ones((n,4,1))],axis=-1)*.5
        corners=np.einsum('tij,tkj->tki',r,local)+p[:,None]
        ray=np.stack([np.broadcast_to(p[:,None],corners.shape),corners],axis=2)
        rim=np.stack([corners,np.roll(corners,-1,axis=1)],axis=2)
        axes=np.stack([np.broadcast_to(p[:,None],(n,3,3)),p[:,None]+r.transpose(0,2,1)*.3],axis=2)
        obs=np.zeros((n,22,9),np.float32);meta=np.zeros((n,22,4),np.float32)
        obs[:,15,:3]=p/2;meta[:,15,[0,2]]=1
        center=np.asarray((v.reshape(-1,3).min(0)+v.reshape(-1,3).max(0))/2,np.float32)
        data=dict(truth_vertices=v,truth_joints=j,faces=body_assets()['faces'],scene_vertices=sv,scene_faces=sf,
            center=center,radius=np.array(3.,np.float32),rest=np.zeros((22,3),np.float32),
            visible_static=np.load(folder/'observed_static_union.npy'),camera_positions=p,
            camera_frustum_segments=np.concatenate([ray,rim],axis=1).astype(np.float32),
            camera_axis_segments=axes.astype(np.float32),camera_path=np.stack([p[:-1],p[1:]],axis=1),
            source_occluded=np.zeros((n,22),bool),source_frame_ids=a('source_frame_ids'),
            head_vertex_world=a('head_vertex_world'),head_vertex_ids=np.load(folder/'head_vertex_ids.npy'),
            source_timestamps_100ns=a('source_timestamps_100ns'),depth=a('depth'),depth_owner=a('depth_owner'))
        for regime in ['clean','noisy']:
            data[regime+'_vertices']=data[regime+'_reference_vertices']=v
            data[regime+'_joints']=data[regime+'_reference_joints']=j
            data[regime+'_observations']=obs;data[regime+'_observation_meta']=meta
        points,owners=a('visible_frame_points'),a('visible_frame_owner')
        for t in range(n):data['current_static_'+str(t)]=points[t][owners[t]==0]
        evidence=dict(scene_errors=[],generation=dict(recorded_guidance='原始数据；未运行模型'),
            display_geometry=dict(camera=dict(source_kind='released_PV_camera',frustum_depth_m=.5)))
        self.cache.clear();self.cache[index]=[dict(entry=case['models'][0],data=data,meta=evidence)]
        return self.cache[index]

    def load_case(self):
        super().load_case()
        self.regime.visible=self.regime.disabled=False
        self.regime.visible=False;self.regime.disabled=True
        for panel in self.panels.values():
            for key in ('gt','reference','uncertainty','reference_source'):
                panel[key].visible=False;panel[key].disabled=True
            panel['gt'].value=panel['reference'].value=panel['uncertainty'].value=False
        for buttons in self.preset_buttons.values():
            for button in buttons.values():button.visible=False;button.disabled=True
        self.panels['exo']['enabled'].value=False
        for rt in self.runtime.values():rt['scan'].opacity=.55
        self.update()

    def update(self):
        if self.loading:return
        super().update()
        case=self.cases[self.index];row=case['source'];t=int(self.frame.value)
        rt=self.runtime['ego'];data=rt['data'];q=row['quality'];source_frame=int(data['source_frame_ids'][t])
        elapsed=(int(data['source_timestamps_100ns'][t])-int(data['source_timestamps_100ns'][0]))/1e7
        self.info.content=(f"**GIMO 原始数据 · {row['split']} · CSV {row['annotation']}**\n\n"
            f"窗口 {t+1}/128 帧 · 原始帧 {source_frame} · PV 实际经过 {elapsed:.3f} 秒。\n\n"
            f"相机/头部固定安装残差 P90：{q['camera_head_position_residual_m']['p90']*100:.1f} cm / "
            f"{q['camera_head_rotation_residual_deg']['p90']:.1f}°。\n\n"
            f"复核标记：{'，'.join(q['review_flags']) or '无自动阈值标记'}。\n\n没有模型预测或预测误差。")
        self.panels['ego']['status'].content='**EGO · 原始人体与实测相机**'
        self.panels['ego']['info'].content=('完整静态扫描用于检查。输入点为当前帧 first-hit 环境采样，'
            '可见并集限定本标注区间。相机轨迹保持数据集原样；头部标定不改变人体或相机。')
        if self.head_handle is not None:
            self.head_handle.points=np.asarray(data['truth_vertices'][t,data['head_vertex_ids']])
            self.link_handle.points=np.array([[data['camera_positions'][t],data['head_vertex_world'][t,:3,3]]],np.float32)
        if self.rgb_toggle is not None:
            depth=np.asarray(data['depth'][t],np.float32);owner=data['depth_owner'][t]
            gray=np.zeros(depth.shape,np.uint8);good=(owner==0)&np.isfinite(depth)
            gray[good]=np.clip((4-depth[good])/4*255,0,255)
            img=np.repeat(gray[:,:,None],3,axis=2);img[owner==100]=[190,40,150]
            self.depth_image.image=img
            self.rgb_image.visible=bool(self.rgb_toggle.value)
            if self.rgb_toggle.value:
                from PIL import Image
                source=Path(next(iter(row['inputs_sha256']))).parent
                path=source/'PV'/f"{int(data['source_timestamps_100ns'][t])}.png"
                if path.exists():self.rgb_image.image=np.asarray(Image.open(path).convert('RGB').resize((380,214)))

    def ego_view(self):
        data=self.runtime['ego']['data'];t=int(self.frame.value)
        folder=Path(self.cases[self.index]['folder']);c=np.load(folder/'camera_world.npy',mmap_mode='r')[self.cases[self.index]['start']+t]
        for client in self.server.get_clients().values():
            with self.server.atomic():
                client.camera.position=c[:3,3];client.camera.look_at=c[:3,3]-c[:3,2];client.camera.up=c[:3,1]


def serve(output,port):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    if not (output/'review.json').exists():review(output)
    cases=json.loads((output/'review.json').read_text())['cases']
    viewer=GimoViewer(cases,port)
    atomic_json(output/'SERVER.json',dict(pid=os.getpid(),url=f'http://localhost:{port}/',cases=len(cases),
        manifest=str(output/'review.json'),source_only=True,model_inference=False))
    print(f'READY http://localhost:{port}/ · {len(cases)} GIMO source windows',flush=True)
    viewer.run()
