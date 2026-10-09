"""One scene, overlaid native GT / paired scene-on / paired scene-off meshes."""
import json,time,threading,argparse
from pathlib import Path
import numpy as np
import viser
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).parent/'runs/native_dynamic_scene20_contact_55k_oct07/scene_effect_demo'
def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8780);p.add_argument('--demo-root',type=Path,default=ROOT);args=p.parse_args();root=args.demo_root;manifest=json.loads((root/'manifest.json').read_text());cases={f"{x['index']} · {x['row']['window']['group']} · {x['row']['identity']['sequence_id']}":(x,dict(np.load(root/f"{x['index']}.npz"))) for x in manifest['cases']}
 repair=manifest.get('comparison')=='orientation_repair'
 display_labels={'有场景绿色':'新增朝向监督绿色','无场景红色':'旧损失续训红色'} if repair else {}
 server=viser.ViserServer(port=args.port,label='SceneMI · 朝向修复' if repair else 'SceneMI 55k · 场景作用');server.scene.set_up_direction('+y')
 intro=('### 朝向修复短训对比\n\n蓝色GT · 绿色新增朝向监督 · 红色旧损失续训。两者均有场景，同一55k初始化、相同1k步训练设置、相同控制与DDIM噪声。' if repair else '### 55k真实生成\n\n蓝色GT · 绿色有场景 · 红色无场景。同一头部条件、人体参数、相同DDIM噪声。')
 server.gui.add_markdown(intro+'\n\n灰色：截至片段末尾观测过的静态表面参考，不是起点全部输入。橙色物体按原轨迹播放；当前可见点可单独显示。手脸默认。案例用于诊断，不代表总体成绩。')
 select=server.gui.add_dropdown('例子',options=list(cases));frame=server.gui.add_slider('帧',min=0,max=127,step=1,initial_value=0);play=server.gui.add_checkbox('播放',initial_value=False);speed=server.gui.add_dropdown('倍速',options=['0.25','0.5','1'],initial_value='0.5');checks={k:server.gui.add_checkbox(display_labels.get(k,k),initial_value=v) for k,v in [('GT蓝色',True),('有场景绿色',True),('无场景红色',True),('静态表面参考',True),('动态物体',True),('当前可见点',False),('相机与轨迹',True)]};opacity=server.gui.add_slider('人体透明度',min=.1,max=1.,step=.05,initial_value=.65);reset=server.gui.add_button('重置外部视角');follow=server.gui.add_checkbox('观察相机视角',initial_value=False);info=server.gui.add_markdown('');lock=threading.RLock();handles={};active=None
 def external(c,g):
  center=g['gt_vertices'][int(frame.value)].mean(0);c.camera.position=tuple(center+[2.0,.8,2.0]);c.camera.look_at=tuple(center);c.camera.up_direction=(0,1,0)
 def update():
  nonlocal active,handles
  with lock,server.atomic():
   m,g=cases[select.value];t=int(frame.value)
   if active!=select.value:
    active=select.value;server.scene.reset();handles={}
    for k,color in [('gt',(45,145,245)),('on',(40,210,85)),('off',(240,60,55))]:handles[k]=server.scene.add_mesh_simple('/'+k,g[k+'_vertices'][t],g['faces'],color=color,opacity=opacity.value)
    if repair:
     for k,color in [('on',(40,210,85)),('off',(240,60,55))]:handles[k+'_heading']=server.scene.add_line_segments('/'+k+'_head_forward',np.zeros((1,2,3),np.float32),colors=color,thickness=3,thickness_units='screen')
    handles['context']=server.scene.add_point_cloud('/context',g['context'],colors=(150,160,155),point_size=.005)
    handles['points']=server.scene.add_point_cloud('/visible',g['visible_points'][t],colors=(230,210,50),point_size=.018)
    handles['objects']=[server.scene.add_mesh_simple(f'/object/{i}',g[f'obj_{i}_v'],g[f'obj_{i}_f'],color=(240,165,40)) for i in range(len(m['objects']))]
    handles['frustum']=server.scene.add_line_segments('/camera',np.zeros((8,2,3),np.float32),colors=(180,50,215),thickness=2,thickness_units='screen')
    handles['path']=server.scene.add_line_segments('/camera_path',np.stack([g['camera'][:-1],g['camera'][1:]],1),colors=(180,50,215),thickness=2,thickness_units='screen')
    center=g['gt_vertices'][0].mean(0);server.initial_camera.look_at=tuple(center);server.initial_camera.position=tuple(center+[2.0,.8,2.0])
    for c in server.get_clients().values():external(c,g)
   for k,label in [('gt','GT蓝色'),('on','有场景绿色'),('off','无场景红色')]:handles[k].vertices=g[k+'_vertices'][t];handles[k].visible=checks[label].value;handles[k].opacity=opacity.value
   if repair:
    for k in ['on','off']:
     pos=g[k+'_joints'][t,15];handles[k+'_heading'].points=np.stack([pos,pos+.3*g[k+'_head_forward'][t]])[None].astype(np.float32);handles[k+'_heading'].visible=handles[k].visible
   handles['context'].visible=checks['静态表面参考'].value;valid=g['visible_owners'][t]>=0;handles['points'].points=g['visible_points'][t][valid];handles['points'].visible=checks['当前可见点'].value
   for i,node in enumerate(handles['objects']):node.position=g[f'obj_{i}_t'][t];node.wxyz=np.roll(Rotation.from_matrix(g[f'obj_{i}_r'][t]).as_quat(),1);node.visible=checks['动态物体'].value
   pos=g['camera'][t];r=g['rotation'][t];corners=np.array([[-.3,-.2,.5],[.3,-.2,.5],[.3,.2,.5],[-.3,.2,.5]])@r.T+pos;handles['frustum'].points=np.concatenate([np.stack([np.tile(pos,(4,1)),corners],1),np.stack([corners,np.roll(corners,-1,axis=0)],1)]).astype(np.float32)
   handles['frustum'].visible=handles['path'].visible=checks['相机与轨迹'].value
   if follow.value:
    for c in server.get_clients().values():c.camera.position=tuple(pos);c.camera.look_at=tuple(pos+r[:,2]);c.camera.up_direction=tuple(r[:,1])
   row=m['row'];on=row['scene_on'];off=row['scene_off'];green='新增监督' if repair else '有';red='旧损失续训' if repair else '无';info.content=f"帧{t}/127 · {t/20:.2f}s · 源起点{row['identity']['source_start_30fps']}\n\nW-MPJPE：{green}{on['mpjpe_cm']:.2f} / {red}{off['mpjpe_cm']:.2f} cm\n\nPA-MPJPE：{green}{on['pa_mpjpe_mm']:.1f} / {red}{off['pa_mpjpe_mm']:.1f} mm\n\n生成差异{row['scene_effect_cm']:.1f}cm，去根平移差异{row['root_relative_scene_effect_cm']:.1f}cm。\n\nGT根路径{on['gt_root_path_length_m']:.2f}m。物体运动量：{[round(o['max_displacement_m'],3) for o in m['objects']]}m。\n\n{('朝向短训对比，两者均有场景；新增监督 vs 旧损失续训，尚未验收几何碰撞。' if repair else '这是场景条件消融，尚未证明替换另一场景后会规划绕障。')}"
 @select.on_update
 def _(_):play.value=False;frame.value=0;update()
 for control in [frame,opacity,*checks.values()]:control.on_update(lambda _:update())
 @reset.on_click
 def _(_):
  follow.value=False
  for c in server.get_clients().values():external(c,cases[select.value][1])
 @server.on_client_connect
 def _(client):external(client,cases[select.value][1])
 update();print(f'READY http://127.0.0.1:{args.port}',flush=True)
 while True:
  time.sleep(.05/float(speed.value))
  if play.value:frame.value=(int(frame.value)+1)%128
if __name__=='__main__':main()
