"""Single-port review of actual native20 training inputs, not generated motions."""
import argparse,json,threading,time
import numpy as np
import viser
from experiments.offline_camera_retrain_v1.prepare_scene20_acceptance import OUT
from experiments.offline_camera_retrain_v1.view_scene_visibility_v2 import owner_rgb
LABELS={'rich_contact':'RICH · 擦椅/接触','rich_motion':'RICH · 瑜伽/运动','trumans_dynamic':'TRUMANS · 当前训练静稳窗口','egobody_wearer':'EgoBody · 实拍PV观察者','egobody_interactee':'EgoBody · 对方虚拟相机','trumans_source_dynamic':'TRUMANS · 恢复训练的移动椅子'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8780);args=p.parse_args();manifest=json.loads((OUT/'manifest.json').read_text());cases={}
 for m in manifest['cases']:
  key=m['key'];cases[LABELS[key]]=(m,dict(np.load(OUT/key/'geometry.npz')),dict(np.load(OUT/key/'training_input.npz')))
 server=viser.ViserServer(port=args.port,label='SceneMI · native20 输入验收');server.scene.set_up_direction('+y');server.initial_camera.position=(2,1,3);server.initial_camera.look_at=(0,0,0)
 server.gui.add_markdown('### 新版20Hz真实训练输入验收\n\n下拉切换案例，**一次只显示一个场景**。这里播放GT，不是模型生成。\n\nTRUMANS动态窗口已重新纳入新版训练。移动椅子案例已通过真实loader核对；这里展示的六例均为合法训练窗口。\n\n蓝色：训练GT原生mesh（22关节，手脸默认）；灰色：完整原始拟合mesh（遮挡用，默认隐藏）；紫红：另一人；橙色mesh：当前动态物体。\n\n绿色点：当帧可见静态表面；黄色点：当帧可见动态表面；青色点：此前静态记忆。完整扫描仅供核对，不代表全部输入。\n\n红框：相机；紫线：光轴；BPS线段是模型实际使用的截断偏移，长度上限1m，不保证端点抵达真实表面。occupancy黄色格点是窗口起点静态观测，0表示未知。')
 select=server.gui.add_dropdown('验收案例',options=list(cases),initial_value=LABELS['rich_contact']);frame=server.gui.add_slider('20Hz帧',min=0,max=127,step=1,initial_value=0);playing=server.gui.add_checkbox('播放',initial_value=False);speed=server.gui.add_dropdown('倍速',options=['0.25','0.5','1','2'],initial_value='1')
 checks={name:server.gui.add_checkbox(name,initial_value=value) for name,value in [('训练GT mesh',True),('完整拟合mesh',False),('另一人',True),('动态物体mesh',True),('当前可见点',True),('此前静态记忆',False),('完整扫描参考',False),('相机',True),('BPS偏移',False),('起点occupancy',False),('区域接触标签',False)]}
 reset=server.gui.add_button('外部视角');ego=server.gui.add_button('跟随观察相机');img=server.gui.add_image(np.zeros((96,128,3),np.uint8),label='当帧首命中归属：灰静态/黄物体/蓝自身/紫红他人');depth_img=server.gui.add_image(np.zeros((96,128,3),np.uint8),label='深度0–4m：近亮远暗，无命中黑色');info=server.gui.add_markdown('');lock=threading.RLock();ego_clients=set();handles={};active=None
 def external(c,g):
  center=g['training_vertices'][int(frame.value)].mean(0);c.camera.position=tuple(center+[2.2,1.4,2.2]);c.camera.look_at=tuple(center);c.camera.up_direction=(0,1,0)
 def set_ego(c,g,t):
  c.camera.position=tuple(g['camera'][t]);c.camera.look_at=tuple(g['camera'][t]+g['rotation'][t,:,2]);c.camera.up_direction=tuple(g['rotation'][t,:,1]);fy=g['intrinsics'][t,1];c.camera.fov=float(2*np.arctan(48/fy))
 def rebuild(g,m):
  nonlocal handles
  server.scene.reset();handles={};handles['gt']=server.scene.add_mesh_simple('/training_gt',g['training_vertices'][0],g['training_faces'],color=(45,140,235),opacity=.85)
  handles['raw']=server.scene.add_mesh_simple('/raw_full_fit',g['body_0_vertices'][0],g['body_0_faces'],color=(170,175,185),opacity=.35,wireframe=True,visible=False)
  handles['others']=[server.scene.add_mesh_simple(f'/other/{i}',g[f'body_{i}_vertices'][0],g[f'body_{i}_faces'],color=(215,65,155),opacity=.7) for i in range(1,m['body_count'])]
  handles['objects']=[server.scene.add_mesh_simple(f'/object/{i}',g[f'object_{i}_vertices'],g[f'object_{i}_faces'],color=(245,170,40)) for i in range(len(m['objects']))]
  for key,color,size in [('static',(40,240,90),.014),('dynamic',(255,210,20),.022),('memory',(40,190,200),.009),('occupancy',(235,180,50),.04),('contacts',(255,70,75),.07)]:handles[key]=server.scene.add_point_cloud('/'+key,np.empty((0,3),np.float32),colors=color,point_size=size,precision='float32')
  handles['scan']=server.scene.add_point_cloud('/scan_reference',g['scene_reference'],colors=(155,165,175),point_size=.006,precision='float32')
  handles['frustum']=server.scene.add_line_segments('/camera_frustum',np.zeros((8,2,3),np.float32),colors=(240,60,50),thickness=2,thickness_units='screen');handles['axis']=server.scene.add_line_segments('/optical_axis',np.zeros((1,2,3),np.float32),colors=(185,45,225),thickness=3,thickness_units='screen');handles['bps']=server.scene.add_line_segments('/bps',np.zeros((67,2,3),np.float32),colors=(150,60,240),thickness=1,thickness_units='screen')
  handles['path']=server.scene.add_line_segments('/camera_path',np.stack((g['camera'][:-1],g['camera'][1:]),1),colors=(210,70,60),thickness=1,thickness_units='screen')
 def update():
  nonlocal active
  with lock,server.atomic():
   m,g,s=cases[select.value];t=int(frame.value)
   if active!=select.value:
    active=select.value;rebuild(g,m);ego_clients.clear()
    center=g['training_vertices'][t].mean(0);server.initial_camera.position=tuple(center+[2.2,1.4,2.2]);server.initial_camera.look_at=tuple(center)
    for c in server.get_clients().values():external(c,g)
   h=handles;h['gt'].vertices=g['training_vertices'][t];h['gt'].faces=g['self_faces'] if ego_clients else g['training_faces'];h['gt'].visible=checks['训练GT mesh'].value;h['raw'].vertices=g['body_0_vertices'][t];h['raw'].faces=g['self_faces'] if ego_clients else g['body_0_faces'];h['raw'].visible=checks['完整拟合mesh'].value
   for i,node in enumerate(h['others'],1):node.vertices=g[f'body_{i}_vertices'][t];node.visible=checks['另一人'].value
   from scipy.spatial.transform import Rotation
   for i,node in enumerate(h['objects']):node.wxyz=np.roll(Rotation.from_matrix(g[f'object_{i}_rotation'][t]).as_quat(),1);node.position=g[f'object_{i}_position'][t];node.visible=checks['动态物体mesh'].value
   own=g['visible_owners'][t];pts=g['visible_points'][t];h['static'].points=pts[own==0];h['dynamic'].points=pts[(own>0)&(own<100)];h['static'].visible=h['dynamic'].visible=checks['当前可见点'].value
   end=np.searchsorted(g['memory_times'],g['source_frame_ids'][t],side='left');memory=g['memory_points'][:end];step=max(1,int(np.ceil(len(memory)/60000)));h['memory'].points=memory[::step];h['memory'].visible=checks['此前静态记忆'].value;h['scan'].visible=checks['完整扫描参考'].value;h['occupancy'].points=g['occupancy_centers'];h['occupancy'].visible=checks['起点occupancy'].value
   valid=s['contact_valid'][t].astype(bool);contact=valid&(s['contact_target'][t]>.5);h['contacts'].points=g['joints'][t][contact];h['contacts'].visible=checks['区域接触标签'].value
   pos=g['camera'][t];rot=g['rotation'][t];fx,fy,cx,cy=g['intrinsics'][t];corners=.65*np.array([[(u-cx)/fx,-(v-cy)/fy,1] for u,v in [(0,0),(128,0),(128,96),(0,96)]])@rot.T+pos
   h['frustum'].points=np.concatenate((np.stack((np.tile(pos,(4,1)),corners),1),np.stack((corners,np.roll(corners,-1,axis=0)),1))).astype(np.float32);h['axis'].points=np.stack((pos,pos+.45*rot[:,2]))[None].astype(np.float32)
   for k in ['frustum','axis','path']:h[k].visible=checks['相机'].value
   bvalid=s['bps_valid'][t].astype(bool).reshape(-1);h['bps'].points=np.stack((g['bps_anchors'][t][bvalid],g['bps_endpoints'][t][bvalid]),1);h['bps'].visible=checks['BPS偏移'].value
   img.image=owner_rgb(g['ray_owner'][t]);depth=g['depth'][t];gray=np.where(np.isfinite(depth),np.clip(1-depth/4,0,1)*255,0).astype(np.uint8);depth_img.image=np.repeat(gray[:,:,None],3,axis=2)
   for cid,c in server.get_clients().items():
    if cid in ego_clients:set_ego(c,g,t)
   object_text='；'.join(f"{o['name']} 位移{o['displacement_m']:.2f}m/转动{o['rotation_change_degrees']:.0f}°" for o in m['objects']) or '无跟踪物体'
   info.content=f"**{m['identity']['sequence_id']}** · {'已恢复的动态训练窗口' if m['key']=='trumans_source_dynamic' else '当前合法训练窗口'}\n\n帧{t}/127 · {t/20:.2f}s · 原始30Hz帧号{g['source_frame_ids'][t]:g}\n\n原生人体：{m['identity']['native_body']['model']}；观察协议：{m['camera_protocol']}\n\n当前静态点{int((own==0).sum())}，动态点{int(((own>0)&(own<100)).sum())}；此前静态记忆{end}点（显示{len(memory[::step])}点）。\n\n他人遮挡像素{int((g['ray_owner'][t]>100).sum())}；自身遮挡{int((g['ray_owner'][t]==100).sum())}。\n\n区域接触：有效{int(valid.sum())}，阳性{int(contact.sum())}（红点是区域关节标记，非实际接触顶点）。\n\n{object_text}\n\n128帧全部重投影复核：深度最大差异{m['cached_visible_point_recast_max_depth_error_m']*1000:.4f}mm，归属不一致{m['cached_visible_point_recast_owner_disagreements']}。"
 @select.on_update
 def _(_):
  playing.value=False;frame.value=0;update()
 for control in [frame,*checks.values()]:control.on_update(lambda _:update())
 @reset.on_click
 def _(_):
  ego_clients.clear();m,g,s=cases[select.value]
  for c in server.get_clients().values():external(c,g)
  update()
 @ego.on_click
 def _(_):
  for cid in server.get_clients():ego_clients.add(cid)
  update()
 @server.on_client_connect
 def _(client):external(client,cases[select.value][1])
 update();print(f'READY http://127.0.0.1:{args.port}',flush=True)
 while True:
  time.sleep(.05/float(speed.value))
  if playing.value:frame.value=(int(frame.value)+1)%128
if __name__=='__main__':main()
