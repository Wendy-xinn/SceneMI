"""Viser mesh + actual ego first-hit view. Green cloud never includes body hits."""
import argparse
import json
import threading
import time
from pathlib import Path
import numpy as np
import viser


def owner_rgb(owner):
    rgb=np.zeros((*owner.shape,3),np.uint8)+240
    rgb[owner==0]=[140,155,165]
    rgb[(owner>0)&(owner<100)]=[245,175,40]
    rgb[owner==100]=[45,155,235]
    rgb[owner>100]=[210,65,155]
    return rgb


def main():
    p=argparse.ArgumentParser();p.add_argument('--clip',type=Path,required=True);p.add_argument('--port',type=int,default=8780);p.add_argument('--initial-frame',type=int,default=0);a=p.parse_args()
    def load(n):return np.load(a.clip/(n+'.npy'),mmap_mode='r')
    meta=json.loads((a.clip/'metadata.json').read_text());verts=load('body_vertices_scenemi_yup');faces=load('body_faces')
    pos=load('camera_position_scenemi_yup');rot=load('camera_rotation_scenemi_yup')
    points=load('visible_frame_points_scenemi_yup');mask=load('visible_frame_mask');owners=load('owner');scene=load('scene_points_scenemi_yup');source=load('source_frame_ids')
    server=viser.ViserServer(port=a.port);server.scene.set_up_direction('+y')
    if not 0 <= a.initial_frame < len(verts):
        raise ValueError('initial-frame outside clip')
    center=np.mean(verts[a.initial_frame],axis=0);initial=center+np.array([2,1.2,2])
    server.initial_camera.position=tuple(initial);server.initial_camera.look_at=tuple(center);server.initial_camera.up=(0,1,0)
    body=server.scene.add_mesh_simple('/body',verts[0],faces,color=(45,155,235))
    objects=[]
    for i,m in enumerate(meta.get('objects',[])):
        v=load(f'object_{i}_vertices');f=load(f'object_{i}_faces');h=server.scene.add_mesh_simple(f'/objects/{i}',v[0],f,color=(245,175,40));objects.append((h,v))
    others=[]
    if (a.clip/'occluder_body_vertices_scenemi_yup.npy').exists():
        v=load('occluder_body_vertices_scenemi_yup');nv=len(verts[0])
        for i in range(1,v.shape[1]//nv):
            h=server.scene.add_mesh_simple(f'/other_people/{i}',v[0,i*nv:(i+1)*nv],faces,color=(210,65,155));others.append((h,v[:,i*nv:(i+1)*nv]))
    scan=server.scene.add_point_cloud('/scan_debug_only',scene,colors=(160,170,180),point_size=.01)
    cloud=server.scene.add_point_cloud('/current_observation',points[0][mask[0]],colors=(30,240,70),point_size=.015)
    # Keep the diagnostic red wireframe.  Do not add a solid camera ball:
    # that marker sits directly on the face and obscures the mesh in external
    # inspection views.
    frustum=server.scene.add_line_segments('/frustum',np.zeros((8,2,3),np.float32),colors=(240,70,40),thickness=2,thickness_units='screen')
    # Optical-axis marker: the red wireframe is a perspective frustum and can
    # look tilted from an arbitrary external observer view.  This short purple
    # line makes the actual viewing direction unambiguous.
    optical_axis=server.scene.add_line_segments('/camera_optical_axis',np.zeros((1,2,3),np.float32),colors=(190,40,220),thickness=3,thickness_units='screen')
    server.gui.add_markdown(f"### {meta.get('dataset','RICH')} · {meta['sequence_id']}\n\n版本：逐帧网格最近交点 v2。完整场景/物体/人体用于检查，**不等于全部输入模型**。\n\n第一视角深度归属：灰=静态，黄=物体，蓝=自身，紫红=他人，白=无有效深度。绿点只含当前帧可见场景/物体。自身只排除头部皮肤，其余身体保留遮挡。")
    slider=server.gui.add_slider('frame',min=0,max=len(verts)-1,step=1,initial_value=a.initial_frame)
    playing=server.gui.add_checkbox('播放',initial_value=False)
    speed=server.gui.add_number('倍速',initial_value=1.,min=.25,max=2.,step=.25)
    show=server.gui.add_checkbox('显示完整扫描（仅诊断）',initial_value=True)
    reset=server.gui.add_button('重置视角');ego=server.gui.add_button('切到虚拟相机');info=server.gui.add_markdown('')
    img=server.gui.add_image(owner_rgb(owners[0]),label='实际 ego 深度归属')
    lock=threading.Lock()
    ego_clients=set()
    def set_ego_camera(c,t):
        c.camera.position=tuple(pos[t]);c.camera.look_at=tuple(pos[t]+rot[t,:,2]);c.camera.up_direction=tuple(rot[t,:,1]);c.camera.fov=np.deg2rad(40.49)
    @show.on_update
    def _(_):scan.visible=show.value
    @reset.on_click
    def _(_):
        body.faces=faces
        ego_clients.clear()
        for c in server.get_clients().values():c.camera.position=tuple(initial);c.camera.look_at=tuple(center);c.camera.up_direction=(0,1,0)
    @ego.on_click
    def _(_):
        t=int(slider.value)
        body.faces=load('self_occlusion_faces')
        for c in server.get_clients().values():
            ego_clients.add(c.client_id)
            set_ego_camera(c,t)
    def update(t):
        with lock,server.atomic():
            body.vertices=verts[t]
            for h,v in objects+others:h.vertices=v[t]
            for client_id,c in server.get_clients().items():
                if client_id in ego_clients:set_ego_camera(c,t)
            cloud.points=points[t][mask[t]]
            tx=np.tan(np.deg2rad(66.56/2));ty=np.tan(np.deg2rad(40.49/2))
            corners=.7*np.array([[-tx,-ty,1],[tx,-ty,1],[tx,ty,1],[-tx,ty,1]])@rot[t].T+pos[t]
            # The line width is screen-space, so the rays can meet at the
            # true optical centre without becoming a world-sized red ball.
            frustum.points=np.concatenate((np.stack((np.tile(pos[t],(4,1)),corners),axis=1),np.stack((corners,np.roll(corners,-1,axis=0)),axis=1))).astype(np.float32)
            optical_axis.points=np.stack((pos[t],pos[t]+.45*rot[t,:,2]),axis=0)[None].astype(np.float32)
            img.image=owner_rgb(owners[t])
            info.content=f"帧 {t} / {len(verts)-1}；源帧 {source[t]:g}；{meta['output_fps']:g} Hz\n\n自身遮挡像素 {(owners[t]==100).sum()}；他人 {(owners[t]>100).sum()}；物体 {((owners[t]>0)&(owners[t]<100)).sum()}\n\n相机协议：{meta['camera_protocol']}"
    @slider.on_update
    def _(_):update(int(slider.value))
    update(a.initial_frame)
    print(f'http://127.0.0.1:{a.port} {a.clip}',flush=True)
    while True:
        time.sleep(1/(meta['output_fps']*float(speed.value)))
        if playing.value:slider.value=(int(slider.value)+1)%len(verts)


if __name__=='__main__':main()
