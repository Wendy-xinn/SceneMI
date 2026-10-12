"""Six-case viser review of simulated exo inputs, occluders and confidence.

No model/checkpoint is loaded. Full scans/bodies are labelled reference-only;
the model environment cloud excludes target skin and unknown surfaces.
"""
import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import viser
import trimesh
import xml.etree.ElementTree as ET

from experiments.offline_camera_retrain_v1.exo_occlusion_review import RUN, JOINTS
from experiments.offline_camera_retrain_v1.data import PARENTS

HIGH = '高置信度筛选 ≥ 0.7（诊断）'
CURRENT = '当前配置：有效关节 + confidence 降权'


def playback_next(frame, count, loop=False):
    """Do not present a noncyclic clip boundary as one adjacent motion frame."""
    if frame+1<count:return frame+1,True
    return (0,True) if loop else (count-1,False)


def owner_rgb(owner):
    color=np.full((*owner.shape,3),240,np.uint8)
    color[owner==0]=[140,155,165]
    color[(owner>0)&(owner<100)]=[235,175,35]
    color[owner==100]=[45,145,230]
    color[owner>100]=[205,55,145]
    return color


def source_image(arrays,t,high=True):
    # Explicitly a geometry rendering, never presented as source RGB.
    image=cv2.resize(owner_rgb(arrays['owner'][t]),None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
    uv=arrays['joint_pixels'][t]*2;inside=arrays['joint_in_fov'][t]
    mask=arrays['high' if high else 'available'][t]
    for j,p in enumerate(uv):
        if not inside[j]:continue
        c=(20,235,90) if mask[j] else (255,75,65)
        cv2.circle(image,tuple(np.round(p).astype(int)),4,c,-1,lineType=cv2.LINE_AA)
        cv2.putText(image,str(j),tuple(np.round(p+[5,-3]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.32,(10,10,10),1,cv2.LINE_AA)
    return image


def trajectory_segments(positions,mask,end,length=90,joint=None):
    first=max(0,end-length)
    pair=mask[first:end]&mask[first+1:end+1]
    if joint is not None:
        selected=np.zeros_like(pair);selected[:,joint]=pair[:,joint];pair=selected
    # Never connect across a missing frame or hide a gap by interpolation.
    segments=np.stack((positions[first:end],positions[first+1:end+1]),-2)[pair]
    return segments.astype(np.float32).reshape(-1,2,3)


def display_scan_mesh(case):
    """GT-bounded crop of the source scan for display only; never a condition."""
    mesh=trimesh.load(case['source_mesh'],force='mesh',process=False)
    vertices=np.asarray(mesh.vertices,np.float32)
    if case['group']=='rich':
        camera=case['camera'];root=ET.parse(camera['calibration']).getroot()
        extrinsic=np.fromstring(root.find('./CameraMatrix/data').text,sep=' ').reshape(3,4)
        r=extrinsic[:,:3];c=-r.T@extrinsic[:,3];wr=np.array(camera['rotation'])@r
        scale=case['native_body']['scale'];wt=np.array(camera['position'])-scale*wr@c
        vertices=(scale*vertices@wr.T+wt).astype(np.float32)
    joints=np.load(Path(case['path'])/'gt_joints_reference.npy',mmap_mode='r')
    lower=joints.min((0,1))-[1.5,.3,1.5];upper=joints.max((0,1))+[1.5,.5,1.5]
    faces=np.asarray(mesh.faces,np.int32);selected=[]
    for first in range(0,len(faces),100000):
        chunk=faces[first:first+100000];triangles=vertices[chunk]
        mask=(triangles.max(1)>=lower).all(1)&(triangles.min(1)<=upper).all(1)
        selected.append(chunk[mask])
    faces=np.concatenate(selected)
    ids,inverse=np.unique(faces,return_inverse=True)
    return vertices[ids],inverse.reshape(-1,3).astype(np.int32)


class ReviewViewer:
    def __init__(self,manifest,host='0.0.0.0',port=8783):
        self.manifest=json.loads(Path(manifest).read_text());self.cases=self.manifest['cases']
        self.server=server=viser.ViserServer(host=host,port=port,label='SceneMI · exo 遮挡与可信轨迹检查')
        server.scene.set_up_direction('+y');self.lock=threading.RLock();self.handles=[];self.arrays={};self.pose_clients=set()
        self.labels=[f"{i+1}. {c['spec']['id']}" for i,c in enumerate(self.cases)]
        server.gui.add_markdown('## Exo 遮挡检查 · 6 个片段\n\n绿=输入关节，红=未输入 GT，蓝=目标，紫红=他人，黄=刚体。**模拟输入检查；训练暂停。**')
        self.select=server.gui.add_dropdown('片段',self.labels,initial_value=self.labels[0])
        self.policy=server.gui.add_dropdown('关节输入策略',[CURRENT,HIGH],initial_value=CURRENT)
        self.frame=server.gui.add_slider('帧',min=0,max=191,step=1,initial_value=0)
        self.header=server.gui.add_markdown('');self.info=server.gui.add_markdown('')
        self.image=server.gui.add_image(np.zeros((216,384,3),np.uint8),label='几何 first-hit 与关节筛选（非 RGB）')
        self.play=server.gui.add_checkbox('播放',initial_value=False)
        self.loop=server.gui.add_checkbox('循环播放（末尾→开头有跳变）',initial_value=False)
        self.speed=server.gui.add_number('倍速',initial_value=1.,min=.25,max=2.,step=.25)
        self.joint=server.gui.add_dropdown('检查关节',['全部']+[f'{j}: {n}' for j,n in enumerate(JOINTS)],initial_value='全部')
        self.body_visible=server.gui.add_checkbox('目标完整网格（GT 参考）',initial_value=True)
        self.others_visible=server.gui.add_checkbox('遮挡人/物体完整网格（参考）',initial_value=True)
        self.scan_visible=server.gui.add_checkbox('扫描点云与近人网格（参考）',initial_value=False)
        self.current_visible=server.gui.add_checkbox('当前帧可见环境输入',initial_value=True)
        self.memory_visible=server.gui.add_checkbox('本窗口观察静态记忆输入',initial_value=False)
        self.hidden_visible=server.gui.add_checkbox('未输入关节（红色 GT 参考）',initial_value=True)
        self.trails_visible=server.gui.add_checkbox('可信轨迹（近 3 秒，不跨缺失）',initial_value=True)
        critical=server.gui.add_button('跳到外部遮挡最强帧')
        camera=server.gui.add_button('切到 exo 光学视角')
        reset=server.gui.add_button('重置外部视角')
        self.table=server.gui.add_markdown('')
        audit_path=RUN/'FOLLOWUP_AUDIT.json'
        self.source_audit=json.loads(audit_path.read_text()) if audit_path.exists() else None

        @self.select.on_update
        def _(_):self.load_case(self.labels.index(self.select.value))
        for control in [self.frame,self.policy,self.joint,self.body_visible,self.others_visible,self.scan_visible,self.current_visible,self.memory_visible,self.hidden_visible,self.trails_visible]:
            control.on_update(lambda _: self.update(int(self.frame.value)))
        @critical.on_click
        def _(_):self.frame.value=self.case['critical_frame']
        @camera.on_click
        def _(_):
            for client in server.get_clients().values():
                self.pose_clients.add(client.client_id);self.set_camera(client)
        @reset.on_click
        def _(_):
            self.pose_clients.clear()
            for client in server.get_clients().values():self.set_camera(client,external=True)
        @server.on_client_connect
        def _(client):self.set_camera(client,external=True)
        self.load_case(0)

    def load(self,name):
        if name not in self.arrays:self.arrays[name]=np.load(self.folder/(name+'.npy'),mmap_mode='r')
        return self.arrays[name]

    def load_case(self,index):
        with self.lock,self.server.atomic():
            self.play.value=False;self.case=self.cases[index];self.folder=Path(self.case['path']);self.arrays={}
            for h in self.handles:h.remove()
            self.handles=[];self.meshes=[];self.objects=[];scene=self.server.scene
            t=0
            for i in range(self.case['people']):
                v=self.load(f'person_{i}_vertices');f=self.load(f'person_{i}_faces')
                handle=scene.add_mesh_simple(f'/case/person_{i}',v[t],f,color=(45,145,230) if i==0 else (205,55,145),opacity=.45 if i==0 else 1.,side='double')
                self.meshes.append((handle,v));self.handles.append(handle)
            for i,obj in enumerate(self.case['objects']):
                v=self.load(f'object_{i}_vertices');r=self.load(f'object_{i}_rotations');p=self.load(f'object_{i}_positions')
                h=scene.add_mesh_simple(f'/case/object_{i}',v@r[t].T+p[t],self.load(f'object_{i}_faces'),color=(235,175,35),side='double')
                self.objects.append((h,v,r,p,self.load(f'object_{i}_valid')));self.handles.append(h)
            # Retain all selected scene samples as data; a voxel union defines memory.
            fp=self.load('frame_points');owner=self.load('frame_owner');static=fp[owner==0]
            if len(static):
                _,ids=np.unique(np.floor(static/.025).astype(np.int32),axis=0,return_index=True);static=static[np.sort(ids)]
            def point(name,p,color,size):
                h=scene.add_point_cloud(name,np.asarray(p,np.float32),colors=color,point_size=size,point_shape='circle',precision='float32');self.handles.append(h);return h
            self.scan=point('/case/complete_scan_reference',self.load('complete_scan_reference'),(105,110,120),.01)
            sv,sf=display_scan_mesh(self.case)
            self.scan_mesh=scene.add_mesh_simple('/case/near_body_scan_reference',sv,sf,color=(165,175,185),side='double',visible=False)
            self.handles.append(self.scan_mesh)
            self.memory=point('/case/observed_static_memory',static,(155,165,170),.004)
            self.current=point('/case/current_visible_environment',np.zeros((0,3)),(230,185,45),.01)
            self.trusted=point('/case/trusted_joint_inputs',np.zeros((0,3)),(20,235,90),.045)
            self.hidden=point('/case/withheld_gt_reference',np.zeros((0,3)),(255,75,65),.025)
            def lines(name,color,thickness):
                h=scene.add_line_segments(name,np.zeros((0,2,3),np.float32),colors=color,thickness=thickness,thickness_units='screen');self.handles.append(h);return h
            self.skeleton=lines('/case/input_links',(20,235,90),3.)
            self.gt_bones=lines('/case/gt_reference_bones',(85,125,210),1.)
            self.trails=lines('/case/trusted_trajectories',(20,235,90),2.)
            camera=self.case['camera'];r=np.asarray(camera['rotation']);p=np.asarray(camera['position'])
            k=np.asarray(camera['K']);sw,sh=camera['source_size']
            pixels=np.array([[0.,0.],[sw,0.],[sw,sh],[0.,sh]])
            xy=cv2.undistortPoints(pixels[:,None],k,np.asarray(camera['distortion']))[:,0]
            corners=np.c_[xy,np.ones(4)]*.65@r.T+p
            self.frustum=lines('/case/exo_frustum',(255,110,30),2.)
            self.frustum.points=np.concatenate((np.stack((np.broadcast_to(p,corners.shape),corners),1),np.stack((corners,np.roll(corners,-1,axis=0)),1))).astype(np.float32)
            self.centre=self.load('gt_joints_reference')[t].mean(0)
            self.initial=self.centre+np.array([2.5,1.5,2.5])
            self.server.initial_camera.position=tuple(self.initial);self.server.initial_camera.look_at=tuple(self.centre);self.server.initial_camera.up=(0,1,0)
            for client in self.server.get_clients().values():self.set_camera(client,external=client.client_id not in self.pose_clients)
            self.frame.max=self.case['frames']-1;self.frame.value=t
            self.header.content=f"### {self.case['spec']['description']}\n\n`{self.case['sequence_id']}`\n\n相机：{camera['name']}；{'真实标定' if camera['kind']=='released_calibrated' else '**合成诊断相机**'}。30 Hz，192 帧。\n\n高置信度阈值仅供本轮检查，**训练配置仍用当前有效关节及降权**。"
            if self.source_audit and self.case['spec']['id']==self.source_audit['ego_arm']['case']:
                self.header.content+='\n\n**源动作质量提示：2481–2501 帧右臂有拟合跳变，原始 SMPL 直接复算与本视图一致；不是生成模型输出。未平滑或替换参考动作。**'
            self.update(t)

    def set_camera(self,client,external=False):
        if external:
            client.camera.position=tuple(self.initial);client.camera.look_at=tuple(self.centre);client.camera.up_direction=(0,1,0);client.camera.fov=np.deg2rad(50)
        else:
            cam=self.case['camera'];p=np.array(cam['position']);r=np.array(cam['rotation']);k=np.array(cam['K'])
            client.camera.position=tuple(p);client.camera.look_at=tuple(p+r[:,2]);client.camera.up_direction=tuple(-r[:,1]);client.camera.fov=2*np.arctan(cam['source_size'][1]/(2*k[1,1]))

    def update(self,t):
        with self.lock,self.server.atomic():
            if not self.arrays:return
            a={k:self.load(k) for k in ['owner','joint_pixels','joint_in_fov','high','available','confidence','reference','visible','person','static','rigid','source_frame_ids','observation_valid']}
            high=self.policy.value==HIGH;mask=a['high' if high else 'available'][t]
            prefix='high' if high else 'current';positions=self.load(prefix+'_input_positions');reference=self.load('gt_joints_reference')
            j=None if self.joint.value=='全部' else int(self.joint.value.split(':')[0])
            selected=mask.copy()
            if j is not None:selected[:]=False;selected[j]=mask[j]
            for i,(handle,vertices) in enumerate(self.meshes):handle.vertices=vertices[t];handle.visible=self.body_visible.value if i==0 else self.others_visible.value
            for handle,v,r,p,valid in self.objects:handle.vertices=v@r[t].T+p[t];handle.visible=self.others_visible.value and bool(valid[t])
            self.scan.visible=self.scan_visible.value;self.scan_mesh.visible=self.scan_visible.value;self.memory.visible=self.memory_visible.value
            owners=self.load('frame_owner')[t];points=self.load('frame_points')[t];valid=owners>=0
            self.current.points=points[valid];colors=np.tile([230,185,45],(valid.sum(),1)).astype(np.uint8);colors[owners[valid]==0]=[180,190,195];colors[owners[valid]>100]=[205,55,145];self.current.colors=colors;self.current.visible=self.current_visible.value
            self.trusted.points=positions[t,selected]
            hidden=~mask
            if j is not None:hidden[:]=False;hidden[j]=not mask[j]
            self.hidden.points=reference[t,hidden];self.hidden.visible=self.hidden_visible.value
            bones=np.stack([reference[t,[parent,joint]] for joint,parent in enumerate(PARENTS) if parent>=0])
            self.gt_bones.points=bones.astype(np.float32);self.gt_bones.visible=self.body_visible.value
            links=[positions[t,[parent,joint]] for joint,parent in enumerate(PARENTS) if parent>=0 and mask[parent] and mask[joint]]
            self.skeleton.points=np.asarray(links,np.float32).reshape(-1,2,3)
            self.trails.points=trajectory_segments(positions,self.load('high' if high else 'available'),t,joint=j);self.trails.visible=self.trails_visible.value
            self.image.image=source_image(a,t,high)
            self.info.content=f"源帧 **{a['source_frame_ids'][t]}**；片段帧 {t}/{self.case['frames']-1}；时间 {t/30:.2f}s\n\n当前有效 {a['available'][t].sum()}/22；高置信度 {a['high'][t].sum()}/22；本视图输入 {mask.sum()}/22。\n\n挡住目标的像素：他人 **{a['person'][t].sum()}**；静态物体/场景 **{a['static'][t].sum()}**；动态刚体 **{a['rigid'][t].sum()}**。\n\n源观测有效：{bool(a['observation_valid'][t])}。本窗口静态记忆可含本帧被遮挡、其他帧见过的表面。"
            rows=['|关节|置信度|输入|可见/目标单独|人/静态/刚体遮挡|','|---|---:|:---:|---:|---:|']
            for index in range(22) if j is None else [j]:
                rows.append(f"|{index} {JOINTS[index]}|{a['confidence'][t,index]:.2f}|{'✓' if mask[index] else '—'}|{a['visible'][t,index]}/{a['reference'][t,index]}|{a['person'][t,index]}/{a['static'][t,index]}/{a['rigid'][t,index]}|")
            self.table.content='\n'.join(rows)+'\n\n可见/目标单独=该部位皮肤像素比例。目标单独没有像素时，原因可能是自身遮挡、出画或分辨率不足；此表不能进一步区分。部分皮肤可见允许输入该关节，不能据此声称内部关节点直接可见。'

    def run(self):
        print('EXO_VISER_READY',flush=True)
        while True:
            time.sleep(1/(30*float(self.speed.value)))
            with self.lock:
                if self.play.value:
                    next_frame,playing=playback_next(int(self.frame.value),self.case['frames'],self.loop.value)
                    self.frame.value=next_frame;self.play.value=playing


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=RUN/'manifest.json');p.add_argument('--host',default='0.0.0.0');p.add_argument('--port',type=int,default=8783)
    args=p.parse_args();ReviewViewer(args.manifest,args.host,args.port).run()


if __name__=='__main__':main()
