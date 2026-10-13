"""Review six occluded validation inputs directly from the formal raw30 cache.

No model or optimizer is loaded. Display-only target joints are clearly
labelled GT; scene points and observed tracks come from the exported inputs.
"""
import argparse
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import viser

from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.native30_memory import ObservedStaticMemory
from experiments.offline_camera_retrain_v1.supervision import PARENTS
from experiments.offline_camera_retrain_v1.view_exo_occlusion_review import playback_next,trajectory_segments


def select_cases(manifest,frames=192):
    entries=[json.loads(line) for line in Path(manifest).read_text().splitlines()]
    candidates={dataset:[] for dataset in ('egobody','rich','trumans')}
    for meta in entries:
        if meta['split']!='validation' or 'exo' not in meta['allowed_modes'] or meta['frames']<frames:
            continue
        folder=Path(meta['path'])
        def load(name):return np.load(folder/(name+'.npy'),mmap_mode='r')
        valid=load('observation_valid');available=load('exo_position_available').any(1)
        reference=load('exo_reference_counts').sum(1)
        blocked=sum(load('exo_'+kind+'_counts').sum(1) for kind in ('person','static','rigid'))
        def window_sum(values):
            prefix=np.r_[0,np.cumsum(values)];return prefix[frames:]-prefix[:-frames]
        eligible=(window_sum(~valid)==0)&(window_sum(available)>=16)
        score=np.where(eligible,window_sum(blocked)/np.maximum(window_sum(reference),1),-1)
        start=int(np.argmax(score))
        if score[start]<0:continue
        candidates[meta['dataset']].append(dict(path=meta['path'],sequence_id=meta['sequence_id'],
            dataset=meta['dataset'],group=meta['group'],start=start,frames=frames,
            source_first=meta['source_first']+start,external_occlusion_fraction=float(score[start]),
            selection='validation display only: strongest external-occlusion window with >=16 observed frames',
            camera_kind=meta['exo_camera']['kind'],view_id=meta['exo_camera']['view_id']))
    chosen=[]
    for dataset in ('egobody','rich','trumans'):
        ranked=sorted(candidates[dataset],key=lambda row:(-row['external_occlusion_fraction'],row['sequence_id']))
        if len(ranked)<2:raise ValueError('Need two formal validation exo sources for '+dataset)
        first=ranked[0]
        # Show both EgoBody roles when available, without duplicating one stream.
        second=next((r for r in ranked[1:] if dataset!='egobody' or r['group']!=first['group']),ranked[1])
        chosen.extend((first,second))
    return entries,chosen


def review_cases(manifest,cases_file=None):
    if cases_file is None:return select_cases(manifest)
    entries=[json.loads(line) for line in Path(manifest).read_text().splitlines()]
    cases=json.loads(Path(cases_file).read_text())['cases']
    for case in cases:
        meta=next(m for m in entries if m['path']==case['path'])
        if meta['split']!='validation' or 'exo' not in meta['allowed_modes'] or meta['sequence_id']!=case['sequence_id']:
            raise ValueError('Frozen review case is not an external validation input')
    return entries,cases


def save_snapshot(manifest,output,cases_file=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from experiments.offline_camera_retrain_v1.native30_cache_mesh_review import reference_geometry
    entries,cases=review_cases(manifest,cases_file);arrays={}
    def load(folder,name):
        key=(folder,name)
        if key not in arrays:arrays[key]=np.load(Path(folder)/(name+'.npy'),mmap_mode='r')
        return arrays[key]
    memory=ObservedStaticMemory(load,entries);fig=plt.figure(figsize=(16,10))
    for i,case in enumerate(cases):
        meta=next(m for m in entries if m['path']==case['path']);folder=case['path']
        sl=slice(case['start'],case['start']+192)
        blocked=sum(load(folder,'exo_'+k+'_counts')[sl].sum(1) for k in ('person','static','rigid'))
        t=case['start']+int(np.argmax(blocked))
        gt=load(folder,'motion_world')[t,135:].reshape(22,3)*2
        identity=dict(meta,start=case['start'],origin=[0,0,0],anchor=np.eye(3).tolist())
        static,_=memory.window(identity,'exo',192)
        # GT-centred crop is for drawing only, never written to an input cache.
        crop=((static>=gt.min(0)-[1.5,.3,1.5])&(static<=gt.max(0)+[1.5,.4,1.5])).all(1)
        static=static[crop][::max(1,int(crop.sum())//6000)]
        ax=fig.add_subplot(2,3,i+1,projection='3d')
        geometry=reference_geometry(meta,case['start'])
        body=geometry['bodies'][0];mesh_vertices=body['vertices'][t-case['start']]
        triangles=mesh_vertices[body['faces']][:,:,[0,2,1]]
        ax.add_collection3d(Poly3DCollection(triangles,facecolors='royalblue',edgecolors='none',alpha=.3))
        if len(static):ax.scatter(static[:,0],static[:,2],static[:,1],s=.4,c='gray',alpha=.3)
        for j,p in enumerate(PARENTS):
            if p>=0:
                edge=gt[[p,j]];ax.plot(edge[:,0],edge[:,2],edge[:,1],c='royalblue',lw=1.6)
        available=load(folder,'exo_position_available')[t];pos=load(folder,'exo_trajectory_world')[t,available]
        confidence=load(folder,'exo_position_confidence')[t,available]
        if len(pos):ax.scatter(pos[:,0],pos[:,2],pos[:,1],c=confidence,vmin=0,vmax=1,cmap='summer',s=25)
        owners=load(folder,'exo_frame_owner')[t];dynamic=load(folder,'exo_frame_points_world')[t,(owners>0)&(owners!=100)]
        if len(dynamic):ax.scatter(dynamic[:,0],dynamic[:,2],dynamic[:,1],s=6,c='darkorange')
        cam=meta['exo_camera'];p=np.asarray(cam['position']);r=np.asarray(cam['rotation'])
        ax.scatter(*p[[0,2,1]],marker='^',c='red',s=30)
        ax.quiver(*p[[0,2,1]],*r[[0,2,1],2],length=.65,color='red')
        ax.set(title=f"{case['dataset']} / {case['group']}\nsource frame {meta['source_first']+t}; {available.sum()}/22 input joints",
            xlabel='X (m)',ylabel='Z (m)',zlabel='Y (m)')
        ax.set_box_aspect((1,1,1));ax.view_init(elev=18,azim=-65)
    fig.suptitle('Formal raw30 inputs: observed scene, trusted tracks, GT reference; TRUMANS camera is synthetic')
    fig.tight_layout();fig.savefig(output,dpi=140);plt.close(fig)


class CacheViewer:
    def __init__(self,manifest,host='0.0.0.0',port=8784,cases_output=None,cases_file=None):
        self.entries,self.cases=review_cases(manifest,cases_file)
        if cases_output:
            Path(cases_output).write_text(json.dumps(dict(manifest=str(Path(manifest).resolve()),
                scope='formal validation input review; no generated motion',optimizer_steps=0,
                cases=self.cases),indent=2)+'\n')
        self.arrays={};self.handles=[];self.lock=threading.RLock();self.optical_clients=set()
        self.geometry_cache={}
        self.memory=ObservedStaticMemory(self.load,self.entries)
        self.server=server=viser.ViserServer(host=host,port=port,label='SceneMI · 正式30Hz exo输入')
        server.scene.set_up_direction('+y')
        server.gui.add_markdown('## 正式缓存 · 六个遮挡窗口\n\n'
            '绿/黄=可信位置输入（亮度随confidence）；红=未输入GT参考；蓝=原生SMPL/SMPL-X参考mesh。'
            '灰=同视点整段已见静态记忆；紫红/黄=当帧可见人/刚体表面。**训练暂停，无生成结果。**')
        self.labels=[f"{i+1}. {c['dataset']} / {c['group']}" for i,c in enumerate(self.cases)]
        self.select=server.gui.add_dropdown('窗口',self.labels,initial_value=self.labels[0])
        self.frame=server.gui.add_slider('帧',min=0,max=191,step=1,initial_value=0)
        self.play=server.gui.add_checkbox('播放',initial_value=False)
        self.gt_visible=server.gui.add_checkbox('共22关节GT参考',initial_value=True)
        self.mesh_visible=server.gui.add_checkbox('目标原生GT mesh',initial_value=True)
        self.occluders_visible=server.gui.add_checkbox('人/刚体完整mesh（仅参考）',initial_value=True)
        self.scan_visible=server.gui.add_checkbox('近人场景mesh（完整扫描参考）',initial_value=False)
        self.memory_visible=server.gui.add_checkbox('整段已见静态记忆',initial_value=True)
        self.current_visible=server.gui.add_checkbox('当前可见环境表面',initial_value=True)
        self.trails_visible=server.gui.add_checkbox('输入轨迹（不跨缺失）',initial_value=True)
        self.threshold=server.gui.add_slider('显示confidence下限（仅显示）',min=0,max=1,step=.05,initial_value=0)
        self.info=server.gui.add_markdown('');self.table=server.gui.add_markdown('')
        strongest=server.gui.add_button('跳到外部遮挡最多的帧')
        optical=server.gui.add_button('切到输入相机视角')
        external=server.gui.add_button('回到外部检查视角')
        @self.select.on_update
        def _(_):self.load_case(self.labels.index(self.select.value))
        for control in (self.frame,self.gt_visible,self.mesh_visible,self.occluders_visible,self.scan_visible,
                        self.memory_visible,self.current_visible,self.trails_visible,self.threshold):
            control.on_update(lambda _:self.update(int(self.frame.value)))
        @strongest.on_click
        def _(_):self.frame.value=int(np.argmax(self.blocked.sum(1)))
        @optical.on_click
        def _(_):
            for client in server.get_clients().values():
                self.optical_clients.add(client.client_id);self.set_camera(client)
        @external.on_click
        def _(_):
            self.optical_clients.clear()
            for client in server.get_clients().values():self.set_camera(client)
        @server.on_client_connect
        def _(client):self.set_camera(client)
        self.load_case(0)

    def load(self,folder,name):
        key=(str(folder),name)
        if key not in self.arrays:self.arrays[key]=np.load(Path(folder)/(name+'.npy'),mmap_mode='r')
        return self.arrays[key]

    def load_case(self,index):
        with self.lock,self.server.atomic():
            self.play.value=False;self.case=self.cases[index];folder=self.case['path'];self.arrays.clear()
            self.meta=next(m for m in self.entries if m['path']==folder)
            start=self.case['start'];sl=slice(start,start+192)
            from experiments.offline_camera_retrain_v1.native30_cache_mesh_review import reference_geometry
            key=(folder,start)
            if key not in self.geometry_cache:
                self.geometry_cache[key]=reference_geometry(self.meta,start)
                if len(self.geometry_cache)>2:del self.geometry_cache[next(iter(self.geometry_cache))]
            self.geometry=self.geometry_cache[key]
            self.reference=np.asarray(self.load(folder,'motion_world')[sl,135:]).reshape(192,22,3)*2
            self.positions=np.asarray(self.load(folder,'exo_trajectory_world')[sl])
            self.available=np.asarray(self.load(folder,'exo_position_available')[sl])
            self.confidence=np.asarray(self.load(folder,'exo_position_confidence')[sl])
            counts=self.available.sum(1)
            edges=np.diff(np.r_[False,counts==0,False].astype(np.int8))
            gaps=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
            self.window_observed_frames=int((counts>0).sum())
            self.window_mean_joints=float(counts.mean())
            self.window_max_empty_frames=int(gaps.max()) if len(gaps) else 0
            self.parts={k:np.asarray(self.load(folder,'exo_'+k+'_counts')[sl]) for k in ('reference','visible','person','static','rigid')}
            self.blocked=sum(self.parts[k] for k in ('person','static','rigid'))
            self.frame_points=self.load(folder,'exo_frame_points_world')[sl];self.owners=self.load(folder,'exo_frame_owner')[sl]
            identity=dict(self.meta,start=start,origin=[0,0,0],anchor=np.eye(3).tolist())
            static,_=self.memory.window(identity,'exo',192)
            if len(static)>80000:static=static[::int(np.ceil(len(static)/80000))]
            for handle in self.handles:handle.remove()
            self.handles=[];scene=self.server.scene
            self.meshes=[];self.object_meshes=[]
            for i,body in enumerate(self.geometry['bodies']):
                handle=scene.add_mesh_simple(f'/review/native_body_{i}',body['vertices'][0],body['faces'],
                    color=(45,145,230) if i==0 else (205,55,145),opacity=.4 if i==0 else .8,side='double')
                self.meshes.append(handle);self.handles.append(handle)
            for i,obj in enumerate(self.geometry['objects']):
                handle=scene.add_mesh_simple(f'/review/rigid_reference_{i}',
                    obj['vertices']@obj['rotations'][0].T+obj['positions'][0],obj['faces'],
                    color=(235,175,35),opacity=.8,side='double')
                self.object_meshes.append(handle);self.handles.append(handle)
            self.scan_mesh=scene.add_mesh_simple('/review/scene_scan_reference',self.geometry['scene_vertices'],
                self.geometry['scene_faces'],color=(155,165,175),opacity=.5,side='double',visible=False)
            self.handles.append(self.scan_mesh)
            def points(name,p,color,size):
                h=scene.add_point_cloud('/review/'+name,np.asarray(p,np.float32),colors=color,
                    point_size=size,point_shape='circle',precision='float32');self.handles.append(h);return h
            def lines(name,color,width):
                h=scene.add_line_segments('/review/'+name,np.zeros((0,2,3),np.float32),colors=color,
                    thickness=width,thickness_units='screen');self.handles.append(h);return h
            self.static=points('observed_static',static,(160,170,180),.007)
            self.current=points('current_environment',np.zeros((0,3)),(220,185,40),.012)
            self.inputs=points('trusted_positions',np.zeros((0,3)),(50,225,80),.04)
            self.hidden=points('withheld_GT_reference',np.zeros((0,3)),(235,75,70),.023)
            self.gt_bones=lines('GT_reference_bones',(80,135,220),2)
            self.input_links=lines('trusted_links',(40,225,80),3)
            self.trails=lines('input_trajectories',(40,225,80),2)
            self.frustum=lines('input_camera',(255,120,40),2)
            cam=self.meta['exo_camera'];r=np.asarray(cam['rotation']);p=np.asarray(cam['position'])
            w,h=cam['source_size'];pixels=np.array([[0.,0.],[w,0.],[w,h],[0.,h]])
            xy=cv2.undistortPoints(pixels[:,None],np.asarray(cam['K']),np.asarray(cam['distortion']))[:,0]
            corners=np.c_[xy,np.ones(4)]*.65@r.T+p
            self.frustum.points=np.concatenate((np.stack((np.broadcast_to(p,corners.shape),corners),1),
                np.stack((corners,np.roll(corners,-1,axis=0)),1))).astype(np.float32)
            self.centre=self.reference[0].mean(0);self.initial=self.centre+[2.5,1.5,2.5]
            self.frame.value=0
            server=self.server
            server.initial_camera.position=tuple(self.initial);server.initial_camera.look_at=tuple(self.centre)
            server.initial_camera.up=(0,1,0)
            for client in server.get_clients().values():self.set_camera(client)
            self.update(0)

    def set_camera(self,client):
        if not hasattr(self,'initial'):return
        if client.client_id in self.optical_clients:
            cam=self.meta['exo_camera'];p=np.asarray(cam['position']);r=np.asarray(cam['rotation'])
            client.camera.position=tuple(p);client.camera.look_at=tuple(p+r[:,2]);client.camera.up_direction=tuple(-r[:,1])
            client.camera.fov=2*np.arctan(cam['source_size'][1]/(2*cam['K'][1][1]))
        else:
            client.camera.position=tuple(self.initial);client.camera.look_at=tuple(self.centre)
            client.camera.up_direction=(0,1,0);client.camera.fov=np.deg2rad(50)

    def update(self,t):
        if not hasattr(self,'reference'):return
        with self.lock,self.server.atomic():
            for i,handle in enumerate(self.meshes):
                handle.vertices=self.geometry['bodies'][i]['vertices'][t]
                handle.visible=self.mesh_visible.value if i==0 else self.occluders_visible.value
            for handle,obj in zip(self.object_meshes,self.geometry['objects']):
                handle.vertices=obj['vertices']@obj['rotations'][t].T+obj['positions'][t]
                handle.visible=self.occluders_visible.value and bool(obj['valid'][t])
            self.scan_mesh.visible=self.scan_visible.value
            mask=self.available[t]&(self.confidence[t]>=self.threshold.value)
            self.inputs.points=self.positions[t,mask]
            colors=np.stack((220*(1-self.confidence[t,mask]),120+115*self.confidence[t,mask],np.full(mask.sum(),65)),1)
            self.inputs.colors=colors.astype(np.uint8)
            self.hidden.points=self.reference[t,~mask];self.hidden.visible=self.gt_visible.value
            self.gt_bones.points=np.stack([self.reference[t,[p,j]] for j,p in enumerate(PARENTS) if p>=0]).astype(np.float32)
            self.gt_bones.visible=self.gt_visible.value
            self.input_links.points=np.asarray([self.positions[t,[p,j]] for j,p in enumerate(PARENTS)
                if p>=0 and mask[p] and mask[j]],np.float32).reshape(-1,2,3)
            masks=self.available&(self.confidence>=self.threshold.value)
            self.trails.points=trajectory_segments(self.positions,masks,t);self.trails.visible=self.trails_visible.value
            owners=self.owners[t];keep=owners>=0;self.current.points=self.frame_points[t,keep]
            colors=np.tile([175,185,195],(keep.sum(),1)).astype(np.uint8)
            colors[(owners[keep]>0)&(owners[keep]<100)]=[230,185,45];colors[owners[keep]>100]=[205,55,145]
            self.current.colors=colors;self.current.visible=self.current_visible.value;self.static.visible=self.memory_visible.value
            kind='发布真实标定' if self.case['camera_kind']=='released_calibrated' else '**合成观察相机；非真实相机benchmark**'
            self.info.content=f"`{self.case['sequence_id']}`\n\n{kind} / {self.case['view_id']}。\n\n"
            self.info.content+=f"源帧 {self.case['source_first']+t}；窗口帧 {t}/191；30Hz。有效关节 {self.available[t].sum()}/22，显示 {mask.sum()}/22。\n\n"
            self.info.content+=f"整窗有输入 {self.window_observed_frames}/192 帧；平均 {self.window_mean_joints:.2f} 个关节；最长完全无输入 {self.window_max_empty_frames} 帧（{self.window_max_empty_frames/30:.2f} 秒）。\n\n"
            self.info.content+=f"相机位置（米）{np.asarray(self.meta['exo_camera']['position']).round(3).tolist()}；本窗口视点固定。切换序列会切换观察相机。\n\n"
            if self.window_max_empty_frames>18:
                self.info.content+='**存在超过0.6秒的完全无关节观测区间，包含长缺失补全；不能据此要求唯一GT恢复。** 0.6秒仅与当前噪声增广长度对照，不是新采样阈值。\n\n'
            self.info.content+=f"遮挡皮肤像素：人 {self.parts['person'][t].sum()}；静态 {self.parts['static'][t].sum()}；刚体 {self.parts['rigid'][t].sum()}。\n\n"
            self.info.content+='位置来自GT可见部位模拟；无exo关节旋转输入。灰色点只含同视点已见表面；空白为未知。完整人/场景mesh仅供参考，不进入场景条件。显示阈值不修改训练配置。'
            self.info.content+=f"\n\n原生mesh与缓存关节最大误差 {self.geometry['maximum_mesh_joint_error_m']*1000:.4f} mm。"
            rows=['|关节|confidence|输入|可见/部位像素|人/静态/刚体遮挡|','|---|---:|:---:|---:|---:|']
            for j in range(22):
                rows.append(f"|{j}|{self.confidence[t,j]:.2f}|{'✓' if self.available[t,j] else '—'}|"
                    f"{self.parts['visible'][t,j]}/{self.parts['reference'][t,j]}|"
                    f"{self.parts['person'][t,j]}/{self.parts['static'][t,j]}/{self.parts['rigid'][t,j]}|")
            self.table.content='\n'.join(rows)+'\n\n部位没有参考像素可能来自自身遮挡、出画或低分辨率，不能归为外部遮挡。'

    def run(self):
        print('FORMAL_NATIVE30_VISER_READY',flush=True)
        while True:
            time.sleep(1/30)
            if self.play.value:
                with self.lock:
                    frame,playing=playback_next(int(self.frame.value),192)
                    self.frame.value=frame;self.play.value=playing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=HERE/'data/native30_official_memory_v3/manifest.jsonl')
    p.add_argument('--host',default='0.0.0.0');p.add_argument('--port',type=int,default=8784)
    p.add_argument('--cases-output',type=Path);p.add_argument('--cases-file',type=Path)
    p.add_argument('--snapshot-only',type=Path);args=p.parse_args()
    if args.snapshot_only:
        save_snapshot(args.manifest,args.snapshot_only,args.cases_file);return
    CacheViewer(args.manifest,args.host,args.port,args.cases_output,args.cases_file).run()


if __name__=='__main__':main()
