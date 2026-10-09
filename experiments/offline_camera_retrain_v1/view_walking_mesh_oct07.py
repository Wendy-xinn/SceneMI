"""Faithful SMPL playback of archived paired evaluations; no pose corrections."""
import inspect,json,time
from pathlib import Path
import numpy as np
for key,value in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('unicode',str),('str',str)]:
    if key not in np.__dict__:setattr(np,key,value)
if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
import torch,smplx,viser
from experiments.offline_camera_retrain_v1.data import OfflineSceneMIData,PARENTS
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d
from experiments.offline_sequence_v1.data_loader import anchor_rotation, camera_cloud
OUT=Path(__file__).parent/'runs/report55k_fullval_oct07'
def main():
    torch.set_num_threads(4)
    rows=[json.loads(l) for l in (OUT/'rows.jsonl').read_text().splitlines() if l]
    data=OfflineSceneMIData('validation',skeleton_profile='canonical_smpl',rich_source='legacy5interp')
    cases=[];audits=[]
    for idx in [1271,2240,1121]:
        row=next(r for r in rows if r['window']['index']==idx);w=row['window'];name=w['sequence_id']
        sample,identity=data.sample(w['length'],w['group'],sequence_index=w['sequence_index'],start_index=w['start_index'])
        assert identity==row['identity']
        arrays=dict(np.load(OUT/row['motion_file']));meta=data.pose_meta[name]
        model=smplx.SMPL('/home/wenxin/projects/ProtoMotions/data/smpl/SMPL_'+meta['gender'].upper()+'.pkl',num_betas=10)
        betas=torch.as_tensor(meta['betas'],dtype=torch.float32).reshape(1,-1)[:,:10].expand(w['length'],-1)
        vertices={};errors={}
        for mode,key in [('有场景生成','scene_on'),('无场景生成','scene_off'),('GT 参数','truth')]:
            motion=arrays[key+'_motion'] if key!='truth' else np.asarray(sample['motion'])
            motion=torch.as_tensor(motion,dtype=torch.float32)
            local=rotation_from_6d(motion[:,3:135].reshape(-1,22,6))
            hands=torch.eye(3).reshape(1,1,3,3).expand(len(motion),2,3,3)
            with torch.no_grad():out=model(betas=betas,global_orient=local[:,:1],body_pose=torch.cat((local[:,1:],hands),1),transl=motion[:,:3]*2,pose2rot=False)
            vertices[mode]=out.vertices.numpy()
            if key!='truth':errors[key]=float(np.max(np.linalg.norm(out.joints.numpy()[:,:22]-arrays[key],axis=-1)))
        assert max(errors.values())<1e-4,errors
        prepared=data.prepared[name];_,p,r,_=data.base._load(prepared)
        offset=int(round((identity['source_start_30fps']-prepared.get('source_first_frame',0))/1.5))
        anchor=anchor_rotation(r[offset:offset+w['length']])
        camera=(p[offset:offset+w['length']]-p[offset])@anchor.T
        rotations=np.einsum('ij,tjk->tik',anchor,r[offset:offset+w['length']])
        points=np.load(data.visible_meta[name]['path']);points=(points-p[offset])@anchor.T
        visible=[]
        for pos,rot in zip(camera,rotations):
            pc,mask=camera_cloud(points,np.zeros_like(points),pos,rot,3072,90,70,.05,4,selection='pixel_uniform')
            visible.append(pc[mask,:3]@rot.T+pos)
        center=arrays['truth'][:,0].mean(0);points=points[np.linalg.norm(points[:,[0,2]]-center[[0,2]],axis=1)<4]
        path=float(np.linalg.norm(np.diff(arrays['scene_on'][:,0][:,[0,2]],axis=0),axis=-1).sum())
        cases.append(dict(row=row,arrays=arrays,vertices=vertices,faces=model.faces,points=points,path=path,camera=camera,rotations=rotations,visible=visible))
        audits.append(dict(index=idx,max_mesh_joint_error_m=errors,generated_root_path_m=path))
    (OUT/'walking_mesh_audit.json').write_text(json.dumps(audits,indent=2))
    server=viser.ViserServer(port=8794);server.scene.set_up_direction('+y')
    server.initial_camera.position=(3,2,4);server.initial_camera.look_at=(0,-.7,0)
    colors={'有场景生成':(55,190,120),'无场景生成':(225,80,70),'GT 参数':(65,140,235)}
    meshes={name:server.scene.add_mesh_simple('/body_'+str(i),cases[0]['vertices'][name][0],cases[0]['faces'],color=color,opacity=.55,side='double') for i,(name,color) in enumerate(colors.items())}
    cloud=server.scene.add_point_cloud('/environment',cases[0]['points'],colors=(155,163,171),point_size=.014)
    server.gui.add_markdown('### 旧 55k：三套 SMPL 同坐标叠加\n\n绿＝有场景生成，红＝无场景生成，蓝＝GT 参数 mesh。三者同步播放，保持原始位置和姿态，没有额外对齐或贴地。可独立隐藏、调整透明度和线框。\n\n同一头部条件、同一初始噪声。形状使用序列已知 betas，末端手关节设为单位旋转。环境为实际条件点云的显示裁剪。前两例展示走动；第三例展示场景影响，不作为避障证据。')
    trajectory=server.scene.add_line_segments('/camera_path',np.zeros((127,2,3),np.float32),colors=(185,65,235),thickness=3,thickness_units='screen')
    frustum=server.scene.add_line_segments('/camera_frustum',np.zeros((8,2,3),np.float32),colors=(185,65,235),thickness=3,thickness_units='screen')
    current_points=server.scene.add_point_cloud('/current_visible',np.zeros((1,3),np.float32),colors=(255,190,35),point_size=.025)
    server.gui.add_markdown('紫色：真实相机轨迹与当前视锥（90°×70°，视锥图标长度 0.4 m）。黄色：从累计可见地图按当前相机、0.05–4 m 深度和 z-buffer 重新投影的点；不是原始逐帧点云，不含人体遮挡复算。灰色：累计可见地图的显示裁剪。')
    opts=['走动：EgoBody 1271（约 2.19 m）','走动：EgoBody 2240（约 1.35 m）','场景影响：EgoBody 1121']
    select=server.gui.add_dropdown('片段',options=opts,initial_value=opts[0])
    visibility={name:server.gui.add_checkbox('显示 '+name,initial_value=True) for name in colors}
    opacity={name:server.gui.add_slider(name+' 透明度',min=.1,max=1.,step=.05,initial_value=.55) for name in colors}
    wire={name:server.gui.add_checkbox(name+' 线框',initial_value=(name=='GT 参数')) for name in colors}
    frame=server.gui.add_slider('帧',min=0,max=127,step=1,initial_value=0)
    play=server.gui.add_checkbox('播放 20 FPS',initial_value=True)
    camera_show=server.gui.add_checkbox('显示相机轨迹与视锥',initial_value=True)
    visible_show=server.gui.add_checkbox('显示当前相机重投影可见点',initial_value=True)
    env=server.gui.add_checkbox('显示环境',initial_value=True);info=server.gui.add_markdown('')
    def update():
        c=cases[opts.index(select.value)];t=int(frame.value)
        with server.atomic():
            for name,mesh in meshes.items():
                mesh.vertices=c['vertices'][name][t];mesh.faces=c['faces'];mesh.visible=visibility[name].value;mesh.opacity=float(opacity[name].value);mesh.wireframe=wire[name].value
            cloud.points=c['points'];cloud.visible=env.value
            cam=c['camera'];rot=c['rotations'][t]
            trajectory.points=np.stack((cam[:-1],cam[1:]),axis=1);trajectory.visible=camera_show.value
            d=.4;h=d*np.tan(np.deg2rad(70)/2)
            corners=np.array([[-d,-h,d],[d,-h,d],[d,h,d],[-d,h,d]],np.float32)@rot.T+cam[t]
            frustum.points=np.array([[cam[t],corners[i]] for i in range(4)]+[[corners[i],corners[(i+1)%4]] for i in range(4)]);frustum.visible=camera_show.value
            current_points.points=c['visible'][t];current_points.visible=visible_show.value
            on=c['row']['scene_on'];off=c['row']['scene_off']
            info.content=f'窗口 {c["row"]["window"]["index"]} · 6.4 秒 · 20 FPS\n\n生成骨盆水平路径 {c["path"]:.2f} m\n\nMPJPE 有／无：{on["mpjpe_cm"]:.2f} / {off["mpjpe_cm"]:.2f} cm\n\nWA-MPJPE 有／无：{on["wa_mpjpe_cm"]:.2f} / {off["wa_mpjpe_cm"]:.2f} cm'
    for h in [select,frame,env,camera_show,visible_show,*visibility.values(),*opacity.values(),*wire.values()]:h.on_update(lambda _:update())
    update();print('http://127.0.0.1:8794',flush=True)
    (OUT/'walking_mesh_viser.pid').write_text(str(__import__('os').getpid()))
    while True:
        time.sleep(.05)
        if play.value:frame.value=(int(frame.value)+1)%128
if __name__=='__main__':main()
