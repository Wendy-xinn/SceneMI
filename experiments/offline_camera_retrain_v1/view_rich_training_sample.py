"""Inspect an actual RICH window from the current training adapter."""
import json,time
from pathlib import Path
import numpy as np
import torch
import viser
from experiments.offline_camera_retrain_v1.data import OfflineSceneMIData,PARENTS
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_sequence_v1.data_loader import anchor_rotation

def main():
    data=OfflineSceneMIData('train',skeleton_profile='canonical_smpl',rich_source='native20_faceout_oct07')
    name='Pavallion_006_sidebalancerun';length=128;start=0
    members=[(m,v) for m,v in data.rich_members if v[length]]
    index=next(i for i,(m,v) in enumerate(members) if m['sequence_id']==name)
    sample,identity=data.sample(length,'rich',sequence_index=index,start_index=start)
    folder=data.rich_root/name;meta=json.loads((folder/'metadata.json').read_text())
    pos=np.load(folder/'camera_position_scenemi_yup.npy')[:length];rot=np.load(folder/'camera_rotation_scenemi_yup.npy')[:length]
    anchor=anchor_rotation(rot);origin=pos[0]
    points=(np.load(folder/'visible_static_points_scenemi_yup.npy')-origin)@anchor.T
    joints=sample['joints']*2;camera=sample['camera'][:,:3]*2
    rotation=np.stack((sample['camera'][:,3:6],sample['camera'][:,6:9],np.cross(sample['camera'][:,3:6],sample['camera'][:,6:9])),axis=-1)
    fk=forward_kinematics(torch.from_numpy(sample['motion'])[None],torch.from_numpy(sample['rest'])[None])[0].numpy()
    error=np.linalg.norm(fk-joints,axis=-1)
    assert all(np.isfinite(v).all() for v in sample.values())
    assert error.max()<1e-4
    assert np.allclose(camera[0],0,atol=1e-6)
    edges=np.array([(PARENTS[i],i) for i in range(1,22)])
    vox=np.argwhere(sample['occupancy']>0);center=camera.mean(0)-[0,.7,0]
    voxels=(vox[:,[1,0,2]]+.5)*[.26666667,.2,.26666667]+center-[6.4,2.4,6.4]
    out=Path(__file__).parent/'runs/rich_training_input_review_oct07';out.mkdir(exist_ok=True)
    report=dict(identity=identity,source_folder=str(folder),length=length,fps=20,fk_max_m=float(error.max()),visible_points=len(points),occupied_voxels=len(voxels),tensor_shapes={k:list(v.shape) for k,v in sample.items()})
    (out/'audit.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'training_sample.npz',**sample)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(13,6));frame=64
    for n,(title,cloud,color) in enumerate([('Actual visible static union',points,'#a5a69c'),('Actual occupancy voxel centres',voxels,'#edac40')]):
        ax=fig.add_subplot(1,2,n+1,projection='3d');q=cloud[::max(1,len(cloud)//8000)]
        ax.scatter(q[:,0],q[:,2],q[:,1],s=1,c=color,alpha=.25)
        for u,v in edges:
            b=joints[frame,[u,v]];ax.plot(b[:,0],b[:,2],b[:,1],color='#1689e5',lw=2)
        c=camera[frame];end=c+.7*rotation[frame,:,2]
        ax.plot([c[0],end[0]],[c[2],end[2]],[c[1],end[1]],c='red',lw=3)
        mid=joints[frame].mean(0);r=2.3
        ax.set(xlim=(mid[0]-r,mid[0]+r),ylim=(mid[2]-r,mid[2]+r),zlim=(mid[1]-1.3,mid[1]+1.7),xlabel='X (m)',ylabel='Z (m)',zlabel='Y up (m)',title=title);ax.view_init(18,35)
    fig.suptitle(f'{name} | actual train loader | 128 frames @ 20 Hz | frame 64');fig.tight_layout();fig.savefig(out/'training_input.png',dpi=170);plt.close(fig)
    server=viser.ViserServer(port=8791);server.scene.set_up_direction('+y')
    mid=joints[64].mean(0);server.initial_camera.position=tuple(mid+[2,1.2,2]);server.initial_camera.look_at=tuple(mid);server.initial_camera.up=(0,1,0)
    skeleton=server.scene.add_line_segments('/canonical_smpl_gt',joints[0,edges],colors=(40,155,240),thickness=4,thickness_units='screen')
    cloud=server.scene.add_point_cloud('/actual_visible_union',points,colors=(235,175,40),point_size=.012)
    occupancy=server.scene.add_point_cloud('/actual_occupancy',voxels.astype(np.float32),colors=(90,200,180),point_size=.08);occupancy.visible=False
    frustum=server.scene.add_line_segments('/camera_frustum',np.zeros((8,2,3),np.float32),colors=(240,65,40),thickness=2,thickness_units='screen')
    axis=server.scene.add_line_segments('/optical_plus_z',np.zeros((1,2,3),np.float32),colors=(190,40,220),thickness=3,thickness_units='screen')
    source=Path(meta['scene_bundle']);source_mesh=np.load(source/'body_vertices_scenemi_yup.npy',mmap_mode='r')
    source_ids=np.load(source/'source_frame_ids.npy');target_ids=np.load(folder/'source_frame_ids.npy')[:length]
    mesh_indices=np.abs(target_ids[:,None]-source_ids[None]).argmin(1)
    mesh=server.scene.add_mesh_simple('/source_smplx_diagnostic',(source_mesh[mesh_indices[0]]-origin)@anchor.T,np.load(source/'body_faces.npy'),color=(145,165,180),opacity=.45)
    server.gui.add_markdown(f'### 当前训练的 RICH 输入\n{name}\n\n128 帧 × 20 Hz（6.4 秒窗口）。蓝色=canonical SMPL GT 骨架；红色=相机视锥；紫色=实际 +Z 光轴；橙色=训练读取的整段可见静态并集。\n\n灰色网格仅为最近 5 Hz 原始 SMPL-X 外观参考，非训练 SMPL GT，可关闭。occupancy 为实际 24×48×48 输入；BPS 已保存在 training_sample.npz。')
    slider=server.gui.add_slider('训练窗口帧',min=0,max=length-1,step=1,initial_value=64)
    playing=server.gui.add_checkbox('播放',initial_value=False)
    show_cloud=server.gui.add_checkbox('可见静态并集',initial_value=True)
    show_vox=server.gui.add_checkbox('实际 occupancy',initial_value=False)
    show_mesh=server.gui.add_checkbox('原始 SMPL-X 参考网格（5 Hz）',initial_value=True)
    ego=server.gui.add_button('切到实际训练虚拟相机');reset=server.gui.add_button('外部视角');info=server.gui.add_markdown('');ego_ids=set()
    @show_cloud.on_update
    def _(_):cloud.visible=show_cloud.value
    @show_vox.on_update
    def _(_):occupancy.visible=show_vox.value
    @show_mesh.on_update
    def _(_):mesh.visible=show_mesh.value
    def set_ego(client,t):
        client.camera.position=tuple(camera[t]);client.camera.look_at=tuple(camera[t]+rotation[t,:,2]);client.camera.up_direction=tuple(rotation[t,:,1]);client.camera.fov=np.deg2rad(40.49)
    @ego.on_click
    def _(_):
        mesh.visible=False;show_mesh.value=False
        for client in server.get_clients().values():ego_ids.add(client.client_id);set_ego(client,int(slider.value))
    @reset.on_click
    def _(_):
        ego_ids.clear()
        for client in server.get_clients().values():client.camera.position=tuple(mid+[2,1.2,2]);client.camera.look_at=tuple(mid);client.camera.up_direction=(0,1,0)
    def update(t):
        with server.atomic():
            skeleton.points=joints[t,edges];mesh.vertices=(source_mesh[mesh_indices[t]]-origin)@anchor.T
            c=camera[t];r=rotation[t];tx=np.tan(np.deg2rad(66.56/2));ty=np.tan(np.deg2rad(40.49/2))
            corners=.7*np.array([[-tx,-ty,1],[tx,-ty,1],[tx,ty,1],[-tx,ty,1]])@r.T+c
            frustum.points=np.concatenate((np.stack((np.tile(c,(4,1)),corners),1),np.stack((corners,np.roll(corners,-1,axis=0)),1))).astype(np.float32)
            axis.points=np.stack((c,c+.45*r[:,2]))[None].astype(np.float32)
            info.content=f'窗口帧 {t}，源 30 fps 帧 {target_ids[t]:g}；FK 最大误差 {error.max()*100:.6f} cm；可见点 {len(points)}；占据格 {len(voxels)}'
            for cid,client in server.get_clients().items():
                if cid in ego_ids:set_ego(client,t)
    @slider.on_update
    def _(_):update(int(slider.value))
    update(64);print(json.dumps(report,ensure_ascii=False),flush=True);print('http://127.0.0.1:8791',flush=True)
    while True:
        time.sleep(.05)
        if playing.value:slider.value=(int(slider.value)+1)%length
if __name__=='__main__':main()
