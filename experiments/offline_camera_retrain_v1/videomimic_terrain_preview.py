"""VideoMimic released G1 terrain + virtual ego camera, NOT SMPL training data.

Read authoritative xyzw/world link transforms. The G1 head_link origin is near
the torso: put the sensor on the actual head mesh, not on that frame's origin.
Keep source rate for this preview; metadata fps is not certified RGB timing.
"""
import argparse, json, time, xml.etree.ElementTree as ET
from pathlib import Path
import h5py, numpy as np, trimesh
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.scene_visibility_v2 import render_scene, sample_scene_hits, static_ray_scene

H=Path(__file__).parent;VM=H.parents[2]/'VideoMimic'
CAPTURES=VM/'simulation/data/videomimic_captures'
URDF=VM/'simulation/videomimic_gym/resources/robots/g1_description/g1_23dof.urdf'
OUT=H/'runs/videomimic_terrain_oct11'
DEFAULT='megahunter_megasam_reconstruction_results_IMG_8044_cam01_frame_0_350_subsample_1'

def visuals():
    result={}
    for link in ET.parse(URDF).getroot().findall('link'):
        meshes=[]
        for visual in link.findall('visual'):
            node=visual.find('geometry/mesh')
            if node is None:continue
            m=trimesh.load(URDF.parent/node.attrib['filename'],force='mesh',process=False)
            origin=visual.find('origin');xyz=np.zeros(3);rpy=np.zeros(3)
            if origin is not None:
                xyz=np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ');rpy=np.fromstring(origin.attrib.get('rpy','0 0 0'),sep=' ')
            scale=np.fromstring(node.attrib.get('scale','1 1 1'),sep=' ')
            m.vertices=(m.vertices*scale)@Rotation.from_euler('xyz',rpy).as_matrix().T+xyz;meshes.append(m)
        if meshes:result[link.attrib['name']]=trimesh.util.concatenate(meshes)
    return result

def load(sequence):
    folder=CAPTURES/sequence
    with h5py.File(folder/'retarget_poses_g1.h5') as f:
        names=[str(x) for x in f.attrs['/link_names']];positions=f['link_pos'][:].astype(np.float32);rotations=Rotation.from_quat(f['link_quat'][:].reshape(-1,4)).as_matrix().reshape(*positions.shape[:2],3,3).astype(np.float32);fps=float(f.attrs['/fps'])
    if not np.isfinite(positions).all():raise ValueError('Nonfinite robot positions')
    mesh=trimesh.load(folder/'background_mesh.obj',force='mesh',process=False);robot=visuals();head=robot['head_link'];hi=names.index('head_link')
    mount=np.array([head.bounds[1,0]+.02,head.vertices[:,1].mean(),np.quantile(head.vertices[:,2],.8)],np.float32)
    # Existing SceneMI renderer: proper SO(3), +Z optical, +Y image up.
    basis=np.array([[0,0,1],[1,0,0],[0,1,0]],np.float32)@Rotation.from_euler('x',20,degrees=True).as_matrix().astype(np.float32)
    cp=positions[:,hi]+np.einsum('tij,j->ti',rotations[:,hi],mount);cr=rotations[:,hi]@basis
    return folder,mesh,robot,names,positions,rotations,cp,cr,fps,mount

def build(sequence):
    OUT.mkdir(exist_ok=True);folder,mesh,robot,names,p,r,cp,cr,fps,mount=load(sequence);static=static_ray_scene(mesh.vertices,mesh.faces);points=[];owners=[];depth=[];start=time.monotonic()
    for frame in range(len(p)):
        occluders=[(robot[name].vertices@r[frame,i].T+p[frame,i],robot[name].faces,100) for i,name in enumerate(names) if name in robot and name!='head_link']
        rendered=render_scene(occluders,cp[frame],cr[frame],static_scene=static);visible,labels=sample_scene_hits(rendered);points.append(visible);owners.append(labels);depth.append(rendered['depth'])
    dest=OUT/'g1_ego_preview.npz';np.savez_compressed(dest,camera_position=cp,camera_rotation=cr,visible_points=points,owners=owners,depth=depth)
    fractions=[float((o==0).mean()) for o in owners]
    audit=dict(sequence=sequence,source=str(folder),body='G1 robot, NOT SMPL or original human',frames=len(p),header_fps=fps,source_rate_preserved=True,
        timing='header FPS only, not certified source RGB timing; export hardcodes30/subsample and simulation config contains overrides. Resolve before20Hz training.',
        gravity_axis='world+Z, retained for preview',quat='xyzw',camera_mount_in_head_visual_frame_m=mount.tolist(),mount_reason='head_link origin is near torso; mesh extent places actual head0.325..0.531m above frame',
        camera_SO3_max_error=float(np.abs(cr.transpose(0,2,1)@cr-np.eye(3)).max()),camera_clearance_m=.02,
        camera_protocol='virtual sensor rigidly mounted on moving G1 head mesh; proper +Z optical / +Y up renderer frame; fixed20deg downward mounting pitch for terrain preview, not a human head calibration',
        geometry='first-hit mesh raycasting; static background plus same-frame robot occlusion with head removed; body owners100 are occluders, never scene hits; no accumulated body or dynamic trails',
        fov_degrees=[66.56,40.49],depth_resolution=[128,96],depth_range_m=[.05,4.],visible_scene_fraction_mean=float(np.mean(fractions)),terrain_height_span_m=float(np.ptp(mesh.vertices[:,2])),elapsed_s=time.monotonic()-start,
        current55k_training_ready=False,missing='native human SMPL parameters, reliable source timestamps and human/world alignment')
    (OUT/'ego_preview_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2),flush=True)

def view(sequence,port):
    import viser, threading
    folder,mesh,robot,names,p,r,cp,cr,fps,mount=load(sequence)
    with np.load(OUT/'g1_ego_preview.npz') as a:points=a['visible_points'];owners=a['owners'];depth=a['depth']
    server=viser.ViserServer(port=port);server.scene.set_up_direction('+z');server.scene.add_mesh_simple('/terrain',vertices=np.asarray(mesh.vertices,np.float32),faces=mesh.faces,color=(155,158,160),opacity=.8,side='double');handles={}
    for i,name in enumerate(names):
        if name in robot:handles[i]=server.scene.add_mesh_simple('/robot/'+name,vertices=robot[name].vertices,faces=robot[name].faces,color=(75,150,215))
    def quat(m):
        q=Rotation.from_matrix(m).as_quat();return q[[3,0,1,2]]
    # OpenCV frustum from this renderer's image-up basis. Flip image both
    # axes for the displayed depth texture; training observations stay raw.
    cv=np.diag([-1.,-1.,1.]);frustum=server.scene.add_camera_frustum('/virtual_ego',fov=np.deg2rad(40.49),aspect=np.tan(np.deg2rad(66.56/2))/np.tan(np.deg2rad(40.49/2)),scale=.35,color=(200,50,210),wxyz=quat(cr[0]@cv),position=cp[0])
    visible=server.scene.add_point_cloud('/visible',points=points[0][owners[0]==0],colors=(250,190,30),point_size=.012)
    server.scene.add_line_segments('/camera_path',points=np.stack((cp[:-1],cp[1:]),1),colors=(180,40,205),line_width=2)
    server.gui.add_markdown('**VideoMimic地形与虚拟ego相机验收**\n\n蓝色是官方G1参考轨迹；不是SMPL人体生成结果。黄色为当前可见地形点。')
    slider=server.gui.add_slider('帧',min=0,max=len(p)-1,step=1,initial_value=0);playing=server.gui.add_checkbox('播放',True);follow=server.gui.add_checkbox('ego视角',False);show=server.gui.add_checkbox('可见点',True);rate=server.gui.add_slider('播放FPS（H5记录值）',min=1,max=60,step=1,initial_value=int(fps));depth_image=server.gui.add_image(np.zeros((96,128,3),np.uint8),label='虚拟ego深度：未知为黑色');lock=threading.Lock()
    @server.on_client_connect
    def connect(client):
        client.camera.position=cp[0]+np.array([3,-3,2]);client.camera.look_at=cp[0];client.camera.up_direction=np.array([0,0,1.])
    def update(frame):
        with lock,server.atomic():
            for i,handle in handles.items():handle.position=p[frame,i];handle.wxyz=quat(r[frame,i])
            frustum.position=cp[frame];frustum.wxyz=quat(cr[frame]@cv);visible.points=points[frame][owners[frame]==0];visible.visible=show.value
            d=depth[frame];a=np.clip(np.where(np.isfinite(d),d/4,0),0,1);img=np.stack((a,a,a),-1);depth_image.image=(img*255).astype(np.uint8)
            if follow.value:
                for client in server.get_clients().values():client.camera.position=cp[frame];client.camera.look_at=cp[frame]+cr[frame,:,2];client.camera.up_direction=cr[frame,:,1]
    @slider.on_update
    def changed(_):update(int(slider.value))
    update(0);print('READY',port,sequence,flush=True)
    while True:
        if playing.value:slider.value=(int(slider.value)+1)%len(p)
        time.sleep(1/rate.value)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--sequence',default=DEFAULT);a.add_argument('--build',action='store_true');a.add_argument('--port',type=int,default=8781);args=a.parse_args()
    if args.build:build(args.sequence)
    else:view(args.sequence,args.port)
