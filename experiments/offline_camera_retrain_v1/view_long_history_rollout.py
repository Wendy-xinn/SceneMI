"""8s native mesh comparisons, including failed candidates; thin arrays only."""
import argparse,json,time,threading
import numpy as np
import viser
from scipy.spatial.transform import Rotation
from pathlib import Path
from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from experiments.offline_camera_retrain_v1.data import anchor_rotation
from experiments.offline_camera_retrain_v1.evaluate_long_history_rollout import OUT,LABELS

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8780);args=parser.parse_args()
    identities=json.loads((OUT/'protocol.json').read_text())['identities'];server=viser.ViserServer(port=args.port);server.scene.set_up_direction('+y')
    server.gui.add_markdown('**8秒连续执行对照，全部新候选未通过脚步门槛。** 蓝GT、红同模型单次192帧生成、绿所选候选。身体只在最初0.8秒由GT初始化，之后用各自生成执行块反馈；头部是理想GT条件，无tracker/真实物理。场景动态点只显示当前帧，不累计。')
    cases=[f'{i} · {d["group"]} · {d["sequence_id"]}' for i,d in enumerate(identities)]
    case=server.gui.add_dropdown('录制',options=cases,initial_value=cases[0]);rep=server.gui.add_dropdown('种子',options=['0','1'],initial_value='0')
    variant=server.gui.add_dropdown('候选',options=[*LABELS[1:],'roll128_warm400'],initial_value='roll128_spline')
    play=server.gui.add_checkbox('播放',initial_value=True);frame=server.gui.add_slider('帧',min=0,max=175,step=1,initial_value=16)
    checks={k:server.gui.add_checkbox(k,initial_value=True) for k in ['GT蓝','参考红','候选绿','静态记忆','当前可见点','相机与轨迹']}
    opacity=server.gui.add_slider('透明度',min=.1,max=1,step=.1,initial_value=.5);info=server.gui.add_markdown('')
    cache={};models={};active=None;lock=threading.RLock()
    rows=json.loads((OUT/'rows.json').read_text())+json.loads((OUT/'warm_rows.json').read_text())
    def update_unlocked():
        nonlocal active,cache
        i=int(case.value.split(' · ')[0]);r=int(rep.value);key=(i,r,variant.value)
        if key!=active:
            active=key;server.scene.reset();d=identities[i];body=d['native_body'];modelkey=(body['model'],body['gender'])
            if modelkey not in models:models[modelkey]=load_model(*modelkey)
            with np.load(OUT/f'motions_rep{r}.npz') as archive:
                motions={'gt':archive['gt'][i],'reference':archive['single192_raw'][i]}
                if variant.value!='roll128_warm400':motions['candidate']=archive[variant.value][i]
            if variant.value=='roll128_warm400':
                with np.load(OUT/f'warm_motions_rep{r}.npz') as archive:motions['candidate']=archive['roll128_warm400'][i]
            cache={'vertices':{},'handles':{}}
            for name,color in [('gt',(40,135,245)),('reference',(240,65,60)),('candidate',(45,205,90))]:
                v,f,_=decode_native_mesh(motions[name],body,models[modelkey]);cache['vertices'][name]=v
                cache['handles'][name]=server.scene.add_mesh_simple('/'+name,v[int(frame.value)],f,color=color,opacity=opacity.value)
            folder=Path(d['scene_bundle']);ids=np.load(folder/'source_frame_ids.npy');ix=int(np.searchsorted(ids,d['source_start_30fps']));camera=np.load(folder/'camera_position_scenemi_yup.npy')[ix:ix+176];rot=np.load(folder/'camera_rotation_scenemi_yup.npy')[ix:ix+176];anchor=anchor_rotation(rot);origin=camera[0]
            cache.update(folder=folder,start=ix,ids=ids[ix:ix+176],anchor=anchor,origin=origin,camera=(camera-origin)@anchor.T,rotation=anchor[None]@rot)
            cache['points']=np.load(folder/'visible_frame_points_scenemi_yup.npy',mmap_mode='r');cache['owners']=np.load(folder/'visible_frame_owner.npy',mmap_mode='r')
            memory=Path(d['memory_bundle']);p=np.load(memory/'static_points.npy',mmap_mode='r');times=np.load(memory/'first_observed_source_frames.npy',mmap_mode='r');known=times<=cache['ids'][-1];p=p[known];times=times[known];take=np.linspace(0,len(p)-1,min(len(p),30000)).astype(int)
            cache.update(static=(p[take]-origin)@anchor.T,static_times=times[take])
            cache['handles']['static']=server.scene.add_point_cloud('/static',np.empty((0,3),np.float32),colors=(150,160,150),point_size=.008)
            cache['handles']['visible']=server.scene.add_point_cloud('/visible',np.empty((0,3),np.float32),colors=(240,190,40),point_size=.012)
            cache['handles']['path']=server.scene.add_line_segments('/camera_path',np.stack((cache['camera'][:-1],cache['camera'][1:]),1),colors=(180,45,215),thickness=2,thickness_units='screen')
            cache['handles']['camera']=server.scene.add_camera_frustum('/camera',fov=1.,aspect=1.4,scale=.15,color=(180,45,215))
            center=cache['vertices']['gt'][16].mean(0);server.initial_camera.look_at=tuple(center);server.initial_camera.position=tuple(center+[2,1,2])
            for client in server.get_clients().values():client.camera.look_at=tuple(center);client.camera.position=tuple(center+[2,1,2])
            cache['row']=next(x for x in rows if x['identity']['sequence_id']==d['sequence_id'] and x['label']==variant.value and x['replicate']==r)
            cache['reference_row']=next(x for x in rows if x['identity']['sequence_id']==d['sequence_id'] and x['label']=='single192_raw' and x['replicate']==r)
        t=int(frame.value);handles=cache['handles']
        for name,label in [('gt','GT蓝'),('reference','参考红'),('candidate','候选绿')]:
            handles[name].vertices=cache['vertices'][name][t];handles[name].visible=checks[label].value;handles[name].opacity=opacity.value
        handles['static'].points=cache['static'][cache['static_times']<=cache['ids'][t]];handles['static'].visible=checks['静态记忆'].value
        owner=cache['owners'][cache['start']+t];p=cache['points'][cache['start']+t];handles['visible'].points=(p[(owner>=0)&(owner<100)]-cache['origin'])@cache['anchor'].T;handles['visible'].visible=checks['当前可见点'].value
        handles['camera'].position=cache['camera'][t];handles['camera'].wxyz=np.roll(Rotation.from_matrix(cache['rotation'][t]).as_quat(),1)
        handles['camera'].visible=handles['path'].visible=checks['相机与轨迹'].value
        old=cache['reference_row']['full'];new=cache['row']['full'];info.content=f"帧{t}，执行时间{(t-16)/20:.2f}s。指标为完整8秒。\n\n脚滑 红{old['gt_stance_slide_cm_frame']:.3f} / 绿{new['gt_stance_slide_cm_frame']:.3f} cm/帧\n\n世界MPJPE 红{old['mpjpe_cm']:.2f} / 绿{new['mpjpe_cm']:.2f} cm\n\n候选仅用于检查失败，绿色不表示验收通过。"
    def update():
        with lock:update_unlocked()
    for control in [case,rep,variant,frame,opacity,*checks.values()]:control.on_update(lambda _:update())
    update();print(f'READY http://127.0.0.1:{args.port}',flush=True)
    while True:
        time.sleep(.05)
        if play.value:frame.value=16 if int(frame.value)>=175 else int(frame.value)+1
if __name__=='__main__':main()
