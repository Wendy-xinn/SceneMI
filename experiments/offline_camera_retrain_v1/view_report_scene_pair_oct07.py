"""Display saved, noise-paired scene-on/off generations and actual conditions."""
import json,time
from pathlib import Path
import numpy as np
import viser
from experiments.offline_camera_retrain_v1.data import OfflineSceneMIData,PARENTS
from experiments.offline_sequence_v1.data_loader import anchor_rotation

HERE=Path(__file__).parent;OUT=HERE/'runs/report55k_fullval_oct07'
def select_cases(rows):
    # Illustrative cases, intentionally selected for a visible intervention effect.
    candidates=[r for r in rows if r['window']['length']==128 and r['window']['group']!='rich' and r['scene_on']['head_cm']<35 and r['scene_on']['mpjpe_cm']<40 and r['scene_off']['head_cm']<30 and r['root_relative_scene_effect_cm']>8]
    def score(r):return (r['scene_off']['support_floating_m']-r['scene_on']['support_floating_m'])*100+r['root_relative_scene_effect_cm']*.5
    chosen=[next(r for r in candidates if r['window']['index']==idx) for idx in [1121,558] if any(r['window']['index']==idx for r in candidates)];seen={r['window']['sequence_id'] for r in chosen}
    for r in sorted(candidates,key=score,reverse=True):
        key=r['window']['sequence_id']
        if key in seen:continue
        chosen.append(r);seen.add(key)
        if len(chosen)==8:break
    if not chosen:chosen=sorted(rows,key=lambda r:r['root_relative_scene_effect_cm'],reverse=True)[:4]
    return chosen

def main():
    rows=[json.loads(l) for l in (OUT/'rows.jsonl').read_text().splitlines() if l];chosen=select_cases(rows)
    (OUT/'viser_selected_cases.json').write_text(json.dumps(chosen,indent=2))
    data=OfflineSceneMIData('validation',skeleton_profile='canonical_smpl',rich_source='legacy5interp')
    edges=np.array([(PARENTS[i],i) for i in range(1,22)])
    server=viser.ViserServer(port=8792);server.scene.set_up_direction('+y')
    server.initial_camera.position=(4,2,4);server.initial_camera.look_at=(2,-.7,0);server.initial_camera.up=(0,1,0)
    skeletons={v:server.scene.add_line_segments('/'+v,np.zeros((21,2,3),np.float32),colors=color,thickness=4,thickness_units='screen') for v,color in [('scene_on',(30,195,95)),('scene_off',(235,70,60)),('truth',(40,145,235))]}
    cloud_on=server.scene.add_point_cloud('/scene_on_visible',np.zeros((1,3),np.float32),colors=(170,175,170),point_size=.015)
    cloud_off=server.scene.add_point_cloud('/scene_off_reference_only',np.zeros((1,3),np.float32),colors=(170,175,170),point_size=.015)
    camera_paths=[server.scene.add_line_segments(f'/camera_trajectory_{i}',np.zeros((127,2,3),np.float32),colors=(180,70,220),thickness=2,thickness_units='screen') for i in range(2)]
    labels=[server.scene.add_label('/on_label','有场景：绿色生成'),server.scene.add_label('/off_label','无场景：红色生成')]
    server.gui.add_markdown('### 旧三数据集 55k：场景输入消融\n\n同一权重、同一窗口、同一头部条件、完全相同初始扩散噪声。左侧保留 occupancy+BPS，右侧关闭两条场景分支。右侧灰点仅是共同环境的显示参照，没有输入无场景模型。\n\n绿=有场景，红=无场景，蓝=GT，紫=头部相机轨迹。展示片段按场景影响较明显筛选，用来说明条件能改变生成；不是随机代表样本，也不等于已验证避障。')
    options=[f'{i}: {r["window"]["group"]} / {r["window"]["sequence_id"]} / {r["window"]["index"]}' for i,r in enumerate(chosen)]
    selector=server.gui.add_dropdown('示例',options=options,initial_value=options[0])
    frame=server.gui.add_slider('帧',min=0,max=127,step=1,initial_value=64);play=server.gui.add_checkbox('播放',initial_value=False)
    overlay=server.gui.add_checkbox('同坐标重叠对比',initial_value=False)
    show_gt=server.gui.add_checkbox('显示 GT',initial_value=True);show_cloud=server.gui.add_checkbox('显示环境参照',initial_value=True)
    info=server.gui.add_markdown('');state={}
    def load_case(i):
        row=chosen[i];w=row['window'];sample,identity=data.sample(w['length'],w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);assert identity==row['identity']
        with np.load(OUT/row['motion_file']) as arrays:state.update({k:arrays[k].copy() for k in ('truth','scene_on','scene_off','camera')})
        if w['group']=='rich':
            folder=data.rich_root/w['sequence_id'];pos=np.load(folder/'camera_position_scenemi_yup.npy')[w['start_index']];rot=np.load(folder/'camera_rotation_scenemi_yup.npy')[w['start_index']:w['start_index']+w['length']];points=np.load(folder/'visible_static_points_scenemi_yup.npy');anchor=anchor_rotation(rot)
        else:
            meta=data.prepared[w['sequence_id']];j,p,r,_=data.base._load(meta);offset=int(round((identity['source_start_30fps']-meta.get('source_first_frame',0))/1.5));pos=p[offset];rot=r[offset:offset+w['length']];anchor=anchor_rotation(rot)
            static=np.load(data.visible_meta[w['sequence_id']]['path']);dynamic=data.base._object_points(meta,identity['source_start_30fps'],p[offset:offset+w['length']],rot);points=np.concatenate((static,dynamic))
        state['points']=(points-pos)@anchor.T;state['row']=row;state['sample']=sample;update(int(frame.value))
    def update(t):
        if not state:return
        shift=np.array([0,0,0] if overlay.value else [4,0,0],np.float32)
        with server.atomic():
            for name,handle in skeletons.items():handle.points=(state[name]+(shift if name=='scene_off' else 0))[t,edges]
            skeletons['truth'].visible=show_gt.value
            cloud_on.points=state['points'];cloud_off.points=state['points']+shift;cloud_on.visible=show_cloud.value;cloud_off.visible=show_cloud.value and not overlay.value
            for i,h in enumerate(camera_paths):
                c=state['camera']+(shift if i else 0);h.points=np.stack((c[:-1],c[1:]),1);h.visible=not(i and overlay.value)
            labels[0].position=tuple(state['truth'][t,15]+[0,.4,0]);labels[1].position=tuple(state['scene_off'][t,15]+shift+[0,.4,0])
            row=state['row'];on=row['scene_on'];off=row['scene_off']
            info.content=f'窗口 {row["window"]["index"]}；帧 {t}/127；6.4 秒\n\nMPJPE：有 {on["mpjpe_cm"]:.2f} / 无 {off["mpjpe_cm"]:.2f} cm\n\nWA-MPJPE：有 {on["wa_mpjpe_cm"]:.2f} / 无 {off["wa_mpjpe_cm"]:.2f} cm\n\n头误差：有 {on["head_cm"]:.2f} / 无 {off["head_cm"]:.2f} cm\n\n支撑悬空代理：有 {on["support_floating_m"]*100:.2f} / 无 {off["support_floating_m"]*100:.2f} cm\n\n场景开关导致的全身差异 {row["scene_effect_cm"]:.2f} cm；去除每帧骨盆平移后的姿态差异 {row["root_relative_scene_effect_cm"]:.2f} cm。'
            info.content+=f'\n\n双脚水平相对位置变化 RMS：有 {on["foot_separation_rms_cm"]:.2f} / 无 {off["foot_separation_rms_cm"]:.2f} / GT {on["gt_foot_separation_rms_cm"]:.2f} cm。'
    @selector.on_update
    def _(_):load_case(options.index(selector.value))
    @frame.on_update
    def _(_):update(int(frame.value))
    @overlay.on_update
    def _(_):update(int(frame.value))
    @show_gt.on_update
    def _(_):update(int(frame.value))
    @show_cloud.on_update
    def _(_):update(int(frame.value))
    (OUT/'viser.pid').write_text(str(__import__('os').getpid()))
    load_case(0);print('http://127.0.0.1:8792',flush=True)
    while True:
        time.sleep(.05)
        if play.value:frame.value=(int(frame.value)+1)%128
if __name__=='__main__':main()
