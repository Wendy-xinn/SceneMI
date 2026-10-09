"""Mesh overlay of native GT and a low-noise fitting diagnostic (not generation)."""
import json,time
import numpy as np
import viser
from experiments.offline_camera_retrain_v1.data import HERE
OUT=HERE/'runs/native_body_coexistence_trial_oct07'
def main():
 groups=['trumans','camera_wearer','interactee','rich'];cases={g:(dict(np.load(OUT/f'{g}_native_gt.npz')),dict(np.load(OUT/f'{g}_low_noise_fit.npz'))) for g in groups}
 server=viser.ViserServer(port=8795);server.scene.set_up_direction('+y');server.initial_camera.position=(2,1,3);server.initial_camera.look_at=(0,-.6,0)
 gt,pred=cases[groups[0]];a=server.scene.add_mesh_simple('/native_gt',gt['vertices'][0],gt['faces'],color=(65,145,235),opacity=.65,wireframe=True);b=server.scene.add_mesh_simple('/fit',pred['vertices'][0],pred['faces'],color=(40,190,110),opacity=.6)
 server.gui.add_markdown('### 原生 SMPL / SMPL-X 共存诊断\n\n蓝线框＝原生 GT mesh；绿色＝同一 U-Net 的低噪声拟合结果。TRUMANS/RICH 用 SMPL-X，EgoBody 用 SMPL。\n\n这是4个片段、300步过拟合验证，不是自由动作生成，也不是泛化结果。仅预测22个躯干关节，手指和表情未预测，展示使用默认手脸姿态。')
 select=server.gui.add_dropdown('数据',options=groups,initial_value='rich');frame=server.gui.add_slider('帧',min=0,max=63,step=1,initial_value=0);play=server.gui.add_checkbox('播放20FPS',initial_value=True);show_gt=server.gui.add_checkbox('GT',initial_value=True);show_pred=server.gui.add_checkbox('拟合结果',initial_value=True)
 def update():
  gt,pred=cases[select.value];t=int(frame.value)
  with server.atomic():
   a.vertices=gt['vertices'][t];a.faces=gt['faces'];a.visible=show_gt.value;b.vertices=pred['vertices'][t];b.faces=pred['faces'];b.visible=show_pred.value
 for h in [select,frame,show_gt,show_pred]:h.on_update(lambda _:update())
 update();print('http://127.0.0.1:8795',flush=True)
 while True:
  time.sleep(.05)
  if play.value:frame.value=(int(frame.value)+1)%64
if __name__=='__main__':main()
