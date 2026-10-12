"""Standalone input animation: observed scene, GT reference, trusted tracks."""
import json
from pathlib import Path
import numpy as np
import torch

from experiments.offline_camera_retrain_v1.supervision import PARENTS
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations


def preview(cases,folder):
    folder=Path(folder);records=[]
    for label,item,identity in cases:
        sample,scene,_,mode=item;motion=torch.from_numpy(sample['motion'][None])
        rotations=global_rotations(motion)[0].numpy()
        static=scene.points[::max(1,len(scene.points)//4000)]
        dynamic=[cloud['points'][::max(1,len(cloud['points'])//80)] for cloud in scene.dynamic]
        record=dict(label=label,mode=mode,sequence=identity['sequence_id'],source_start=identity['source_start_30fps'],
            static=static.tolist(),dynamic=[p.tolist() for p in dynamic],body=(sample['joints']*2).tolist(),
            tracks=(sample['trajectory'][...,:3]*2).tolist(),valid=(sample['observation_meta'][...,0]>0).tolist(),
            camera=sample['camera'].tolist(),head_forward=rotations[:,15,:,2].tolist(),
            root_forward=rotations[:,0,:,2].tolist(),mount_uncertainty=sample['mount_uncertainty'].tolist())
        records.append(record)
    # Compact numeric precision is display-only; audited source arrays retain float32.
    def rounded(value):
        if isinstance(value,float):return round(value,4)
        if isinstance(value,list):return [rounded(v) for v in value]
        if isinstance(value,dict):return {k:rounded(v) for k,v in value.items()}
        return value
    data=json.dumps(rounded(records),separators=(',',':'),ensure_ascii=False)
    html='''<!doctype html><meta charset="utf-8"><title>30Hz ego/exo 输入对齐</title>
<style>body{margin:0;background:#f5f6f8;color:#263442;font:16px system-ui}main{max-width:1250px;margin:20px auto}h1{font-size:23px}canvas{width:100%;height:650px;background:white;border:1px solid #ccd3dd;touch-action:none}select,button,input{font:inherit;margin:8px}#info{white-space:pre-line}small{line-height:1.7}</style>
<main><h1>30Hz / 192 帧：相机、可信关节与可见场景</h1>
<small>此处回放输入与参考标注，未展示未训练模型的生成质量。灰色＝该 6.4 秒视角看过的静态点云，青色＝当前可见动态表面；蓝色骨架＝GT 参考，橙色＝可信输入。紫色箭头＝参考 pelvis 朝向，绿色＝参考头部朝向，红色＝观察相机光轴。拖动旋转视角，滚轮缩放。</small><br>
<select id="case"></select><button id="play">播放</button><input id="frame" type="range" min="0" max="191" value="0"><span id="time"></span><canvas id="view"></canvas><p id="info"></p></main>
<script>const cases=__DATA__, parents=__PARENTS__;
const canvas=document.getElementById('view'),ctx=canvas.getContext('2d'),select=document.getElementById('case'),slider=document.getElementById('frame');
let az=-.7,el=.3,zoom=90,playing=false,drag=null;
cases.forEach((x,i)=>{let o=document.createElement('option');o.value=i;o.textContent=x.label;select.add(o)}); select.value=cases.findIndex(x=>x.label==='validation/camera_wearer/exo');
function draw(){let c=cases[+select.value],t=+slider.value;canvas.width=canvas.clientWidth*devicePixelRatio;canvas.height=650*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);ctx.clearRect(0,0,canvas.clientWidth,650);let w=canvas.clientWidth,center=c.body[t][0];
function project(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],u=Math.cos(az)*x-Math.sin(az)*z,v=Math.sin(az)*x+Math.cos(az)*z;return [w/2+u*zoom,340-(Math.cos(el)*y-Math.sin(el)*v)*zoom]}
function dot(p,color,r){let a=project(p);ctx.fillStyle=color;ctx.beginPath();ctx.arc(a[0],a[1],r,0,7);ctx.fill()}
function line(a,b,color,width){a=project(a);b=project(b);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}
function arrow(p,v,color){let q=p.map((x,i)=>x+.45*v[i]);line(p,q,color,4);dot(q,color,4)}
c.static.forEach(p=>dot(p,'#9ca3af',1));c.dynamic[t].forEach(p=>dot(p,'#04a9af',2));
for(let j=1;j<22;j++)line(c.body[t][parents[j]],c.body[t][j],'#367dcc',3);c.body[t].forEach(p=>dot(p,'#367dcc',3));
let n=0;for(let j=0;j<22;j++)if(c.valid[t][j]){dot(c.tracks[t][j],'#ed960c',6);n++}
let camera=c.camera[t],p=camera.slice(0,3).map(x=>x*2),a=camera.slice(3,6),b=camera.slice(6,9),f=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
dot(p,'#d64242',7);arrow(p,f,'#d64242');arrow(c.body[t][15],c.head_forward[t],'#39a159');arrow(c.body[t][0],c.root_forward[t],'#8a4dba');
document.getElementById('time').textContent=`第 ${t} 帧 · ${(t/30).toFixed(2)} 秒`;
document.getElementById('info').textContent=`${c.sequence}\n原始 30Hz 起点 ${c.source_start}，当前可信位置关节 ${n}/22。${c.mode==='exo'?'exo 未提供关节旋转，目标皮肤仅用于参考显示和可见性模拟，不进入场景。':'ego 光学相机已经转换为解剖学 head；相机与 head 安装误差会保留在监督尺度中。'}\n静态点云只使用本窗口的可见表面；空白区域视为未知。`;
}
select.onchange=draw;slider.oninput=draw;document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'暂停':'播放'};
setInterval(()=>{if(playing){slider.value=(+slider.value+1)%192;draw()}},1000/30);
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointerup=()=>drag=null;canvas.onpointermove=e=>{if(drag){az+=(e.clientX-drag[0])*.008;el=Math.max(-1.3,Math.min(1.3,el+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];draw()}};
canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(15,Math.min(600,zoom*Math.exp(-e.deltaY*.001)));draw()};window.onresize=draw;draw();</script>'''
    (folder/'INPUT_REPLAY.html').write_text(html.replace('__DATA__',data).replace('__PARENTS__',json.dumps(PARENTS)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[r for r in records if r['label'].startswith('validation/')]
    fig=plt.figure(figsize=(16,10))
    for i,c in enumerate(chosen):
        ax=fig.add_subplot(2,3,i+1,projection='3d');pts=np.asarray(c['static'])
        if len(pts):ax.scatter(pts[:,0],pts[:,2],pts[:,1],s=.3,c='gray',alpha=.25)
        body=np.asarray(c['body'][96]);track=np.asarray(c['tracks'][96]);valid=np.asarray(c['valid'][96])
        for j in range(1,22):
            edge=body[[PARENTS[j],j]];ax.plot(edge[:,0],edge[:,2],edge[:,1],c='royalblue',lw=2)
        ax.scatter(track[valid,0],track[valid,2],track[valid,1],c='orange',s=30)
        dynamic=np.asarray(c['dynamic'][96])
        if len(dynamic):ax.scatter(dynamic[:,0],dynamic[:,2],dynamic[:,1],c='cyan',s=3)
        p=np.asarray(c['camera'][96][:3])*2;ax.scatter(*p[[0,2,1]],c='red',s=40,marker='^')
        root=body[0];head=body[15]
        for pos,forward,color in [(root,c['root_forward'][96],'purple'),(head,c['head_forward'][96],'green')]:
            ax.quiver(*pos[[0,2,1]],*np.asarray(forward)[[0,2,1]],length=.4,color=color)
        ax.set(title=c['label'],xlabel='X (m)',ylabel='Z (m)',zlabel='Y (m)')
        ax.view_init(elev=18,azim=-65)
    fig.suptitle('Raw30 input alignment — reference body, trusted joints, observed surfaces; no trained predictions')
    fig.tight_layout();fig.savefig(folder/'INPUT_ALIGNMENT.png',dpi=140);plt.close(fig)
