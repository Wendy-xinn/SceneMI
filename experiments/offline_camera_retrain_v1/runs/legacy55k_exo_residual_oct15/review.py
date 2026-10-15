"""Full-panel plots and self-contained synchronized skeleton replay."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,PARENTS

ROOT=Path(__file__).resolve().parent


def main(prediction_file='predictions.pt',output_prefix=''):
    torch.set_num_threads(2)
    report=json.loads((ROOT/'RESULTS.json').read_text())
    identities=json.loads((ROOT/'IDENTITIES.json').read_text())['validation']
    data=torch.load(ROOT/prediction_file,map_location='cpu',weights_only=False)
    bank=torch.load(ROOT/'replay_inputs.pt',map_location='cpu',weights_only=False)['validation']
    motions=dict(initial=data['initial'],no_scene=data['outputs']['no_scene'],scene=data['outputs']['scene'],truth=data['truth'])
    joints={k:forward_kinematics(v,data['rest']).numpy() for k,v in motions.items()}
    cases=[]
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for axis,regime in zip(axes,('clean','noisy')):
        ix=[i for i,r in enumerate(identities) if r['regime']==regime]
        for name,color in (('initial','#999999'),('no_scene','#c0399e'),('scene','#168e68')):
            errors=np.linalg.norm(joints[name]-joints['truth'],axis=-1)*1000
            values=[errors[i][data['missing'][i].numpy()].mean() for i in ix]
            axis.plot(range(len(ix)),values,'o-',color=color,label=name)
        axis.set_xticks(range(len(ix)),[identities[i]['group']+' '+str(identities[i]['start']) for i in ix],rotation=55,ha='right',fontsize=7)
        axis.set_ylabel('Occluded-joint MPJPE (mm)');axis.set_title(regime+' | all registered windows');axis.grid(alpha=.2);axis.legend()
    fig.savefig(ROOT/(output_prefix+'OCCLUSION_RESULTS.png'),dpi=160);plt.close(fig)
    for i,identity in enumerate(identities):
        step=slice(None,None,2)
        points=joints['initial'][i,:, :,None,:]+bank['scene'][i].numpy().reshape(192,22,2,12)[...,:3]
        geom=bank['scene'][i].numpy().reshape(192,22,2,12)
        cases.append(dict(label=identity['group']+' | '+identity['regime']+' | '+identity['sequence_id']+' | missing '+str(round(100*identity['missing_fraction'],1))+'%',
            joints={k:np.round(v[i,step],4).tolist() for k,v in joints.items()},
            observed=np.round(bank['trajectory'][i,step,:,:3].numpy()*2,4).tolist(),
            available=(bank['observation_meta'][i,step,:,0].numpy()>0).astype(int).tolist(),
            points=np.round(points[step].reshape(96,-1,3),3).tolist(),
            scene_valid=geom[step,...,7].reshape(96,-1).astype(int).tolist(),
            scene_other=geom[step,...,11].reshape(96,-1).astype(int).tolist()))
    payload=json.dumps(dict(cases=cases,parents=[int(p) for p in PARENTS],status=report['status']),separators=(',',':'))
    html='''<!doctype html><meta charset="utf-8"><title>55k EXO residual review</title>
<style>body{font:15px system-ui;margin:18px;background:#fafafa}select{max-width:95vw}#panels{display:flex;gap:10px}article{flex:1;min-width:0}canvas{width:100%;background:white;border:1px solid #ddd}button,input,select{margin:6px}small{color:#444}</style>
<h2>55k + EXO residual: development review</h2><div id="status"></div>
<select id="case"></select><br><button id="play">Play</button><input id="frame" type="range" min="0" max="95" value="0" style="width:45vw"><span id="time"></span>
<br>Yaw <input id="yaw" type="range" min="-180" max="180" value="25"> Pitch <input id="pitch" type="range" min="-70" max="70" value="15"> Zoom <input id="zoom" type="range" min="0.5" max="3" step="0.1" value="1.4">
<p><small>Blue = GT; magenta = prediction; orange = available observations. Gray/cyan = nearest observed scene patches (cyan: other person). Sparse patches are not a full scene scan. Display sampled at15Hz, metrics at30Hz. Known shape, ideal visible-joint frontend simulation; legacy55k overlap prevents clean official benchmark claims.</small></p>
<div id="panels"><article>Frozen55k initialization<canvas id="initial" width="600" height="650"></canvas></article><article>Residual trained without scene<canvas id="no_scene" width="600" height="650"></canvas></article><article>Residual trained with scene<canvas id="scene" width="600" height="650"></canvas></article></div>
<script>const D=PAYLOAD;const $=x=>document.getElementById(x);D.cases.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=c.label;$('case').append(o)});$('status').textContent=D.status;
let playing=false,last=0;const names=['initial','no_scene','scene'];
function draw(){let c=D.cases[+$('case').value],f=+$('frame').value,y=+$('yaw').value*Math.PI/180,p=+$('pitch').value*Math.PI/180;
let centre=c.joints.truth[f][0].slice();centre[1]-=.1;let scale=180*+$('zoom').value;
function project(q){let x=q[0]-centre[0],h=q[1]-centre[1],z=q[2]-centre[2],a=x*Math.cos(y)+z*Math.sin(y),b=-x*Math.sin(y)+z*Math.cos(y);return [300+a*scale,325-(h*Math.cos(p)-b*Math.sin(p))*scale]}
for(let name of names){let ctx=$(name).getContext('2d');ctx.clearRect(0,0,600,650);c.points[f].forEach((q,j)=>{if(c.scene_valid[f][j]){let a=project(q);ctx.fillStyle=c.scene_other[f][j]?'#00aab0':'#b0b5bd';ctx.fillRect(a[0]-2,a[1]-2,4,4)}});
function skeleton(j,color,w){ctx.strokeStyle=color;ctx.lineWidth=w;for(let k=1;k<22;k++){let a=project(j[k]),b=project(j[D.parents[k]]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}}
skeleton(c.joints.truth[f],'#3976c0',3);skeleton(c.joints[name][f],'#c328a0',3);c.observed[f].forEach((q,j)=>{if(c.available[f][j]){let a=project(q);ctx.fillStyle='#e68a16';ctx.beginPath();ctx.arc(...a,3,0,Math.PI*2);ctx.fill()}})}$('time').textContent=(f/15).toFixed(2)+' s / 6.4 s'}
for(let id of ['case','frame','yaw','pitch','zoom'])$(id).oninput=draw;$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause':'Play'};function tick(t){if(playing&&t-last>1000/15){$('frame').value=(+$('frame').value+1)%96;last=t;draw()}requestAnimationFrame(tick)};draw();requestAnimationFrame(tick);
</script>'''.replace('PAYLOAD',payload)
    (ROOT/(output_prefix+'REPLAY.html')).write_text(html)
    print('Review ready',ROOT/(output_prefix+'OCCLUSION_RESULTS.png'),ROOT/(output_prefix+'REPLAY.html'))


if __name__=='__main__':main()
