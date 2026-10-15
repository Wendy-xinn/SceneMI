"""Portable30Hz fixed-panel replay: previous hard guard vs current continuation."""
import ast
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,PARENTS

ROOT=Path(__file__).resolve().parent
PILOT=ROOT.parent/'legacy55k_exo_residual_oct15'


def key(row):return (row['group'],row['sequence_id'],row['start'],row['regime'])


def main():
    torch.set_num_threads(2)
    latest=torch.load(ROOT/'validation_latest.pt',map_location='cpu',weights_only=False)
    bank=torch.load(ROOT/'validation_inputs.pt',map_location='cpu',weights_only=False)
    old=torch.load(PILOT/'reference_anchor/predictions_protected.pt',map_location='cpu',weights_only=False)
    old_ids=json.loads((PILOT/'reference_anchor/IDENTITIES.json').read_text())['validation']
    old_map={key(r):i for i,r in enumerate(old_ids)}
    cases=[]
    for index,(batch,row) in enumerate(bank):
        if row['panel']!='fixed':continue
        previous=old_map[key(row)]
        torch.testing.assert_close(batch['motion'][0],old['truth'][previous],atol=1e-6,rtol=1e-6)
        motions=dict(initial=old['outputs']['scene'][previous:previous+1],
            no_scene=latest['prediction'][index:index+1],scene=batch['initial'],truth=batch['motion'])
        joints={k:forward_kinematics(v,batch['rest'])[0].numpy() for k,v in motions.items()}
        geometry=batch['scene'][0].numpy().reshape(192,22,2,12)
        points=joints['scene'][:,:,None]+geometry[...,:3]
        cases.append(dict(label=row['group']+' | '+row['regime']+' | '+row['sequence_id'],
            joints={k:np.round(v,4).tolist() for k,v in joints.items()},
            observed=np.round(batch['trajectory'][0,:,:,:3].numpy()*2,4).tolist(),
            available=(batch['observation_meta'][0,:,:,0].numpy()>0).astype(int).tolist(),
            points=np.round(points.reshape(192,-1,3),3).tolist(),
            scene_valid=geometry[...,7].reshape(192,-1).astype(int).tolist(),
            scene_other=geometry[...,11].reshape(192,-1).astype(int).tolist()))
    payload=json.dumps(dict(cases=cases,parents=[int(p) for p in PARENTS],status='回放评估步数：'+str(latest['step'])),separators=(',',':'))
    tree=ast.parse((PILOT/'review.py').read_text())
    template=next(node.value for node in ast.walk(tree) if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.startswith('<!doctype html>'))
    html=template.replace('PAYLOAD',payload).replace('max="95"','max="191"')
    html=html.replace('(f/15).toFixed(2)','(f/30).toFixed(2)')
    html=html.replace('let playing=false,last=0;','let playing=false,playStart=0,playFrame=0;')
    html=html.replace("playing=!playing;$('play').textContent", "playing=!playing;if(playing){playStart=performance.now();playFrame=+$('frame').value;}$('play').textContent")
    html=html.replace("function tick(t){if(playing&&t-last>1000/15){$('frame').value=(+$('frame').value+1)%96;last=t;draw()}requestAnimationFrame(tick)}", "function tick(t){if(playing){let f=(playFrame+Math.floor((t-playStart)*.03))%192;if(+$('frame').value!==f){$('frame').value=f;draw()}}requestAnimationFrame(tick)}")
    html=html.replace('55k + EXO residual: development review','55k＋exo 后台续训对照')
    html=html.replace('Frozen55k initialization','旧版1000步（硬保护）')
    html=html.replace('Residual trained without scene','新版续训 '+str(latest['step'])+' 步')
    html=html.replace('Residual trained with scene','新版平滑初始参考')
    html=html.replace('Blue = GT; magenta = prediction; orange = available observations. Gray/cyan = nearest observed scene patches (cyan: other person). Sparse patches are not a full scene scan. Display sampled at15Hz, metrics at30Hz. Known shape, ideal visible-joint frontend simulation; legacy55k overlap prevents clean official benchmark claims.',
        '蓝色＝GT；品红＝各栏预测／参考；橙色＝输入可见关节。灰／青色＝观测场景patch。显示原始30Hz帧，以实际时间播放。三栏共用观察视角，预测坐标未对齐。左中比较同时包含方法与训练步数变化。刷新页面可载入下一次评估。')
    html=html.replace('<select id="case">','<div id="training"></div><select id="case">')
    html=html.replace('</script>',"async function progress(){try{let s=await (await fetch('STATUS.json',{cache:'no-store'})).json();$('training').textContent='训练：'+s.status+'，当前 '+s.latest_update_step+' / '+s.max_step+'，已保存 '+s.checkpoint_step;}catch(e){}}progress();setInterval(progress,10000);</script>")
    target=ROOT/'REPLAY.html';temp=target.with_suffix('.html.tmp');temp.write_text(html);temp.replace(target)
    (ROOT/'VIEWER_STATUS.json').write_text(json.dumps(dict(step=latest['step'],cases=len(cases),fps=30,raw_frames=192,
        url='http://127.0.0.1:8791/REPLAY.html'),indent=2)+'\n')
    print('VIEWER',latest['step'],len(cases),flush=True)


if __name__=='__main__':main()
