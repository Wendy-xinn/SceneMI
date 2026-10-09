"""Join old/new predictions only when fixed clips, truth and camera match."""
import argparse
import json
from pathlib import Path

from experiments.offline_camera_retrain_v1.gallery import write_gallery


def read_gallery(path):
    return json.loads(Path(path).read_text().split('const data=', 1)[1].split(';\nconst $=', 1)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old, new = read_gallery(args.old), read_gallery(args.new)
    if len(old['cases']) != len(new['cases']):
        raise ValueError('Different clip counts')
    merged, walking = [], []
    for i, (a, b) in enumerate(zip(old['cases'], new['cases'])):
        for key in ('identity', 'group', 'camera'):
            if a[key] != b[key]:
                raise ValueError(f'Clip {i}: mismatched {key}')
        if a['tracks']['GT'] != b['tracks']['GT']:
            raise ValueError(f'Clip {i}: changed ground truth')
        overlap = (a['tracks'].keys() & b['tracks'].keys()) - {'GT'}
        if overlap:
            raise ValueError(f'Duplicate variants: {overlap}')
        case = dict(a, tracks={**a['tracks'], **b['tracks']}, metrics={**a['metrics'], **b['metrics']}, old_index=i)
        metric = next(iter(a['metrics'].values()))
        case['gt_path_m'] = metric['gt_root_path_length_m']
        merged.append(case)
        # Same GT-only criterion as earlier reports: never select by a model's
        # apparent improvement. Includes all five previously reported walks.
        if metric['gt_root_path_length_m'] > 1 and metric['gt_foot_separation_rms_cm'] > 10:
            walking.append(case)
    args.output.mkdir(parents=True, exist_ok=False)
    write_gallery(args.output / 'all12.html', merged, '原固定12段 · 30k / 34k / 54k / 55k')
    manifest = dict(old=str(args.old), new=str(args.new), verified_identical=['identity', 'group', 'camera', 'GT'],
                    selection='GT root path > 1m AND GT foot separation RMS > 10cm; no prediction-based selection',
                    old_indices=[c['old_index'] for c in walking], identities=[c['identity'] for c in walking])
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    template = r'''<!doctype html><meta charset="utf-8">
<title>原来的五段走动 · 同步对比</title>
<style>body{font:16px system-ui;background:#111827;color:#e5e7eb;margin:20px}a{color:#7dd3fc}select,button,input{padding:5px;margin:6px}canvas{width:100%;background:#172033;border:1px solid #475569}p{max-width:1400px}pre{white-space:pre-wrap}label{display:inline-block}</style>
<h2>原来的五段走动：真值 / 旧版本 / 新版本同步播放</h2>
<p>使用之前固定12段中的全部5段行走代理样本（原编号0、1、3、7、8）。序列、起始帧、相机轨迹、真值均逐项核对一致；DDIM20、采样种子777与之前相同。没有按新模型表现挑片段。</p>
<p>默认中间为你之前看的脚步修复版34k，右边为新全身协调版55k。判断新增协调损失的作用，请把中间切换到“原损失对照55k”。三栏共享时间、视角和比例尺，未对动作重定位或重定时。左肢青色、右肢紫色、躯干蓝色；橙线是输入相机轨迹。</p>
<label>旧片段<select id="case"></select></label>
<label>中间<select id="before"></select></label><label>右边<select id="after"></select></label>
<label>视角<select id="view"><option value="oblique">斜视</option><option value="side">侧视x/y</option><option value="front">正视z/y</option><option value="top">俯视x/z</option></select></label>
<button id="play">暂停</button><button id="prev">上一帧</button><button id="next">下一帧</button>
<label>速度<select id="speed"><option value="1">1倍</option><option value="0.5">0.5倍</option></select></label>
<input id="frame" type="range" min="0" max="127" value="0"><span id="time"></span>
<label><input id="trails" type="checkbox" checked>脚迹</label>
<canvas id="canvas" width="1500" height="620"></canvas><pre id="info"></pre>
<p>网格为GT脚关节高度参考，不是真实地面；同相/反相不能单独判断所有动作的自然性。<a href="all12.html">完整原12段、所有版本</a> · <a href="manifest.json">片段一致性核对记录</a></p>
<script>
const data=PAYLOAD;
const $=id=>document.getElementById(id),ctx=$('canvas').getContext('2d');
const parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19];
const left=new Set([1,4,7,10,13,16,18,20]),right=new Set([2,5,8,11,14,17,19,21]);
const labels={'original_ddim20_seed777':'最初原版30k','support_v2_ddim20_seed777':'之前脚步修复34k','warmstart_54k_ddim20_seed777':'本轮起点54k','control_55k_ddim20_seed777':'原损失对照55k','coordination_55k_ddim20_seed777':'新全身协调55k'};
data.cases.forEach((c,i)=>$('case').add(new Option(`原#${c.old_index} · ${c.identity.sequence_id} · 帧${c.identity.source_start_30fps} · GT走过${c.gt_path_m.toFixed(1)}m`,i)));
for(const k of Object.keys(data.cases[0].tracks).filter(k=>k!=='GT'))for(const id of ['before','after'])$(id).add(new Option(labels[k]||k,k));
$('before').value='support_v2_ddim20_seed777';$('after').value='coordination_55k_ddim20_seed777';
let frame=0,playing=true,last=0,bounds;
function project(p){switch($('view').value){case 'side':return[p[0],p[1]];case 'front':return[p[2],p[1]];case 'top':return[p[0],p[2]];default:return[.8*p[0]-.6*p[2],p[1]-.25*(p[0]+p[2])]}}
function update(){const c=data.cases[+$('case').value];let lo=[Infinity,Infinity],hi=[-Infinity,-Infinity];
for(const track of [...Object.values(c.tracks),c.camera.map(p=>[p])])for(const pose of track)for(const p of pose){const q=project(p);for(let d=0;d<2;d++){lo[d]=Math.min(lo[d],q[d]);hi[d]=Math.max(hi[d],q[d])}}
const heights=c.tracks.GT.flatMap(p=>[p[10][1],p[11][1]]).sort((a,b)=>a-b),roots=c.tracks.GT.map(p=>p[0]);
bounds={center:lo.map((v,d)=>(v+hi[d])/2),scale:Math.min(440/Math.max(.5,hi[0]-lo[0]),490/Math.max(.5,hi[1]-lo[1])),height:heights[Math.floor(.05*(heights.length-1))],xz:[0,2].map(d=>[Math.min(...roots.map(p=>p[d]))-.3,Math.max(...roots.map(p=>p[d]))+.3])};
$('frame').max=c.tracks.GT.length-1;frame=Math.min(frame,c.tracks.GT.length-1);draw()}
function xy(p,panel){const q=project(p);return[panel*500+250+(q[0]-bounds.center[0])*bounds.scale,335-(q[1]-bounds.center[1])*bounds.scale]}
function line(a,b,color,width,panel){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...xy(a,panel));ctx.lineTo(...xy(b,panel));ctx.stroke()}
function draw(){ctx.clearRect(0,0,1500,620);const c=data.cases[+$('case').value];
['GT',$('before').value,$('after').value].forEach((name,panel)=>{ctx.save();ctx.beginPath();ctx.rect(panel*500,0,500,620);ctx.clip();ctx.fillStyle='#e5e7eb';ctx.font='20px system-ui';ctx.fillText(name==='GT'?'真值 GT':labels[name]||name,panel*500+20,32);
const [[x0,x1],[z0,z1]]=bounds.xz,y=bounds.height;for(let i=0;i<=8;i++){let x=x0+(x1-x0)*i/8,z=z0+(z1-z0)*i/8;line([x,y,z0],[x,y,z1],'#334155',1,panel);line([x0,y,z],[x1,y,z],'#334155',1,panel)}
for(let t=1;t<c.camera.length;t++)line(c.camera[t-1],c.camera[t],'#b98130',1,panel);
const track=c.tracks[name],pose=track[frame];if($('trails').checked)for(let t=Math.max(1,frame-19);t<=frame;t++){line(track[t-1][10],track[t][10],'#22d3ee',1.5,panel);line(track[t-1][11],track[t][11],'#c084fc',1.5,panel)}
for(let j=1;j<22;j++)line(pose[parents[j]],pose[j],left.has(j)?'#22d3ee':right.has(j)?'#c084fc':'#60a5fa',3,panel);
pose.forEach((p,j)=>{ctx.fillStyle=left.has(j)?'#22d3ee':right.has(j)?'#c084fc':'#60a5fa';ctx.beginPath();ctx.arc(...xy(p,panel),3,0,Math.PI*2);ctx.fill()});ctx.restore()});
$('frame').value=frame;$('time').textContent=`${frame}/${c.tracks.GT.length-1}帧 · ${(frame/20).toFixed(2)}s`;
$('info').textContent=`原编号 ${c.old_index} | ${c.identity.sequence_id} | 源起始帧 ${c.identity.source_start_30fps} | GT根部路径 ${c.gt_path_m.toFixed(2)}m`}
function pause(){playing=false;$('play').textContent='播放'}
for(const id of ['case','before','after','view'])$(id).onchange=update;
$('trails').onchange=draw;$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'暂停':'播放'};
$('frame').oninput=()=>{pause();frame=+$('frame').value;draw()};
for(const [id,delta] of [['prev',-1],['next',1]])$(id).onclick=()=>{pause();const n=data.cases[+$('case').value].tracks.GT.length;frame=(frame+delta+n)%n;draw()};
function tick(now){if(playing&&now-last>=50/+$('speed').value){frame=(frame+1)%data.cases[+$('case').value].tracks.GT.length;last=now;draw()}requestAnimationFrame(tick)}
update();requestAnimationFrame(tick);
</script>'''
    payload = json.dumps(dict(cases=walking), ensure_ascii=True).replace('<', '\\u003c')
    (args.output / 'gallery.html').write_text(template.replace('PAYLOAD', payload))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
